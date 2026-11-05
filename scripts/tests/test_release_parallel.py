"""Release checks; run from any directory after building the native libraries.

python scripts/tests/test_release_parallel.py [--json path] [--installed]
Approximate comparisons use fresh processes to isolate inherited native state.
"""

import argparse
import ctypes
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
INSTALLED = "--installed" in sys.argv
if not INSTALLED:
    sys.path.insert(0, str(ROOT / "source" / "python"))

from fast_apspn_kernel import (  # noqa: E402
    brandes_apspn_kernel,
    cc_apspn_kernel_approx,
    cc_apspn_kernel_par,
    cc_sebccopts_apspn_kernel,
    cm_gc,
)

EXACT_MODES = ("apspn", "bc", "gc", "phd", "apslp")
OBSERVATIONS = {"exact": {}, "approximate": {}}


def fixture(name):
    return loadmat(Path(__file__).parent / "fixtures" / (name + ".mat"))["x"].tocsr()


def reference_distance(n):
    index = np.arange(1, n + 1, dtype=float)
    xy = np.column_stack((0.19 * index + 0.003 * index**2,
                          np.sin(0.61 * index) + 0.07 * np.cos(1.13 * index)
                          + 0.017 * index**2))
    return np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(axis=2))


def representation(x, kind):
    boolean = kind.endswith("_bool")
    name = kind.removesuffix("_bool").removesuffix("_double")
    dtype = bool if boolean else np.float64
    if name in ("dense", "fortran", "strided"):
        result = x.toarray().astype(dtype)
        if name == "fortran":
            result = np.asfortranarray(result)
        elif name == "strided":
            storage = np.zeros((result.shape[0] * 2, result.shape[1] * 2), dtype=dtype)
            storage[::2, ::2] = result
            result = storage[::2, ::2]
        return result
    return getattr(sparse, name)(x, dtype=dtype)


FORMATS = ("dense_double", "dense_bool", "fortran_double", "strided_double",
           "csr_matrix_double", "csr_matrix_bool", "csc_matrix_double",
           "coo_matrix_bool", "csr_array_double", "csc_array_bool", "coo_array_double")


def options(mode, rd):
    result = {"problem_type": mode}
    if mode in ("gc", "apslp", "gc_approx"):
        result["rd_matrix"] = rd
    if mode == "gc_approx":
        result["sampling_factor"] = 2.0
    if mode == "bc_approx":
        result.update(delta=0.1, err=0.1, emp_peeling_a=2.0,
                      memoization_depth="log_n_w", seed=42)
    return result


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def approx_probe(target, mode, kind, threads, directed):
    x = fixture("cycle_n11" if directed else "grid_4x5")
    if directed:
        x = sparse.triu(x).tolil()
        x[0, -1] = 0
        x[-1, 0] = 1
        x = x.tocsr()
    adjacency = representation(x, kind)
    before = adjacency.copy()
    kw = options(mode, reference_distance(x.shape[0]))
    if target == "parallel":
        result = cc_apspn_kernel_par(adjacency, num_threads=threads, verbose=False, **kw)
        assert result["requested_threads"] == threads
        assert result["effective_threads"] == 1
    else:
        result = cc_apspn_kernel_approx(adjacency, verbose=False, **kw)
    if sparse.issparse(before):
        assert (before != adjacency).nnz == 0
    else:
        np.testing.assert_array_equal(before, adjacency)
    expected_shape = (x.shape[0],) if mode == "bc_approx" else ()
    value = np.asarray(result["result"])
    assert value.shape == expected_shape, (mode, value.shape)
    assert np.isfinite(value).all() and (value >= 0).all()
    assert np.isfinite(result["elapsed_time"]) and result["elapsed_time"] >= 0
    print("PROBE_JSON=" + json.dumps(value.tolist(), allow_nan=False))


def fresh_approx(target, mode, kind, threads=4, directed=False):
    command = [sys.executable, str(Path(__file__).resolve()), "--probe",
               target, mode, kind, str(threads), str(int(directed))]
    if INSTALLED:
        command.append("--installed")
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise AssertionError(f"Approximate subprocess failed ({command}):\n{result.stdout}\n{result.stderr}")
    lines = [line.removeprefix("PROBE_JSON=") for line in result.stdout.splitlines()
             if line.startswith("PROBE_JSON=")]
    if len(lines) != 1:
        raise AssertionError(f"Missing approximate result: {result.stdout}\n{result.stderr}")
    return np.asarray(json.loads(lines[0]))


class ParallelReleaseTests(unittest.TestCase):
    def check_metadata(self, result, threads, mode, n):
        value = np.asarray(result["result"])
        shape = (n, n) if mode in ("apspn", "apslp") else ((n,) if mode in ("bc", "phd") else ())
        self.assertEqual(value.shape, shape)
        self.assertTrue(np.isfinite(value).all())
        self.assertTrue(np.isfinite(result["elapsed_time"]))
        self.assertGreaterEqual(result["elapsed_time"], 0)
        self.assertEqual(result["requested_threads"], threads)
        self.assertGreaterEqual(result["effective_threads"], 1)
        self.assertLessEqual(result["effective_threads"], threads)

    def test_exact_all_modes_formats_and_thread_changes(self):
        for name in ("grid_4x5", "cycle_n11"):
            x = fixture(name)
            rd = reference_distance(x.shape[0])
            for mode in EXACT_MODES:
                kw = options(mode, rd)
                expected = brandes_apspn_kernel(x, **kw)["result"]
                for kind in FORMATS:
                    with self.subTest(fixture=name, mode=mode, representation=kind):
                        adjacency = representation(x, kind)
                        before = adjacency.copy()
                        sequential = cc_apspn_kernel_approx(adjacency, verbose=False, **kw)
                        self.assertEqual(np.asarray(sequential["result"]).shape,
                                         np.asarray(expected).shape)
                        assert_close(sequential["result"], expected)
                        self.assertTrue(np.isfinite(sequential["elapsed_time"]))
                        self.assertGreaterEqual(sequential["elapsed_time"], 0)
                        saved_output = None
                        for threads in (1, 4, 1):
                            result = cc_apspn_kernel_par(adjacency, num_threads=threads,
                                                         verbose=False, **kw)
                            self.check_metadata(result, threads, mode, x.shape[0])
                            if mode == "apspn":
                                np.testing.assert_array_equal(result["result"], expected)
                            else:
                                assert_close(result["result"], expected)
                            if saved_output is not None:
                                assert_close(saved_output[0], saved_output[1])
                            value = np.asarray(result["result"])
                            saved_output = (value, value.copy())
                        if sparse.issparse(adjacency):
                            self.assertEqual((adjacency != before).nnz, 0)
                        else:
                            np.testing.assert_array_equal(adjacency, before)
                OBSERVATIONS["exact"][name + "/" + mode] = np.asarray(result["result"]).tolist()

    def test_sparse_exact_does_not_densify_adjacency(self):
        x = fixture("grid_4x5")
        rd = reference_distance(x.shape[0])
        with ExitStack() as stack:
            for cls in (sparse.csr_matrix, sparse.csr_array):
                for method in ("toarray", "todense"):
                    stack.enter_context(patch.object(cls, method, side_effect=AssertionError("CSR adjacency was densified")))
            for mode in EXACT_MODES:
                cc_apspn_kernel_par(x, num_threads=4, verbose=False, **options(mode, rd))

    def test_other_release_libraries_smoke(self):
        x = fixture("grid_4x5")
        rd = reference_distance(x.shape[0])
        expected_gc = brandes_apspn_kernel(x, problem_type="gc", rd_matrix=rd)["result"]
        for adjacency in (x, x.toarray().astype(bool)):
            result = cm_gc(adjacency, rd)
            assert_close(result["result"], expected_gc)
            self.assertTrue(np.isfinite(result["elapsed_time"]))
            self.assertGreaterEqual(result["elapsed_time"], 0)
        for mode in ("apspn", "bc"):
            expected = brandes_apspn_kernel(x, problem_type=mode)["result"]
            for adjacency in (x, x.toarray().astype(bool)):
                for use_se, use_bcc in ((False, False), (True, False), (False, True), (True, True)):
                    with self.subTest(mode=mode, use_se=use_se, use_bcc=use_bcc):
                        result = cc_sebccopts_apspn_kernel(adjacency, problem_type=mode,
                                                          use_se=use_se, use_bcc=use_bcc)
                        assert_close(result["result"], expected)
                        self.assertTrue(np.isfinite(result["elapsed_time"]))
                        self.assertGreaterEqual(result["elapsed_time"], 0)

    def test_reference_distance_storage(self):
        x = fixture("grid_4x5")
        n = x.shape[0]
        rd = reference_distance(n)
        expected = cc_apspn_kernel_approx(x, problem_type="apslp", rd_matrix=rd,
                                          verbose=False)["result"]
        storage = np.zeros((2 * n, 2 * n)); storage[::2, ::2] = rd
        for layout in (rd, np.asfortranarray(rd), storage[::2, ::2], sparse.csc_matrix(rd)):
            before = layout.copy()
            result = cc_apspn_kernel_par(x, problem_type="apslp", rd_matrix=layout,
                                         num_threads=4, verbose=False)["result"]
            assert_close(result, expected)
            if sparse.issparse(layout):
                self.assertEqual((before != layout).nnz, 0)
            else:
                np.testing.assert_array_equal(before, layout)

    def test_column_major_output_layout(self):
        # Valid graph outputs are symmetric. A nonsymmetric sentinel written
        # by a mock ABI detects a wrapper transpose without pretending that
        # asymmetric referential distances are supported by the algorithm.
        x = fixture("grid_4x5")
        n = x.shape[0]
        expected = np.arange(n * n, dtype=np.float64).reshape(n, n)

        def native_sentinel(*args):
            self.assertEqual(args[4], n)
            output = np.ctypeslib.as_array(args[8], shape=(n * n,))
            output[:] = expected.ravel(order="F")
            ctypes.cast(args[9], ctypes.POINTER(ctypes.c_double))[0] = 0.0
            return 0

        library = SimpleNamespace(cc_apspn_kernel_par_py=native_sentinel,
                                  cc_apspn_kernel_par_effective_threads=lambda: 1)
        module = importlib.import_module("fast_apspn_kernel.cc_apspn_kernel_par")
        with patch.object(module, "_get_lib", return_value=library):
            result = cc_apspn_kernel_par(x, num_threads=1, verbose=False)["result"]
        np.testing.assert_array_equal(result, expected)

        def sequential_sentinel(*args):
            self.assertEqual(args[3].value, n)
            output = np.ctypeslib.as_array(args[5], shape=(n * n,))
            output[:] = expected.ravel(order="F")
            ctypes.cast(args[6], ctypes.POINTER(ctypes.c_double))[0] = 0.0
            return 0

        library = SimpleNamespace(cc_apspn_kernel_py=sequential_sentinel)
        module = importlib.import_module("fast_apspn_kernel.cc_apspn_kernel_approx")
        with patch.object(module, "_get_lib", return_value=library):
            for mode in ("apspn", "apslp"):
                result = cc_apspn_kernel_approx(
                    x, verbose=False, **options(mode, reference_distance(n)))["result"]
                np.testing.assert_array_equal(result, expected)

    def test_unified_sequential_api_and_ablations(self):
        package = importlib.import_module("fast_apspn_kernel")
        self.assertNotIn("cc_apspn_kernel", package.__all__)
        self.assertFalse(hasattr(package, "cc_apspn_kernel"))
        self.assertIsNone(importlib.util.find_spec("fast_apspn_kernel.cc_apspn_kernel"))

        x = fixture("grid_4x5")
        expected = brandes_apspn_kernel(x, problem_type="bc")["result"]
        for ablation in ("None", "Pri", "Mem", "Symm", "DAP"):
            with self.subTest(ablation=ablation):
                result = cc_apspn_kernel_approx(
                    x, problem_type="bc", ablation_type=ablation, verbose=False)
                assert_close(result["result"], expected)

        # Confirm the selected ablation reaches the existing native entry point.
        calls = []
        def record_dispatch(*args):
            calls.append((args[0], args[1]))
            ctypes.cast(args[6], ctypes.POINTER(ctypes.c_double))[0] = 0.0
            return 0

        module = importlib.import_module("fast_apspn_kernel.cc_apspn_kernel_approx")
        library = SimpleNamespace(cc_apspn_kernel_py=record_dispatch)
        with patch.object(module, "_get_lib", return_value=library):
            for ablation in ("None", "Pri", "Mem", "Symm", "DAP"):
                cc_apspn_kernel_approx(x, problem_type="bc", ablation_type=ablation,
                                       verbose=False)
            # Keep the pre-existing default GC_APPROX mode.
            cc_apspn_kernel_approx(x, rd_matrix=reference_distance(x.shape[0]), verbose=False)
        self.assertEqual(calls[:-1], [(b"BC", value.encode())
                                      for value in ("None", "Pri", "Mem", "Symm", "DAP")])
        self.assertEqual(calls[-1], (b"GC_APPROX", b"None"))

        for mode in ("gc", "gc_approx", "apslp"):
            for rd in (None, np.zeros((2, 2))):
                with self.subTest(mode=mode, rd_shape=None if rd is None else rd.shape):
                    with self.assertRaises(ValueError):
                        cc_apspn_kernel_approx(x, problem_type=mode, rd_matrix=rd, verbose=False)
        for kw in ({"problem_type": "unknown"}, {"problem_type": "bc", "ablation_type": "unknown"}):
            with self.subTest(options=kw), self.assertRaises(ValueError):
                cc_apspn_kernel_approx(x, verbose=False, **kw)

    def test_invalid_input_rejected(self):
        x = fixture("grid_4x5")
        weighted = x.toarray(); weighted[0, 1] = 0.5
        asymmetric = x.toarray(); i, j = sparse.triu(x).nonzero(); asymmetric[i[0], j[0]] = 0
        nonfinite = x.toarray(); nonfinite[0, 1] = np.nan
        invalid = (np.zeros((2, 3)), np.zeros((1, 1)), weighted, sparse.csr_matrix(weighted),
                   asymmetric, sparse.csc_matrix(asymmetric), nonfinite, x.astype(complex) * (1 + 1j))
        for value in invalid:
            with self.subTest(shape=value.shape), self.assertRaises((ValueError, TypeError, RuntimeError)):
                cc_apspn_kernel_par(value, problem_type="bc", num_threads=1, verbose=False)
        for threads in (0, -1, 1.5, np.nan, np.inf, "4", [1, 2]):
            with self.subTest(threads=threads), self.assertRaises((ValueError, TypeError, RuntimeError)):
                cc_apspn_kernel_par(x, problem_type="bc", num_threads=threads, verbose=False)
        bad_rd = reference_distance(x.shape[0]); bad_rd[0, 1] += 0.25
        for kw in ({"problem_type": "unknown"}, {"problem_type": "gc"},
                   {"problem_type": "gc", "rd_matrix": np.zeros((2, 2))},
                   {"problem_type": "apslp", "rd_matrix": bad_rd},
                   {"problem_type": "bc_approx", "err": 0},
                   {"problem_type": "gc_approx", "rd_matrix": reference_distance(x.shape[0]), "sampling_factor": 0}):
            with self.subTest(options=kw), self.assertRaises((ValueError, TypeError, RuntimeError)):
                cc_apspn_kernel_par(x, num_threads=1, verbose=False, **kw)

    def test_approximate_fresh_process_compatibility(self):
        for mode in ("bc_approx", "gc_approx"):
            expected = fresh_approx("sequential", mode, "dense_double", 1)
            OBSERVATIONS["approximate"][mode] = expected.tolist()
            for kind in ("dense_double", "dense_bool", "csr_matrix_double", "csc_matrix_bool",
                         "coo_array_double"):
                with self.subTest(mode=mode, representation=kind):
                    assert_close(fresh_approx("parallel", mode, kind), expected)
            for threads in (1, 4, 1):
                assert_close(fresh_approx("parallel", mode, "csr_matrix_double", threads), expected)
        expected = fresh_approx("sequential", "bc_approx", "dense_double", 1, True)
        for kind in ("dense_double", "csr_matrix_double", "coo_array_double"):
            assert_close(fresh_approx("parallel", "bc_approx", kind, 4, True), expected)
        OBSERVATIONS["approximate"]["directed_bc_approx"] = expected.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional results file outside the tracked source tree")
    parser.add_argument("--installed", action="store_true", help="Test the installed package, not the source checkout")
    parser.add_argument("--probe", nargs=5, metavar=("TARGET", "MODE", "FORMAT", "THREADS", "DIRECTED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        target, mode, kind, threads, directed = args.probe
        approx_probe(target, mode, kind, int(threads), bool(int(directed)))
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ParallelReleaseTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if result.wasSuccessful() and args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(OBSERVATIONS, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        sys.exit(0 if result.wasSuccessful() else 1)
