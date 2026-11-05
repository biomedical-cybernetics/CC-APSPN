"""Build the Python native libraries on Windows/MSVC or Linux/GCC.

From the repository root:
    python source/python/build_native.py --eigen /usr/include/eigen3
    python -m pip install ./source/python

On Windows the script discovers Visual Studio's x64 compiler environment.
EIGEN3_INCLUDE_DIR (or EIGEN_PATH) can replace --eigen. No MATLAB is required.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
COMMON = ROOT / "source" / "common_include"
CPP = ROOT / "source" / "cpp"  # one folder per algorithm, named after its target
TARGETS = ("brandes_apspn_kernel", "cm_gc", "cc_sebccopts_apspn_kernel",
           "cc_apspn_kernel_approx", "cc_apspn_kernel_par")


def approximation_sources(folder):
    bc = folder / "cc_approxbc_lib"
    return [folder / "cc_approxgc_lib" / "cc_approxgc.cpp"] + [
        bc / name for name in ("cc_approxbc_wrapper.cpp", "Probabilistic.cpp",
                              "Graph.cpp", "Rand_gen.cpp", "utilities.cpp",
                              "Sp_sampler.cpp", "Ranking_list.cpp")]


def sources_for(name):
    folder = CPP / name
    if name == "cc_apspn_kernel_par":
        return [folder / "cc_apspn_kernel_par.cpp", COMMON / "graph_utils.cpp",
                COMMON / "cc_bfs_p_apsp_l.cpp"] + approximation_sources(folder)
    if name == "cc_apspn_kernel_approx":
        return [folder / "cc_apspn_kernel_cpp_ablation_approx.cpp",
                COMMON / "graph_utils.cpp"] + approximation_sources(folder)
    return [folder / (name + "_cpp.cpp"), COMMON / "graph_utils.cpp"]


def msvc_environment(vcvars=None):
    env = {key.upper(): value for key, value in os.environ.items()}
    if shutil.which("cl", path=env.get("PATH")):
        return env
    if vcvars is None:
        installer = Path(env.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if not installer.is_file():
            raise RuntimeError("MSVC not found. Install Visual Studio C++ Build Tools or use an x64 Native Tools prompt.")
        install = subprocess.check_output([
            str(installer), "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"
        ], text=True).strip()
        vcvars = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
    vcvars = Path(vcvars)
    if not vcvars.is_file():
        raise RuntimeError(f"MSVC environment script not found: {vcvars}")
    configured = subprocess.check_output(
        f'cmd /d /s /c ""{vcvars}" >nul && set"', text=True
    )
    for line in configured.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.upper()] = value
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eigen", type=Path, help="Directory containing Eigen/Core")
    parser.add_argument("--target", choices=("all",) + TARGETS, default="all")
    parser.add_argument("--compiler", help="C++ compiler (default: cl on Windows, g++ on Linux)")
    parser.add_argument("--msvc-env", type=Path, help="Optional vcvars64.bat location")
    args = parser.parse_args()
    if sys.platform not in {"win32", "linux"}:
        parser.error("This release build script supports Windows and Linux.")
    eigen = args.eigen
    if eigen is None:
        configured = os.environ.get("EIGEN3_INCLUDE_DIR") or os.environ.get("EIGEN_PATH")
        eigen = Path(configured) if configured else Path("/usr/include/eigen3")
    eigen = eigen.resolve()
    if not (eigen / "Eigen" / "Core").is_file():
        parser.error("Provide --eigen or EIGEN3_INCLUDE_DIR pointing to the directory containing Eigen/Core.")
    windows = sys.platform == "win32"
    compiler = args.compiler or ("cl" if windows else "g++")
    env = msvc_environment(args.msvc_env) if windows else os.environ.copy()
    compiler_path = shutil.which(compiler, path=env.get("PATH"))
    if not compiler_path:
        parser.error(f"Compiler not found: {compiler}")
    compiler = compiler_path
    output = HERE / "fast_apspn_kernel" / "lib"
    output.mkdir(parents=True, exist_ok=True)
    include_dirs = [eigen, COMMON,
                    ROOT / "third_party" / "cc_bfs_p" / "cpp" / "include"]
    targets = TARGETS if args.target == "all" else (args.target,)
    for name in targets:
        sources = sources_for(name)
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing source files: " + ", ".join(missing))
        library = output / (name + (".dll" if windows else ".so"))
        # Compiler intermediates stay outside the source tree. Only completed
        # runtime libraries are retained in the installable package.
        with tempfile.TemporaryDirectory(prefix="cc-apspn-build-") as scratch:
            built_library = Path(scratch) / library.name
            if windows:
                command = [compiler, "/nologo", "/LD", "/EHsc", "/O2", "/MD",
                           "/std:c++20", "/DPYTHON_BUILD", "/openmp", "/DNOMINMAX"]
                command += ["/I" + str(path) for path in include_dirs]
                command += [str(path) for path in sources]
                command += ["/link", "/OUT:" + str(built_library)]
            else:
                command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", "-DPYTHON_BUILD",
                           "-fPIC", "-shared", "-fopenmp"]
                command += ["-I" + str(path) for path in include_dirs]
                command += [str(path) for path in sources]
                command += ["-o", str(built_library)]
            print(f"Building {name} ({compiler})", flush=True)
            subprocess.run(command, cwd=scratch, env=env, check=True)
            shutil.copy2(built_library, library)
        print(f"Built {library}", flush=True)


if __name__ == "__main__":
    main()
