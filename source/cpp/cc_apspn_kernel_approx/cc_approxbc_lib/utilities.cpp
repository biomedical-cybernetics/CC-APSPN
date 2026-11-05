// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#include "utilities.h"
#include "Probabilistic.h"
#include <chrono>
#include <vector>
#include <variant>
#include <stdexcept>
#ifdef MATLAB_BUILD
#include "matrix.h"
#endif

using namespace std;

namespace cc_approxbc_lib {

double get_time_sec() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
}

#ifdef MATLAB_BUILD
bool is_sparse_symmetric(const mxArray* mxA) {
    mwSize n = mxGetM(mxA);
    mwIndex* ir = mxGetIr(mxA);
    mwIndex* jc = mxGetJc(mxA);

    for (mwSize j = 0; j < n; j++) {
        for (mwIndex k_idx = jc[j]; k_idx < jc[j + 1]; k_idx++) {
            mwIndex i = ir[k_idx];
            if (i == j) continue;

            bool found = false;
            for (mwIndex k_idx2 = jc[i]; k_idx2 < jc[i + 1]; k_idx2++) {
                if (ir[k_idx2] == j) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    return true;
}

bool is_dense_symmetric_double(const double* data, mwSize n) {
    for (mwSize i = 0; i < n; i++) {
        for (mwSize j = i + 1; j < n; j++) {
            if (data[j * n + i] != data[i * n + j]) {
                return false;
            }
        }
    }
    return true;
}

bool is_dense_symmetric_logical(const mxLogical* data, mwSize n) {
    for (mwSize i = 0; i < n; i++) {
        for (mwSize j = i + 1; j < n; j++) {
            if (data[j * n + i] != data[i * n + j]) {
                return false;
            }
        }
    }
    return true;
}

bool detect_directed(const mxArray* mxA) {
    if (mxIsSparse(mxA)) {
        return !is_sparse_symmetric(mxA);
    }
    if (mxIsDouble(mxA)) {
        return !is_dense_symmetric_double(mxGetPr(mxA), mxGetM(mxA));
    }
    return !is_dense_symmetric_logical(mxGetLogicals(mxA), mxGetM(mxA));
}
#endif

bool detect_directed(const Eigen::MatrixXd& adj_matrix) {
    return !adj_matrix.isApprox(adj_matrix.transpose());
}

#ifdef MATLAB_BUILD
void count_degrees_sparse(const mxArray* mxA, std::vector<uint32_t>& degrees,
                          std::vector<uint32_t>* in_degrees) {
    mwSize n = mxGetM(mxA);
    mwIndex* ir = mxGetIr(mxA);
    mwIndex* jc = mxGetJc(mxA);
    mwSize nz = jc[n];
    for (mwSize idx = 0; idx < nz; idx++) {
        mwIndex row = ir[idx];
        if (row < n) {
            degrees[row]++;
        }
    }
    if (in_degrees) {
        for (mwSize j = 0; j < n; j++) {
            (*in_degrees)[j] = static_cast<uint32_t>(jc[j + 1] - jc[j]);
        }
    }
}

void count_degrees_dense_double(const double* data, mwSize n, std::vector<uint32_t>& degrees,
                                std::vector<uint32_t>* in_degrees) {
    for (mwSize j = 0; j < n; j++) {
        for (mwSize i = 0; i < n; i++) {
            if (data[j * n + i] != 0) {
                degrees[i]++;
                if (in_degrees) (*in_degrees)[j]++;
            }
        }
    }
}

void count_degrees_dense_logical(const mxLogical* data, mwSize n, std::vector<uint32_t>& degrees,
                                 std::vector<uint32_t>* in_degrees) {
    for (mwSize j = 0; j < n; j++) {
        for (mwSize i = 0; i < n; i++) {
            if (data[j * n + i] != 0) {
                degrees[i]++;
                if (in_degrees) (*in_degrees)[j]++;
            }
        }
    }
}

void compute_degrees_from_mxA(const mxArray* mxA, bool directed, std::vector<uint32_t>& degrees,
                              std::vector<uint32_t>& in_degrees) {
    mwSize n = mxGetM(mxA);

    if (mxIsSparse(mxA)) {
        count_degrees_sparse(mxA, degrees, directed ? &in_degrees : nullptr);
        return;
    }

    if (mxIsDouble(mxA)) {
        count_degrees_dense_double(mxGetPr(mxA), n, degrees, directed ? &in_degrees : nullptr);
    } else {
        count_degrees_dense_logical(mxGetLogicals(mxA), n, degrees, directed ? &in_degrees : nullptr);
    }
}
#endif

void compute_degrees_from_eigen(const Eigen::MatrixXd& adj_matrix, bool directed, std::vector<uint32_t>& degrees,
                                std::vector<uint32_t>& in_degrees) {
    const Eigen::Index n = adj_matrix.rows();
    for (Eigen::Index j = 0; j < n; j++) {
        for (Eigen::Index i = 0; i < n; i++) {
            if (adj_matrix(i, j) != 0.0) {
                degrees[static_cast<size_t>(i)]++;
                if (directed) {
                    in_degrees[static_cast<size_t>(j)]++;
                }
            }
        }
    }
}

#ifdef MATLAB_BUILD
void add_edges_dense_double(const double* data, mwSize n, Probabilistic& G) {
    for (mwSize j = 0; j < n; j++) {
        for (mwSize i = 0; i < n; i++) {
            if (data[j * n + i] != 0) {
                G.add_edge(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
            }
        }
    }
}

void add_edges_dense_logical(const mxLogical* data, mwSize n, Probabilistic& G) {
    for (mwSize j = 0; j < n; j++) {
        for (mwSize i = 0; i < n; i++) {
            if (data[j * n + i] != 0) {
                G.add_edge(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
            }
        }
    }
}

void add_edges_from_mxA(const mxArray* mxA, Probabilistic& G) {
    mwSize n = mxGetM(mxA);
    if (mxIsSparse(mxA)) {
        mwIndex* ir = mxGetIr(mxA);
        mwIndex* jc = mxGetJc(mxA);
        for (mwSize j = 0; j < n; j++) {
            for (mwIndex k_idx = jc[j]; k_idx < jc[j + 1]; k_idx++) {
                mwIndex i = ir[k_idx];
                G.add_edge(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
            }
        }
        return;
    }

    if (mxIsDouble(mxA)) {
        add_edges_dense_double(mxGetPr(mxA), n, G);
    } else {
        add_edges_dense_logical(mxGetLogicals(mxA), n, G);
    }
}
#endif

void add_edges_from_eigen(const Eigen::MatrixXd& adj_matrix, Probabilistic& G) {
    const Eigen::Index n = adj_matrix.rows();
    for (Eigen::Index j = 0; j < n; j++) {
        for (Eigen::Index i = 0; i < n; i++) {
            if (adj_matrix(i, j) != 0.0) {
                G.add_edge(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
            }
        }
    }
}

void preprocess_adj_input(const std::variant<const mxArray*, const Eigen::MatrixXd*>& adj_input,
                          uint32_t& n,
                          bool& directed,
                          std::vector<uint32_t>& degrees,
                          std::vector<uint32_t>& in_degrees) {
    const bool is_eigen = std::holds_alternative<const Eigen::MatrixXd*>(adj_input);

    if (is_eigen) {
        const Eigen::MatrixXd* adj = std::get<const Eigen::MatrixXd*>(adj_input);
        if (!adj) {
            throw std::invalid_argument("Adjacency matrix pointer is null.");
        }
        if (adj->rows() != adj->cols()) {
            throw std::invalid_argument("Adjacency matrix must be square.");
        }
        if (adj->rows() <= 1) {
            throw std::invalid_argument("Adjacency matrix must have n > 1.");
        }

        n = static_cast<uint32_t>(adj->rows());
        directed = detect_directed(*adj);
        degrees.assign(n, 0);
        in_degrees.assign(n, 0);
        compute_degrees_from_eigen(*adj, directed, degrees, in_degrees);
        return;
    }

#ifdef MATLAB_BUILD
    const mxArray* mxA = std::get<const mxArray*>(adj_input);
    if (!mxA) {
        throw std::invalid_argument("Adjacency matrix pointer is null.");
    }
    if (!mxIsDouble(mxA) && !mxIsLogical(mxA)) {
        throw std::invalid_argument("Adjacency matrix must be double or logical (sparse or dense).");
    }

    mwSize mx_n = mxGetM(mxA);
    if (mxGetN(mxA) != mx_n) {
        throw std::invalid_argument("Adjacency matrix must be square.");
    }
    if (mx_n <= 1) {
        throw std::invalid_argument("Adjacency matrix must have n > 1.");
    }

    n = static_cast<uint32_t>(mx_n);
    directed = detect_directed(mxA);
    degrees.assign(n, 0);
    in_degrees.assign(n, 0);
    compute_degrees_from_mxA(mxA, directed, degrees, in_degrees);
#else
    throw std::runtime_error("CcApproxBc called with mxArray input but MATLAB_BUILD is not enabled.");
#endif
}

} // namespace cc_approxbc_lib
