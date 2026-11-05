// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#ifndef UTILITIES_H
#define UTILITIES_H
#include <vector>
#include <cstdint>
#include <variant>
#include <Eigen/Dense>

struct mxArray_tag;
typedef struct mxArray_tag mxArray;

namespace cc_approxbc_lib {

class Probabilistic;

double get_time_sec();
bool detect_directed(const mxArray* mxA);
bool detect_directed(const Eigen::MatrixXd& adj_matrix);
void compute_degrees_from_mxA(const mxArray* mxA, bool directed, std::vector<uint32_t>& degrees,
                              std::vector<uint32_t>& in_degrees);
void compute_degrees_from_eigen(const Eigen::MatrixXd& adj_matrix, bool directed, std::vector<uint32_t>& degrees,
                                std::vector<uint32_t>& in_degrees);
void add_edges_from_mxA(const mxArray* mxA, Probabilistic& G);
void add_edges_from_eigen(const Eigen::MatrixXd& adj_matrix, Probabilistic& G);
void preprocess_adj_input(const std::variant<const mxArray*, const Eigen::MatrixXd*>& adj_input,
                          uint32_t& n,
                          bool& directed,
                          std::vector<uint32_t>& degrees,
                          std::vector<uint32_t>& in_degrees);

} // namespace cc_approxbc_lib

#endif // UTILITIES_H
