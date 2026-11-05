//
// Created by Junda on 2024/4/1.
//

#ifndef CM_GC_H
#define CM_GC_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>

// Result types for different algorithms
struct Result {
    double gc;
    double total_time;
};

Result CM_GC_Cpp(const Eigen::MatrixXd& adj_matrix,
                 const Eigen::MatrixXd& RD);

#endif //CM_GC_H

