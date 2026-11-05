//
// Created by Junda on 2025/4/17.
//

#ifndef CPP_MEX_UTILS_H
#define CPP_MEX_UTILS_H
#include "mex.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Function to parse adjacency matrix from MATLAB input
Eigen::MatrixXd parseSparseDenseMatrix(const mxArray *matlabMatrix) {
    mwSize numRows = mxGetM(matlabMatrix);
    mwSize numCols = mxGetN(matlabMatrix);

    Eigen::MatrixXd adjMatrix(numRows, numCols);

    if (mxIsSparse(matlabMatrix)) {
        // Handle sparse matrix
        mwIndex* rowIndices = mxGetIr(matlabMatrix);
        mwIndex* colPointers = mxGetJc(matlabMatrix);

        // Initialize with zeros
        adjMatrix.setZero();

        if (mxIsLogical(matlabMatrix)) {
            // Handle sparse logical matrix
            mxLogical* logicalValues = mxGetLogicals(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    adjMatrix(rowIndices[idx], col) = static_cast<double>(logicalValues[idx]);
                }
            }
        } else {
            // Handle sparse double matrix
            double* values = mxGetPr(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    adjMatrix(rowIndices[idx], col) = values[idx];
                }
            }
        }
    } else {
        // Handle dense matrix
        if (mxIsLogical(matlabMatrix)) {
            // Convert logical to double
            mxLogical* logicalData = mxGetLogicals(matlabMatrix);
            for (mwSize i = 0; i < numRows; ++i) {
                for (mwSize j = 0; j < numCols; ++j) {
                    adjMatrix(i, j) = static_cast<double>(logicalData[i + j * numRows]);
                }
            }
        } else {
            // Map double matrix
            Eigen::Map<Eigen::MatrixXd> map(mxGetPr(matlabMatrix), numRows, numCols);
            adjMatrix = map;
        }
    }

    return adjMatrix;
}




#endif //CPP_MEX_UTILS_H
