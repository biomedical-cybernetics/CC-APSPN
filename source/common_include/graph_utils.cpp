#include "graph_utils.h"

#include <exception>

// Definitions for graph utils functions should be placed here.

BiconnectedComponents find_biconnected_components_tree(const std::vector<std::vector<int64_t>>& A) {
    int64_t n = A.size();
    std::vector<int64_t> disc(n, 0);
    std::vector<int64_t> low(n, 0);
    std::vector<int64_t> parent(n, -1);
    int64_t time = 0;

    Eigen::MatrixXd edge_in_stack = Eigen::MatrixXd::Zero(n, n);
    std::vector<std::pair<int64_t, int64_t>> edgeStack;
    int64_t blockCount = 0;

    std::vector<bool> is_cut_vertex(n, false);
    std::vector<std::vector<bool>> blockToNode;

    std::function<void(int64_t)> DFSBiconnected = [&](int64_t u) {
        time++;
        disc[u] = time;
        low[u] = time;
        int64_t childCount = 0;
        bool isArticulation = false;

        for (int64_t v : A[u]) {
            if (disc[v] == 0) {
                childCount++;
                parent[v] = u;
                edgeStack.push_back({u, v});
                edge_in_stack(u, v) = edge_in_stack(v, u) = 1;
                DFSBiconnected(v);

                low[u] = std::min<int64_t>(low[u], low[v]);

                if ((parent[u] == -1 && childCount > 1) ||
                    (parent[u] != -1 && low[v] >= disc[u])) {
                    isArticulation = true;

                    blockToNode.push_back(std::vector<bool>(n, false));

                    while (!edgeStack.empty()) {
                        auto [e1, e2] = edgeStack.back();
                        edgeStack.pop_back();
                        edge_in_stack(e1, e2) = edge_in_stack(e2, e1) = 0;

                        blockToNode[blockCount][e1] = true;
                        blockToNode[blockCount][e2] = true;

                        if ((e1 == u && e2 == v) || (e1 == v && e2 == u)) {
                            break;
                        }
                    }
                    blockCount++;

                }
            } else if (v != parent[u] && disc[v] < disc[u]) {
                low[u] = std::min<int64_t>(low[u], disc[v]);
                if (edge_in_stack(u, v) == 0) {
                    edgeStack.push_back({u, v});
                    edge_in_stack(u, v) = edge_in_stack(v, u) = 1;
                }
            }
        }

        if (isArticulation && !is_cut_vertex[u]) {
            is_cut_vertex[u] = true;
        }
    };

    // Run DFS for each unvisited node
    for (int64_t u = 0; u < n; u++) {
        if (disc[u] == 0) {
            DFSBiconnected(u);

            if (!edgeStack.empty()) {
                blockToNode.emplace_back(n, false);

                while (!edgeStack.empty()) {
                    auto [e1, e2] = edgeStack.back();
                    edgeStack.pop_back();
                    edge_in_stack(e1, e2) = edge_in_stack(e2, e1) = 0;

                    blockToNode[blockCount][e1] = true;
                    blockToNode[blockCount][e2] = true;
                }
                blockCount++;
            }
        }
    }

    // Prepare return values
    std::vector<int64_t> cutVertices;
    for (int64_t i = 0; i < n; i++) {
        if (is_cut_vertex[i]) cutVertices.push_back(i);
    }

    std::unordered_map<int64_t, std::vector<int64_t>> block2nodes;
    for (int64_t blockId = 0; blockId < blockCount; blockId++) {
        for (int64_t i = 0; i < n; i++) {
            if (blockToNode[blockId][i]) {
                block2nodes[blockId].push_back(i);
            }
        }
    }

    // Build block-cut tree
    std::vector<std::vector<int64_t>> bcTreeAdj(blockCount + n, std::vector<int64_t>());
    for (int64_t cv : cutVertices) {
        int64_t cvTreeId = blockCount + cv;
        for (int64_t b = 0; b < blockCount; b++) {
            if (blockToNode[b][cv]) {
                bcTreeAdj[b].push_back(cvTreeId);
                bcTreeAdj[cvTreeId].push_back(b);
            }
        }
    }

    return {block2nodes, cutVertices, bcTreeAdj, blockToNode};
}

// Function to find structural equivalence classes
std::tuple<std::vector<std::vector<int64_t>>, std::vector<int64_t>>
find_structural_equivalence(const std::vector<std::vector<int64_t>>& A)
{
    int64_t n = A.size();

    // Use unordered_map to group vertices with identical neighbor patterns
    std::unordered_map<std::string, std::vector<int64_t>> pattern_map;

    // Process each vertex
    for (int64_t v = 0; v < n; ++v) {
        // Create a key from neighbor pattern
        std::vector<int64_t> neighbors = A[v]; // Assume neighbors are sorted
        std::ostringstream oss;
        for (int64_t neighbor : neighbors) {
            oss << neighbor << ",";
        }
        std::string key = oss.str();

        // If this pattern exists, append vertex to existing class
        pattern_map[key].push_back(v);
    }

    // Convert map values to equivalence classes
    int64_t eqc_n = pattern_map.size();
    std::vector<std::vector<int64_t>> equivalence_classes(eqc_n);
    std::vector<int64_t> eqc_node_mul(eqc_n);

    int64_t i = 0;
    for (const auto& pair : pattern_map) {
        equivalence_classes[i] = pair.second;
        eqc_node_mul[i] = pair.second.size();
        ++i;
    }
    return {equivalence_classes, eqc_node_mul};
}

/*
 * This function decomposes the graph into biconnected components and computes the block-cut tree
 * And tree weights D_r and D_l
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
compute_eqc_comp_tree_weights(
        const std::unordered_map<int64_t, std::vector<int64_t>>& eqc_block2nodes,
        const std::vector<std::vector<int64_t>>& eqc_bcTreeAdj,
        const int64_t eqc_n,
        const std::vector<int64_t>& eqc_node_mul,
        const int64_t n) {

    int64_t numBlocks = eqc_block2nodes.size();
    int64_t offset = numBlocks;

    std::queue<std::pair<int64_t, int64_t>> Q;
    Eigen::MatrixXd D_r = Eigen::MatrixXd::Constant(numBlocks, eqc_n, std::numeric_limits<double>::quiet_NaN());
    Eigen::MatrixXd D_l = Eigen::MatrixXd::Constant(numBlocks, eqc_n, std::numeric_limits<double>::quiet_NaN());

    // Initialize the queue with leaf nodes
    for (int64_t B_id = 0; B_id < numBlocks; ++B_id) {
        const auto& bcvs = eqc_bcTreeAdj[B_id];
        if (bcvs.size() == 1) {
            Q.emplace(B_id, bcvs[0]);
        }
    }

    while (!Q.empty()) {
        auto [first, second] = Q.front();
        Q.pop();

        if (first < numBlocks) { // pair is a component-vertex pair (B, v)
            int64_t B = first;
            int64_t v = second;
            int64_t B_cardi = std::accumulate(eqc_block2nodes.at(B).begin(), eqc_block2nodes.at(B).end(), 0, [&](int64_t sum, int64_t idx) {
                return sum + eqc_node_mul[idx];
            });
            int64_t size = B_cardi - eqc_node_mul[v - offset]; // number of nodes in block B, except for v itself
            for (int64_t cv : eqc_bcTreeAdj[B]) {
                if (!std::isnan(D_r(B, cv - offset)) && cv != v) {
                    size += n - D_r(B, cv - offset) - eqc_node_mul[cv - offset];
                }
            }
            D_r(B, v - offset) = size;

            // Update Q
            const auto& adj_blocks = eqc_bcTreeAdj[v];
            std::vector<int64_t> unknown_adj_blocks;
            for (int64_t B_dot : adj_blocks) {
                if (std::isnan(D_r(B_dot, v - offset))) {
                    unknown_adj_blocks.push_back(B_dot);
                }
            }
            if (unknown_adj_blocks.size() == 1) {
                Q.emplace(v, unknown_adj_blocks[0]);
            }

        } else { // pair is a vertex-component pair (v, B)
            int64_t v = first;
            int64_t B = second;

            int64_t size = eqc_node_mul[v - offset];
            for (int64_t b : eqc_bcTreeAdj[v]) {
                if (!std::isnan(D_r(b, v - offset)) && b != B) {
                    size += (int64_t)D_r(b, v - offset);
                }
            }
            D_r(B, v - offset) = n - size;

            // Update Q
            const auto& adj_cvs = eqc_bcTreeAdj[B];
            std::vector<int64_t> unknown_adj_cvs;
            for (int64_t cv_dot : adj_cvs) {
                if (std::isnan(D_r(B, cv_dot - offset))) {
                    unknown_adj_cvs.push_back(cv_dot);
                }
            }
            if (unknown_adj_cvs.size() == 1) {
                Q.emplace(B, unknown_adj_cvs[0]);
            }
        }
    }

    // Compute D_l using D_r
    for (int64_t B = 0; B < numBlocks; ++B) {
        for (int64_t v : eqc_bcTreeAdj[B]) {
            D_l(B, v - offset) = n - eqc_node_mul[v-offset] - D_r(B, v - offset);
        }
    }

    return {D_r, D_l};
}

// Function to compute/update the traffic matrix
void compute_comm_intensity_matrix(
        const std::vector<std::vector<int64_t>>& A,
        const std::vector<std::vector<int64_t>>& equivalence_classes,
        const std::vector<std::vector<int64_t>>& eqc_A,
        int64_t eqc_B_id,
        const std::vector<int64_t>& eqc_block,
        const Eigen::MatrixXd& eqc_D_l,
        Eigen::MatrixXd& h,
        const std::vector<bool>& eqc_cv_mask,
        const std::vector<int64_t>& eqc_node_mul) {
    for (int64_t x : eqc_block) {
        for (int64_t y : eqc_block) {
            if (x != y) {
                int64_t C_x_cardi = eqc_node_mul[x];
                int64_t C_y_cardi = eqc_node_mul[y];
                if (!eqc_cv_mask[x] && !eqc_cv_mask[y]) {
                    h(x, y) = C_x_cardi * C_y_cardi;
                } else if (!eqc_cv_mask[x] && eqc_cv_mask[y]) {
                    h(x, y) = (eqc_D_l(eqc_B_id, y) + eqc_node_mul[y]) * C_x_cardi;
                } else if (eqc_cv_mask[x] && !eqc_cv_mask[y]) {
                    h(x, y) = (eqc_D_l(eqc_B_id, x) + eqc_node_mul[x]) * C_y_cardi;
                } else if (eqc_cv_mask[x] && eqc_cv_mask[y]) {
                    h(x, y) = (eqc_D_l(eqc_B_id, x) + eqc_node_mul[x]) * (eqc_D_l(eqc_B_id, y) + eqc_node_mul[y]);
                } else {
                    throw std::runtime_error("unexpected condition");
                }
            } else {
                int64_t C_w_cardi = eqc_node_mul[x];
                double acc = 0;
                for (int64_t C_v : eqc_A[x]) { // C_v is the eq class adjacent to i
                    int64_t C_v_cardi = eqc_node_mul[C_v];
                    int64_t v = equivalence_classes[C_v][0];
                    int64_t NN_v = A[v].size();
                    acc += C_w_cardi * C_v_cardi * (C_v_cardi - 1) / static_cast<double>(NN_v);
                }
                h(x, y) = acc;
            }
        }
    }
}

int64_t compute_eqc_block_m(const std::vector<int64_t>& eqc_block,
                            const std::vector<std::vector<int64_t>>& eqc_A,
                            const std::vector<bool>& eqc_cv_mask,
                            const std::vector<bool>& eqc_block_mask) {
    int64_t eqc_block_m = 0;
    for (int64_t eqc_node : eqc_block) {
        if (eqc_cv_mask[eqc_node]) {
            for (int64_t neighbor : eqc_A[eqc_node]) {
                if (eqc_block_mask[neighbor]) {
                    eqc_block_m++;
                }
            }
        } else {
            eqc_block_m += eqc_A[eqc_node].size();
        }
    }
    return eqc_block_m;
}



void compute_inter_block_Pcount(
        const std::unordered_map<int64_t, std::vector<int64_t>>& eqc_block2nodes,
        const std::vector<std::vector<int64_t>>& eqc_bcTreeAdj,
        const std::vector<int64_t>& eqc_node_mul,
        Eigen::MatrixXd& eqc_Pcount,
        int64_t offset,
        int64_t eqc_n) {

    for (const auto& pair : eqc_block2nodes) {
        auto& eqc_block_id = pair.first;
        auto& eqc_block = pair.second;

        // expand from current block
        std::vector<bool> visited(offset + eqc_n, false);
        visited[eqc_block_id] = true;
        std::queue<std::pair<int64_t, int64_t>> queue;
        for (int64_t cv : eqc_bcTreeAdj[eqc_block_id]) {
            queue.emplace(cv, -1);
        }

        while (!queue.empty()) {
            auto curr = queue.front();
            queue.pop();
            visited[curr.first] = true;

            if (curr.first < offset) { // block
                for (int64_t cv : eqc_bcTreeAdj[curr.first]) {
                    if (!visited[cv]) {
                        visited[cv] = true;
                        queue.emplace(cv, -1);
                    }
                }
                // update Pcount(s,t) for s in eqc_block to t in current block
                for (int64_t eqc_s : eqc_block) {
                    for (int64_t eqc_t : eqc_block2nodes.at(curr.first)) {
                        eqc_Pcount(eqc_s, eqc_t) = eqc_Pcount(eqc_s, curr.second - offset) *
                                                   eqc_Pcount(curr.second - offset, eqc_t) / eqc_node_mul[curr.second - offset];
                    }
                }
            } else { // cut vertex
                for (int64_t block : eqc_bcTreeAdj[curr.first]) {
                    if (!visited[block]) {
                        visited[block] = true;
                        queue.emplace(block, curr.first);
                    }
                }
            }
        }
    }
}

std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> ComputeDegreeAndAdjList(
    const Eigen::MatrixXd& adj_matrix,
    const int64_t n)
{
    Eigen::VectorXd DEG(n);
    for (int64_t i = 0; i < n; ++i)
    {
        DEG(i) = adj_matrix.col(i).sum();
    }

    std::vector<std::vector<int64_t>> A(n);
    for (int64_t i = 0; i < n; ++i)
    {
        A[i].reserve((int64_t)DEG(i));
    }

    for (int64_t i = 0; i < n; ++i)
    {
        auto col = adj_matrix.col(i);
        for (int64_t j = 0; j < n; ++j)
        {
            if (col(j) > 0)
            {
                A[i].push_back(j);
            }
        }
    }

    return {DEG, A};
}

#ifdef MATLAB_BUILD
namespace graph_utils {

// Parse independent MATLAB columns in parallel. The result has sorted
// neighbor order and accepts real binary double or logical square matrices.
ParsedGraphInput ParseGraphInputParallel(const mxArray* input, const int64_t num_threads) {
    if ((!mxIsDouble(input) && !mxIsLogical(input)) || mxIsComplex(input)) {
        throw std::runtime_error(
            "Adjacency matrix must be a real double or logical matrix.");
    }
    if (mxGetNumberOfDimensions(input) != 2) {
        throw std::runtime_error("Adjacency matrix must be two-dimensional.");
    }
    const int64_t n = static_cast<int64_t>(mxGetM(input));
    if (n != static_cast<int64_t>(mxGetN(input))) {
        throw std::runtime_error("Adjacency matrix must be square.");
    }
    if (n < 1) {
        throw std::runtime_error("Adjacency matrix must be nonempty.");
    }

    const bool is_sparse = mxIsSparse(input);
    const bool is_logical = mxIsLogical(input);
    const mxLogical* logical_values = is_logical ? mxGetLogicals(input) : nullptr;
    const double* double_values = is_logical ? nullptr : mxGetPr(input);
    const mwIndex* row_indices = is_sparse ? mxGetIr(input) : nullptr;
    const mwIndex* column_offsets = is_sparse ? mxGetJc(input) : nullptr;

    std::vector<size_t> column_degrees(static_cast<size_t>(n), 0);
    int invalid_binary = 0;
#pragma omp parallel for num_threads(num_threads) schedule(static) reduction(|:invalid_binary)
    for (int64_t column = 0; column < n; ++column) {
        size_t degree = 0;
        if (is_sparse) {
            for (mwIndex index = column_offsets[column];
                 index < column_offsets[column + 1]; ++index) {
                if (is_logical) {
                    degree += logical_values[index] ? 1 : 0;
                } else {
                    const double value = double_values[index];
                    if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
                        invalid_binary = 1;
                    }
                    degree += value != 0.0 ? 1 : 0;
                }
            }
        } else {
            const size_t offset = static_cast<size_t>(column) * static_cast<size_t>(n);
            for (int64_t row = 0; row < n; ++row) {
                const size_t index = offset + static_cast<size_t>(row);
                if (is_logical) {
                    degree += logical_values[index] ? 1 : 0;
                } else {
                    const double value = double_values[index];
                    if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
                        invalid_binary = 1;
                    }
                    degree += value != 0.0 ? 1 : 0;
                }
            }
        }
        column_degrees[static_cast<size_t>(column)] = degree;
    }
    if (invalid_binary) {
        throw std::runtime_error(
            "Adjacency matrix must contain only finite binary values 0 or 1.");
    }

    ParsedGraphInput result;
    result.adjacency.resize(static_cast<size_t>(n));
    result.degrees.resize(n);
    std::exception_ptr allocation_error;
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t column = 0; column < n; ++column) {
        try {
            auto& neighbors = result.adjacency[static_cast<size_t>(column)];
            neighbors.resize(column_degrees[static_cast<size_t>(column)]);
            result.degrees(column) = static_cast<double>(neighbors.size());
            size_t position = 0;
            if (is_sparse) {
                for (mwIndex index = column_offsets[column];
                     index < column_offsets[column + 1]; ++index) {
                    const bool present =
                        is_logical ? logical_values[index] : double_values[index] != 0.0;
                    if (present) {
                        neighbors[position++] = static_cast<int64_t>(row_indices[index]);
                    }
                }
            } else {
                const size_t offset = static_cast<size_t>(column) * static_cast<size_t>(n);
                for (int64_t row = 0; row < n; ++row) {
                    const size_t index = offset + static_cast<size_t>(row);
                    const bool present =
                        is_logical ? logical_values[index] : double_values[index] != 0.0;
                    if (present) {
                        neighbors[position++] = row;
                    }
                }
            }
        } catch (...) {
#pragma omp critical(graph_utils_input_allocation_error)
            {
                if (!allocation_error) {
                    allocation_error = std::current_exception();
                }
            }
        }
    }
    if (allocation_error) {
        std::rethrow_exception(allocation_error);
    }

    return result;
}

} // namespace graph_utils
#endif

std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>> ComputeAdjAndNonAdjList(
    const Eigen::MatrixXd& adj_matrix,
    const int64_t n)
{
    std::vector<std::vector<int64_t>> A(n);
    std::vector<std::vector<int64_t>> nonA(n);
    for (int64_t i = 0; i < n; ++i) {
        int64_t deg = 0;
        auto col = adj_matrix.col(i);
        for (int64_t j = 0; j < n; ++j) {
            if (col(j) > 0) {
                deg++;
            }
        }
        A[i].reserve(deg);
        if (n > 1) {
            nonA[i].reserve((n - 1) - deg);
        }
        for (int64_t j = 0; j < n; ++j) {
            if (col(j) > 0) {
                A[i].push_back(j);
            } else if (j != i) {
                nonA[i].push_back(j);
            }
        }
    }
    return {A, nonA};
}

std::vector<std::vector<int64_t>> ComputeAdjList(
    const Eigen::MatrixXd& adj_matrix,
    const int64_t n)
{
    // compute adjacency list
    std::vector<std::vector<int64_t>> A(n);
    for (int64_t i = 0; i < n; ++i) {
        A[i].reserve(adj_matrix.col(i).count());
    }
    for (int64_t i = 0; i < n; ++i) {
        auto col = adj_matrix.col(i);
        for (int64_t j = 0; j < n; ++j) {
            if (col(j) > 0) {
                A[i].push_back(j);
            }
        }
    }

    return A;
}
