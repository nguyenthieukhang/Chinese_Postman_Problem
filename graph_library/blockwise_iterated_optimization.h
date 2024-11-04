#include "meta_heuristics.h"
#include <cmath>
#include <algorithm>

// Helper function to divide edges into blocks of size sqrt(n)
vector<vector<Edge>> divide_into_blocks(const vector<Edge>& edges, int block_size) {
    vector<vector<Edge>> blocks;
    for (int i = 0; i < edges.size(); i += block_size) {
        vector<Edge> block(edges.begin() + i, edges.begin() + std::min(i + block_size, (int)edges.size()));
        blocks.push_back(block);
    }
    return blocks;
}

double calculate_cost(Graph *graph, const vector<Edge> sequence) {
    return dynamic_programming(graph, sequence).first[0][0];
}

double block_cost(Graph *graph, const vector<Edge> &sequence) {
    // A heuristics cost function to guide the local search on each block
	// Check if the graph has computed Floyd's algorithm before
    assert(graph -> shortest_path.size() == graph -> num_nodes);
	const double W = graph -> W;

	// Number of edges in this list
	const int m = sequence.size();

	// Initialize
    vector< vector<double> > f(m, vector<double>(2, INF));

	// Compute the Q
	vector<double> Q(m, 0);
	Q[m - 1] = sequence[m - 1].q;
	for (int k = m - 2; k >= 0; --k) {
		Q[k] = Q[k + 1] + sequence[k].q;
	}

    f[m - 1][0] = f[m - 1][1] = (Q[m - 1] / 2 + W) * sequence.back().d;

    for(int k = m - 2; k >= 0; k--) {
        int u = sequence[k].first, v = sequence[k].second;
        double q = sequence[k].q, d = sequence[k].d;
        int prev_u = sequence[k + 1].first, prev_v = sequence[k + 1].second;
        f[k][0] = (W + Q[k] - q / 2) * d + std::min((W + Q[k] - q) * graph->shortest_path[v][prev_u] + f[k + 1][0], 
                                                (W + Q[k] - q) * graph->shortest_path[v][prev_v] + f[k + 1][1]);
        f[k][1] = (W + Q[k] - q / 2) * d + std::min((W + Q[k] - q) * graph->shortest_path[u][prev_u] + f[k + 1][0], 
                                                (W + Q[k] - q) * graph->shortest_path[u][prev_v] + f[k + 1][1]);                       
    }

	return std::min(f[0][0], f[0][1]);
}

vector<Edge> LocalSearch(Graph *graph, const vector<Edge> &block) {
    vector<Edge> current_block = block;
    double current_cost = block_cost(graph, current_block);
    
    bool improved = true;
    while (improved) {
        improved = false;
        
        for (int i = 0; i < current_block.size() - 1; ++i) {
            // Swap adjacent edges to explore neighborhood
            swap(current_block[i], current_block[i + 1]);
            double new_cost = block_cost(graph, current_block);
            
            if (new_cost < current_cost) {
                current_cost = new_cost;
                improved = true;
            } else {
                // Undo the swap if no improvement
                swap(current_block[i], current_block[i + 1]);
            }
        }
    }
    return current_block;
}

vector<Edge> Perturbation(const vector<Edge> &block) {
    vector<Edge> perturbed_block = block;
    int size = perturbed_block.size();
    
    if (size > 2) {
        // Randomly select two edges to swap as a simple perturbation
        int idx1 = rand() % size;
        int idx2 = (idx1 + 1 + rand() % (size - 1)) % size;
        swap(perturbed_block[idx1], perturbed_block[idx2]);
    }
    return perturbed_block;
}

vector<Edge> Blockwise_ILS(Graph *graph, const vector<Edge> &block, const int max_iterations = 10) {
    vector<Edge> best_block = LocalSearch(graph, block);
    double best_cost = block_cost(graph, best_block);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Apply perturbation
        vector<Edge> perturbed_block = Perturbation(best_block);
        
        // Local search on the perturbed solution
        vector<Edge> optimized_block = LocalSearch(graph, perturbed_block);
        double optimized_cost = block_cost(graph, optimized_block);
        
        // Check if we have found a new best solution
        if (optimized_cost < best_cost) {
            best_block = optimized_block;
            best_cost = optimized_cost;
        }
    }
    
    return best_block;
}

vector<Edge> Optimize_Block(Graph *graph, const vector<Edge> &block) {
    return Blockwise_ILS(graph, block);
}

// Block-based optimization function
pair<vector<Edge>, double> Blockwise_Iterated_Optimization(Graph *graph) {
    // Step 1: Create initial solution using a greedy heuristic
    pair<vector<Edge>, double> greedy = Greedy_Constructive_Heuristic(graph);
    vector<Edge> sigma_star = greedy.first;
    double best_cost = greedy.second;
    
    int n = sigma_star.size();
    int block_size = std::sqrt(n);

    // Step 2: Divide edges into blocks of size sqrt(n)
    vector<vector<Edge>> blocks = divide_into_blocks(sigma_star, block_size);

    // Step 3: Optimize each block independently
    for (auto &block : blocks) {
        block = Optimize_Block(graph, block); // Using 1-OPT as an example
    }

    // Step 4: Combine blocks and optimize the permutation of blocks
    // Flatten the optimized blocks

    // Try to find a better block permutation to minimize cost
    vector<int> block_order(blocks.size());
    for(int i = 0; i < blocks.size(); i++) { // Initialize with 0, 1, ..., block_count - 1
        block_order[i] = i;
    }
    double best_permutation_cost = best_cost;

    // Simple heuristic: permute blocks to find the best order
    do {
        vector<Edge> permuted_edges;
        for (int idx : block_order) {
            permuted_edges.insert(permuted_edges.end(), blocks[idx].begin(), blocks[idx].end());
        }
        
        double current_cost = calculate_cost(graph, permuted_edges); // Assume calculate_cost() computes the cost of a solution
        if (current_cost < best_permutation_cost) {
            best_permutation_cost = current_cost;
            sigma_star = permuted_edges;
        }
    } while (std::next_permutation(block_order.begin(), block_order.end()));

    return make_pair(sigma_star, best_permutation_cost);
}