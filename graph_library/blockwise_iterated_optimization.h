#include "meta_heuristics.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <set>

const int MAX_LOCAL_SEARCH_ITERS = 100;  // Limit for local search iterations
const int MAX_BLOCK_ORDER_ITERS = 1000;    // Limit for block order optimization iterations

// Helper function to divide edges into blocks of size sqrt(n)
vector<vector<Edge>> divide_into_blocks(const vector<Edge>& edges, int block_size) {
    vector<vector<Edge>> blocks;
    for (int i = 0; i < edges.size(); i += block_size) {
        vector<Edge> block(edges.begin() + i, edges.begin() + std::min(i + block_size, (int)edges.size()));
        blocks.push_back(block);
    }
    return blocks;
}

vector<vector<Edge>> divide_into_blocks_odd(const vector<Edge>& edges, int block_size) {
    vector<vector<Edge>> blocks;

    if (edges.empty() || block_size <= 0) {
        return blocks; // Handle edge cases: empty input or invalid block size
    }

    // Calculate the size of the first block
    int first_block_size = std::max(1, block_size / 2); // Ensure at least one edge in the first block
    int remaining_size = edges.size();

    // Create the first block
    blocks.push_back(vector<Edge>(edges.begin(), edges.begin() + std::min(first_block_size, remaining_size)));
    remaining_size -= first_block_size;

    // Create the subsequent blocks of size block_size
    for (int i = first_block_size; i < edges.size(); i += block_size) {
        vector<Edge> block(edges.begin() + i, edges.begin() + std::min(i + block_size, (int)edges.size()));
        blocks.push_back(block);
    }

    return blocks;
}


vector<Edge> concatenate_blocks(const vector<vector<Edge> > blocks) {
    vector<Edge> ret;
    for (auto block : blocks) {
        ret.insert(ret.end(), block.begin(), block.end());
    }
    return ret;
}

double calculate_cost(Graph *graph, const vector<Edge> &sequence) {
    return dynamic_programming(graph, sequence).first[0][0];
}

double calculate_cost(Graph *graph, const vector<vector<Edge> > &blocks) {
    return calculate_cost(graph, concatenate_blocks(blocks));
}


pair<vector<Edge>, double> Random_Tour(Graph *graph) {
    auto edges = graph->edges;
    // Create a random number generator
    std::random_device rd;  // Seed for the random number generator
    std::mt19937 g(rd());   // Mersenne Twister engine initialized with rd

    // Shuffle the vector
    std::shuffle(edges.begin(), edges.end(), g);
    auto cost = dynamic_programming(graph, edges).first[0][0];
    return {edges, cost};
}

pair< vector<Edge>, double> Greedy_Sorting_Heuristic(Graph *graph) {
	// Information
	const int num_nodes = graph -> num_nodes;
	const int m = graph -> num_deliver_edges;
		
	vector<Edge> edges;
	edges.clear();
	for (int k = 0; k < m; ++k) {
		edges.push_back(Edge(graph -> deliver_edges[k]));
	}

	// Sort the list of edges
	sort(edges.begin(), edges.end());
    double cost = dynamic_programming(graph, edges).first[0][0];
    return {edges, cost};
}

vector<Edge> Block_2_OPT(Graph *graph, const vector<vector<Edge> > &blocks, int index) {
    auto block = blocks[index];
    auto lblocks = blocks;
    int m = block.size();

    auto best_block = block;
    double best_cost = calculate_cost(graph, blocks);

    for (int i = 0; i < m; i++) {
        for (int j=i + 1; j < m; j++) {
            swap(block[i], block[j]);
            lblocks[index] = block;
            auto new_cost = calculate_cost(graph, lblocks);
            if (new_cost < best_cost) {
                best_cost = new_cost;
                best_block = block;
            }
            swap(block[i], block[j]);
        }
        block = best_block;
    }
    return block;
}

vector<Edge> Optimize_Block(Graph *graph, const vector<vector<Edge> > &blocks, int index) {
    assert(index >= 0 && index < blocks.size());
    return Block_2_OPT(graph, blocks, index);
}

pair< vector<Edge>, double> Method_2_EXCHANGE_BLOCK(Graph *graph, const vector<Edge> sigma) {
    double best = calculate_cost(graph, sigma);
    vector<Edge> result = sigma;

    int best_i = -1, best_j = -1;

    // Search
    for (int i = 0; i < sigma.size(); ++i) {
        for (int j = i + 1; j < sigma.size(); ++j) {
            // Reverse the order of edges from i-th to j-th
            vector<Edge> sequence;
            sequence.clear();
            for (int k = 0; k < i; ++k) {
                sequence.push_back(Edge(sigma[k]));
            }
            for (int k = j; k >= i; --k) {
                sequence.push_back(Edge(sigma[k]));
            }
            for (int k = j + 1; k < sigma.size(); ++k) {
                sequence.push_back(Edge(sigma[k]));
            }
            assert(sequence.size() == sigma.size());

            // Dynamic programming
            pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sequence);

            const double cost = dp.first[0][0];
            if (cost < best) {
                best = cost;
                result = sequence;
                best_i = i;
                best_j = j;
            }
        }
    }
    cout << "best i is " << best_i << ", best_j is " << best_j << ", distance is " << best_j - best_i << " blockj size is " << sqrt(sigma.size()) << endl;
    return make_pair(result, best);
}


// Block-based optimization function
pair<vector<Edge>, double> Blockwise_Iterated_Optimization_2(Graph *graph, int block_size=-1, std::string init_heuristics = "GCH", int max_iterations = 10) {
    // Step 1: Create initial solution
    pair<vector<Edge>, double> greedy;
    if (init_heuristics == "GCH") {
        greedy = Greedy_Constructive_Heuristic(graph);
    }
    else if (init_heuristics == "GSH") {
        greedy = Greedy_Sorting_Heuristic(graph);
    }
    else if (init_heuristics == "RND") {
        greedy = Random_Tour(graph);
    }
    vector<Edge> sigma_star = greedy.first;
    double best_cost = greedy.second;
    
    if (block_size == -1) {
        int n = sigma_star.size();
        block_size = sqrt(n);
    }

    vector<vector<Edge>> blocks;
    vector<Edge> sigma_temp = sigma_star;

    // Step 3: Optimize each block independently
    for (int iter = 0; iter < max_iterations; iter++) {
        cout << "Iteration " << iter << ": " << endl;
        cout << "Initial cost is " << calculate_cost(graph, sigma_star) << endl;
        sigma_temp = Method_2_EXCHANGE_BLOCK(graph, sigma_star).first;
        cout << "Initial cost after Method_2_EXCHANGE_BLOCK is " << calculate_cost(graph, sigma_temp) << endl;
        blocks = divide_into_blocks(sigma_temp, block_size);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = Optimize_Block(graph, blocks, i);
        }
        sigma_temp = concatenate_blocks(blocks);
        auto new_cost = calculate_cost(graph, sigma_temp);
        cout << "The new cost is " << new_cost << endl;
        if (new_cost < best_cost) {
            cout << "improved at k = " << iter << endl;
            best_cost = new_cost;
            sigma_star = sigma_temp;
        }
    }

    return make_pair(sigma_star, best_cost);
}

pair< vector<Edge>, double> Method_2_EXCHANGE_BLOCK_3(Graph *graph, const vector<Edge> sigma) {
    double best = calculate_cost(graph, sigma);
    vector<Edge> result = sigma;

    int block_size = sqrt(sigma.size());

    int best_i = -1, best_j = -1;

    // Search
    for (int i = rand() % block_size; i < sigma.size(); i+=block_size) {
        for (int j = i + block_size; j < sigma.size(); j+=block_size) {
            // Reverse the order of edges from i-th to j-th
            vector<Edge> sequence;
            sequence.clear();
            for (int k = 0; k < i; ++k) {
                sequence.push_back(Edge(sigma[k]));
            }
            for (int k = j; k >= i; --k) {
                sequence.push_back(Edge(sigma[k]));
            }
            for (int k = j + 1; k < sigma.size(); ++k) {
                sequence.push_back(Edge(sigma[k]));
            }
            assert(sequence.size() == sigma.size());

            // Dynamic programming
            pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sequence);

            const double cost = dp.first[0][0];
            if (cost < best) {
                best = cost;
                result = sequence;
                best_i = i;
                best_j = j;
            }
        }
    }
    cout << "best i is " << best_i << ", best_j is " << best_j << ", distance is " << best_j - best_i << " blockj size is " << sqrt(sigma.size()) << endl;
    return make_pair(result, best);
}

// Block-based optimization function
pair<vector<Edge>, double> Blockwise_Iterated_Optimization_3(Graph *graph, int block_size=-1, std::string init_heuristics = "GCH", int max_iterations = 50) {
    // Step 1: Create initial solution
    pair<vector<Edge>, double> greedy;
    if (init_heuristics == "GCH") {
        greedy = Greedy_Constructive_Heuristic(graph);
    }
    else if (init_heuristics == "GSH") {
        greedy = Greedy_Sorting_Heuristic(graph);
    }
    else if (init_heuristics == "RND") {
        greedy = Random_Tour(graph);
    }
    vector<Edge> sigma_star = greedy.first;
    double best_cost = greedy.second;
    
    if (block_size == -1) {
        int n = sigma_star.size();
        block_size = sqrt(n);
    }

    vector<vector<Edge>> blocks;
    vector<Edge> sigma_temp = sigma_star;

    // Step 3: Optimize each block independently
    for (int iter = 0; iter < max_iterations; iter++) {
        cout << "Iteration " << iter << ": " << endl;
        cout << "Initial cost is " << calculate_cost(graph, sigma_star) << endl;
        sigma_temp = sigma_star;
        cout << "Initial cost after Method_2_EXCHANGE_BLOCK is " << calculate_cost(graph, sigma_temp) << endl;
        blocks = divide_into_blocks(sigma_temp, block_size);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = Optimize_Block(graph, blocks, i);
        }
        sigma_temp = concatenate_blocks(blocks);
        auto new_cost = calculate_cost(graph, sigma_temp);
        cout << "The new cost is " << new_cost << endl;
        if (new_cost < best_cost) {
            cout << "improved at k = " << iter << endl;
            best_cost = new_cost;
            sigma_star = sigma_temp;
        }
    }

    return make_pair(sigma_star, best_cost);
}