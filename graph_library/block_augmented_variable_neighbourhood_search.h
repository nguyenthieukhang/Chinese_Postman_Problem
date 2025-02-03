#include "blockwise_iterated_optimization.h"

pair< vector<Edge>, double> Block_Augmented_Variable_Neighborhood_Search(Graph *graph, const int k_max = 30, const bool verbose = false) {
    // Greedy constructive heuristic
    pair< vector<Edge>, double > greedy = Greedy_Constructive_Heuristic(graph);
    vector<Edge> sigma_star = greedy.first;
    double best = greedy.second;
    const int block_size = sqrt(sigma_star.size());

    // Iterative
    for (int k = 1; k <= k_max; ++k) {
        // Random exchange
        vector<Edge> sigma = sigma_star;
		
		// Compute the cost of sigma
		pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sigma);
		double cost = dp.first[0][0];

		// 2-EXCHANGE
        pair< vector<Edge>, double> Result_2_EXCHANGE = Method_2_EXCHANGE(graph, sigma);
		
		if (Result_2_EXCHANGE.second < cost) {
			cost = Result_2_EXCHANGE.second;
			sigma = Result_2_EXCHANGE.first;
		} else {
        	// 1-OPT
        	pair< vector<Edge>, double> Result_1_OPT = Method_1_OPT(graph, sigma);

			if (Result_1_OPT.second < cost) {
				cost = Result_1_OPT.second;
				sigma = Result_1_OPT.first;		
			} else {
				// 2-OPT
        		pair< vector<Edge>, double> Result_2_OPT = Method_2_OPT(graph, sigma);

				if (Result_2_OPT.second < cost) {
					cost = Result_2_OPT.second;
					sigma = Result_2_OPT.first;
				}
			}
		}

		auto blocks = divide_into_blocks(sigma, block_size);
        for (size_t i = 0; i < blocks.size(); ++i) {
            blocks[i] = Optimize_Block(graph, blocks, i);
        }
        sigma = concatenate_blocks(blocks);
        cost = calculate_cost(graph, sigma);

    	// Update
		if (cost < best) {
			best = cost;
			sigma_star = sigma;
		}

		if (verbose) {
            cout << "Done " << k << " iterations." << endl;
        }
	}

    return make_pair(sigma_star, best);
}