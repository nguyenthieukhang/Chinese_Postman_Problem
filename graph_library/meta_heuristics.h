// Meta-heuristics (e.g., greedy, ILS, and VNS) for the load-dependent Chinese postman problem
// Author: Dr. Truong Son Hy
// Copyright 2023

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>
#include <assert.h>
#include <thread>
#include <map>
#include <set>

#include "../graph_library/Graph.h"

using namespace std;

void my_assert(bool condition, const std::string& message) {
    if (!condition) {
        std::cout << "Assertion failed: " << message << std::endl;
        exit(-1);
    }
}

// +--------------------------------+
// | Greedy Constructive Heuristics |
// +--------------------------------+
pair< vector<Edge>, double> Greedy_Constructive_Heuristic(Graph *graph) {
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

	// Result
	vector<Edge> sigma_star;
	sigma_star.clear();
	double best;

	// Algorithm
	for (int i = 0; i < m; ++i) {
		double z_min = INF;
		vector<Edge> sigma_prime;

		// The i-th edge
		Edge e = Edge(edges[i]);

		for (int j = 0; j <= i; ++j) {
			// Create another list of edges by adding the i-th edge into the j-th position of sigma_star
			vector<Edge> sigma;
			sigma.clear();
			for (int k = 0; k < j; ++k) {
				sigma.push_back(Edge(sigma_star[k]));
			}
			sigma.push_back(e);
			for (int k = j + 1; k <= i; ++k) {
				sigma.push_back(Edge(sigma_star[k - 1]));
			}

			// Dynamic programming
			pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sigma);

			// Update
			const double z = dp.first[0][0];
			if (z < z_min) {
				z_min = z;
				sigma_prime = sigma;
			}
		}

		// Update sigma_star
		sigma_star = sigma_prime;
		best = z_min;
	}

	return make_pair(sigma_star, best);
}


// +-----------------------------------------------+
// | Greedy Constructive Heuristics (new proposal) |
// +-----------------------------------------------+
pair< vector<Edge>, double> Greedy_Constructive_Heuristic_2(Graph *graph) {
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

	// Mask
	vector<bool> mask;
	mask.clear();
	for (int k = 0; k < m; ++k) {
		mask.push_back(false);
	}

    // Result
    vector<Edge> sigma_star;
    sigma_star.clear();
    double best;

	// Algorithm
    for (int i = 0; i < m; ++i) {
		// If this edge is used already, move on
		if (mask[i] == true) {
			continue;
		}
		mask[i] = true;

		// We need to find where to put it in
        double z_min = INF;
        vector<Edge> sigma_prime;

        // The i-th edge
        Edge e = Edge(edges[i]);

		int position = -1;
		int start = -1;
		int finish = -1;

        for (int j = 0; j <= sigma_star.size(); ++j) {
            // Create another list of edges by adding the i-th edge into the j-th position of sigma_star
            vector<Edge> sigma;
            sigma.clear();
            for (int k = 0; k < j; ++k) {
                sigma.push_back(Edge(sigma_star[k]));
            }

			// Add the edge
            sigma.push_back(e);
            			
			for (int k = j; k < sigma_star.size(); ++k) {
                sigma.push_back(Edge(sigma_star[k]));
            }

            // Dynamic programming
            pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sigma);

            // Update
            const double z = dp.first[0][0];
            if (z < z_min) {
                z_min = z;
                sigma_prime = sigma;
				
				position = j;
				if (position == 0) {
					start = graph -> start_node;
				} else {
					if (dp.second[j - 1] == 0) {
						start = sigma_prime[j - 1].second;
					} else {
						start = sigma_prime[j - 1].first;
					}
				}

				if (dp.second[j] == 0) {
					finish = sigma_prime[j].first;
				} else {
					finish = sigma_prime[j].second;
				}
            }
        }

		assert(position != -1);
	
		// Update sigma_star with the shortest path trick
		sigma_star.clear();
		for (int k = 0; k < position; ++k) {
			sigma_star.push_back(Edge(sigma_prime[k]));
		}

		// Add the shortest path
		assert(position != -1);
		assert(start != -1);
		assert(finish != -1);

		const vector<int> path = graph -> dijkstra_path[start][finish];

		for (int k = 1; k < path.size(); ++k) {
			const int u = path[k - 1];
			const int v = path[k];
			for (int t = i + 1; t < m; ++t) {
				if (!mask[t]) {
					if ((u == edges[t].first) && (v == edges[t].second)) {
						sigma_star.push_back(Edge(edges[t]));
						mask[t] = true;
						break;
					}
					if ((u == edges[t].second) && (v == edges[t].first)) {
                        sigma_star.push_back(Edge(edges[t]));
						mask[t] = true;
                        break;
                    }
				}
			}
		}

		// Add the rest
		for (int k = position; k < sigma_prime.size(); ++k) {
			sigma_star.push_back(Edge(sigma_prime[k]));
		}

		// Dynamic programming
		pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sigma_star);
		best = dp.first[0][0];
    }

    return make_pair(sigma_star, best);
}


// +-------------------------------------------+
// | Random exchange with probability 1/factor |
// +-------------------------------------------+
vector<Edge> random_exchange(const vector<Edge> edges, const int factor = 5) {
	// Copy
	vector<Edge> result;
	result.clear();
	for (int i = 0; i < edges.size(); ++i) {
		result.push_back(Edge(edges[i]));
	}

	// Random exchange
	for (int i = 0; i < edges.size(); ++i) {
		if (rand() % factor == 0) {
			const int j = rand() % edges.size();
			swap(result[i], result[j]);
		}
	}

	return result;
}


// +---------------------------+
// | 2-MOVE-OPT (new proposal) |
// +---------------------------+

// For single-thread
pair< vector<Edge>, double> Method_2_MOVE_OPT(Graph *graph, const vector<Edge> sigma) {
    double best = INF;
    vector<Edge> result;

	// Search
	for (int i = 0; i < sigma.size(); ++i) {
		for (int j = 0; j < sigma.size(); ++j) {
			if (i == j) {
				continue;
			}

			// The rest elements except i-th and j-th
			vector<Edge> A;
			A.clear();
			for (int k = 0; k < sigma.size(); ++k) {
				if ((k != i) && (k != j)) {
					A.push_back(Edge(sigma[k]));
				}
			}
			assert(A.size() == sigma.size() - 2);

			// Search for the best place to put the i-th in
			vector<Edge> B;
			B.clear();
			double B_value = INF;

			for (int k = 0; k < A.size(); ++k) {
				vector<Edge> candidate;
				candidate.clear();
				for (int t = 0; t < k; ++t) {
					candidate.push_back(Edge(A[t]));
				}
				candidate.push_back(Edge(sigma[i]));
				for (int t = k; t < A.size(); ++t) {
					candidate.push_back(Edge(A[t]));
				}

				// Update
				pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, candidate);
				const double cost = dp.first[0][0];
				if (cost < B_value) {
					B_value = cost;
					B = candidate;
				}
			}

			assert(B.size() == sigma.size() - 1);

			// Search for the best place to put the j-th in
			vector<Edge> C;
			C.clear();
			double C_value = INF;

			for (int k = 0; k < B.size(); ++k) {
				vector<Edge> candidate;
				candidate.clear();
				for (int t = 0; t < k; ++t) {
					candidate.push_back(Edge(B[t]));
				}
				candidate.push_back(Edge(sigma[j]));
				for (int t = k; t < B.size(); ++t) {
					candidate.push_back(Edge(B[t]));
				}

				// Update
				pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, candidate);
				const double cost = dp.first[0][0];
				if (cost < C_value) {
					C_value = cost;
					C = candidate;
				}
			}

			assert(C.size() == sigma.size());

			// Update
			if (C_value < best) {
				best = C_value;
				result = C;
			}
		}
	}

    return make_pair(result, best);
}


// +-------+
// | 1-OPT |
// +-------+

// For single-thread
pair< vector<Edge>, double> Method_1_OPT(Graph *graph, const vector<Edge> sigma) {
	double best = INF;
	vector<Edge> result;
	
	// Search
	for (int i = 0; i < sigma.size(); ++i) {
		Edge e = sigma[i];

		// Move the i-th edge to the j-th position
		for (int j = 0; j < sigma.size(); ++j) {
			vector<Edge> sequence;
			sequence.clear();
			for (int k = 0; k < j; ++k) {
				if (k != i) {
					sequence.push_back(Edge(sigma[k]));
				}
			}
			sequence.push_back(Edge(e));
			for (int k = j; k < sigma.size(); ++k) {
				if (k != i) {
					sequence.push_back(Edge(sigma[k]));
				}
			}
			assert(sequence.size() == sigma.size());

			// Dynamic programming
			pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sequence);

			// Update
			const double cost = dp.first[0][0];
			if (cost < best) {
				best = cost;
				result = sequence;
			}
		}
	}

	return make_pair(result, best);
}


// +-------+
// | 2-OPT |
// +-------+

// For single-thread
pair< vector<Edge>, double> Method_2_OPT(Graph *graph, const vector<Edge> sigma) {
    double best = INF;
    vector<Edge> result;

	// Copy
	vector<Edge> sequence;
	sequence.clear();
	for (int k = 0; k < sigma.size(); ++k) {
		sequence.push_back(Edge(sigma[k]));
	}

	// Search
    for (int i = 0; i < sigma.size(); ++i) {
        Edge e = sigma[i];

        // Swap the i-th and j-th edges
        for (int j = i + 1; j < sigma.size(); ++j) {
			// Swap
			swap(sequence[i], sequence[j]);

            // Dynamic programming
            pair< vector< vector<double> >, vector<int> > dp = dynamic_programming(graph, sequence);

            const double cost = dp.first[0][0];
            if (cost < best) {
                best = cost;
                result = sequence;
            }

			// Swap back
			swap(sequence[j], sequence[i]);
        }
    }

    return make_pair(result, best);
}


// +------------+
// | 2-EXCHANGE |
// +------------+

// For single-thread
pair< vector<Edge>, double> Method_2_EXCHANGE(Graph *graph, const vector<Edge> sigma) {
    double best = INF;
    vector<Edge> result;

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
            }
        }
    }

    return make_pair(result, best);
}


// +--------------------------------------------+
// | Iterated Local Search (ILS) Meta-heuristic |
// +--------------------------------------------+

// Single-thread implementation
pair< vector<Edge>, double> Iterated_Local_Search(Graph *graph, const int k_max = 75, const bool verbose = false) {
	// Greedy constructive heuristic
	pair< vector<Edge>, double > greedy = Greedy_Constructive_Heuristic(graph);
	vector<Edge> sigma_star = greedy.first;
	double best = greedy.second;

	// Iterative
	for (int k = 1; k <= k_max; ++k) {
		// Random exchange
		vector<Edge> sigma = random_exchange(sigma_star);
		
		// 1-OPT
		pair< vector<Edge>, double> Result_1_OPT = Method_1_OPT(graph, sigma);

		// 2-OPT
		pair< vector<Edge>, double> Result_2_OPT = Method_2_OPT(graph, sigma);

		// 2-EXCHANGE
		pair< vector<Edge>, double> Result_2_EXCHANGE = Method_2_EXCHANGE(graph, sigma);

		// Update
		if (Result_1_OPT.second < best) {
			best = Result_1_OPT.second;
			sigma_star = Result_1_OPT.first;
		}

		if (Result_2_OPT.second < best) {
            best = Result_2_OPT.second;
            sigma_star = Result_2_OPT.first;
        }

		if (Result_2_EXCHANGE.second < best) {
            best = Result_2_EXCHANGE.second;
            sigma_star = Result_2_EXCHANGE.first;
        }

		if (verbose) {
			cout << "Done " << k << " iterations." << endl;
		}
	}

	return make_pair(sigma_star, best);
}


// +-----------------------------------------------------------+
// | New proposal - Iterated Local Search (ILS) Meta-heuristic |
// +-----------------------------------------------------------+

// Single-thread implementation
pair< vector<Edge>, double> Iterated_Local_Search_2(Graph *graph, const int k_max = 75, const bool verbose = false) {
    // New Greedy Constructive Heuristic
    pair< vector<Edge>, double > greedy = Greedy_Constructive_Heuristic_2(graph);
    vector<Edge> sigma_star = greedy.first;
    double best = greedy.second;

    // Iterative
    for (int k = 1; k <= k_max; ++k) {
        // Random exchange
        vector<Edge> sigma = random_exchange(sigma_star);

        // 2-MOVE
        pair< vector<Edge>, double> Result_2_MOVE_OPT = Method_2_MOVE_OPT(graph, sigma);

        // Update
        if (Result_2_MOVE_OPT.second < best) {
            best = Result_2_MOVE_OPT.second;
            sigma_star = Result_2_MOVE_OPT.first;
        }

        if (verbose) {
            cout << "Done " << k << " iterations." << endl;
        }
    }

    return make_pair(sigma_star, best);
}


// +---------------------------------------------------+
// | Variable Neighborhood Search (VNS) Meta-heuristic |
// +---------------------------------------------------+
pair< vector<Edge>, double> Variable_Neighborhood_Search(Graph *graph, const int k_max = 75, const bool verbose = false) {
    // Greedy constructive heuristic
    pair< vector<Edge>, double > greedy = Greedy_Constructive_Heuristic(graph);
    vector<Edge> sigma_star = greedy.first;
    double best = greedy.second;

    // Iterative
    for (int k = 1; k <= k_max; ++k) {
        // Random exchange
        vector<Edge> sigma = random_exchange(sigma_star);
		
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

std::vector<Edge> get_path_edges(Graph* graph, int u, int v) {
	const auto vertices = graph->dijkstra_path[u][v];
	int cur = u;
	std::vector<Edge> edges;
	for (int i = 1; i < (int) vertices.size(); i++) {
		auto vertex = vertices[i];
		auto edge = graph->get_edge(cur, vertex);
		if (edge.first == -1) {edge = graph->get_edge(vertex, cur);}
		edges.push_back(edge);
		cur = vertex;
		my_assert(edges.back().first != -1, "This is wrong!");
	}
	return edges;
}

// Return the optimal edge direction and corresponding cost by using dynamic programming
std::pair<std::vector<int>, double> cost_tour(Graph* network, std::vector<Edge> edges) {
	auto dp = dynamic_programming(network, edges);
	return {dp.second, dp.first[0][0]};
}

// Greedy Constructive Heuristics Implementation
pair< vector<Edge>, double> new_greedy_constructive_heuristics(Graph* network) {
   // Lambda comparator for edges
    auto edge_comparator = [](const Edge& a, const Edge& b) {
        double productA = a.d * a.q;
        double productB = b.d * b.q;
        return productA > productB; // Larger products come first
    };

    // Multiset of edges with custom comparator
    std::multiset<Edge, decltype(edge_comparator)> edge_pool(edge_comparator);
	for (const auto& edge : network->deliver_edges) {
		edge_pool.insert(edge);
	}

    // Construct solution iteratively
    std::vector<Edge> constructed_solution;

    while (!edge_pool.empty()) {
        // Extract and remove the best edge from the pool
        Edge current_edge = *edge_pool.begin();
        edge_pool.erase(edge_pool.begin());

        // Apply the simple shift operator		
		auto min_cost = std::numeric_limits<double>::infinity();
		std::vector<Edge> best_solution = constructed_solution;
		int n = constructed_solution.size();
		int k_prime = -1;

		for (int i = 0; i <= n ; i++) {
			constructed_solution.insert(constructed_solution.begin() + i, current_edge);
			auto cost = cost_tour(network, constructed_solution).second;
			if (cost < min_cost) {
				min_cost = cost;
				best_solution = constructed_solution;
				k_prime = i;
			}
			constructed_solution.erase(constructed_solution.begin() + i);
		}
    	constructed_solution = best_solution;
        my_assert(k_prime >= 0, "k_prime must be >= 0");

		auto theres_more = k_prime != (int) constructed_solution.size() - 1;

    	// Update the path from k_prime - 1 to k_prime
		auto temp_direction = cost_tour(network, constructed_solution).first;
    	auto u = network->start_node;
    	if (k_prime != 0) {
    		my_assert(k_prime > 0, "k_prime must be > 0");
    		u = temp_direction[k_prime - 1]
					   ? constructed_solution[k_prime - 1].first
					   : constructed_solution[k_prime - 1].second;
    	}
        auto v = temp_direction[k_prime]
                   ? constructed_solution[k_prime].second
                   : constructed_solution[k_prime].first;

        int insert_position = k_prime;
		auto path = get_path_edges(network, u, v);
        for (const auto& edge : path) {
	        auto it = edge_pool.end();
        	for (it = edge_pool.begin(); it != edge_pool.end(); ++it) {
        		if ((*it) == edge) {break;}
        	}
            if (it != edge_pool.end()) {
                constructed_solution.insert(constructed_solution.begin() + insert_position, *it);
                edge_pool.erase(it);
                insert_position++;
            }
        	else {
        		int index = -1;
        		for (int i = insert_position + 1; i < (int) constructed_solution.size(); i++) {
					if (constructed_solution[i] == edge) {index = i; break;}
        		}
        		if (index == -1) {continue;}
        		constructed_solution.erase(constructed_solution.begin() + index);
				constructed_solution.insert(constructed_solution.begin() + insert_position, edge);
        		insert_position++;
        	}
        }

    	if (theres_more) {
			// Update the tour from the shortest path between k_prime and k_prime + 1
    		u = temp_direction[k_prime]
					   ? constructed_solution[insert_position].first
					   : constructed_solution[insert_position].second;
    		v = temp_direction[k_prime + 1]
				   ? constructed_solution[insert_position + 1].second
				   : constructed_solution[insert_position + 1].first;
    		my_assert(current_edge == constructed_solution[insert_position], "Insert position wrong!");
    		insert_position = insert_position + 1;
			path = get_path_edges(network, u, v);
    		for (const auto& edge : path) {
    			auto it = edge_pool.end();
    			for (it = edge_pool.begin(); it != edge_pool.end(); ++it) {
    				if ((*it) == edge) {break;}
    			}
    			if (it != edge_pool.end()) {
    				constructed_solution.insert(constructed_solution.begin() + insert_position, *it);
    				edge_pool.erase(it);
    				insert_position++;
    			}
    			else {
    				int index = -1;
    				for (int i = insert_position + 1; i < (int) constructed_solution.size(); i++) {
    					if (constructed_solution[i] == edge) {index = i; break;}
    				}
    				if (index == -1) {continue;}
    				constructed_solution.erase(constructed_solution.begin() + index);
    				constructed_solution.insert(constructed_solution.begin() + insert_position, edge);
    				insert_position++;
    			}
    		}
    	}
    }
	
	my_assert(constructed_solution.size() == network->deliver_edges.size(), "They must have the same size");

	// Check if the result contains distinct elements
	std::set<std::pair<int, int> > s;
	for (const auto& edge : constructed_solution) {s.insert({edge.first, edge.second});}
	my_assert(s.size() == network->deliver_edges.size(), "They have the same size");

    // Finalize and return the constructed solution
    auto direction = cost_tour(network, constructed_solution).first;
    std::vector<Edge> perm;
    auto cur = network->start_node;

	auto comp = [](const Edge& a, const Edge& b) { return std::make_pair(a.first, a.second) < std::make_pair(b.first, b.second);};

    std::map<Edge, bool, decltype(comp)> delivery_edges(comp);

    for (int i = 0; i < (int) constructed_solution.size(); i++) {
        if (delivery_edges.count(constructed_solution[i]) > 0) continue;
        auto next = direction[i]? constructed_solution[i].second : constructed_solution[i].first;
		auto path = get_path_edges(network, cur, next);
        for (auto edge : path) {
            if (edge.q > 0.0) {
                // This is a delivery edge
                if (delivery_edges.count(edge) == 0) {
                    delivery_edges[edge] = true;
                    perm.push_back(edge);
                }
            }
        }
        perm.push_back(constructed_solution[i]);
        delivery_edges[perm.back()] = true;
        cur = direction[i]? perm.back().first : perm.back().second;
        if (perm.size() == constructed_solution.size()) break;
    }

	auto cost = cost_tour(network, constructed_solution).second;

	return {constructed_solution, cost};
}