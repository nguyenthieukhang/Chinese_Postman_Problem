// Testing the Variable Neighborhood Search (VNS) for the load-dependent Chinese postman problem
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
#include <assert.h>

#include "../graph_library/Graph.h"
#include "../graph_library/blockwise_iterated_optimization.h"

using namespace std;
using namespace std::chrono;

// Function to run and time a heuristic
void run_heuristic(Graph* graph, const string& heuristic_name,
                   pair<vector<Edge>, double> (*heuristic_func)(Graph*)) {
    cout << "Running heuristic: " << heuristic_name << endl;
    
    // Start timer
    auto start = high_resolution_clock::now();

    // Run heuristic
    pair<vector<Edge>, double> result = heuristic_func(graph);
    const vector<Edge> sigma = result.first;
    const double cost = result.second;

    // Run dynamic programming to verify the cost
    pair<vector<vector<double>>, vector<int>> dp = dynamic_programming(graph, sigma);
    cout << "Cost (" << heuristic_name << "): " << dp.first[0][0] << endl;
    cout << "Cost from the function is " << cost << ". And cost from outer is " << dp.first[0][0] << std::endl;
    assert(abs(cost - dp.first[0][0]) < 1e-6);

    // Stop timer and print duration
    auto stop = high_resolution_clock::now();
    auto ms_duration = duration_cast<milliseconds>(stop - start);
    cout << "Running time (" << heuristic_name << ") - milliseconds: " << ms_duration.count() << endl;
    auto duration = duration_cast<seconds>(stop - start);
    cout << "Running time (" << heuristic_name << ") - seconds: " << duration.count() << endl << endl;
}

int main(int argc, char** argv) {
    // Fix random seed
    srand(0);

    // Starting timepoint for entire program
    auto program_start = high_resolution_clock::now();

    cout << "File name: " << argv[1] << endl;

    // Load the input graph
    Graph* graph = new Graph(argv[1]);

    cout << "Number of nodes: " << graph->num_nodes << endl;
    cout << "Number of edges: " << graph->num_edges << endl;
    cout << "Number of deliver-edges (q > 0): " << graph->num_deliver_edges << endl;

    // Run the Floyd's algorithm
    graph->Floyd_algorithm();

    // List of heuristic functions to test
    vector<pair<string, pair<vector<Edge>, double> (*)(Graph*)>> heuristics = {
        {"Blockwise Iterated Optimization", Blockwise_Iterated_Optimization}
    };

    // Run each heuristic and print results
    for (const auto& [name, func] : heuristics) {
        run_heuristic(graph, name, func);
    }

    // Ending timepoint for entire program
    auto program_stop = high_resolution_clock::now();
    auto program_duration = duration_cast<seconds>(program_stop - program_start);
    cout << "Total running time (seconds): " << program_duration.count() << endl << endl;

    delete graph;
    return 0;
}
