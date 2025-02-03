#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <filesystem>
#include <assert.h>
#include <chrono>

#include "../graph_library/Graph.h"
#include "../graph_library/blockwise_iterated_optimization.h"
// #include "../graph_library/block_augmented_ant_colony_optimization.h"
// #include "../graph_library/meta_heuristics.h"
// #include "../graph_library/ant_colony_optimization_multithreads.h"
// #include "../graph_library/block_augmented_variable_neighbourhood_search.h"
// #include "../graph_library/block_augmented_evolutionary_algorithm.h"
// #include "../graph_library/evolutionary_algorithm_multithreads.h"


using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

// Function to run and time a heuristic
pair<double, double> run_heuristic(Graph* graph, const string& heuristic_name,
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
    assert(abs(cost - dp.first[0][0]) < 1e-6);

    // Stop timer and calculate duration
    auto stop = high_resolution_clock::now();
    auto ms_duration = duration_cast<milliseconds>(stop - start);
    auto sec_duration = duration_cast<seconds>(stop - start);
    
    cout << "Cost (" << heuristic_name << "): " << dp.first[0][0] << endl;
    cout << "Running time (" << heuristic_name << ") - milliseconds: " << ms_duration.count() << endl;
    cout << "Running time (" << heuristic_name << ") - seconds: " << sec_duration.count() << endl << endl;

    return {cost, ms_duration.count()};
}

pair< vector<Edge>, double> Optimization_MultiThreads_(Graph* graph) {
    return Greedy_Constructive_Heuristic(graph);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Please specify a folder containing graph files." << endl;
        return 1;
    }

    // Path to folder containing input graph files
    string folder_path = argv[1];

    // Output CSV file
    ofstream csv_file("Greedy_Constructive_Heuristic_sample.csv");
    csv_file << "Name,Number of nodes,Number of edges,Number of deliver edges,Cost,Time (ms),Time (s)" << endl;

    // Loop through all files in the folder
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            string file_path = entry.path().string();
            string file_name = entry.path().filename().string();
            
            cout << "Processing file: " << file_name << endl;

            // Load the input graph
            Graph* graph = new Graph(file_path.c_str());

            // Collect graph statistics
            int num_nodes = graph->num_nodes;
            int num_edges = graph->num_edges;
            int num_deliver_edges = graph->num_deliver_edges;

            graph->Floyd_algorithm();

            // Run the heuristic function and capture cost and time
            double cost, ms_time;
            tie(cost, ms_time) = run_heuristic(graph, "Blockwise_Iterated_Optimization", Optimization_MultiThreads_);
            double sec_time = ms_time / 1000.0;

            // Write results to CSV
            csv_file << file_name << "," << num_nodes << "," << num_edges << "," 
                     << num_deliver_edges << "," << cost << "," << ms_time << "," << sec_time << endl;

            delete graph;
        }
    }

    csv_file.close();
    cout << "Results saved to Greedy_Constructive_Heuristic_sample=100.csv" << endl;

    return 0;
}
