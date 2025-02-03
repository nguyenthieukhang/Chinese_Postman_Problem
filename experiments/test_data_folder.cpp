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
#include "../graph_library/meta_heuristics.h"

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
    std::cout << "Here" << std::endl;
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

pair<vector<Edge>, double> Optimize(Graph* graph) {
    return new_greedy_constructive_heuristics(graph);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <graph_folder> <output_file>" << endl;
        return 1;
    }

    string folder_path = argv[1];
    string output_file = argv[2];

    // Check if file exists, if not, write the header
    bool file_exists = fs::exists(output_file);
    ofstream csv_file(output_file, ios::app);
    if (!file_exists) {
        csv_file << "Name,Number of nodes,Number of edges,Number of deliver edges,Cost,Time (ms),Time (s)" << endl;
    }

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            string file_path = entry.path().string();
            string file_name = entry.path().filename().string();
            
            cout << "Processing file: " << file_name << endl;

            Graph* graph = new Graph(file_path.c_str());

            int num_nodes = graph->num_nodes;
            int num_edges = graph->num_edges;
            int num_deliver_edges = graph->num_deliver_edges;

            graph->Floyd_algorithm();
            graph->Dijkstra_algorithm();

            double cost, ms_time;
            tie(cost, ms_time) = run_heuristic(graph, "new_greedy_constructive_heuristics", Optimize);
            double sec_time = ms_time / 1000.0;

            csv_file << file_name << "," << num_nodes << "," << num_edges << "," 
                     << num_deliver_edges << "," << cost << "," << ms_time << "," << sec_time << endl;

            csv_file.flush(); // Ensure data is written immediately
            delete graph;
        }
    }

    csv_file.close();
    cout << "Results appended to " << output_file << endl;

    return 0;
}