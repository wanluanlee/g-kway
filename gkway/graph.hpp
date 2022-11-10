//
// Created by Wan Luan Lee on 11/8/22.
//

#ifndef G_KWAY_GRAPH_HPP
#define G_KWAY_GRAPH_HPP
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

class graph {
public:
    graph(std::string input_path);
    void parse();
    std::vector<int> get_adjncy();
    std::vector<int> get_adjp();
    std::vector<int> get_adjwgt();
    std::vector<int> get_vwgt();
    int get_num_vertex();
    int get_num_edge();
    int get_max_degree();

private:
    int _num_vertex = 0;
    int _num_edge = 0;
    int _max_degree = 0;
    std::string _input_path;
    std::vector<int> _adjncy; //adjacency list
    std::vector<int> _adjp; //adjacency pointer
    std::vector<int> _adjwgt; //edge weight
    std::vector<int> _vwgt; //vertx weight

};


#endif //G_KWAY_GRAPH_HPP
