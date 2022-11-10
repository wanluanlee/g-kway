//
// Created by Wan Luan Lee on 11/8/22.
//

#include "graph.hpp"

// TODO: understand what the difference is between this and yours
// in-place construction
graph::graph(std::string input_path) : 
  _input_path {input_path} {
}

// TODO: tabe size = 2
void graph::parse() {
    std::ifstream file(_input_path);
    std::string line;
    int line_number = 0;
    int adjncy_count = 0;
    int vertex_count = 1;
    if(file.is_open()) {
        //std::cout << "file open " << std::endl;
        while (std::getline(file,line)) {
            //std::cout << "line number " << line_number << std::endl;
            std::istringstream ss(line);
            std::string word;
            if(line_number == 0) {
                ss >> _num_vertex;
                ss >> _num_edge;
                _adjp.resize(_num_vertex, 0);
                _vwgt.resize(_num_vertex, 0);
                _adjncy.resize(2 * _num_edge, 0);
                _adjwgt.resize(2 * _num_edge, 0);
            }
            else {
                int degree_count = 0;
                while (ss >> word) {
                   // std::cout << "word: " << word << std::endl;
                     degree_count++;

                    // TODO: _adjncy[adjncy_count] vs _adjncy.at(adjncy_count)
                    _adjncy.at(adjncy_count) = std::stoi(word);
                    _adjwgt.at(adjncy_count) = 1;
                    adjncy_count++;
                }
                if(_max_degree < degree_count) {_max_degree = degree_count;}
                if(vertex_count < _num_vertex) {
                    _adjp.at(vertex_count) = adjncy_count;
                    _adjwgt.at(vertex_count) = 1;
                    vertex_count++;
                }
            }
            line_number++;
        }
    }
    file.close();

    //for(auto&& i : _adjncy) {
        //std::cout << i << "  ";
    //}
    //std::cout << " new line" << std::endl;
    //for(auto&& i : _adjp) {
        //std::cout << i << "  ";
    //}
}

std::vector<int> graph::get_adjncy() {
    return _adjncy;
}

std::vector<int> graph::get_adjp() {
    return _adjp;
}

std::vector<int> graph::get_adjwgt() {
    return _adjwgt;
}

std::vector<int> graph::get_vwgt() {
    return _vwgt;
}

int graph::get_num_vertex() {
    return _num_vertex;
}

int graph::get_num_edge() {
    return _num_edge;
}

int graph::get_max_degree() {
    return _max_degree;
}
