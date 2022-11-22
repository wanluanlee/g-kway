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
    // TODO: const std::string&
    graph(const std::string& input_path);
    ~graph();
    void parse();

    // TODO: std::vector<int>&
    std::vector<int>& get_adjncy();
    std::vector<int>& get_adjncy_source();
    std::vector<int>& get_adjp();
    std::vector<int>& get_adjwgt();
    std::vector<int>& get_vwgt();
    size_t get_num_vertex();
    size_t get_num_edge();
    int get_max_degree();
    void allocate_gpu_memory();
    //int* get_d_adjncy();
    //int* get_d_adjp();
    //int* get_d_adjwgt();
    //int* get_d_vwgt();

private:
    size_t _num_vertex = 0;
    size_t _num_edge = 0;
    int _max_degree = 0;
    std::string _input_path;
    std::vector<int> _adjncy; //adjacency list
    std::vector<int> _adjp; //adjacency pointer
    std::vector<int> _adjwgt; //edge weight
    std::vector<int> _vwgt; //vertx weight
    std::vector<int> _adjncy_source;
};

// TODO: understand what the difference is between this and yours
// in-place construction
graph::graph(const std::string& input_path) : 
  _input_path {input_path} {
}

graph::~graph() {
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
        _adjp.resize(_num_vertex + 1, 0);
        _vwgt.resize(_num_vertex, 0);
        _adjncy.resize(2 * _num_edge, 0);
        _adjncy_source.resize(2 * _num_edge, 0);
        _adjwgt.resize(2 * _num_edge, 0);
      }
      else {
        int degree_count = 0;
        while (ss >> word) {
         // std::cout << "word: " << word << std::endl;
          degree_count++;
          // TODO: _adjncy[adjncy_count] vs _adjncy.at(adjncy_count)
          _adjncy[adjncy_count] = std::stoi(word);
          _adjncy_source[adjncy_count] = line_number - 1;
          //printf("_adjncy_source at i %d, is %d \n",adjncy_count, _adjncy_source[adjncy_count]);
          _adjwgt[adjncy_count] = 1;
          adjncy_count++;
        }
        if(_max_degree < degree_count) {_max_degree = degree_count;}
        if(vertex_count < _num_vertex) {
          _adjp[vertex_count] = adjncy_count;
          _adjwgt[vertex_count] = 1;
          vertex_count++;
        }
      }
      line_number++;
    }
  }
  file.close();
  _adjp[_adjp.size() - 1] = adjncy_count;
}

std::vector<int>& graph::get_adjncy() {
  return _adjncy;
}

std::vector<int>& graph::get_adjncy_source() {
  return _adjncy_source;
}

std::vector<int>& graph::get_adjp() {
  return _adjp;
}

std::vector<int>& graph::get_adjwgt() {
  return _adjwgt;
}

std::vector<int>& graph::get_vwgt() {
  return _vwgt;
}

size_t graph::get_num_vertex() {
  return _num_vertex;
}

size_t graph::get_num_edge() {
  return _num_edge;
}

int graph::get_max_degree() {
  return _max_degree;
}

#endif //G_KWAY_GRAPH_HPP
