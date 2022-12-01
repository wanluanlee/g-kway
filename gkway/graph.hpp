#pragma once

#include "../declarations.h"

namespace gk { // begin of namespace gk ============================================

// ======================================================
//
// Declaration of Graph
//
// This class is used to grap graph information.
//
// ======================================================

class Graph {

  public:

      // TODO: const std::string&
      Graph(const std::string& input_path);
      ~Graph();

      // TODO: std::vector<int>&
      const int* get_adjncy();
      const int* get_adjncy_source();
      const int* get_adjp();
      const int* get_adjwgt();
      const int* get_vwgt();
      size_t get_num_vertex();
      size_t get_num_edge();
      void allocate_gpu_memory();
      //int* get_d_adjncy();
      //int* get_d_adjp();
      //int* get_d_adjwgt();
      //int* get_d_vwgt();

  private:

      void _parse();

      size_t _num_vertex = 0;
      size_t _num_edge = 0;
      std::string _input_path;
      std::vector<int> _adjncy; //adjacency list
      std::vector<int> _adjp; //adjacency pointer
      std::vector<int> _adjwgt; //edge weight
      std::vector<int> _vwgt; //vertx weight
      std::vector<int> _adjncy_source;
};

// ======================================================
//
// Definition of Graph
//
// ======================================================

// TODO: understand what the difference is between this and yours
// in-place construction
Graph::Graph(const std::string& input_path) : 
  _input_path {input_path} {
   _parse();
}

Graph::~Graph() {
}

// TODO: tabe size = 2
void Graph::_parse() {
  std::ifstream file(_input_path);
  std::string line;
  int line_number = 0;
  int adjncy_count = 0;
  int vertex_count = 1;
  int format_count = 0;
  int if_weighted_edge = false;
  if(file.is_open()) {
    //std::cout << "file open " << std::endl;
    while (std::getline(file,line)) {
        //std::cout << "line number " << line_number << std::endl;
      std::istringstream ss(line);
      std::string word;
      if(line_number == 0) {
        while(ss >> word) {
          if(format_count == 0) {
           _num_vertex = std::stoi(word); 
          }
          else if(format_count == 1) {
           _num_edge = std::stoi(word); 
          }
          else if(format_count == 2) {
           if(word == "1") {
             printf("weighted graph \n");
             if_weighted_edge = true;
           }
          }
          format_count++;
        }
        printf("number of vertex %d \n", _num_vertex);
        printf("number of edge %d \n", _num_edge);
        //ss >> _num_vertex;
        //ss >> _num_edge;
        _adjp.resize(_num_vertex + 1, 0);
        _vwgt.resize(_num_vertex, 1);
        _adjncy.resize(2 * _num_edge, 0);
        _adjncy_source.resize(2 * _num_edge, 0);
        _adjwgt.resize(2 * _num_edge, 0);
      }
      else {
        int token_count = 0;
        while (ss >> word) {
          if(if_weighted_edge == false) {
            // TODO: _adjncy[adjncy_count] vs _adjncy.at(adjncy_count)
            _adjncy[adjncy_count] = std::stoi(word);
            _adjncy_source[adjncy_count] = line_number - 1;
            _adjwgt[adjncy_count] = 1;
            adjncy_count++;
          }
           /*
           * If the Graph's edges have weightes, each line will have the following format
           * v1 e1 v2 e2 ...
           * Where v1 is the first connected vertex and e1 is the edge weight betwwen v1 and source vertex
           * */
          else {
            if(token_count %2 == 0) {
            //std::cout << "count: " << token_count << ", word: " << word << "\n";
              _adjncy[adjncy_count] = std::stoi(word);
              _adjncy_source[adjncy_count] = line_number - 1;
              //printf("adjncy at %d is %d \n", adjncy_count, _adjncy[adjncy_count]);
            }
            else {
              _adjwgt[adjncy_count] = std::stoi(word);
              adjncy_count++;
            }
            token_count++;
          }
        }
        if(vertex_count < _num_vertex) {
          _adjp[vertex_count] = adjncy_count;
          //_vwgt[vertex_count] = 1;
          //_adjwgt[vertex_count] = 1;
          vertex_count++;
        }
      }
      line_number++;
    }
  }
  file.close();
  _adjp[_adjp.size() - 1] = adjncy_count;
  //for(auto i : _adjwgt) {
    //printf("_adjwgt is %d \n", i);
  //}
  //for(auto i : _vwgt) {
    //printf("_vwgt is %d \n", i);
  //}
  //for(auto i : _adjncy) {
    //printf("_adjncy is %d \n", i);
  //}
  //for(auto i : _adjncy_source) {
    //printf("_adjncy_source is %d \n", i);
  //}
}

const int* Graph::get_adjncy() {
  return _adjncy.data();
}

const int* Graph::get_adjncy_source() {
  return _adjncy_source.data();
}

const int* Graph::get_adjp() {
  return _adjp.data();
}

const int* Graph::get_adjwgt() {
  return _adjwgt.data();
}

const int* Graph::get_vwgt() {
  return _vwgt.data();
}

size_t Graph::get_num_vertex() {
  return _num_vertex;
}

size_t Graph::get_num_edge() {
  return _num_edge;
}


} // end of namespace gk ==========================================
