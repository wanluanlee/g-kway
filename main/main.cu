#include "../declarations.h"
#include "../gkway/graph.hpp"
#include "../gkway/kernel.hpp"


int main(int argc, char** argv) {
  //parsing data to arra
  gk::Graph mygraph(argv[1]);

  size_t num_vertex = mygraph.get_num_vertex();
  size_t num_edge = mygraph.get_num_edge();    

  //const std::vector<int> adjp = mygraph.get_adjp();
  const int *h_adjp = mygraph.get_adjp();
  const int *h_vwgt = mygraph.get_vwgt();
  const int *h_adjncy = mygraph.get_adjncy();
  const int *h_adjncy_source = mygraph.get_adjncy_source();
  const int *h_adjwgt = mygraph.get_adjwgt();

  gk::match(num_vertex, num_edge, h_adjncy, h_adjncy_source, h_adjp, h_vwgt, h_adjwgt);

  return 0;
}


