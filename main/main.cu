#include <iostream>
#include<vector>
#include "../gkway/graph.hpp"
#include <thrust/scan.h>

cudaError_t check_cuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("CUDA Runtime Error : "s + cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

__global__ void matching_candidate_discovery(int* adjncy, int* adjp, int num_vertex) {
  const int ADJP_SIZE = 1024 * sizeof(int);
  __shared__ int sdjp[ADJP_SIZE];
  extern __shared__ int sdjncy[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //fectch 32 corresponding adjp into sdjp
   sdjp[i] = adjp[i];
  __syncthreads();
  //fetch 32 vertice's adjanct list
  int num_vertices = adjp[i - 1] - adjp[i];
  int start = adjp[i];
  for(int i = 0; i < num_vertices; ++i) {
    sdjncy[i+start] = adjncy[i+start];
  }
  //use binary search over adjp to find source vertex for an edge
}

void prefiex_sum(int* in_data, int* out_data, int size) {
  thrust::exclusive_scan(in_data, in_data + size, out_data);
}

void match(int num_vertex, int num_edge, int* adjncy, int* adjp, int max_degree) {
//allocate matching array and cmap array
  //int* match;
  //int* cmap;

  int threadsPerBlock = 1024;
  int numBlocks = (num_vertex + threadsPerBlock - 1)/ threadsPerBlock;
  //calculate neighbor list length using prefix sum; assume num_vertex > 32
  int share_memory_size = threadsPerBlock * sizeof(int);
  matching_candidate_discovery <<< numBlocks, threadsPerBlock, share_memory_size >>> (adjncy, adjp, num_vertex);
  //fectch the neighbor lists of 32 vertices
  cudaDeviceSynchronize();
}


int main(int argc, char** argv) {
  //parsing data to arra
  graph mygraph(argv[1]);
  mygraph.parse();

  int* adjp;    
  int* vwgt;    
  int* adjncy;    
  int* adjwgt;
  int num_vertex = mygraph.get_num_vertex();
  int num_edge = mygraph.get_num_edge();    
  int max_degree = mygraph.get_max_degree();
  printf("num vertex %d \n", num_vertex);
  printf("num edge %d \n", num_edge);
  std::vector<int> v_adjp = mygraph.get_adjp();
  std::vector<int> v_vwgt = mygraph.get_vwgt();
  std::vector<int> v_adjncy = mygraph.get_adjncy();
  std::vector<int> v_adjwgt = mygraph.get_adjwgt();
  check_cuda(cudaMallocManaged(&adjp, sizeof(int) * num_vertex));
  check_cuda(cudaMallocManaged(&vwgt, sizeof(int) * num_vertex));
  check_cuda(cudaMallocManaged(&adjncy, sizeof(int) * num_edge * 2));
  check_cuda(cudaMallocManaged(&adjwgt, sizeof(int) * num_edge * 2));
  for(int i = 0; i < num_vertex; ++i) {
    adjp[i] = v_adjp.at(i); 
    vwgt[i] = v_vwgt.at(i); 
  }
  for(int i = 0; i < (num_edge * 2); ++i) {
    adjncy[i] = v_adjncy.at(i); 
    adjwgt[i] = v_adjwgt.at(i); 
  }
  printf("max degree %d \n",max_degree);
  match(num_vertex, num_edge, adjncy, adjp, max_degree);
  printf("after match \n");
  cudaFree(adjp); 
  cudaFree(vwgt); 
  cudaFree(adjncy); 
  cudaFree(adjwgt); 
  //copy_graph_to_gpu(h_adjp, h_vwgt, h_adjncy, h_adjwgt, num_vertex, num_edge);

  return 0;
}
