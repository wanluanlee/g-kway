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
  extern __shared__ int sadj[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //use binary search over adjp to find source vertex for an edge
  if(i < num_vertex) {
    int num_neighbor = adjp[i+1] - adjp[i];
    int start = adjp[i];
    printf("tid is %d, i is %d\n", tid, i);
    printf("tid is %d, i is %d, adjp is %d \n", tid, i, adjp[i]);
    for(int i = 0; i < num_neighbor; ++i) {
      sadj[start+i] = adjncy[start+i];
    }
  }
  __syncthreads();
}

void prefiex_sum(int* in_data, int* out_data, int size) {
  thrust::exclusive_scan(in_data, in_data + size, out_data);
}

void match(int num_vertex, int num_edge, int* adjncy, int* adjp, int max_degree) {
//allocate matching array and cmap array
  //int* match;
  //int* cmap;

  //check_cuda(cudaMallocManaged(&match, sizeof(int) * num_vertex));
  //check_cuda(cudaMallocManaged(&cmap, sizeof(int) * num_vertex));
  int threadsPerBlock = 32;
  int numBlocks = (num_vertex + threadsPerBlock - 1)/ threadsPerBlock;
  //calculate neighbor list length using prefix sum; assume num_vertex > 32
  int share_memory_size = threadsPerBlock * max_degree * sizeof(int);
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
  //adjp = mygraph.get_adjp().data();    
  //vwgt = mygraph.get_vwgt().data();    
  //adjncy = mygraph.get_adjncy().data();    
  //adjwgt = mygraph.get_adjwgt().data();
  printf("max degree %d \n",max_degree);
  match(num_vertex, num_edge, adjncy, adjp, max_degree);
  cudaFree(adjp); 
  cudaFree(vwgt); 
  cudaFree(adjncy); 
  cudaFree(adjwgt); 
  //copy_graph_to_gpu(h_adjp, h_vwgt, h_adjncy, h_adjwgt, num_vertex, num_edge);

  return 0;
}
