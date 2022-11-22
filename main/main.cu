#include <iostream>
#include<vector>
#include "../gkway/graph.hpp"
#include <thrust/scan.h>
#define SHARE_ADJP_SIZE 32

cudaError_t check_cuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("CUDA Runtime Error : "s + cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

//find the source vertex using binary search
__device__ inline int find_source_index(int* share_adjp, const int current_idx, int start, int end) {;
  //printf("current idx %d, start %d, end %d \n", current_idx, start, end);
  int start_idx = 0;
  int end_idx = end;
  while(true) {
    //printf("start %d \n", start_idx);
    //printf("end %d \n", end_idx);
    //printf("mid %d \n", end_idx);
    if((end_idx - start_idx) == 1) {
      break;
    }
    int mid = (start_idx + end_idx ) / 2;
    if(current_idx == share_adjp[mid]) {
      return mid;
    }
    else if(current_idx < share_adjp[mid]) {
      end_idx = mid;
    }
    else {
      start_idx = mid;
    }
  }
  return start_idx;  
}

__device__ int find_max(int val1, int val2, int val1_idx, int val2_idx) {
  if(val1 >= val2) {return val1_idx;}
  else {return val2_idx;}
}

__global__ void matching_candidate_discovery(int* adjwgt, int* adjncy, int* adjncy_source, int* adjp, int* d_match_candidate, int start, int end) {
  int num_neighbor = end - start;
  int start_idx = blockIdx.x * blockDim.x;
  printf("num_neighbor %d \n", num_neighbor);
  extern __shared__ int s[];
  int* s_adjwgt = s;
  int* s_adjncy = &s_adjwgt[blockDim.x * sizeof(int)];
  int* s_source = &s_adjncy[blockDim.x * sizeof(int)];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("tid %d, gid %d \n",tid,gid);
  for(int i = tid; i < end; i += blockDim.x) {
    //printf("i is %d, end is %d \n",i,end);
    s_adjwgt[i] = adjwgt[start+i];
    s_adjncy[i] = adjncy[start+i];
    printf("s_adjncy at %d is %d \n", i, s_adjwgt[i]);
    s_source[i] = adjncy_source[start+i];
    printf("s_source at %d is %d \n", i, s_source[i]);
  }
  printf("before sync \n");
  __syncthreads();
  //segmented reduce
  unsigned int lk, rk;
  int max_idx;
  for(int i = 1; i < num_neighbor; i <<= 1) {
    if(tid + i < num_neighbor) {
      lk = s_source[tid];
      rk = s_source[tid + i];
      if(lk == rk) {
        max_idx = find_max(s_adjwgt[tid], s_adjwgt[tid + i], tid, tid + i);
        s_adjwgt[tid] = s_adjwgt[max_idx];
        s_adjncy[tid] = s_adjncy[max_idx];
      }
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    for(int i = 0; i < end; ++i) {
      printf("adjncy at %d is %d \n", i, s_adjncy[i]);
    }
  }
  //write candidate to match candidate
  d_match_candidate[threadIdx.x] = s_adjncy[adjp[threadIdx.x]];
  printf("threadIdx %d, match candidate %d \n",threadIdx.x, d_match_candidate[threadIdx.x]); 
}


void prefiex_sum(int* in_data, int* out_data, int size) {
  thrust::exclusive_scan(in_data, in_data + size, out_data);
}
void match(int num_vertex, int num_edge, int* h_adjncy, int* h_adjncy_source, int* h_adjp, int* h_vwgt, int* h_adjwgt) {
  
  int* d_adjp;
  int* d_vwgt;
  int* d_adjncy;
  int* d_adjncy_source;
  int* d_adjwgt;
  int* d_match_candidate;
  //allocate space on GPU global memory
  check_cuda(cudaMalloc((void **)&d_adjp, sizeof(int) * (num_vertex + 1)));
  check_cuda(cudaMalloc((void **)&d_match_candidate, sizeof(int) * num_vertex));
  check_cuda(cudaMalloc((void **)&d_vwgt, sizeof(int) * num_vertex));
  check_cuda(cudaMalloc((void **)&d_adjncy, sizeof(int) * 2 * num_edge));
  check_cuda(cudaMalloc((void **)&d_adjncy_source, sizeof(int) * 2 * num_edge));
  check_cuda(cudaMalloc((void **)&d_adjwgt, sizeof(int) * 2 * num_edge));

  //copy data from CPU to GPU
  check_cuda(cudaMemcpy(d_adjp, h_adjp, sizeof(int) * (num_vertex + 1), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_vwgt, h_vwgt, sizeof(int) * num_vertex, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_adjncy, h_adjncy, sizeof(int) * 2 * num_edge, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_adjncy_source, h_adjncy_source, sizeof(int) * 2 * num_edge, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(d_adjwgt, h_adjwgt, sizeof(int) * 2 * num_edge, cudaMemcpyHostToDevice));
  
  //TODO: consider when need to launch more than one block;
  if(num_vertex > 1024) {
  } 
  else {
   int threadsPerBlock = num_vertex;
  //calculate neighbor list length using prefix sum; assume num_vertex > 32
   int num_neighbor = h_adjp[num_vertex];
   size_t share_mem_size = (num_neighbor * 3) * sizeof(int);
   printf("num_neighbor %d \n", num_neighbor);
   printf("share mempry size %zu \n", share_mem_size);
   matching_candidate_discovery <<< 1, threadsPerBlock, share_mem_size >>> (d_adjwgt, d_adjncy, d_adjncy_source, d_adjp, d_match_candidate, 0, num_neighbor);
  }
  //fectch the neighbor lists of 32 vertices
  cudaDeviceSynchronize();
  cudaFree(d_adjp);
  cudaFree(d_match_candidate);
  cudaFree(d_vwgt);
  cudaFree(d_adjncy);
  cudaFree(d_adjncy_source);
  cudaFree(d_adjwgt);
}

int main(int argc, char** argv) {
  //parsing data to arra
  graph mygraph(argv[1]);
  mygraph.parse();

  size_t num_vertex = mygraph.get_num_vertex();
  size_t num_edge = mygraph.get_num_edge();    
  int max_degree = mygraph.get_max_degree();
  printf("num vertex %zd \n", num_vertex);
  printf("num edge %zd \n", num_edge);

  int *h_adjp = mygraph.get_adjp().data();
  int *h_vwgt = mygraph.get_vwgt().data();
  int *h_adjncy = mygraph.get_adjncy().data();
  int *h_adjncy_source = mygraph.get_adjncy_source().data();
  int *h_adjwgt = mygraph.get_adjwgt().data();
  // check inputs
  //for(int i = 0; i <= num_vertex; ++i) {
    //printf("adjp at %d is %d \n", i, h_adjp[i]);
  //}
  for(int i = 0; i < num_edge * 2; ++i) {
    printf("adjncy at %d is %d \n", i, h_adjncy[i]);
  }
  //for(int i = 0; i < num_edge * 2; ++i) {
    //printf("adj source at %d is %d \n", i, h_adjncy_source[i]);
  //}

  match(num_vertex, num_edge, h_adjncy, h_adjncy_source, h_adjp, h_vwgt, h_adjwgt);

  return 0;
}
