#pragma once

#include <thrust/scan.h>

#define MAX_VERTEX_PER_THREAD 256
#define SHARE_SIZE 10000

namespace gk { // begin of namespace gk ============================================

// ======================================================
//
// Declaration/Definition of GPU kernels
//
// ======================================================

cudaError_t check_cuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("CUDA Runtime Error : "s + cudaGetErrorString(result));
  }
  return result;
}

// ======================================================
//
// For each vertex v and its adjancy list weight, use segmated reduction to
// find the vertex u which is connected by the  heaviest edge weight to merge
//  and write it to d_match_candidate array
//
// ======================================================
__global__ 
void matching_candidate_discovery(int* adjwgt, int* adjncy, int* adjncy_source, int* adjp, int* d_match_candidate, int num_vertex) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x; 
  int start = adjp[blockIdx.x * blockDim.x];
  int end = adjp[blockIdx.x * blockDim.x + blockDim.x];
  int num_neighbor = end - start;
  extern __shared__ int s[];
  int* s_adjwgt = s;
  int* s_source = &s_adjwgt[num_neighbor];
  int* s_match_candidate = &s_source[num_neighbor];

  for(int i = tid; i < end; i += blockDim.x) {
    s_adjwgt[i] = adjwgt[start+i];
    s_source[i] = adjncy_source[start+i];
  }
  __syncthreads();
  s_match_candidate[threadIdx.x] = d_match_candidate[gid];
  //segmented reduce
  unsigned int lk, rk;
  for(int i = 1; i < num_neighbor; i <<= 1) {
    for(int j = tid; j < num_neighbor; j += blockDim.x) {
      if(j + i < num_neighbor) {
        lk = s_source[j];
        rk = s_source[j + i];
        if(lk == rk) {
          if(s_adjwgt[j] < s_adjwgt[j+i]) {
            s_adjwgt[j] = s_adjwgt[j+i];
            s_match_candidate[lk] = j+i; 
            //printf("checking %d, %d, update candidate at lk %d to %d \n",j, j+i, lk, s_match_candidate[lk]); 
          }
          else {
             if(s_adjwgt[s_match_candidate[lk]] < s_adjwgt[j]) {
              s_match_candidate[lk] = j; 
             }
             //printf("checking %d, %d, keep candidate same  at lk %d to %d \n",j, j+i, lk, s_match_candidate[lk]); 
          }
        }
      }
    }
  }
  __syncthreads();
  if(threadIdx.x == 0) {
    //for(int i = 0; i < num_vertex; ++i) {
      //printf("d_match_candidate at %d is %d \n", i, s_match_candidate[i]);
    //}
    if(s_adjwgt[s_match_candidate[0]] < s_adjwgt[tid]) {
       s_match_candidate[0] = 0;
    }
  }
  //write candidate to match candidat
  if(threadIdx.x < num_vertex) {
    //int idx =  d_match_candidate[threadIdx.x];
    d_match_candidate[gid] = adjncy[s_match_candidate[threadIdx.x]];
  }
}

// ======================================================
//
// For each vertex v, check if d_match_candidate has a match. Ex: d_match_candidate[v] = u,
// and d_natch_candidate[u] = v. If not match, set d_match_candidate[v] = v.
// calcaulate the new adjncy list
//
// ======================================================
__global__ 
void check_candidate(int* d_match_candidate, int num_vertex) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; 
  if(gid < num_vertex) {
    int match_vertex = d_match_candidate[gid];
    if(d_match_candidate[match_vertex] != gid) {
      d_match_candidate[gid] = gid;
      printf("mismatch at %d gid \n", gid);
    }
    //calcaulte a new adjncy list

  }
}

// ======================================================
//
// Calculate new reduced num_vertex and write merge infomation
// to cmap
// Delete depulicated adjancy vertex
//
// ======================================================
__global__ 
void contraction(int* match_array, int* d_cmap, int num_vertex, int reduced_num_vertex) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; 
  extern __shared__ int s[];
  if(gid < num_vertex) {
     if(gid <= match_array[gid]) {
       s[gid] = 1;
     }
     else {
       s[gid] = 0;
     }
  }
}

void prefiex_sum(int* in_data, int* out_data, int size) {
  thrust::exclusive_scan(in_data, in_data + size, out_data);
}

void match(
  int num_vertex, 
  int num_edge, 
  const int* h_adjncy, 
  const int* h_adjncy_source, 
  const int* h_adjp, 
  const int* h_vwgt, 
  const int* h_adjwgt
) {
  
  int* d_adjp;
  int* d_vwgt;
  int* d_adjncy;
  int* d_adjncy_source;
  int* d_adjwgt;
  int* d_match_candidate;
  int* d_cmap;

  //allocate space on GPU global memory
  check_cuda(cudaMalloc((void **)&d_adjp, sizeof(int) * (num_vertex + 1)));
  check_cuda(cudaMalloc((void **)&d_match_candidate, sizeof(int) * num_vertex));
  check_cuda(cudaMalloc((void **)&d_cmap, sizeof(int) * num_vertex));
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
  check_cuda(cudaMemcpy(d_match_candidate, h_adjp, sizeof(int) * num_vertex, cudaMemcpyHostToDevice));
  int num_block = (num_vertex + MAX_VERTEX_PER_THREAD - 1) / MAX_VERTEX_PER_THREAD;
  size_t share_mem_size = SHARE_SIZE * sizeof(int);

  if(num_vertex > MAX_VERTEX_PER_THREAD) {
   matching_candidate_discovery <<<num_block, MAX_VERTEX_PER_THREAD, share_mem_size >>> (d_adjwgt, d_adjncy, d_adjncy_source, d_adjp, d_match_candidate, num_vertex);
  } 
  else {
   int threadsPerBlock = num_vertex;
  //calculate neighbor list length using prefix sum; assume num_vertex > 32
   int num_neighbor = h_adjp[num_vertex];
   //size_t share_mem_size = (num_neighbor * 2 + num_vertex) * sizeof(int);
   matching_candidate_discovery <<< 1, threadsPerBlock, share_mem_size >>> (d_adjwgt, d_adjncy, d_adjncy_source, d_adjp, d_match_candidate, num_vertex);
   //check_candidate<<<1, num_vertex >>>(d_match_candidate);
  }

  check_candidate<<<num_block, MAX_VERTEX_PER_THREAD >>>(d_match_candidate, num_vertex);
  //fectch the neighbor lists of 32 vertices

  cudaDeviceSynchronize();

  cudaFree(d_adjp);
  cudaFree(d_match_candidate);
  cudaFree(d_cmap);
  cudaFree(d_vwgt);
  cudaFree(d_adjncy);
  cudaFree(d_adjncy_source);
  cudaFree(d_adjwgt);
}

//__device__ inline int find_source_index(int* share_adjp, const int current_idx, int start, int end) {;
  ////printf("current idx %d, start %d, end %d \n", current_idx, start, end);
  //int start_idx = 0;
  //int end_idx = end;
  //while(true) {
    ////printf("start %d \n", start_idx);
    ////printf("end %d \n", end_idx);
    ////printf("mid %d \n", end_idx);
    //if((end_idx - start_idx) == 1) {
      //break;
    //}
    //int mid = (start_idx + end_idx ) / 2;
    //if(current_idx == share_adjp[mid]) {
      //return mid;
    //}
    //else if(current_idx < share_adjp[mid]) {
      //end_idx = mid;
    //}
    //else {
      //start_idx = mid;
    //}
  //}
  //return start_idx;  
//}

} // end of namespace gk ==========================================
