#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { cudaError_t _err=(call); if(_err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); return 1;} } while(0)

__global__ void blur_global(const float* in, float* out, int W, int H){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if(x>=W || y>=H) return;

  float sum=0; int cnt=0;
  for(int dy=-1; dy<=1; dy++){
    for(int dx=-1; dx<=1; dx++){
      int xx = x + dx, yy = y + dy;
      if(xx>=0 && xx<W && yy>=0 && yy<H){
        sum += in[yy*W + xx];
        cnt++;
      }
    }
  }
  out[y*W + x] = sum / cnt;
}

template<int TILE>
__global__ void blur_shared(const float* in, float* out, int W, int H){
  __shared__ float tile[TILE+2][TILE+2]; // halo of 1 around

  int x = blockIdx.x*TILE + threadIdx.x;
  int y = blockIdx.y*TILE + threadIdx.y;

  int lx = threadIdx.x + 1;
  int ly = threadIdx.y + 1;

  auto load = [&](int gx,int gy,int tx,int ty){
    if(gx>=0 && gx<W && gy>=0 && gy<H) tile[ty][tx] = in[gy*W + gx];
    else tile[ty][tx] = 0.0f;
  };

  if(threadIdx.x < TILE && threadIdx.y < TILE){
    load(x,y,lx,ly);
    if(threadIdx.x==0)        load(x-1,y,0,ly);
    if(threadIdx.x==TILE-1)   load(x+1,y,TILE+1,ly);
    if(threadIdx.y==0)        load(x,y-1,lx,0);
    if(threadIdx.y==TILE-1)   load(x,y+1,lx,TILE+1);

    if(threadIdx.x==0 && threadIdx.y==0)                 load(x-1,y-1,0,0);
    if(threadIdx.x==TILE-1 && threadIdx.y==0)            load(x+1,y-1,TILE+1,0);
    if(threadIdx.x==0 && threadIdx.y==TILE-1)            load(x-1,y+1,0,TILE+1);
    if(threadIdx.x==TILE-1 && threadIdx.y==TILE-1)       load(x+1,y+1,TILE+1,TILE+1);
  }

  __syncthreads();

  if(x>=W || y>=H) return;

  float sum=0;
  sum += tile[ly-1][lx-1]; sum += tile[ly-1][lx]; sum += tile[ly-1][lx+1];
  sum += tile[ly][lx-1];   sum += tile[ly][lx];   sum += tile[ly][lx+1];
  sum += tile[ly+1][lx-1]; sum += tile[ly+1][lx]; sum += tile[ly+1][lx+1];
  out[y*W + x] = sum / 9.0f;
}

int main(){
  int W=1024, H=1024;
  size_t bytes = (size_t)W*H*sizeof(float);
  float *din,*dout;
  CHECK_CUDA(cudaMalloc(&din, bytes));
  CHECK_CUDA(cudaMalloc(&dout, bytes));
  CHECK_CUDA(cudaMemset(din, 1, bytes));

  dim3 block(16,16);
  dim3 grid((W+15)/16, (H+15)/16);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // global
  CHECK_CUDA(cudaEventRecord(start));
  blur_global<<<grid,block>>>(din,dout,W,H);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms1=0; CHECK_CUDA(cudaEventElapsedTime(&ms1,start,stop));

  // shared (tile 16)
  dim3 blockT(16,16);
  dim3 gridT((W+15)/16, (H+15)/16);
  CHECK_CUDA(cudaEventRecord(start));
  blur_shared<16><<<gridT,blockT>>>(din,dout,W,H);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms2=0; CHECK_CUDA(cudaEventElapsedTime(&ms2,start,stop));

  printf("Blur global  : %.3f ms\n", ms1);
  printf("Blur shared16: %.3f ms\n", ms2);

  cudaFree(din); cudaFree(dout);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return 0;
}