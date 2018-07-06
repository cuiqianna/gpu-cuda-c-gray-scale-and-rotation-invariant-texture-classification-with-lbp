#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tiffio.h>
#include <cuda_runtime.h>
#include "common/common.h"

#define PI 3.14159265
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLACK -16777216;


// struttura dati nella quale vengono salvate le traslazioni da fare intorno al pixel centrale in base a P e R
typedef struct Neighbor {
   int x;
   int y;
} Neighbor;

//allocazione dinamica della shared memory
extern __shared__ int array[];

__global__ void localBinaryPattern(uint32 *raster, int *histogram, Neighbor *neighbors, int width, int height, int P, int R);
void generateNeighbors(int R, int P, Neighbor *neighbors);
void printHistogram(int size, int *hisotram);
void initializeHistogram(int size, int *histogram);

int main(int argc, char **argv){
  int gridSizeX, gridSizeY, P, R, *width, *height, *histogram, smemSize, *dev_histogram;
  Neighbor *neighbors, *dev_neighbors;
  uint32 *raster, *dev_raster;
  size_t npixels;

  P = *argv[2] - '0';
  R = *argv[3] - '0';

  printf("R: %d, P: %d\n", R, P);

  histogram = (int*)malloc((P + 2) * sizeof(int));
  height = (int*)malloc(sizeof(int));
  width = (int*)malloc(sizeof(int));
  neighbors = (Neighbor*)malloc((P + 1) * sizeof(Neighbor));
  smemSize = ((BLOCK_SIZE_X * BLOCK_SIZE_Y) + (2 * BLOCK_SIZE_X * R) + (2 * BLOCK_SIZE_Y * R) + (4 * R * R)) * sizeof(uint32);

  initializeHistogram(P + 2, histogram);

  TIFF* tif = TIFFOpen(argv[1], "r");
  if (tif) {
      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);
      npixels = *width * *height;
      raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
      if (raster != NULL)
          TIFFReadRGBAImage(tif, *width, *height, raster, 0);
  }

  gridSizeX = *width/BLOCK_SIZE_X + ((*width % BLOCK_SIZE_X == 0) ? 0 : 1);
  gridSizeY = *height/BLOCK_SIZE_Y + ((*height % BLOCK_SIZE_Y == 0) ? 0 : 1);

  printf("Grid dimention: (%d, %d)\n", gridSizeX, gridSizeY);
  printf("Block dimention: (%d, %d)\n", BLOCK_SIZE_X, BLOCK_SIZE_Y);
  printf("Number of threads: %d\n", BLOCK_SIZE_X * BLOCK_SIZE_Y * gridSizeX * gridSizeY);
  printf("Image size: (%d, %d)\n", *height, *width);
  printf("Number of pixels: %d\n", npixels);
  printf("SMEM size: %d\n", smemSize);

  generateNeighbors(R, P, neighbors);

  CHECK(cudaMalloc((void **) &dev_histogram, (P + 2) * sizeof(int)));
  CHECK(cudaMalloc((void **) &dev_raster, npixels * sizeof (uint32)));
  CHECK(cudaMalloc((void **) &dev_neighbors, (P + 1) * sizeof(Neighbor)));

  CHECK(cudaMemcpy(dev_histogram, histogram, (P + 2) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_raster, raster, npixels * sizeof (uint32), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_neighbors, neighbors, (P + 1) * sizeof(Neighbor), cudaMemcpyHostToDevice));

  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid(gridSizeX, gridSizeY);

  localBinaryPattern <<<dimGrid, dimBlock, smemSize>>> (dev_raster, dev_histogram, dev_neighbors, *width, *height, P, R);
  CHECK(cudaDeviceSynchronize());    // host wait for device to finish the kernel
  CHECK(cudaMemcpy(histogram, dev_histogram, (P + 2) * sizeof(int), cudaMemcpyDeviceToHost));

  printHistogram(P + 2, histogram);

  free(height);
  free(width);
  free(histogram);
  free(neighbors);
  _TIFFfree(raster);
  TIFFClose(tif);

  CHECK(cudaFree(dev_histogram));
  CHECK(cudaFree(dev_raster));
  CHECK(cudaFree(dev_neighbors));
  return 0;
}


void generateNeighbors(int R, int P, Neighbor *neighbors){
  int p;
  for(p = 0; p < P; p ++){
    neighbors[p].y = R * cos(2 * PI * ((float)p/(float)P));
    neighbors[p].x = (-1 * R) * sin(2 * PI * ((float)p/(float)P));
  }
}

void initializeHistogram(int size, int *histogram){
  int i;
  for(i = 0; i < size; i ++)
    histogram[i] = 0;
}

void printHistogram(int size, int *histogram){
  int i;
  for(i = 0; i < size; i++)
    printf("[%d] frequency = %d\n", i, histogram[i]);
}

__device__ uint32 getPixel(uint32 *raster, int x, int y, int col, int row, int width, int height){
  return ((row + y) < 0) || ((col + x) < 0) || ((row + y) >= height) || ((col + x) >= width) ? -16777216 : raster[((row + y) * width) + col + x];
}

__device__ void loadSMEM(uint32 *raster, uint32 *subImg, int height, int width, int R, int idx, int smemIdx, int smemWidth, int col, int row){
  int x, y;

  subImg[((R + threadIdx.y) * smemWidth) + threadIdx.x + R] = raster[idx];

  if(threadIdx.x == 0){
    for(x = 0; x < R; x ++)
      subImg[smemIdx + x + (R * smemWidth)] = getPixel(raster, x - R, 0, col, row, width, height);
    if(threadIdx.y == 0){
      for(x = R; x > 0; x --)
        for(y = R; y > 0; y --)
          subImg[smemIdx + (R - x) + ((R - y) * smemWidth)] = getPixel(raster, -x, -y, col, row, width, height);
    }else if(threadIdx.y == (blockDim.y - 1) || row == (height -1)){
      for(x = R; x > 0; x --)
        for(y = 1; y <= R; y ++)
          subImg[smemIdx + (R - x) + ((R + y) * smemWidth)] = getPixel(raster, -x, y, col, row, width, height);
    }
  }else if(threadIdx.x == (blockDim.x - 1) || col == (width - 1)){
    for(x = 1; x <= R; x ++)
      subImg[smemIdx + (R * smemWidth) + (R + x)] = getPixel(raster, x, 0, col, row, width, height);
    if(threadIdx.y == (blockDim.y - 1) || row == (height -1)){
      for(x = 1; x <= R; x ++)
        for(y = 1; y <= R; y ++)
          subImg[smemIdx + x + R + ((y + R) * smemWidth)] = getPixel(raster, x, y, col, row, width, height);
    }else if(threadIdx.y == 0){
      for(x = 1; x <= R; x ++)
        for(y = R; y > 0; y --)
          subImg[smemIdx + x + R + ((R - y) * smemWidth)] = getPixel(raster, x, -y, col, row, width, height);
    }
  }
  if(threadIdx.y == (blockDim.y - 1) || row == (height -1))
    for(y = 1; y <= R; y ++)
      subImg[smemIdx + ((R + y) * smemWidth) + R] = getPixel(raster, 0, y, col, row, width, height);
  else if(threadIdx.y == 0)
    for(y = 0; y < R; y ++)
      subImg[smemIdx + (y * smemWidth) + R] = getPixel(raster, 0, y - R, col, row, width, height);
}

__device__ int getUniformValue(Neighbor *neighbors, int P, uint32 *subImg, int smemWidth, int center){
  int i1, i2, p, u;

  i1 = center + (neighbors[P - 1].y * smemWidth) + neighbors[P - 1].x;
  i2 = center + (neighbors[0].y * smemWidth) + neighbors[0].x;

  u = abs(((subImg[i1] >= subImg[center]) ? 1 : 0) - ((subImg[i2] >= subImg[center]) ? 1 : 0));

  for(p = 1; p < P; p++){
    i1 = center + (neighbors[p].y * smemWidth) + neighbors[p].x;
    i2 = center + (neighbors[p - 1].y * smemWidth) + neighbors[p - 1].x;

    u += abs(((subImg[i1] >= subImg[center]) ? 1 : 0) - ((subImg[i2] >= subImg[center]) ? 1 : 0));
  }

  return u;
}

__global__ void localBinaryPattern(uint32 *raster, int *histogram, Neighbor *neighbors, int width, int height, int P, int R){

  //declaration
  int col, row, idx, n, sum, smemWidth, center, smemIdx;
  uint32* subImg;

  //initialization
  col = blockDim.x * blockIdx.x + threadIdx.x;
  row = blockDim.y * blockIdx.y + threadIdx.y;
  idx = row * width + col;
  sum = 0;

  if(col < width && row < height){
    subImg = (uint32*)array;
    smemWidth = blockDim.x + (2 * R);
    smemIdx = threadIdx.y * smemWidth + threadIdx.x;
    center = smemIdx + R + (R * smemWidth);

    loadSMEM(raster, subImg, height, width, R, idx, smemIdx, smemWidth, col, row); //load SMEM content

	   __syncthreads(); // waits the SMEM load to be completed

    for(n = 0; n < P; n++)
      if(subImg[center + neighbors[n].x + (neighbors[n].y * smemWidth)] >= subImg[center])
        sum += 1;

    (getUniformValue(neighbors, P, subImg, smemWidth, center) <= 2) ? atomicAdd(&histogram[sum], 1) : atomicAdd(&histogram[P + 1], 1);
  }
}














//
