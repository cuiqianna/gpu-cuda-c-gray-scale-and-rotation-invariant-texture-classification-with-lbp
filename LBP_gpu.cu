#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tiffio.h>
#include "common/common.h"

#define PI 3.14159265
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLACK -16777216;

//
#define INDEX(idx, x, y, width) (idx + (y * width) + x)

typedef struct Neighbor {
   int x;
   int y;
} Neighbor;


__global__ void localBinaryPattern(uint32 *raster, int *histogram, Neighbor *neighbors, int width, int height, int P);
void generateNeighbors(int R, int P, Neighbor *neighbors);
__device__ int getUniformValue(Neighbor *neighbors, int P, uint32 *raster, int width, int height, int col, int row, int size, int idx);
void printHistogram(int size, int *hisotram);
void initializeHistogram(int size, int *histogram);

int main(int argc, char **argv){
  int gridSizeX, gridSizeY, P, R, *width, *height, *histogram, *dev_histogram;
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

  generateNeighbors(R, P, neighbors);

  CHECK(cudaMalloc((void **) &dev_histogram, (P + 2) * sizeof(int)));
  CHECK(cudaMalloc((void **) &dev_raster, npixels * sizeof (uint32)));
  CHECK(cudaMalloc((void **) &dev_neighbors, (P + 1) * sizeof(Neighbor)));

  CHECK(cudaMemcpy(dev_histogram, histogram, (P + 2) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_raster, raster, npixels * sizeof (uint32), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_neighbors, neighbors, (P + 1) * sizeof(Neighbor), cudaMemcpyHostToDevice));

  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 dimGrid(gridSizeX, gridSizeY);

  localBinaryPattern <<<dimGrid, dimBlock>>> (dev_raster, dev_histogram, dev_neighbors, *width, *height, P);
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
  for(i = 0; i < size; i ++) {
    histogram[i] = 0;
  }
}

void printHistogram(int size, int *histogram){
  int i;
  for(i = 0; i < size; i++){
    printf("[%d] frequency = %d\n", i, histogram[i]);
  }
}

__device__ uint32 getPixel(uint32 *raster, int x, int y, int col, int row, int width, int height){
  return ((row + y) < 0) || ((col + x) < 0) || ((row + y) >= height) || ((col + x) >= width) ? -16777216 : raster[((row + y) * width) + col + x];
}

__device__ int getUniformValue(Neighbor *neighbors, int P, uint32 *raster, int width, int height, int col, int row, int idx){
  int p, u, p1, p2;

  p1 = getPixel(raster, neighbors[P - 1].x, neighbors[P - 1].y, col, row, width, height);
  p2 = getPixel(raster, neighbors[0].x, neighbors[0].y, col, row, width, height);

  u = abs(((p1 >= raster[idx]) ? 1 : 0) - ((p2 >= raster[idx]) ? 1 : 0));

  for(p = 1; p < P; p++){
    p1 = getPixel(raster, neighbors[p].x, neighbors[p].y, col, row, width, height);
    p2 = getPixel(raster, neighbors[p - 1].x, neighbors[p - 1].y, col, row, width, height);
    u += abs(((p1 >= raster[idx]) ? 1 : 0) - ((p2 >= raster[idx]) ? 1 : 0));
  }

  return u;
}

__global__ void localBinaryPattern(uint32 *raster, int *histogram, Neighbor *neighbors, int width, int height, int P){
  int n, sum, idx, col, row;

  col = blockDim.x * blockIdx.x + threadIdx.x;
  row = blockDim.y * blockIdx.y + threadIdx.y;
  idx = row * width + col;
  sum = 0;

  if(col < width && row < height){

    for(n = 0; n < P; n++){
      if(getPixel(raster, neighbors[n].x, neighbors[n].y, col, row, width, height) >= raster[idx]){
        sum += 1;
      }
    }

    (getUniformValue(neighbors, P, raster, width, height, col, row, idx) <= 2) ? atomicAdd(&histogram[sum], 1) : atomicAdd(&histogram[P + 1], 1);
  }
}














//
