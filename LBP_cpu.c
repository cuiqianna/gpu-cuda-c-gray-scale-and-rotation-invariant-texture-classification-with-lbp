#include <stdio.h>
#include <stdlib.h>
#include <tiffio.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265
#define BLACK -16777216;

typedef struct Neighbor {
   int x;
   int y;
} Neighbor;

uint32 getPixel(uint32 *raster, int x, int y, int idx, int width, int height);
void generateNeighbors(int R, int P, Neighbor *neighbors);
void localBinaryPattern(int width, int height, int P, uint32 *raster, Neighbor *neighbors, int * histogram);
int getUniformValue(Neighbor *neighbors, int P, uint32 *raster, int width, int height, int idx);
void printHistogram(int size, int *hisotram);
void initializeHistogram(int size, int *histogram);

int main(int argc, char **argv){
  clock_t start, end;
  double cpu_time_used;
  int P, R, i, *width, *height, *histogram;
  Neighbor *neighbors;
  uint32 *raster;
  size_t npixels;

  P = *argv[2] - '0';
  R = *argv[3] - '0';

  printf("R: %d, P: %d\n", R, P);

  histogram = malloc((P + 2) * sizeof(int));
  height = malloc(sizeof(int));
  width = malloc(sizeof(int));
  neighbors = malloc((P + 1) * sizeof(Neighbor));

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

  printf("width: %d, height: %d\n", *width, *height);

  generateNeighbors(R, P, neighbors);

  start = clock();
  localBinaryPattern(*width, *height, P, raster, neighbors, histogram);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printHistogram(P + 2, histogram);
  printf("Time (s): %f\n", cpu_time_used*1000);

  free(height);
  free(width);
  free(histogram);
  free(neighbors);
  _TIFFfree(raster);
  TIFFClose(tif);
  return 0;
}

void localBinaryPattern(int width, int height, int P, uint32 *raster, Neighbor *neighbors, int *histogram){
  int n, idx, sum, size;
  size = width * height;

  for(idx = 0; idx < size; idx ++){
    sum = 0;
    for(n = 0; n < P; n++)
      if(getPixel(raster, neighbors[n].x, neighbors[n].y, width, height, idx) >= raster[idx])
        sum += 1;

    (getUniformValue(neighbors, P, raster, width, height, idx) <= 2) ? (histogram[sum] += 1) : (histogram[P + 1] += 1);
  }
}

uint32 getPixel(uint32 *raster, int x, int y, int width, int height, int idx){
  int col = idx % width;
  int row = idx / width;
  return ((row + y) < 0) || ((col + x) < 0) || ((row + y) >= height) || ((col + x) >= width) ? -16777216 : raster[((row + y) * width) + col + x];
}

void generateNeighbors(int R, int P, Neighbor *neighbors){
  int p;
  for(p = 0; p < P; p ++){
    neighbors[p].y = R * cos(2 * PI * ((float)p/(float)P));
    neighbors[p].x = (-1 * R) * sin(2 * PI * ((float)p/(float)P));
  }
}


int getUniformValue(Neighbor *neighbors, int P, uint32 *raster, int width, int height, int idx){
  int p, u, p1, p2;

  p1 = getPixel(raster, neighbors[P - 1].x, neighbors[P - 1].y, width, height, idx);
  p2 = getPixel(raster, neighbors[0].x, neighbors[0].y, width, height, idx);

  u = abs(((p1 >= raster[idx]) ? 1 : 0) - ((p2 >= raster[idx]) ? 1 : 0));

  for(p = 1; p < P; p++){
    p1 = getPixel(raster, neighbors[p].x, neighbors[p].y, width, height, idx);
    p2 = getPixel(raster, neighbors[p - 1].x, neighbors[p - 1].y, width, height, idx);
    u += abs(((p1 >= raster[idx]) ? 1 : 0) - ((p2 >= raster[idx]) ? 1 : 0));
  }

  return u;
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


























//
