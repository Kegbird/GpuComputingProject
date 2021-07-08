#pragma once

int sobel_kernel_3x3_h[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
int sobel_kernel_3x3_v[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

unsigned char* image;
unsigned char* filtered_image;

int robert_kernel_3x3_h[3][3] = { {1, 0, 0}, {0, -1, 0}, {0, 0, 0} };
int robert_kernel_3x3_v[3][3] = { {0, 1, 0},{-1, 0, 0}, {0, 0, 0} };

int cpu_convolution(unsigned char* pixel, int channels, int* kernel, int width, int height, int kernel_size);
void cpu_filter(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel, int kernel_size, unsigned char* result);
void cpu_module(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel_h, int* kernel_v, int kernel_size, unsigned char* result);