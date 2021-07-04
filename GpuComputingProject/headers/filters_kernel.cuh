#pragma once
#include <stdio.h>

__global__ void kernel_convolution(unsigned char* image,
	unsigned char* filtered_image,
	int width,
	int height,
	int channels,
	int* kernel,
	int kernel_size);