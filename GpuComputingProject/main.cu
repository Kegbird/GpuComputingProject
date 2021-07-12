#include <stdio.h>
#include<math.h>
#include <time.h>
#include "utils.h"
#include "cpu_imp.cuh"
#include "gpu_imp.cuh"

#define KERNEL_SIZE 3
#define KERNEL_RADIUS KERNEL_SIZE/2
#define OUTPUT true

int sobel_kernel_3x3_h[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
int sobel_kernel_3x3_v[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

int robert_kernel_3x3_h[3][3] = { {1, 0, 0}, {0, -1, 0}, {0, 0, 0} };
int robert_kernel_3x3_v[3][3] = { {0, 1, 0},{-1, 0, 0}, {0, 0, 0} };

char filename[] = "Sample.png";

int main()
{
	if (!check_input(filename))
		return 0;
	print_file_details(filename);

	printf("============================\n");
	printf("	CPU Convolution(Robert)	\n");
	printf("============================\n\n");
	cpu_filter(filename, "Sample_Robert.png", &robert_kernel_3x3_h[0][0], KERNEL_SIZE, KERNEL_RADIUS, OUTPUT);

	load_constant_memory_robert_h(&robert_kernel_3x3_h[0][0], KERNEL_SIZE);

	printf("============================\n");
	printf("	GPU naive Convolution(Robert)	\n");
	printf("============================\n\n");

	naive_robert_gpu_convolution(filename, &robert_kernel_3x3_h[0][0], KERNEL_SIZE, KERNEL_RADIUS, OUTPUT);

	printf("============================\n");
	printf("	GPU Convolution(Smem)	\n");
	printf("============================\n\n");

	smem_gpu_convolution(filename, &robert_kernel_3x3_h[0][0], KERNEL_SIZE, KERNEL_RADIUS, OUTPUT);

	printf("============================\n");
	printf("	GPU Convolution(Stream)	\n");
	printf("============================\n\n");

	stream_gpu_convolution(filename, &robert_kernel_3x3_h[0][0], KERNEL_SIZE, KERNEL_RADIUS, OUTPUT);

	printf("============================\n");
	printf("	CPU Module(Sobel)	\n");
	printf("============================\n\n");

	cpu_module(filename, "Sample_Module_Sobel.png", &sobel_kernel_3x3_h[0][0], &sobel_kernel_3x3_v[0][0], KERNEL_SIZE, KERNEL_RADIUS, OUTPUT);

	return 0;
}

