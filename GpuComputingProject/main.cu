#include <stdio.h>
#include<math.h>
#include <time.h>
#include "utils.h"
#include "cpu_imp.cuh"
#include "gpu_imp.cuh"

#define KERNEL_SIDE 3
#define KERNEL_RADIUS KERNEL_SIDE/2

#define GAUSSIAN_KERNEL_SIDE 7
#define GAUSSIAN_KERNEL_RADIUS GAUSSIAN_KERNEL_SIDE/2
#define SIGMA 1
#define LOW_THRESHOLD_RATIO 0.05
#define HIGH_THRESHOLD_RATIO 0.5
#define OUTPUT true

float sobel_kernel_3x3_h[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
float sobel_kernel_3x3_v[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

float robert_kernel_3x3_h[3][3] = { {1, 0, 0}, {0, -1, 0}, {0, 0, 0} };
float robert_kernel_3x3_v[3][3] = { {0, 1, 0},{-1, 0, 0}, {0, 0, 0} };

float gaussian_kernel_7x7[7][7];

const char filename[] = "Sample.png";

const char filename_cpu_conv[] = "CPU_Conv_Robert.png";
const char filename_cpu_module[] = "CPU_Module.png";
const char filename_cpu_canny[] = "CPU_Canny.png";

void print_device_props()
{
	printf("- Device Info -\n\n");
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device name: %s\n", prop.name);
	printf("Memory Clock Rate (KHz): %d\n",
		prop.memoryClockRate);
	printf("Memory Bus Width (bits): %d\n",
		prop.memoryBusWidth);
	printf("Peak Memory Bandwidth (GB/s): %f\n\n",
		2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
}

int main()
{
	print_device_props();

	if (!check_input(filename))
		return 0;
	print_file_details(filename);

	load_constant_memory_robert_h(&sobel_kernel_3x3_h[0][0], KERNEL_SIDE);
	load_constant_memory_robert_v(&robert_kernel_3x3_v[0][0], KERNEL_SIDE);
	load_constant_memory_sobel_h(&sobel_kernel_3x3_h[0][0], KERNEL_SIDE);
	load_constant_memory_sobel_v(&sobel_kernel_3x3_v[0][0], KERNEL_SIDE);
	calculate_gaussian_kernel(&gaussian_kernel_7x7[0][0], SIGMA, GAUSSIAN_KERNEL_SIDE, GAUSSIAN_KERNEL_RADIUS);
	load_constant_memory_gaussian(&gaussian_kernel_7x7[0][0], GAUSSIAN_KERNEL_SIDE);

	printf("- CPU Robert -\n\n");

	filter_cpu(filename, filename_cpu_conv, &sobel_kernel_3x3_h[0][0], KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Convolution Robert -\n\n");

	naive_robert_convolution_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Convolution Robert (smem) -\n\n");

	smem_robert_convolution_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Convolution Robert (stream) -\n\n");

	stream_robert_convolution_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Convolution Robert (stream and smem) -\n\n");

	stream_smem_robert_convolution_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- CPU Sobel Module -\n\n");

	module_cpu(filename, filename_cpu_module, &sobel_kernel_3x3_h[0][0], &sobel_kernel_3x3_v[0][0], KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Sobel Module -\n\n");

	naive_module_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Sobel Module (smem) -\n\n");

	smem_module_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Sobel Module (stream) -\n\n");

	stream_module_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- GPU Sobel Module (stream and smem) -\n\n");

	stream_smem_module_gpu(filename, KERNEL_SIDE, KERNEL_RADIUS, OUTPUT);

	printf("- CPU Canny -\n\n");

	canny_cpu(filename, filename_cpu_canny, &sobel_kernel_3x3_h[0][0], &sobel_kernel_3x3_v[0][0], &gaussian_kernel_7x7[0][0], SIGMA, GAUSSIAN_KERNEL_SIDE, GAUSSIAN_KERNEL_RADIUS, LOW_THRESHOLD_RATIO, HIGH_THRESHOLD_RATIO, OUTPUT);

	printf("- GPU Canny -\n\n");

	naive_canny_gpu(filename, SIGMA, GAUSSIAN_KERNEL_SIDE, GAUSSIAN_KERNEL_RADIUS, LOW_THRESHOLD_RATIO, HIGH_THRESHOLD_RATIO, OUTPUT);

	printf("- GPU Canny (smem) -\n\n");

	smem_canny_gpu(filename, SIGMA, GAUSSIAN_KERNEL_SIDE, GAUSSIAN_KERNEL_RADIUS, LOW_THRESHOLD_RATIO, HIGH_THRESHOLD_RATIO, OUTPUT);

	printf("- GPU Canny (stream) -\n\n");

	stream_canny_gpu(filename, SIGMA, GAUSSIAN_KERNEL_SIDE, GAUSSIAN_KERNEL_RADIUS, LOW_THRESHOLD_RATIO, HIGH_THRESHOLD_RATIO, OUTPUT);

	printf("- GPU Canny (stream and smem) -\n\n");

	stream_smem_canny_gpu(filename, SIGMA, GAUSSIAN_KERNEL_SIDE, GAUSSIAN_KERNEL_RADIUS, LOW_THRESHOLD_RATIO, HIGH_THRESHOLD_RATIO, OUTPUT);

	return 0;
}

