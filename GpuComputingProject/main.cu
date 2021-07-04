
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<math.h>
#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include <stb_image_write.h>
#include <time.h>

#define BLOCK_SIZE 32
#define KERNEL_SIZE 3
#define KERNEL_RADIUS KERNEL_SIZE/2

int robert_kernel_3x3_h[3][3] = { {1, 0, 0}, {0, -1, 0}, {0, 0, 0} };
int robert_kernel_3x3_v[3][3] = { {0, 1, 0},{-1, 0, 0}, {0, 0, 0} };

__constant__ int d_robert_kernel_3x3_h[3][3];
__constant__ int d_robert_kernel_3x3_v[3][3];

int sobel_kernel_3x3_h[3][3] = { {1, 0, -1}, {2, 0, -2}, {1, 0, -1} };
int sobel_kernel_3x3_v[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

unsigned char* image;
unsigned char* d_image;
unsigned char* filtered_image;
unsigned char* d_filtered_image;

#pragma region CPU_IMPLEMENTATIONS

int cpu_convolution(unsigned char* pixel, int channels, int* kernel, int width, int height, int kernel_size)
{
	int result = 0;
	int kernel_length = kernel_size * kernel_size;
	int color = 0;
	for (int j = 0; j < kernel_size; j++)
	{
		for (int k = 0; k < kernel_size; k++)
		{
			for (int i = 0; i < channels; i++)
				color += pixel[i];
			color /= channels;
			result += color * kernel[j*kernel_size + k];
			pixel += channels;
			color = 0;
		}
		pixel += (width * channels) - channels * (kernel_size - 1);
	}

	return result;
}


void cpu_filter(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel, int kernel_size, unsigned char* result)
{
	unsigned char* pixel = image;
	unsigned char* r = result;
	int kernel_radius = kernel_size / 2;
	int value = 0;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			value = cpu_convolution(pixel, channels, kernel, width, height, kernel_size);
			if (value < 0)
				r[0] = 0;
			else
				r[0] = value;
			r += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
}

void cpu_module(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel_h, int* kernel_v, int kernel_size, unsigned char* result)
{
	int gh = 0;
	int gv = 0;
	int modulo = 0;
	int kernel_radius = kernel_size / 2;

	unsigned char* pixel = image;
	unsigned char* r = result;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			gh = cpu_convolution(pixel, channels, kernel_h, width, height, kernel_size);
			gv = cpu_convolution(pixel, channels, kernel_v, width, height, kernel_size);
			r[0] = sqrt(gh*gh + gv * gv);
			r += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
}

#pragma endregion

#pragma region GPU_IMPLEMENTATIONS
__global__ void kernel_robert_h_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (KERNEL_RADIUS) * 2 <= col || height <= row - (KERNEL_RADIUS) * 2)
		return;

	int color = 0;
	int result = 0;
	int index = row * width + col - ((KERNEL_RADIUS) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	for (int i = 0; i < KERNEL_SIZE; i++)
	{
		for (int j = 0; j < KERNEL_SIZE; j++)
		{
			for (int k = 0; k < channels; k++)
				color += pixel[k];
			color /= 3;
			result += color * d_robert_kernel_3x3_h[i][j];
			pixel += channels;
			color = 0;
		}
		pixel += (width * channels) - channels * (KERNEL_SIZE - 1);
	}
	if (result < 0)
		result = 0;
	(filtered_image + index)[0] = result;
}

#pragma endregion

void freeHostMemory()
{
	free(image);
	free(filtered_image);
}

void freeDeviceMemory()
{
	cudaFree(d_image);
	cudaFree(d_filtered_image);
}

int main()
{
	dim3 grid;
	dim3 block;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;
	char filename[] = "Sample.png";
	time_t begin;
	time_t end;
	double cpu_time = 1;
	double elapsed_time = 0;
	cudaError_t status;
	//Image loading and common check
	image = stbi_load(filename, &width, &height, &channels, 0);
	if (image == NULL)
	{
		printf("No image provided!");
		return 0;
	}
	if (width < 3 || height < 3)
	{
		printf("The image provided is too small; the minimum resolution is 3x3 pixels.\n");
		return 0;
	}
	image_size = width * height * channels;
	printf("============================\n");
	printf("	Input Details	\n");
	printf("============================\n\n");
	printf("Width: %d\n", width);
	printf("Height: %d\n", height);
	printf("Channels: %d\n", channels);
	printf("Size: %d bytes\n\n", image_size);
	//Convolution decreases the resolution of the result
	f_width = width - (KERNEL_SIZE / 2) * 2;
	f_height = height - (KERNEL_SIZE / 2) * 2;
	filtered_image_size = f_width * f_height;
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	//initialization of constant memory
	cudaMemcpyToSymbol(d_robert_kernel_3x3_h, &robert_kernel_3x3_h, 3 * 3 * sizeof(int));
	cudaMemcpyToSymbol(d_robert_kernel_3x3_v, &robert_kernel_3x3_v, 3 * 3 * sizeof(int));
	printf("============================\n");
	printf("	CPU Convolution(Robert)	\n");
	printf("============================\n\n");
	begin = clock();
	cpu_filter(image, width, height, channels, image_size, &(robert_kernel_3x3_h[0][0]), KERNEL_SIZE, filtered_image);
	end = clock();
	cpu_time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("CPU Convolution:%f seconds\n\n", cpu_time);
	stbi_write_png("Sample_Convolution_Robert.png", f_width, f_height, 1, filtered_image, f_width);

	printf("============================\n");
	printf("	GPU naive Convolution(Robert)	\n");
	printf("============================\n\n");
	printf("Allocation of image on GPU gmem...\n\n");
	//Allocation gmem
	status = cudaMalloc((void**)&d_image, image_size);
	if (status != cudaSuccess)
	{
		printf("First cudaMalloc failed!\n");
		return 0;
	}

	status = cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		printf("First cudaMemcpy failed!\n");
		return 0;
	}

	status = cudaMalloc((void**)&d_filtered_image, filtered_image_size);
	if (status != cudaSuccess)
	{
		printf("Second cudaMalloc failed!\n");
		return 0;
	}

	status=cudaMemset(d_filtered_image, 0, filtered_image_size);

	if (status != cudaSuccess)
	{
		printf("First cudaMemset failed!\n");
		return 0;
	}

	printf("32x32 blocks\n");
	block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
	begin = clock();
	kernel_robert_h_convolution << < grid, block >> > (d_image, d_filtered_image, width, height, channels);
	cudaDeviceSynchronize();
	end = clock();
	status = cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost);

	/*if (status != cudaSuccess)
	{
		printf(cudaGetErrorString(status));
		return 0;
	}*/

	stbi_write_png("Sample_Naive_Convolution_Robert_32x32_block.png", f_width, f_height, 1, filtered_image, f_width);
	elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("GPU naive Convolution:%f seconds\n", elapsed_time);
	printf("Speedup: %f\n\n", elapsed_time / cpu_time);

	printf("============================\n");
	printf("	CPU Module(Sobel)	\n");
	printf("============================\n\n");
	begin = clock();
	cpu_module(image, width, height, channels, image_size, &(sobel_kernel_3x3_h[0][0]), &(sobel_kernel_3x3_v[0][0]), KERNEL_SIZE, filtered_image);
	end = clock();
	elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("CPU Module:%f seconds\n\n", elapsed_time);
	stbi_write_png("Sample_Module_Sobel.png", f_width, f_height, 1, filtered_image, f_width);

	freeHostMemory();
	freeDeviceMemory();
	return 0;
}

