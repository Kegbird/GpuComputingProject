#include"gpu_imp.cuh"
#include "utils.h"
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STREAMS 8
#define BLOCK_SIZES 2

#define GAUSS_KERNEL_CODE 0
#define SOBEL_KERNEL_CODE_H 1
#define SOBEL_KERNEL_CODE_V 2
#define ROBERT_KERNEL_CODE_H 3
#define ROBERT_KERNEL_CODE_V 3

__constant__ float d_robert_kernel_3x3_h[3][3];
__constant__ float d_robert_kernel_3x3_v[3][3];

__constant__ float d_sobel_kernel_3x3_h[3][3];
__constant__ float d_sobel_kernel_3x3_v[3][3];

__constant__ float d_gaussian_kernel_7x7[7][7];

const int block_sizes[BLOCK_SIZES] = { 16, 32 };

unsigned char* image;
unsigned char* filtered_image;
unsigned char* pinned_image;
unsigned char* pinned_filtered_image;
unsigned char* d_image;
unsigned char* d_filtered_image;
unsigned char* d_gaussian_image;
unsigned char* d_module_image;
unsigned char* d_non_max_image;
float* d_orientations;

int width;
int height;
int f_width;
int f_height;
int f_width_gaussian;
int f_height_gaussian;
int channels;
size_t image_size;
size_t filtered_image_size;
size_t gaussian_image_size;
size_t orientations_size;

const char* output_filename_robert[] = { "Conv_Robert_Naive_16x16.png",
									"Conv_Robert_Naive_32x32.png" };

const char* output_filename_robert_smem[] = { "Conv_Robert_Smem_16x16.png",
									"Conv_Robert_Smem_32x32.png" };

const char* output_filename_robert_stream[] = { "Conv_Robert_Stream_16x16.png",
									"Conv_Robert_Stream_32x32.png" };

const char* output_filename_robert_stream_smem[] = { "Conv_Robert_Smem_Stream_16x16.png",
									"Conv_Robert_Smem_Stream_32x32.png" };

const char* output_filename_module[] = { "Module_Naive_16x16.png",
									"Module_Naive_32x32.png" };

const char* output_filename_module_smem[] = { "Module_Smem_16x16.png",
									"Module_Smem_32x32.png" };

const char* output_filename_module_stream[] = { "Module_Stream_16x16.png",
									"Module_Stream_32x32.png" };

const char* output_filename_module_stream_smem[] = { "Module_Stream_Smem_16x16.png",
									"Module_Stream_Smem_32x32.png" };

const char* output_filename_canny[] = { "Canny_Naive_16x16.png",
										"Canny_Naive_32x32.png" };

const char* output_filename_canny_smem[] = { "Canny_Smem_16x16.png",
										"Canny_Smem_32x32.png" };

const char* output_filename_canny_stream[] = { "Canny_Stream_16x16.png",
										"Canny_Stream_32x32.png" };

void load_constant_memory_robert_h(float* kernel, int kernel_size)
{
	CHECK(cudaMemcpyToSymbol(d_robert_kernel_3x3_h, kernel, kernel_size * kernel_size * sizeof(float)));
}

void load_constant_memory_robert_v(float* kernel, int kernel_size)
{
	CHECK(cudaMemcpyToSymbol(d_robert_kernel_3x3_v, kernel, kernel_size * kernel_size * sizeof(float)));
}

void load_constant_memory_sobel_h(float* kernel, int kernel_size)
{
	CHECK(cudaMemcpyToSymbol(d_sobel_kernel_3x3_h, kernel, kernel_size * kernel_size * sizeof(float)));
}

void load_constant_memory_sobel_v(float* kernel, int kernel_size)
{
	CHECK(cudaMemcpyToSymbol(d_sobel_kernel_3x3_v, kernel, kernel_size * kernel_size * sizeof(float)));
}

void load_constant_memory_gaussian(float * kernel, int kernel_size)
{
	CHECK(cudaMemcpyToSymbol(d_gaussian_kernel_7x7, kernel, kernel_size * kernel_size * sizeof(float)));
}

__device__ float device_grayscale(unsigned char* pixel, int channels)
{
	float color = 0;
	for (int j = 0; j < channels; j++)
		color += pixel[j] / channels;
	return color;
}

__device__ void fill_shared_memory_tile(unsigned char* pixel, unsigned char* image_tile, int width, int height, int channels, int tile_side, int tile_index, int row, int col, int kernel_radius)
{
	image_tile[tile_index] = device_grayscale(pixel, channels);

	if ((threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) || (row == height - (kernel_radius) * 2 - 1) || (col == width - (kernel_radius) * 2 - 1))
	{
		//Bottom right corner thread
		for (int i = 1; i <= kernel_radius * 2; i++)
		{
			image_tile[tile_index + i] = device_grayscale(pixel + channels * i, channels);
			image_tile[tile_index + tile_side * i] = device_grayscale(pixel + width * channels * i, channels);

			for (int j = 1; j <= kernel_radius * 2; j++)
				image_tile[tile_index + tile_side * i + j] = device_grayscale(pixel + width * channels * i + channels * j, channels);
		}
	}
	else if (threadIdx.x == blockDim.x - 1 || (col == width - (kernel_radius) * 2 - 1))
	{
		//Right edge thread
		for (int i = 1; i <= kernel_radius * 2; i++)
			image_tile[tile_index + i] = device_grayscale(pixel + channels * i, channels);
	}
	else if (threadIdx.y == blockDim.y - 1 || (row == height - (kernel_radius) * 2 - 1))
	{
		//Bottom left corner thread
		for (int i = 1; i <= kernel_radius * 2; i++)
			image_tile[tile_index + tile_side * i] = device_grayscale(pixel + width * channels * i, channels);
	}

}

__device__ float device_convolution(unsigned char* pixel, int channels, int width, float* kernel, int kernel_size, int kernel_radius)
{
	float result = 0;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			result += device_grayscale(pixel, channels) * kernel[i*kernel_size + j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}
	if (result < 0)
		result = 0;
	return result;
}

__device__ float device_convolution_smem(float* kernel, unsigned char* image_tile, int tile_index, int tile_side, int kernel_size, int kernel_radius)
{
	float result = 0.0;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++, tile_index++)
			result += image_tile[tile_index] * kernel[i*kernel_size + j];
		tile_index += tile_side - kernel_radius * 2 - 1;
	}
	if (result < 0)
		result = 0;
	return result;
}

__device__ bool device_strong_neighbour(unsigned char* pixel, int width, int strong_color)
{
	if (*(pixel - width - 1) == strong_color || *(pixel - width) == strong_color || *(pixel - width + 1) == strong_color ||
		*(pixel - 1) == strong_color || *(pixel + 1) == strong_color ||
		*(pixel + width - 1) == strong_color || *(pixel + width) == strong_color || *(pixel + width + 1) == strong_color)
		return true;
	return false;
}

__device__ float device_module(unsigned char* pixel, int channels, int width)
{
	int kernel_size = 3;
	float gh = 0.0, gv = 0.0;
	for (int i = 0; i < kernel_size; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < kernel_size; j++, pixel += channels)
		{
			gh += device_grayscale(pixel, channels) * d_sobel_kernel_3x3_h[i][j];
			gv += device_grayscale(pixel, channels) * d_sobel_kernel_3x3_v[i][j];
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}

	return sqrtf(gh*gh + gv * gv);
}

__device__ float* pick_kernel(int kernel_code)
{
	switch (kernel_code)
	{
	case GAUSS_KERNEL_CODE:
		return &d_gaussian_kernel_7x7[0][0];
		break;
	case SOBEL_KERNEL_CODE_H:
		return &d_sobel_kernel_3x3_h[0][0];
		break;
	case SOBEL_KERNEL_CODE_V:
		return &d_sobel_kernel_3x3_v[0][0];
		break;
	case ROBERT_KERNEL_CODE_H:
		return &d_robert_kernel_3x3_h[0][0];
		break;
	default:
		return &d_robert_kernel_3x3_v[0][0];
		break;
	}
}

__global__ void kernel_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int kernel_size, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;

	*(filtered_image + index) = device_convolution(pixel, channels, width, pick_kernel(kernel_code), kernel_size, kernel_radius);
}

__global__ void kernel_convolution_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int kernel_size, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char *pixel = image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, kernel_radius);

	__syncthreads();

	float *kernel = pick_kernel(kernel_code);

	int index = row * width + col - ((kernel_radius) * 2)*row;

	(filtered_image + index)[0] = device_convolution_smem(kernel, image_tile, tile_index, tile_side, kernel_size, kernel_radius);
}

__global__ void kernel_convolution_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int row_offset, int image_offset, int filtered_image_offset, int kernel_size, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row + row_offset)
		return;

	unsigned char* pixel = image + (row * width * channels + col * channels) + image_offset;

	int index = row * (width - kernel_radius * 2) + col + filtered_image_offset;

	*(filtered_image + index) = device_convolution(pixel, channels, width, pick_kernel(kernel_code), kernel_size, kernel_radius);
}

__global__ void kernel_convolution_stream_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int row_offset, int image_offset, int filtered_image_offset, int kernel_size, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row + row_offset)
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char* pixel = image + (row * width * channels + col * channels) + image_offset;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row + row_offset, col, kernel_radius);

	__syncthreads();

	float *kernel = pick_kernel(kernel_code);

	int index = (row * (width - kernel_radius * 2) + col) + filtered_image_offset;
	(filtered_image + index)[0] = device_convolution_smem(kernel, image_tile, tile_index, tile_side, kernel_size, kernel_radius);
}

__global__ void kernel_module(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row)
		return;

	int index = row * width + col - 2 * row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	(filtered_image + index)[0] = device_module(pixel, channels, width);
}

__global__ void kernel_module_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - 2 <= col || height - 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char *pixel = image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, 1);

	__syncthreads();

	float gh = 0, gv = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++, tile_index++)
		{
			gh += image_tile[tile_index] * d_sobel_kernel_3x3_h[i][j];
			gv += image_tile[tile_index] * d_sobel_kernel_3x3_v[i][j];
		}
		tile_index += tile_side - 3;
	}

	int index = row * width + col - 2 * row;
	(filtered_image + index)[0] = sqrtf(gh*gh + gv * gv);
}

__global__ void kernel_module_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int row_offset, int image_offset, int filtered_image_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row + row_offset)
		return;

	unsigned char* pixel = image + row * width * channels + col * channels + image_offset;
	int index = filtered_image_offset + (row * width + col - 2 * row);
	(filtered_image + index)[0] = device_module(pixel, channels, width);
}

__global__ void kernel_module_stream_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int row_offset, int image_offset, int filtered_image_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row + row_offset)
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char* pixel = image + (row * width * channels + col * channels) + image_offset;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row + row_offset, col, 1);

	__syncthreads();

	float gh = 0, gv = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++, tile_index++)
		{
			gh += image_tile[tile_index] * d_sobel_kernel_3x3_h[i][j];
			gv += image_tile[tile_index] * d_sobel_kernel_3x3_v[i][j];
		}
		tile_index += tile_side - 3;
	}

	int index = filtered_image_offset + (row * width + col - 2 * row);
	(filtered_image + index)[0] = sqrtf(gh*gh + gv * gv);
}

__global__ void kernel_module_orientation(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row)
		return;

	unsigned char* pixel = gaussian_filtered_image + row * width * channels + col * channels;

	float gh = 0.0, gv = 0.0;
	for (int i = 0; i < 3; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < 3; j++)
		{
			gh += *pixel * d_sobel_kernel_3x3_h[i][j];
			gv += *pixel * d_sobel_kernel_3x3_v[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * 2 - channels;
	}

	int index = row * width + col - 2 * row;
	(module_image + index)[0] = sqrtf(gh*gh + gv * gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_module_orientation_smem(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - 2 <= col || height - 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];
	unsigned char *pixel = gaussian_filtered_image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;
	//Image_tile contains the grayscale portion of the image on which the module will be applied
	image_tile[tile_index] = *(pixel);

	if ((threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) || (row == height - 3) || (col == width - 3))
	{
		//Bottom right corner thread
		image_tile[tile_index + 1] = *(pixel + channels);
		image_tile[tile_index + 2] = *(pixel + channels * 2);
		image_tile[tile_index + tile_side] = *(pixel + width * channels);
		image_tile[tile_index + tile_side * 2] = *(pixel + (width*channels) * 2);

		image_tile[tile_index + tile_side + 1] = *(pixel + width * channels + channels);
		image_tile[tile_index + tile_side + 2] = *(pixel + width * channels + channels * 2);
		image_tile[tile_index + tile_side * 2 + 1] = *(pixel + width * channels * 2 + channels);
		image_tile[tile_index + tile_side * 2 + 2] = *(pixel + width * channels * 2 + channels * 2);
	}
	else if (threadIdx.x == blockDim.x - 1 || (col == width - 3))
	{
		//Right edge thread
		image_tile[tile_index + 1] = *(pixel + channels);
		image_tile[tile_index + 2] = *(pixel + channels * 2);
	}
	else if (threadIdx.y == blockDim.y - 1 || (row == height - 3))
	{
		//Bottom left corner thread
		image_tile[tile_index + tile_side] = *(pixel + width * channels);
		image_tile[tile_index + tile_side * 2] = *(pixel + (width*channels) * 2);
	}

	__syncthreads();

	float gh = 0, gv = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			gh += image_tile[tile_index] * d_sobel_kernel_3x3_h[i][j];
			gv += image_tile[tile_index] * d_sobel_kernel_3x3_v[i][j];
			tile_index++;
		}
		tile_index += tile_side - 3;
	}

	int index = row * width + col - 2 * row;
	(module_image + index)[0] = sqrtf(gh*gh + gv * gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_module_orientation_stream(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels, int row_offset, int image_offset, int filtered_image_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row + row_offset)
		return;

	unsigned char* pixel = gaussian_filtered_image + row * width * channels + col * channels + image_offset;

	float gh = 0.0, gv = 0.0;
	for (int i = 0; i < 3; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < 3; j++)
		{
			gh += *pixel * d_sobel_kernel_3x3_h[i][j];
			gv += *pixel * d_sobel_kernel_3x3_v[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * 2 - channels;
	}

	int index = filtered_image_offset + (row * width + col - 2 * row);
	(module_image + index)[0] = sqrtf(gh*gh + gv * gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_non_max_suppression(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int weak_color, int strong_color, int low_threshold, int high_threshold)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row * width + col;

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		non_max_image[index] = module_image[index];
	}
	else
	{
		float angle = orientations[index];
		int r, q;

		if ((0.0 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= 180))
		{
			r = module_image[index + 1];
			q = module_image[index - 1];
		}
		else if (22.5 < angle && angle <= 67.5)
		{
			r = module_image[index + 1 - width];
			q = module_image[index - 1 + width];
		}
		else if (67.5 < angle && angle <= 112.5)
		{
			r = module_image[index - width];
			q = module_image[index + width];
		}
		else
		{
			r = module_image[index - width - 1];
			q = module_image[index + width + 1];
		}

		if (module_image[index] >= r && module_image[index] >= q)
			non_max_image[index] = module_image[index];
		else
			non_max_image[index] = 0;
	}

	if (non_max_image[index] < low_threshold)
		non_max_image[index] = 0;
	else if (non_max_image[index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= non_max_image[index] && non_max_image[index] < high_threshold)
		non_max_image[index] = weak_color;
}

__global__ void kernel_non_max_suppression_smem(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int weak_color, int strong_color, int low_threshold, int high_threshold, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	extern __shared__ unsigned char image_tile[];

	int index = row * width + col;
	int tile_index = (threadIdx.y + 1)*tile_side + threadIdx.x + 1;

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		//Image corners
		image_tile[tile_index] = *(module_image + index);
	}
	else if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//Filling block top left corner
		image_tile[0] = *(module_image + index - width - 1);
		image_tile[1] = *(module_image + index - width);
		image_tile[tile_side - 1] = *(module_image + index - 1);
		image_tile[tile_index] = *(module_image + index);
	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
	{
		//Filling top right corner
		image_tile[tile_index - tile_side] = *(module_image + index - width);
		image_tile[tile_index - tile_side + 1] = *(module_image + index - width + 1);
		image_tile[tile_index] = *(module_image + index);
		image_tile[tile_index + 1] = *(module_image + index + 1);
	}
	else if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
	{
		//Filling bottom left
		image_tile[tile_index - 1] = *(module_image + index - 1);
		image_tile[tile_index] = *(module_image + index);
		image_tile[tile_index + tile_side - 1] = *(module_image + index + width - 1);
		image_tile[tile_index + tile_side] = *(module_image + index + width);

	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.x - 1)
	{
		//Filling bottom right
		image_tile[tile_index] = *(module_image + index);
		image_tile[tile_index + 1] = *(module_image + index + 1);
		image_tile[tile_index + tile_side] = *(module_image + index + width);
		image_tile[tile_index + tile_side + 1] = *(module_image + index + width + 1);
	}
	else if (threadIdx.y == 0)
	{
		//Top edge
		image_tile[tile_index - tile_side] = *(module_image + index - width);
		image_tile[tile_index] = *(module_image + index);
	}
	else if (threadIdx.x == 0)
	{
		//Left edge
		image_tile[tile_index - 1] = *(module_image + index - 1);
		image_tile[tile_index] = *(module_image + index);
	}
	else if (threadIdx.x == blockDim.x - 1)
	{
		//Right edge
		image_tile[tile_index] = *(module_image + index);
		image_tile[tile_index + 1] = *(module_image + index + 1);
	}
	else
	{
		//Bottom edge
		image_tile[tile_index] = *(module_image + index);
		image_tile[tile_index + tile_side] = *(module_image + index + width);
	}

	__syncthreads();

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		non_max_image[index] = image_tile[tile_index];
	}
	else
	{
		float angle = orientations[index];
		int r, q;

		if ((0.0 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= 180))
		{
			r = image_tile[tile_index + 1];
			q = image_tile[tile_index - 1];
		}
		else if (22.5 < angle && angle <= 67.5)
		{
			r = image_tile[tile_index + 1 - tile_side];
			q = image_tile[tile_index - 1 + tile_side];
		}
		else if (67.5 < angle && angle <= 112.5)
		{
			r = image_tile[tile_index - tile_side];
			q = image_tile[tile_index + tile_side];
		}
		else
		{
			r = image_tile[tile_index - tile_side - 1];
			q = image_tile[tile_index + tile_side + 1];
		}

		if (image_tile[tile_index] >= r && image_tile[tile_index] >= q)
		{
			non_max_image[index] = image_tile[tile_index];
		}
		else
		{
			image_tile[tile_index] = 0;
			non_max_image[index] = 0;
		}
	}

	if (image_tile[tile_index] < low_threshold)
		non_max_image[index] = 0;
	else if (image_tile[tile_index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= image_tile[tile_index] && image_tile[tile_index] < high_threshold)
		non_max_image[index] = weak_color;
}

__global__ void kernel_non_max_suppression_stream(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int row_offset, int image_offset, int weak_color, int strong_color, int low_threshold, int high_threshold)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row + row_offset)
		return;

	int index = row * width + col + image_offset;

	if (row + row_offset == 0 || col == 0 || row + row_offset == height - 1 || col == width - 1)
	{
		non_max_image[index] = module_image[index];
	}
	else
	{
		float angle = orientations[index];
		int r, q;

		if ((0.0 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= 180))
		{
			r = module_image[index + 1];
			q = module_image[index - 1];
		}
		else if (22.5 < angle && angle <= 67.5)
		{
			r = module_image[index + 1 - width];
			q = module_image[index - 1 + width];
		}
		else if (67.5 < angle && angle <= 112.5)
		{
			r = module_image[index - width];
			q = module_image[index + width];
		}
		else
		{
			r = module_image[index - width - 1];
			q = module_image[index + width + 1];
		}

		if (module_image[index] >= r && module_image[index] >= q)
			non_max_image[index] = module_image[index];
		else
			non_max_image[index] = 0;
	}

	if (non_max_image[index] < low_threshold)
		non_max_image[index] = 0;
	else if (non_max_image[index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= non_max_image[index] && non_max_image[index] < high_threshold)
		non_max_image[index] = weak_color;
}

__global__ void kernel_hysteresis(unsigned char* non_max_image, unsigned char* filtered_image, int width, int height, int weak_color, int strong_color)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row * width + col;

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		filtered_image[index] = 0;
	}
	else
	{
		unsigned char* pixel = non_max_image + index;
		if (*pixel == strong_color || (*pixel == weak_color && device_strong_neighbour(pixel, width, strong_color)))
			filtered_image[index] = strong_color;
		else
			filtered_image[index] = 0;
	}
}

__global__ void kernel_hysteresis_smem(unsigned char* non_max_image, unsigned char* filtered_image, int width, int height, int weak_color, int strong_color, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row * width + col;
	extern __shared__ unsigned char image_tile[];

	int tile_index = (threadIdx.y + 1)*tile_side + threadIdx.x + 1;

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		//Image corners
		image_tile[tile_index] = non_max_image[index];
	}
	else if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//Filling block top left corner
		image_tile[0] = non_max_image[index - width - 1];
		image_tile[1] = non_max_image[index - width];
		image_tile[tile_side - 1] = non_max_image[index - 1];
		image_tile[tile_index] = non_max_image[index];

	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
	{
		//Filling top right corner
		image_tile[tile_index - tile_side] = non_max_image[index - width];
		image_tile[tile_index - tile_side + 1] = non_max_image[index - width + 1];
		image_tile[tile_index] = non_max_image[index];
		image_tile[tile_index + 1] = non_max_image[index + 1];
	}
	else if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
	{
		//Filling bottom left
		image_tile[tile_index - 1] = non_max_image[index - 1];
		image_tile[tile_index] = non_max_image[index];
		image_tile[tile_index + tile_side - 1] = non_max_image[index + width - 1];
		image_tile[tile_index + tile_side] = non_max_image[index + width];
	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.x - 1)
	{
		//Filling bottom right
		image_tile[tile_index] = non_max_image[index];
		image_tile[tile_index + 1] = non_max_image[index + 1];
		image_tile[tile_index + tile_side] = non_max_image[index + width];
		image_tile[tile_index + tile_side + 1] = non_max_image[index + width + 1];
	}
	else if (threadIdx.y == 0)
	{
		//Top edge
		image_tile[tile_index - tile_side] = non_max_image[index - width];
		image_tile[tile_index] = non_max_image[index];
	}
	else if (threadIdx.x == 0)
	{
		//Left edge
		image_tile[tile_index - 1] = non_max_image[index - 1];
		image_tile[tile_index] = non_max_image[index];
	}
	else if (threadIdx.x == blockDim.x - 1)
	{
		//Right edge
		image_tile[tile_index] = non_max_image[index];
		image_tile[tile_index + 1] = non_max_image[index + 1];
	}
	else
	{
		//Bottom edge
		image_tile[tile_index] = non_max_image[index];
		image_tile[tile_index + tile_side] = non_max_image[index + width];
	}

	__syncthreads();

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		filtered_image[index] = 0;
	}
	else
	{
		unsigned char* pixel = image_tile + tile_index;
		if (*pixel == strong_color || (*pixel == weak_color && device_strong_neighbour(pixel, tile_side, strong_color)))
			filtered_image[index] = strong_color;
		else
			filtered_image[index] = 0;
	}
}

__global__ void kernel_hysteresis_stream(unsigned char* non_max_image, unsigned char* filtered_image, int width, int height, int row_offset, int image_offset, int weak_color, int strong_color)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row + row_offset)
		return;

	int index = row * width + col + image_offset;

	if (row + row_offset == 0 || col == 0 || row + row_offset == height - 1 || col == width - 1)
	{
		filtered_image[index] = 0;
	}
	else
	{
		unsigned char* pixel = non_max_image + index;
		if (*pixel == strong_color || (*pixel == weak_color && device_strong_neighbour(pixel, width, strong_color)))
			filtered_image[index] = strong_color;
		else
			filtered_image[index] = 0;
	}
}

void naive_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
		kernel_convolution << <grid, block >> > (d_image, d_filtered_image, width, height, channels, kernel_size, kernel_radius, ROBERT_KERNEL_CODE_H);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file(output_filename_robert[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void smem_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
		int tile_side = block_sizes[i] + kernel_radius * 2;
		size_t tile_size = tile_side * tile_side;

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));

		kernel_convolution_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side, kernel_size, kernel_radius, ROBERT_KERNEL_CODE_H);

		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file(output_filename_robert_smem[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void stream_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = image_size / STREAMS;
	//Chunk_size_result is the chunk of data written by kernels in the output
	size_t chunk_size_result = filtered_image_size / STREAMS;

	//Stream creation
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);
		printf("Streams: %d\n", STREAMS);
		//Offset_input is the offset from which a kernel starts to read input image data
		int offset_input = 0;
		//Since the input potentially has more channels than the output(the output is always in grayscale), we need a different offset.
		int offset_output = 0;
		int row_offset;
		int image_offset;
		int filtered_image_offset;

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		for (int j = 0; j < STREAMS; j++)
		{
			row_offset = j * (height / STREAMS);
			image_offset = j * width*(height / STREAMS)*channels;
			filtered_image_offset = j * (width - kernel_radius * 2)*(height / STREAMS);
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_convolution_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height, channels, row_offset, image_offset, filtered_image_offset, kernel_size, kernel_radius, ROBERT_KERNEL_CODE_H);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += chunk_size;
			offset_output += chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file(output_filename_robert_stream[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}


	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void stream_smem_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = (image_size / STREAMS);
	//Chunk_size_result is the chunk of data written by kernels in the output
	size_t chunk_size_result = filtered_image_size / STREAMS;

	//Stream creation
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);
		printf("Streams: %d\n", STREAMS);
		int tile_side = block_sizes[i] + kernel_radius * 2;
		size_t tile_size = tile_side * tile_side;
		//Offset_input is the offset from which a kernel starts to read input image data
		int offset_input = 0;
		//Since the input potentially has more channels than the output(the output is always in grayscale), we need a different offset.
		int offset_output = 0;
		int row_offset;
		int image_offset;
		int filtered_image_offset;


		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		for (int j = 0; j < STREAMS; j++)
		{
			row_offset = j * (height / STREAMS);
			image_offset = j * width*(height / STREAMS)*channels;
			filtered_image_offset = j * (width - kernel_radius * 2)*(height / STREAMS);
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_convolution_stream_smem << <grid, block, tile_size, stream[j] >> > (d_image, d_filtered_image, width, height, channels, tile_side, row_offset, image_offset, filtered_image_offset, kernel_size, kernel_radius, ROBERT_KERNEL_CODE_H);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += image_size / STREAMS;
			offset_output += chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file(output_filename_robert_stream_smem[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}

	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void naive_module_gpu(const char * filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
		kernel_module << < grid, block >> > (d_image, d_filtered_image, width, height, channels);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file(output_filename_module[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void smem_module_gpu(const char * filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
		int tile_side = block_sizes[i] + kernel_radius * 2;
		size_t tile_size = tile_side * tile_side;

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));

		kernel_module_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side);

		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file(output_filename_module_smem[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void stream_module_gpu(const char * filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = image_size / STREAMS;
	//Chunk_size_result is the chunk of data written by kernels in the output
	size_t chunk_size_result = filtered_image_size / STREAMS;

	//Stream creation
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);
		printf("Streams: %d\n", STREAMS);
		//Offset_input is the offset from which a kernel starts to read input image data
		int offset_input = 0;
		//Since the input potentially has more channels than the output(the output is always in grayscale), we need a different offset.
		int offset_output = 0;
		int row_offset;
		int image_offset;
		int filtered_image_offset;

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		for (int j = 0; j < STREAMS; j++)
		{
			row_offset = j * (height / STREAMS);
			image_offset = j * width*(height / STREAMS)*channels;
			filtered_image_offset = j * (width - kernel_radius * 2)*(height / STREAMS);
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_module_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height, channels, row_offset, image_offset, filtered_image_offset);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += chunk_size;
			offset_output += chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file(output_filename_module_stream[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}

	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void stream_smem_module_gpu(const char * filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = image_size / STREAMS;
	//Chunk_size_result is the chunk of data written by kernels in the output
	size_t chunk_size_result = filtered_image_size / STREAMS;

	//Stream creation
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
		int tile_side = block_sizes[i] + kernel_radius * 2;
		size_t tile_size = tile_side * tile_side;

		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);
		printf("Streams: %d\n", STREAMS);
		//Offset_input is the offset from which a kernel starts to read input image data
		int offset_input = 0;
		//Since the input potentially has more channels than the output(the output is always in grayscale), we need a different offset.
		int offset_output = 0;
		int row_offset;
		int image_offset;
		int filtered_image_offset;

		begin_timer();
		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		for (int j = 0; j < STREAMS; j++)
		{
			row_offset = j * (height / STREAMS);
			image_offset = j * width*(height / STREAMS)*channels;
			filtered_image_offset = j * (width - kernel_radius * 2)*(height / STREAMS);
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_module_stream_smem << <grid, block, tile_size, stream[j] >> > (d_image, d_filtered_image, width, height, channels, tile_side, row_offset, image_offset, filtered_image_offset);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += chunk_size;
			offset_output += chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file(output_filename_module_stream_smem[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}

	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void naive_canny_gpu(const char * filename, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	int sobel_kernel_size = 3;
	int sobel_kernel_radius = 1;
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);

	f_width = f_width_gaussian - sobel_kernel_radius * 2;
	f_height = f_height_gaussian - sobel_kernel_radius * 2;
	filtered_image_size = f_width * f_height;

	size_t orientations_size = sizeof(float) * f_width*f_height;
	filtered_image = (unsigned char*)malloc(filtered_image_size);

	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_height_gaussian + block.y - 1) / block.y);
		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block.x, block.y);
		begin_timer();
		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
		CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
		//Gaussian filter
		kernel_convolution << <grid, block >> > (d_image, d_gaussian_image, width, height, channels, kernel_size, kernel_radius, GAUSS_KERNEL_CODE);
		grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_width_gaussian + block.y - 1) / block.y);
		//Module and orientations
		kernel_module_orientation << <grid, block >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1);
		grid = dim3((f_width + block.x - 3) / block.x, (f_height + block.y - 3) / block.y);
		//Non max suppression
		kernel_non_max_suppression << <grid, block >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, weak_color, strong_color, low_threshold, high_threshold);
		//Hysteresis
		kernel_hysteresis << <grid, block >> > (d_non_max_image, d_module_image, f_width, f_height, weak_color, strong_color);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_module_image, filtered_image_size, cudaMemcpyDeviceToHost));
		end_timer();

		if (output)
			save_file(output_filename_canny[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		CHECK(cudaFree(d_image));
		CHECK(cudaFree(d_gaussian_image));
		CHECK(cudaFree(d_module_image));
		CHECK(cudaFree(d_non_max_image));
		CHECK(cudaFree(d_orientations));
	}

	free(image);
	free(filtered_image);
}

void smem_canny_gpu(const char * filename, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	int sobel_kernel_size = 3;
	int sobel_kernel_radius = 1;
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);

	f_width = f_width_gaussian - sobel_kernel_radius * 2;
	f_height = f_height_gaussian - sobel_kernel_radius * 2;

	filtered_image_size = f_width * f_height;

	orientations_size = sizeof(float) * f_width*f_height;
	filtered_image = (unsigned char*)malloc(filtered_image_size);

	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;
	size_t tile_size;
	int tile_side;

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_height_gaussian + block.y - 1) / block.y);
		tile_side = block_sizes[i] + kernel_radius * 2;
		tile_size = tile_side * tile_side;
		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block.x, block.y);
		begin_timer();
		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
		CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
		//Gaussian filter
		kernel_convolution_smem << <grid, block, tile_size >> > (d_image, d_gaussian_image, width, height, channels, tile_side, kernel_size, kernel_radius, GAUSS_KERNEL_CODE);
		//Module and orientations
		grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_width_gaussian + block.y - 1) / block.y);
		tile_side = block_sizes[i] + sobel_kernel_radius * 2;
		tile_size = tile_side * tile_side;
		kernel_module_orientation_smem << <grid, block, tile_size >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1, tile_side);
		//Non max suppression
		kernel_non_max_suppression_smem << <grid, block, tile_size >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, weak_color, strong_color, low_threshold, high_threshold, tile_side);
		//Hysteresis
		kernel_hysteresis_smem << <grid, block, tile_size >> > (d_non_max_image, d_module_image, f_width, f_height, weak_color, strong_color, tile_side);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_module_image, filtered_image_size, cudaMemcpyDeviceToHost));
		end_timer();

		if (output)
			save_file(output_filename_canny_smem[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		CHECK(cudaFree(d_image));
		CHECK(cudaFree(d_gaussian_image));
		CHECK(cudaFree(d_module_image));
		CHECK(cudaFree(d_non_max_image));
		CHECK(cudaFree(d_orientations));
	}

	free(image);
	free(filtered_image);
}

void stream_canny_gpu(const char * filename, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	int sobel_kernel_size = 3;
	int sobel_kernel_radius = 1;
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);

	f_width = f_width_gaussian - sobel_kernel_radius * 2;
	f_height = f_height_gaussian - sobel_kernel_radius * 2;

	orientations_size = sizeof(float) * f_width*f_height;
	filtered_image_size = f_width * f_height;

	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;

	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, gaussian_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = image_size / STREAMS;
	//Chunk_size_result is the chunk of data written by kernels in the output
	size_t chunk_size_result = filtered_image_size / STREAMS;

	//Stream creation
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		dim3 block = dim3(block_sizes[i], block_sizes[i]);
		printf("Streams: %d\n", STREAMS);
		//Offset_input is the offset from which a kernel starts to read input image data
		int offset_input = 0;
		//Since the input potentially has more channels than the output(the output is always in grayscale), we need a different offset.
		int offset_output = 0;
		int row_offset;
		int image_offset;
		int filtered_image_offset;

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
		CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_orientations, orientations_size));

		for (int j = 0; j < STREAMS; j++)
		{
			//Gaussian filtering
			dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, ((f_height_gaussian / STREAMS) + block.y - 1) / block.y);
			row_offset = j * (height / STREAMS);
			image_offset = j * width*(height / STREAMS)*channels;
			filtered_image_offset = j * f_width_gaussian*(height / STREAMS);
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_convolution_stream << <grid, block, 0, stream[j] >> > (d_image, d_gaussian_image, width, height, channels, row_offset, image_offset, filtered_image_offset, kernel_size, kernel_radius, GAUSS_KERNEL_CODE);
			//Module
			grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
			row_offset = j * (f_height_gaussian / STREAMS);
			image_offset = j * f_width_gaussian*(f_height_gaussian / STREAMS);
			filtered_image_offset = j * f_width*(f_height_gaussian / STREAMS);
			kernel_module_orientation_stream << <grid, block, 0, stream[j] >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1, row_offset, image_offset, filtered_image_offset);
			//Non max suppression
			row_offset = j * (f_height / STREAMS);
			image_offset = j * f_width*(f_height / STREAMS);
			kernel_non_max_suppression_stream << <grid, block, 0, stream[j] >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, row_offset, image_offset, weak_color, strong_color, low_threshold, high_threshold);
			//Hysteresis
			kernel_hysteresis_stream << <grid, block, 0, stream[j] >> > (d_non_max_image, d_module_image, f_width, f_height, row_offset, image_offset, weak_color, strong_color);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_module_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += chunk_size;
			offset_output += chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file(output_filename_canny_stream[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_module_image);
		cudaFree(d_filtered_image);
		cudaFree(d_non_max_image);
		cudaFree(d_gaussian_image);
		cudaFree(d_orientations);
	}


	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
