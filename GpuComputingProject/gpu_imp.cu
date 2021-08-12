#include"gpu_imp.cuh"
#include "utils.h"
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STREAMS 4

#define GAUSS_KERNEL_CODE 0
#define SOBEL_KERNEL_CODE_H 1
#define SOBEL_KERNEL_CODE_V 2
#define ROBERT_KERNEL_CODE_H 3
#define ROBERT_KERNEL_CODE_V 3
#define BLOCK_SIDE 8

__constant__ float d_robert_kernel_3x3_h[3][3];
__constant__ float d_robert_kernel_3x3_v[3][3];

__constant__ float d_sobel_kernel_3x3_h[3][3];
__constant__ float d_sobel_kernel_3x3_v[3][3];

__constant__ float d_gaussian_kernel_7x7[7][7];

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

const char output_filename_robert[] = "Conv_Robert_Naive.png";

const char output_filename_robert_smem[] = "Conv_Robert_Smem.png";

const char output_filename_robert_stream[] = "Conv_Robert_Stream.png";

const char output_filename_robert_stream_smem[] = "Conv_Robert_Smem_Stream.png";

const char output_filename_module[] = "Module_Naive.png";

const char output_filename_module_smem[] = "Module_Smem.png";

const char output_filename_module_stream[] = "Module_Stream.png";

const char output_filename_module_stream_smem[] = "Module_Stream_Smem.png";

const char output_filename_canny[] ="Canny_Naive.png";

const char output_filename_canny_smem[] = "Canny_Smem.png";

const char output_filename_canny_stream[] = "Canny_Stream.png";

const char output_filename_canny_stream_smem[] = "Canny_Stream_Smem.png";											

void load_constant_memory_robert_h(float* kernel, int kernel_side)
{
	CHECK(cudaMemcpyToSymbol(d_robert_kernel_3x3_h, kernel, kernel_side * kernel_side * sizeof(float)));
}

void load_constant_memory_robert_v(float* kernel, int kernel_side)
{
	CHECK(cudaMemcpyToSymbol(d_robert_kernel_3x3_v, kernel, kernel_side * kernel_side * sizeof(float)));
}

void load_constant_memory_sobel_h(float* kernel, int kernel_side)
{
	CHECK(cudaMemcpyToSymbol(d_sobel_kernel_3x3_h, kernel, kernel_side * kernel_side * sizeof(float)));
}

void load_constant_memory_sobel_v(float* kernel, int kernel_side)
{
	CHECK(cudaMemcpyToSymbol(d_sobel_kernel_3x3_v, kernel, kernel_side * kernel_side * sizeof(float)));
}

void load_constant_memory_gaussian(float * kernel, int kernel_side)
{
	CHECK(cudaMemcpyToSymbol(d_gaussian_kernel_7x7, kernel, kernel_side * kernel_side * sizeof(float)));
}

__device__ float device_grayscale(unsigned char* pixel, int channels)
{
	float color = 0;
	for (int j = 0; j < channels; j++)
		color += pixel[j] / channels;
	return color;
}

__device__ void device_fill_shared_memory_tile(unsigned char* pixel, unsigned char* image_tile, int width, int height, int channels, int tile_side, int tile_index, int row, int col, int kernel_radius)
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

__device__ void device_fill_shared_memory_tile_as_frame(unsigned char* pixel, unsigned char* image_tile, int index, int tile_side, int tile_index, int row, int col, int width, int height)
{
	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		//Image corners
		image_tile[tile_index] = *(pixel + index);
	}
	else if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//Filling block top left corner
		image_tile[0] = *(pixel + index - width - 1);
		image_tile[1] = *(pixel + index - width);
		image_tile[tile_side - 1] = *(pixel + index - 1);
		image_tile[tile_index] = *(pixel + index);
	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
	{
		//Filling top right corner
		image_tile[tile_index - tile_side] = *(pixel + index - width);
		image_tile[tile_index - tile_side + 1] = *(pixel + index - width + 1);
		image_tile[tile_index] = *(pixel + index);
		image_tile[tile_index + 1] = *(pixel + index + 1);
	}
	else if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
	{
		//Filling bottom left
		image_tile[tile_index - 1] = *(pixel + index - 1);
		image_tile[tile_index] = *(pixel + index);
		image_tile[tile_index + tile_side - 1] = *(pixel + index + width - 1);
		image_tile[tile_index + tile_side] = *(pixel + index + width);

	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.x - 1)
	{
		//Filling bottom right
		image_tile[tile_index] = *(pixel + index);
		image_tile[tile_index + 1] = *(pixel + index + 1);
		image_tile[tile_index + tile_side] = *(pixel + index + width);
		image_tile[tile_index + tile_side + 1] = *(pixel + index + width + 1);
	}
	else if (threadIdx.y == 0)
	{
		//Top edge
		image_tile[tile_index - tile_side] = *(pixel + index - width);
		image_tile[tile_index] = *(pixel + index);
	}
	else if (threadIdx.x == 0)
	{
		//Left edge
		image_tile[tile_index - 1] = *(pixel + index - 1);
		image_tile[tile_index] = *(pixel + index);
	}
	else if (threadIdx.x == blockDim.x - 1)
	{
		//Right edge
		image_tile[tile_index] = *(pixel + index);
		image_tile[tile_index + 1] = *(pixel + index + 1);
	}
	else
	{
		//Bottom edge
		image_tile[tile_index] = *(pixel + index);
		image_tile[tile_index + tile_side] = *(pixel + index + width);
	}
}

__device__ float device_convolution(unsigned char* pixel, int channels, int width, float* kernel, int kernel_side, int kernel_radius)
{
	float result = 0;
	for (int i = 0; i < kernel_side; i++)
	{
		for (int j = 0; j < kernel_side; j++)
		{
			result += device_grayscale(pixel, channels) * kernel[i*kernel_side + j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_side - 1) - channels;
	}
	if (result < 0)
		result = 0;
	return result;
}

__device__ float device_convolution_smem(float* kernel, unsigned char* image_tile, int tile_index, int tile_side, int kernel_side, int kernel_radius)
{
	float result = 0.0;
	for (int i = 0; i < kernel_side; i++)
	{
		for (int j = 0; j < kernel_side; j++, tile_index++)
			result += image_tile[tile_index] * kernel[i*kernel_side + j];
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
	int kernel_side = 3;
	float gh = 0.0, gv = 0.0;
	for (int i = 0; i < kernel_side; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < kernel_side; j++, pixel += channels)
		{
			float gray = device_grayscale(pixel, channels);
			gh += gray * d_sobel_kernel_3x3_h[i][j];
			gv += gray * d_sobel_kernel_3x3_v[i][j];
		}
		pixel += (width * channels) - channels * (kernel_side - 1) - channels;
	}

	return sqrtf(gh*gh + gv * gv);
}

__device__ float device_module_smem(unsigned char* image_tile, int tile_index, int tile_side)
{
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
	return sqrtf(gh*gh + gv * gv);
}

__device__ float device_module(unsigned char* pixel, int channels, int width, float* gh, float* gv)
{
	*gh = 0, *gv = 0;
	for (int i = 0; i < 3; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < 3; j++)
		{
			*gh += *pixel * d_sobel_kernel_3x3_h[i][j];
			*gv += *pixel * d_sobel_kernel_3x3_v[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * 2 - channels;
	}
	return sqrtf((*gh)*(*gh) + (*gv)*(*gv));
}

__device__ float device_module_smem(unsigned char* image_tile, int tile_index, int tile_side, float* gh, float* gv)
{
	*gh = 0, *gv = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++, tile_index++)
		{
			*gh += image_tile[tile_index] * d_sobel_kernel_3x3_h[i][j];
			*gv += image_tile[tile_index] * d_sobel_kernel_3x3_v[i][j];
		}
		tile_index += tile_side - 3;
	}
	return sqrtf((*gh)*(*gh) + (*gv)*(*gv));
}

__device__ void device_non_max_suppression(unsigned char* non_max_image, unsigned char* module_image, float* orientations, int index, int width, int height, int row, int col)
{
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
}

__device__ void device_non_max_suppression_smem(unsigned char* non_max_image, unsigned char* image_tile, int tile_index, int tile_side, float* orientations, int index, int row, int col, int width, int height)
{
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

__global__ void kernel_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int kernel_side, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;

	*(filtered_image + index) = device_convolution(pixel, channels, width, pick_kernel(kernel_code), kernel_side, kernel_radius);
}

__global__ void kernel_convolution_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int kernel_side, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char *pixel = image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	device_fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, kernel_radius);

	__syncthreads();

	float *kernel = pick_kernel(kernel_code);

	int index = row * width + col - ((kernel_radius) * 2)*row;

	(filtered_image + index)[0] = device_convolution_smem(kernel, image_tile, tile_index, tile_side, kernel_side, kernel_radius);
}

__global__ void kernel_convolution_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int row_offset, int kernel_side, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	unsigned char* pixel = image + (row * width * channels + col * channels);

	int index = row * (width - kernel_radius * 2) + col;

	*(filtered_image + index) = device_convolution(pixel, channels, width, pick_kernel(kernel_code), kernel_side, kernel_radius);
}

__global__ void kernel_convolution_stream_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int row_offset, int kernel_side, int kernel_radius, int kernel_code)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char* pixel = image + (row * width * channels + col * channels);

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	device_fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, kernel_radius);

	__syncthreads();

	float *kernel = pick_kernel(kernel_code);

	int index = (row * (width - kernel_radius * 2) + col);
	(filtered_image + index)[0] = device_convolution_smem(kernel, image_tile, tile_index, tile_side, kernel_side, kernel_radius);
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

	device_fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, 1);

	__syncthreads();

	int index = row * width + col - 2 * row;

	(filtered_image + index)[0] = device_module_smem(image_tile, tile_index, tile_side);
}

__global__ void kernel_module_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int row_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height-2 <= row)
		return;

	unsigned char* pixel = image + row * width * channels + col * channels;
	int index = row * width + col - 2 * row;
	(filtered_image + index)[0] = device_module(pixel, channels, width);
}

__global__ void kernel_module_stream_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int row_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row)
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char* pixel = image + row * width * channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	device_fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, 1);

	__syncthreads();

	int index = row * width + col - 2 * row;

	(filtered_image + index)[0] = device_module_smem(image_tile, tile_index, tile_side);
}

__global__ void kernel_module_orientation(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row)
		return;

	unsigned char* pixel = gaussian_filtered_image + row * width * channels + col * channels;

	float gh, gv;
	int index = row * width + col - 2 * row;
	(module_image + index)[0] = device_module(pixel, channels, width, &gh, &gv);
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

	device_fill_shared_memory_tile(pixel, image_tile, width, height, channels, tile_side, tile_index, row, col, 1);

	__syncthreads();

	float gh, gv;
	int index = row * width + col - 2 * row;
	(module_image + index)[0] = device_module_smem(image_tile, tile_index, tile_side, &gh, &gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_module_orientation_stream(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels, int row_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row)
		return;

	unsigned char* pixel = gaussian_filtered_image + row * width * channels + col * channels;

	float gh, gv;
	int index = row * width + col - 2 * row;
	(module_image + index)[0] = device_module(pixel, channels, width, &gh, &gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_module_orientation_stream_smem(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels, int tile_side, int row_offset)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - 2 <= col || height - 2 <= row)
		return;

	extern __shared__ unsigned char image_tile[];
	unsigned char *pixel = gaussian_filtered_image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;
	device_fill_shared_memory_tile(pixel, image_tile, width, height, 1, tile_side, tile_index, row, col, 1);

	__syncthreads();

	float gh, gv;
	int index = row * width + col - 2 * row;
	(module_image + index)[0] = device_module_smem(image_tile, tile_index, tile_side, &gh, &gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_non_max_suppression(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int weak_color, int strong_color, float low_threshold, float high_threshold)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row * width + col;

	device_non_max_suppression(non_max_image, module_image, orientations, index, width, height, row, col);

	if (non_max_image[index] < low_threshold)
		non_max_image[index] = 0;
	else if (non_max_image[index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= non_max_image[index] && non_max_image[index] < high_threshold)
		non_max_image[index] = weak_color;
}

__global__ void kernel_non_max_suppression_smem(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int weak_color, int strong_color, float low_threshold, float high_threshold, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	extern __shared__ unsigned char image_tile[];

	int index = row * width + col;

	int tile_index = (threadIdx.y + 1)*tile_side + threadIdx.x + 1;

	device_fill_shared_memory_tile_as_frame(module_image, image_tile, index, tile_side, tile_index, row, col, width, height);

	__syncthreads();

	device_non_max_suppression_smem(non_max_image, image_tile, tile_index, tile_side, orientations, index, row, col, width, height);

	if (image_tile[tile_index] < low_threshold)
		non_max_image[index] = 0;
	else if (image_tile[tile_index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= image_tile[tile_index] && image_tile[tile_index] < high_threshold)
		non_max_image[index] = weak_color;
}

__global__ void kernel_non_max_suppression_stream(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int row_offset, int weak_color, int strong_color, float low_threshold, float high_threshold)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row * width + col;

	device_non_max_suppression(non_max_image, module_image, orientations, index, width, height, row, col);

	if (non_max_image[index] < low_threshold)
		non_max_image[index] = 0;
	else if (non_max_image[index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= non_max_image[index] && non_max_image[index] < high_threshold)
		non_max_image[index] = weak_color;
}

__global__ void kernel_non_max_suppression_stream_smem(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int row_offset, int weak_color, int strong_color, float low_threshold, float high_threshold, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	extern __shared__ unsigned char image_tile[];

	int index = row * width + col;

	int tile_index = (threadIdx.y + 1)*tile_side + threadIdx.x + 1;

	device_fill_shared_memory_tile_as_frame(module_image, image_tile, index, tile_side, tile_index, row, col, width, height);

	__syncthreads();

	device_non_max_suppression_smem(non_max_image, image_tile, tile_index, tile_side, orientations, index, row, col, width, height);

	if (image_tile[tile_index] < low_threshold)
		non_max_image[index] = 0;
	else if (image_tile[tile_index] >= high_threshold)
		non_max_image[index] = strong_color;
	else if (low_threshold <= image_tile[tile_index] && image_tile[tile_index] < high_threshold)
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

	device_fill_shared_memory_tile_as_frame(non_max_image, image_tile, index, tile_side, tile_index, row, col, width, height);

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

__global__ void kernel_hysteresis_stream(unsigned char* non_max_image, unsigned char* filtered_image, int width, int height, int row_offset, int weak_color, int strong_color)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
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

__global__ void kernel_hysteresis_stream_smem(unsigned char* non_max_image, unsigned char* filtered_image, int width, int height, int row_offset, int weak_color, int strong_color, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y + row_offset;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row * width + col;
	extern __shared__ unsigned char image_tile[];

	int tile_index = (threadIdx.y + 1)*tile_side + threadIdx.x + 1;

	device_fill_shared_memory_tile_as_frame(non_max_image, image_tile, index, tile_side, tile_index, row, col, width, height);

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

void save_result(bool output, unsigned char* image, const char* filename, int width, int height)
{
	if (output)
	{
		printf("Saving saving the result...\n");
		save_file(filename, image, width, height, 1);
		printf("Result saved as %s.\n", filename);
	}
}

//Parallel and most straightforward approach to apply a convolution on a image.
void naive_robert_convolution_gpu(const char* filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Allocation of the memory on which the filtered image will be saved.
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	//The execution of the kernel is done with different block dimensions: 16x16 and 32x32.
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
	printf("Begin execution(block %dx%d)...\n", block.x, block.y);
	begin_timer();
	//Allocation of device memory for the image and the filtered image, followed by a memory copy.
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
	//Kernel execution.
	kernel_convolution << <grid, block >> > (d_image, d_filtered_image, width, height, channels, kernel_side, kernel_radius, ROBERT_KERNEL_CODE_H);
	CHECK(cudaDeviceSynchronize());
	//Storing the result from device memory to host memory.
	CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result as png image.
	save_result(output, filtered_image, output_filename_robert, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);

	//Deallocation of host memory.
	free(image);
	free(filtered_image);
}
//Parallel approach to apply a convolution on a image enhanced by shared memory usage.
void smem_robert_convolution_gpu(const char* filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Allocation of the memory on which the filtered image will be saved.
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
	/*The image is loaded in shared memory tiles, allocated with each block.
	//The memory tile requires extra rows and columns; the number of extra rows and columns depend by the kernel radius (this is tile_side = block_size + kernel_radius*2)
	____________________
	|                 | |extra
	|                 | |column
	|                 | |
	|                 | |
	|                 | |
	|                 | |
	|_________________| |
	|______extra row____|
	*/
	//By loading each pixel into the shared memory, convolutions can be executed faster(reading from shared memory is faster than from global memory).
	int tile_side = BLOCK_SIDE + kernel_radius * 2;
	size_t tile_size = tile_side * tile_side;
	printf("Begin execution(block %dx%d)...\n", block.x, block.y);
	begin_timer();
	//Allocation of device memory for the image and the filtered image, followed by a memory copy.
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
	//Kernel execution.
	kernel_convolution_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side, kernel_side, kernel_radius, ROBERT_KERNEL_CODE_H);
	CHECK(cudaDeviceSynchronize());
	//Storing the result from device memory to host memory.
	CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result as png image.
	save_result(output, filtered_image, output_filename_robert_smem, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Deallocation of host memory.
	free(image);
	free(filtered_image);
}
//Parallel approach to apply a convolution on a image enhanced by cuda streams.
void stream_robert_convolution_gpu(const char* filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Cuda streams allows to subdivide the execution of a task in more indipendent streams.
	//The image is subdivided into STREAMS part, each one managed by a stream.
	//Each part of the provided image will be loaded asynchronously into global memory, thanks to streams.
	//To use this feature, we must first create and allocate pinned memory (for the result and the source image).
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);
	//Since the loading is asynchronous, each loading loads chunk_size data of the original image into the global memory.
	size_t chunk_size = image_size / STREAMS;
	//Since the result is in grayscale and with less rows and columns(due to the convolution), we have a different chunk size for the result.
	size_t chunk_size_result = filtered_image_size / STREAMS;
	//Stream creation.
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
	int offset_input = 0;
	int offset_output = 0;
	int row_offset;
	printf("Begin execution(block %dx%d, streams %d)...\n", block.x, block.y, STREAMS);
	begin_timer();
	//Allocation of device memory for the image and the filtered image, followed by a memory copy.
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	for (int j = 0; j < STREAMS; j++)
	{
		row_offset = j * (height / STREAMS);
		//Asynchronous loading of chunk size bytes of the input image.
		CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
		//Kernel execution.
		kernel_convolution_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height, channels, row_offset, kernel_side, kernel_radius, ROBERT_KERNEL_CODE_H);
		//Asynchronous loading of chunk size result bytes of the filtered image.
		CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
		offset_input += (int)chunk_size;
		offset_output += (int)chunk_size_result;
	}
	//The computations end as soon as all streams end.
	for (int j = 0; j < STREAMS; j++)
		CHECK(cudaStreamSynchronize(stream[j]));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result as png image.
	save_result(output, pinned_filtered_image, output_filename_robert_stream, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Deallocation of streams.
	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);
	//Deallocation of host memory.
	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
//Parallel approach enhanced by shared memory combined with cuda streams.
void stream_smem_robert_convolution_gpu(const char* filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Allocation of pinned memory.
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);
	size_t chunk_size = (image_size / STREAMS);
	size_t chunk_size_result = filtered_image_size / STREAMS;
	//Stream creation.
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
	int tile_side = BLOCK_SIDE + kernel_radius * 2;
	size_t tile_size = tile_side * tile_side;
	int offset_input = 0;
	int offset_output = 0;
	int row_offset;
	printf("Begin execution(block %dx%d, streams %d)...\n", block.x, block.y, STREAMS);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	for (int j = 0; j < STREAMS; j++)
	{
		row_offset = j * (height / STREAMS);
		CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
		//Kernel execution.
		kernel_convolution_stream_smem << <grid, block, tile_size, stream[j] >> > (d_image, d_filtered_image, width, height, channels, tile_side, row_offset, kernel_side, kernel_radius, ROBERT_KERNEL_CODE_H);
		CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
		offset_input += (int)(image_size) / STREAMS;
		offset_output += (int)chunk_size_result;
	}
	//The computations end as soon as all streams end.
	for (int j = 0; j < STREAMS; j++)
		CHECK(cudaStreamSynchronize(stream[j]));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result as png image.
	save_result(output, pinned_filtered_image, output_filename_robert_stream_smem, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Deallocation of streams.
	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);
	//Deallocation of host memory.
	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
//Parallel approach of Sobel module.
void naive_module_gpu(const char * filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
	printf("Begin execution(block %dx%d)...\n", block.x, block.y);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
	//Kernel execution.
	kernel_module << < grid, block >> > (d_image, d_filtered_image, width, height, channels);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));
	end_timer();
	printf("Execution ended.\n");

	if (output)
	{
		printf("Saving saving the result...\n");
		save_file(output_filename_module, filtered_image, f_width, f_height, 1);
		printf("Result saved as %s.\n", output_filename_module);
	}
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Deallocation of host memory.
	free(image);
	free(filtered_image);
}
//Parallel approach of Sobel module enhanced by shared memory.
void smem_module_gpu(const char * filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
	//The image is loaded in shared memory tiles, in the same fashion as it's written in the smem convolution kernel.
	int tile_side = BLOCK_SIDE + kernel_radius * 2;
	size_t tile_size = tile_side * tile_side;
	printf("Begin execution(block %dx%d)...\n", block.x, block.y);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
	//Kernel execution.
	kernel_module_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));
	end_timer();
	printf("Execution ended.\n\n");
	save_result(output, filtered_image, output_filename_module_smem, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Deallocation of host memory.
	free(image);
	free(filtered_image);
}
//Parallel approach of Sobel module enhanced by cuda streams.
void stream_module_gpu(const char * filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Allocation of pinned memory for input and result.
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);
	//Input and output data are subdivided into STREAMS portions of chunk_size and chunk_size_result bytes respectively.
	size_t chunk_size = image_size / STREAMS;
	size_t chunk_size_result = filtered_image_size / STREAMS;
	//Stream creation.
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
	int offset_input = 0;
	int offset_output = 0;
	int row_offset;
	printf("Begin execution(block %dx%d, streams %d)...\n", block.x, block.y, STREAMS);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	for (int j = 0; j < STREAMS; j++)
	{
		row_offset = j * (height / STREAMS);
		CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
		//Kernel execution.
		kernel_module_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height, channels, row_offset);
		CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
		offset_input += (int)chunk_size;
		offset_output += (int)chunk_size_result;
	}
	//The computations end as soon as all streams end.
	for (int j = 0; j < STREAMS; j++)
		CHECK(cudaStreamSynchronize(stream[j]));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result as png image.
	save_result(output, pinned_filtered_image, output_filename_module_stream, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Stream deallocation.
	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);
	//Deallocation of host memory.
	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
//Parallel approach of Sobel module enhanced by shared memory combined with cuda streams.
void stream_smem_module_gpu(const char * filename, int kernel_side, int kernel_radius, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Allocation of pinned memory for input and result.
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);
	size_t chunk_size = image_size / STREAMS;
	size_t chunk_size_result = filtered_image_size / STREAMS;
	//Stream creation.
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
	int tile_side = BLOCK_SIDE + kernel_radius * 2;
	size_t tile_size = tile_side * tile_side;
	int offset_input = 0;
	int offset_output = 0;
	int row_offset;
	printf("Begin execution(block %dx%d, streams %d)...\n", block.x, block.y, STREAMS);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
	for (int j = 0; j < STREAMS; j++)
	{
		row_offset = j * (height / STREAMS);
		CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
		//Kernel execution.
		kernel_module_stream_smem << <grid, block, tile_size, stream[j] >> > (d_image, d_filtered_image, width, height, channels, tile_side, row_offset);
		CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
		offset_input += (int)chunk_size;
		offset_output += (int)chunk_size_result;
	}
	//The computations end as soon as all streams end.
	for (int j = 0; j < STREAMS; j++)
		CHECK(cudaStreamSynchronize(stream[j]));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result as png image.
	save_result(output, pinned_filtered_image, output_filename_module_stream_smem, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_filtered_image);
	//Stream deallocation.
	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);
	//Deallocation of host memory.
	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
//Parallel implementation of the Canny Filter.
void naive_canny_gpu(const char * filename, float sigma, int kernel_side, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);
	/*The canny filter can be described with 5 operations:
	- 1 - Gaussian Filter
	- 2 - Sobel Module and Gradient orientations
	- 3 - Non max suppression
	- 4 - Hysteresis
	*/
	//f_width is the width of the filtered image.
	f_width = f_width_gaussian - 2;
	//f_height is the height of the filtered image.
	f_height = f_height_gaussian - 2;
	//f_width is the size of the filtered image.
	filtered_image_size = f_width * f_height;
	size_t orientations_size = sizeof(float) * f_width*f_height;
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	//Strong color is the color given to all edges of the filtered image.
	int strong_color = 255;
	//Weak color is the color given to all pixels which color is greater than the low_threshold.
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_height_gaussian + block.y - 1) / block.y);
	printf("Begin execution(block %dx%d)...\n", block.x, block.y);
	begin_timer();
	/*
	-d_image stores the input image.
	-d_gaussian_image stores the image filtered with the gaussian kernel.
	-d_module_image stores the module of the gaussian image.
	-d_non_max_image stores the result of non max suppression.
	-d_orientations stores the orientations of all gradients.
	*/
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
	CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
	CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
	// - 1 - Gaussian Filter
	kernel_convolution << <grid, block >> > (d_image, d_gaussian_image, width, height, channels, kernel_side, kernel_radius, GAUSS_KERNEL_CODE);
	grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_width_gaussian + block.y - 1) / block.y);
	//	- 2 - Sobel Module and Gradient orientations
	kernel_module_orientation << <grid, block >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1);
	grid = dim3((f_width + block.x - 3) / block.x, (f_height + block.y - 3) / block.y);
	// - 3 - Non max suppression 
	kernel_non_max_suppression << <grid, block >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, weak_color, strong_color, low_threshold, high_threshold);
	// - 4 - Hysteresis
	kernel_hysteresis << <grid, block >> > (d_non_max_image, d_module_image, f_width, f_height, weak_color, strong_color);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(filtered_image, d_module_image, filtered_image_size, cudaMemcpyDeviceToHost));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result.
	save_result(output, filtered_image, output_filename_canny, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	CHECK(cudaFree(d_image));
	CHECK(cudaFree(d_gaussian_image));
	CHECK(cudaFree(d_module_image));
	CHECK(cudaFree(d_non_max_image));
	CHECK(cudaFree(d_orientations));
	//Deallocation of host memory.
	free(image);
	free(filtered_image);
}
//Parallel implementation of the Canny Filter enhanced by shared memory.
void smem_canny_gpu(const char * filename, float sigma, int kernel_side, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);
	f_width = f_width_gaussian - 2;
	f_height = f_height_gaussian - 2;
	filtered_image_size = f_width * f_height;
	orientations_size = sizeof(float) * f_width*f_height;
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;
	size_t tile_size;
	int tile_side;
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_height_gaussian + block.y - 1) / block.y);
	tile_side = BLOCK_SIDE + kernel_radius * 2;
	tile_size = tile_side * tile_side;
	printf("Begin execution(block %dx%d)...\n", block.x, block.y);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
	CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
	CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
	// - 1 - Gaussian Filter
	kernel_convolution_smem << <grid, block, tile_size >> > (d_image, d_gaussian_image, width, height, channels, tile_side, kernel_side, kernel_radius, GAUSS_KERNEL_CODE);
	grid = dim3((f_width_gaussian + block.x - 1) / block.x, (f_width_gaussian + block.y - 1) / block.y);
	tile_side = BLOCK_SIDE + 2;
	tile_size = tile_side * tile_side;
	//	- 2 - Sobel Module and Gradient orientations
	kernel_module_orientation_smem << <grid, block, tile_size >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1, tile_side);
	// - 3 - Non max suppression 
	kernel_non_max_suppression_smem << <grid, block, tile_size >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, weak_color, strong_color, low_threshold, high_threshold, tile_side);
	// - 4 - Hysteresis
	kernel_hysteresis_smem << <grid, block, tile_size >> > (d_non_max_image, d_module_image, f_width, f_height, weak_color, strong_color, tile_side);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(filtered_image, d_module_image, filtered_image_size, cudaMemcpyDeviceToHost));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving result.
	save_result(output, filtered_image, output_filename_canny_smem, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	CHECK(cudaFree(d_image));
	CHECK(cudaFree(d_gaussian_image));
	CHECK(cudaFree(d_module_image));
	CHECK(cudaFree(d_non_max_image));
	CHECK(cudaFree(d_orientations));
	//Deallocation of host memory.
	free(image);
	free(filtered_image);
}
//Parallel implementation of the Canny Filter enhanced by streams.
void stream_canny_gpu(const char * filename, float sigma, int kernel_side, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);
	f_width = f_width_gaussian - 2;
	f_height = f_height_gaussian - 2;
	orientations_size = sizeof(float) * f_width*f_height;
	filtered_image_size = f_width * f_height;
	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;
	//Allocation of pinned memory for input and result.
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);
	size_t chunk_size = image_size / STREAMS;
	size_t chunk_size_result = filtered_image_size / STREAMS;
	//Stream creation.
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	int offset_input = 0;
	int offset_output = 0;
	int row_offset;
	printf("Begin execution(block %dx%d, streams %d)...\n", block.x, block.y, STREAMS);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
	CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
	for (int j = 0; j < STREAMS; j++)
	{
		dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, ((f_height_gaussian / STREAMS) + block.y - 1) / block.y);
		row_offset = j * (height / STREAMS);
		CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
		// - 1 - Gaussian Filter
		kernel_convolution_stream << <grid, block, 0, stream[j] >> > (d_image, d_gaussian_image, width, height, channels, row_offset, kernel_side, kernel_radius, GAUSS_KERNEL_CODE);
		grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
		row_offset = j * (f_height_gaussian / STREAMS);
		//	- 2 - Sobel Module and Gradient orientations
		kernel_module_orientation_stream << <grid, block, 0, stream[j] >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1, row_offset);
		row_offset = j * (f_height / STREAMS);
		// - 3 - Non max suppression 
		kernel_non_max_suppression_stream << <grid, block, 0, stream[j] >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, row_offset, weak_color, strong_color, low_threshold, high_threshold);
		// - 4 - Hysteresis
		kernel_hysteresis_stream << <grid, block, 0, stream[j] >> > (d_non_max_image, d_module_image, f_width, f_height, row_offset, weak_color, strong_color);
		CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_module_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
		offset_input += (int)chunk_size;
		offset_output += (int)chunk_size_result;
	}
	//The computations end as soon as all streams end.
	for (int j = 0; j < STREAMS; j++)
		CHECK(cudaStreamSynchronize(stream[j]));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving the result as png image.
	save_result(output, pinned_filtered_image, output_filename_canny_stream, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_module_image);
	cudaFree(d_filtered_image);
	cudaFree(d_non_max_image);
	cudaFree(d_gaussian_image);
	cudaFree(d_orientations);
	//Deallocation of streams.
	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);
	//Deallocation of host memory.
	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
//Parallel implementation of the Canny Filter enhanced by streams and shared memory.
void stream_smem_canny_gpu(const char * filename, float sigma, int kernel_side, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	//Loading of the image.
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);
	f_width = f_width_gaussian - 2;
	f_height = f_height_gaussian - 2;
	orientations_size = sizeof(float) * f_width*f_height;
	filtered_image_size = f_width * f_height;
	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;
	//Allocation of pinned memory for input and result.
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);
	size_t chunk_size = image_size / STREAMS;
	size_t chunk_size_result = filtered_image_size / STREAMS;
	size_t tile_size;
	int tile_side;
	//Stream creation.
	cudaStream_t stream[STREAMS];
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));
	dim3 block = dim3(BLOCK_SIDE, BLOCK_SIDE);
	printf("Streams: %d\n", STREAMS);
	int offset_input = 0;
	int offset_output = 0;
	int row_offset;
	printf("Begin execution(block %dx%d, streams %d)...\n", block.x, block.y, STREAMS);
	begin_timer();
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
	CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
	CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
	for (int j = 0; j < STREAMS; j++)
	{
		dim3 grid = dim3((f_width_gaussian + block.x - 1) / block.x, ((f_height_gaussian / STREAMS) + block.y - 1) / block.y);
		row_offset = j * (height / STREAMS);
		tile_side = BLOCK_SIDE + kernel_radius * 2;
		tile_size = tile_side * tile_side;
		CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
		// - 1 - Gaussian Filter
		kernel_convolution_stream_smem << <grid, block, tile_size, stream[j] >> > (d_image, d_gaussian_image, width, height, channels, tile_side, row_offset, kernel_side, kernel_radius, GAUSS_KERNEL_CODE);
		grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
		row_offset = j * (f_height_gaussian / STREAMS);
		tile_side = BLOCK_SIDE + 2;
		tile_size = tile_side * tile_side;
		//	- 2 - Sobel Module and Gradient orientations
		kernel_module_orientation_stream_smem << <grid, block, tile_size, stream[j] >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1, tile_side, row_offset);
		row_offset = j * (f_height / STREAMS);
		// - 3 - Non max suppression 
		kernel_non_max_suppression_stream_smem << < grid, block, tile_size, stream[j] >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, row_offset, weak_color, strong_color, low_threshold, high_threshold, tile_side);
		// - 4 - Hysteresis
		kernel_hysteresis_stream_smem << <grid, block, tile_size, stream[j] >> > (d_non_max_image, d_module_image, f_width, f_height, row_offset, weak_color, strong_color, tile_side);
		CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_module_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
		offset_input += (int)chunk_size;
		offset_output += (int)chunk_size_result;
	}
	//The computations end as soon as all streams end.
	for (int j = 0; j < STREAMS; j++)
		CHECK(cudaStreamSynchronize(stream[j]));
	end_timer();
	printf("Execution ended.\n\n");
	//Saving the result as png image.
	save_result(output, pinned_filtered_image, output_filename_canny_stream_smem, f_width, f_height);
	printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
	printf("Speedup: %f\n\n", speedup());
	//Deallocation of device memory.
	cudaFree(d_image);
	cudaFree(d_module_image);
	cudaFree(d_filtered_image);
	cudaFree(d_non_max_image);
	cudaFree(d_gaussian_image);
	cudaFree(d_orientations);
	//Deallocation of cuda streams.
	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);
	//Deallocation of host memory.
	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}
