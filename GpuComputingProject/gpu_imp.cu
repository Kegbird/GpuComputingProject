#include"gpu_imp.cuh"
#include "utils.h"
#include <stdio.h>
#include <time.h>
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STREAMS 2
#define BLOCK_SIZES 2
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
int channels;
size_t image_size;
size_t filtered_image_size;

const char* output_filename_robert[] = { "Sample_Naive_Convolution_Robert_16x16_block.png",
									"Sample_Naive_Convolution_Robert_32x32_block.png" };

const char* output_filename_robert_smem[] = { "Sample_Convolution_Robert_Smem_16x16_block.png",
									"Sample_Convolution_Robert_Smem_32x32_block.png" };

const char* output_filename_robert_stream[] = { "Sample_Convolution_Robert_Stream_16x16_block.png",
									"Sample_Convolution_Robert_Stream_32x32_block.png" };


const char* output_filename_module[] = { "Sample_Naive_Sobel_Module_16x16_block.png",
									"Sample_Naive_Sobel_Module_32x32_block.png" };

const char* output_filename_module_smem[] = { "Sample_Sobel_Module_Smem_16x16_block.png",
									"Sample_Sobel_Module_Smem_32x32_block.png" };

const char* output_filename_module_stream[] = { "Sample_Sobel_Module_Stream_16x16_block.png",
									"Sample_Sobel_Module_Stream_32x32_block.png" };

const char* output_filename_canny[] = { "Sample_Canny_16x16_block.png",
										"Sample_Canny_32x32_block.png" };

__device__ float grayscale(unsigned char* pixel, int channels)
{
	float color = 0;
	for (int j = 0; j < channels; j++)
		color += pixel[j] / channels;
	return color;
}

__device__ float convolution(unsigned char* pixel, int channels, int width, float* kernel, int kernel_size, int kernel_radius)
{
	float result = 0;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			result += grayscale(pixel, channels) * kernel[i*kernel_size + j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}
	if (result < 0)
		result = 0;
	return result;
}

__device__ bool strong_neighbour(unsigned char* pixel, int width, int strong_color)
{
	if (*(pixel - width - 1) == strong_color || *(pixel - width) == strong_color || *(pixel - width + 1) == strong_color ||
		*(pixel - 1) == strong_color || *(pixel + 1) == strong_color ||
		*(pixel + width - 1) == strong_color || *(pixel + width) == strong_color || *(pixel + width + 1) == strong_color)
		return true;
	return false;
}

__device__ int module(unsigned char* pixel, int channels, int width, float* kernel_h, float* kernel_v, int kernel_size, int kernel_radius)
{
	float gh = 0.0, gv = 0.0;
	for (int i = 0; i < kernel_size; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < kernel_size; j++)
		{
			gh += grayscale(pixel, channels) * kernel_h[i*kernel_size + j];
			gv += grayscale(pixel, channels) * kernel_v[i*kernel_size + j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}

	return sqrtf(gh*gh + gv * gv);
}

__global__ void kernel_robert_h_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	(filtered_image + index)[0] = convolution(pixel, channels, width, &d_robert_kernel_3x3_h[0][0], kernel_size, kernel_radius);
}

__global__ void kernel_robert_h_convolution_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char *pixel = image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	image_tile[tile_index] = grayscale(pixel, channels);

	if ((threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) || (row == height - (kernel_radius) * 2 - 1) || (col == width - (kernel_radius) * 2 - 1))
	{
		//Bottom right corner thread
		image_tile[tile_index + 1] = grayscale(pixel + channels, channels);
		image_tile[tile_index + 2] = grayscale(pixel + channels * 2, channels);
		image_tile[tile_index + tile_side] = grayscale(pixel + width * channels, channels);
		image_tile[tile_index + tile_side * 2] = grayscale(pixel + (width*channels) * 2, channels);

		image_tile[tile_index + tile_side + 1] = grayscale(pixel + width * channels + channels, channels);
		image_tile[tile_index + tile_side + 2] = grayscale(pixel + width * channels + channels * 2, channels);
		image_tile[tile_index + tile_side * 2 + 1] = grayscale(pixel + width * channels * 2 + channels, channels);
		image_tile[tile_index + tile_side * 2 + 2] = grayscale(pixel + width * channels * 2 + channels * 2, channels);
	}
	else if (threadIdx.x == blockDim.x - 1 || (col == width - (kernel_radius) * 2 - 1))
	{
		//Right edge thread
		image_tile[tile_index + 1] = grayscale(pixel + channels, channels);
		image_tile[tile_index + 2] = grayscale(pixel + channels * 2, channels);
	}
	else if (threadIdx.y == blockDim.y - 1 || (row == height - (kernel_radius) * 2 - 1))
	{
		//Bottom left corner thread
		image_tile[tile_index + tile_side] = grayscale(pixel + width * channels, channels);
		image_tile[tile_index + tile_side * 2] = grayscale(pixel + (width*channels) * 2, channels);
	}

	__syncthreads();

	int result = 0;

	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++, tile_index++)
			result += image_tile[tile_index] * d_robert_kernel_3x3_h[i][j];
		tile_index += tile_side - kernel_radius * 2 - 1;
	}
	if (result < 0)
		result = 0;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	(filtered_image + index)[0] = result;
}

__global__ void kernel_robert_h_convolution_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int offset_input, int offset_output, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 < row)
		return;

	unsigned char* pixel = image + row * width * channels + col * channels + offset_input;
	int index = offset_output + (row * width + col - ((kernel_radius) * 2)*row);
	(filtered_image + index)[0] = convolution(pixel, channels, width, &d_robert_kernel_3x3_h[0][0], kernel_size, kernel_radius);
}

__global__ void kernel_module_sobel(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	(filtered_image + index)[0] = module(pixel, channels, width, &d_sobel_kernel_3x3_h[0][0], &d_sobel_kernel_3x3_v[0][0], kernel_size, kernel_radius);
}

__global__ void kernel_module_sobel_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char *pixel = image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;
	//Image_tile contains the grayscale portion of the image on which the module will be applied
	image_tile[tile_index] = grayscale(pixel, channels);

	if ((threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) || (row == height - (kernel_radius) * 2 - 1) || (col == width - (kernel_radius) * 2 - 1))
	{
		//Bottom right corner thread
		image_tile[tile_index + 1] = grayscale(pixel + channels, channels);
		image_tile[tile_index + 2] = grayscale(pixel + channels * 2, channels);
		image_tile[tile_index + tile_side] = grayscale(pixel + width * channels, channels);
		image_tile[tile_index + tile_side * 2] = grayscale(pixel + (width*channels) * 2, channels);

		image_tile[tile_index + tile_side + 1] = grayscale(pixel + width * channels + channels, channels);
		image_tile[tile_index + tile_side + 2] = grayscale(pixel + width * channels + channels * 2, channels);
		image_tile[tile_index + tile_side * 2 + 1] = grayscale(pixel + width * channels * 2 + channels, channels);
		image_tile[tile_index + tile_side * 2 + 2] = grayscale(pixel + width * channels * 2 + channels * 2, channels);
	}
	else if (threadIdx.x == blockDim.x - 1 || (col == width - (kernel_radius) * 2 - 1))
	{
		//Right edge thread
		image_tile[tile_index + 1] = grayscale(pixel + channels, channels);
		image_tile[tile_index + 2] = grayscale(pixel + channels * 2, channels);
	}
	else if (threadIdx.y == blockDim.y - 1 || (row == height - (kernel_radius) * 2 - 1))
	{
		//Bottom left corner thread
		image_tile[tile_index + tile_side] = grayscale(pixel + width * channels, channels);
		image_tile[tile_index + tile_side * 2] = grayscale(pixel + (width*channels) * 2, channels);
	}

	__syncthreads();

	int gh = 0;
	int gv = 0;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++, tile_index++)
		{
			gh += image_tile[tile_index] * d_sobel_kernel_3x3_h[i][j];
			gv += image_tile[tile_index] * d_sobel_kernel_3x3_v[i][j];
		}
		tile_index += tile_side - kernel_radius * 2 - 1;
	}

	int result = sqrtf(gh*gh + gv * gv);
	int index = row * width + col - ((kernel_radius) * 2)*row;
	(filtered_image + index)[0] = result;
}

__global__ void kernel_module_sobel_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int offset_input, int offset_output, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 < row)
		return;

	unsigned char* pixel = image + row * width * channels + col * channels + offset_input;
	int index = offset_output + (row * width + col - ((kernel_radius) * 2)*row);
	(filtered_image + index)[0] = module(pixel, channels, width, &d_sobel_kernel_3x3_h[0][0], &d_sobel_kernel_3x3_v[0][0], kernel_size, kernel_radius);
}

__global__ void kernel_gaussian_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	(filtered_image + index)[0] = convolution(pixel, channels, width, &d_gaussian_kernel_7x7[0][0], kernel_size, kernel_radius);
}

__global__ void kernel_module_orientation(unsigned char* gaussian_filtered_image, unsigned char* module_image, float* orientations, int width, int height, int channels, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = gaussian_filtered_image + row * width * channels + col * channels;

	float gh = 0.0, gv = 0.0;
	for (int i = 0; i < kernel_size; i++)
	{
		//Evaluating gh and gv
		for (int j = 0; j < kernel_size; j++)
		{
			gh += grayscale(pixel, channels) * d_sobel_kernel_3x3_h[i][j];
			gv += grayscale(pixel, channels) * d_sobel_kernel_3x3_v[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}

	(module_image + index)[0] = sqrtf(gh*gh + gv * gv);
	orientations[index] = atan2(gv, gh);
}

__global__ void kernel_non_max_suppression(unsigned char* module_image, unsigned char* non_max_image, float* orientations, int width, int height, int weak_color, int strong_color, int low_threshold, int high_threshold)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width <= col || height <= row)
		return;

	int index = row*width + col;

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		non_max_image[index] = module_image[index];
	}
	else
	{
		float angle = orientations[index];
		float r, q;

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
	unsigned char* pixel = non_max_image + index;

	if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
	{
		filtered_image[index] = 0;
	}
	else
	{
		if(*pixel == strong_color || (*pixel == weak_color && strong_neighbour(pixel, width, strong_color)))
			filtered_image[index] = strong_color;
		else
			filtered_image[index] = 0;
	}
}

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

void naive_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output)
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
		kernel_robert_h_convolution << < grid, block >> > (d_image, d_filtered_image, width, height, channels, kernel_size, kernel_radius);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file((char*)output_filename_robert[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void smem_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output)
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

		kernel_robert_h_convolution_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side, kernel_size, kernel_radius);

		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file((char*)output_filename_robert_smem[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void stream_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = (image_size / STREAMS) + width * channels;
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

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		for (int j = 0; j < STREAMS; j++)
		{
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_robert_h_convolution_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height / STREAMS, channels, offset_input, offset_output, kernel_size, kernel_radius);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += (int)((image_size / STREAMS) - width * channels);
			offset_output += (int)chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file((char*)output_filename_robert_stream[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}


	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void naive_sobel_module_gpu(char * filename, int kernel_size, int kernel_radius, bool output)
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
		kernel_module_sobel << < grid, block >> > (d_image, d_filtered_image, width, height, channels, kernel_size, kernel_radius);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file((char*)output_filename_module[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void smem_sobel_module_gpu(char * filename, int kernel_size, int kernel_radius, bool output)
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

		kernel_module_sobel_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side, kernel_size, kernel_radius);

		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file((char*)output_filename_module_smem[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void stream_sobel_module_gpu(char * filename, int kernel_size, int kernel_radius, bool output)
{
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	//Pinned memory allocation
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	memcpy(pinned_image, image, image_size);

	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = (image_size / STREAMS) + width * channels;
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

		begin_timer();

		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));
		for (int j = 0; j < STREAMS; j++)
		{
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_module_sobel_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height / STREAMS, channels, offset_input, offset_output, kernel_size, kernel_radius);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += (int)((image_size / STREAMS) - width * channels);
			offset_output += (int)chunk_size_result;
		}

		for (int j = 0; j < STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file((char*)output_filename_module_stream[i], pinned_filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}


	for (int i = 0; i < STREAMS; i++)
		cudaStreamDestroy(stream[i]);

	free(image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void naive_canny_gpu(char * filename, float * kernel_h, float * kernel_v, float * gaussian_kernel, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	int sobel_kernel_size = 3;
	int sobel_kernel_radius = 1;
	size_t gaussian_image_size;
	int f_width_gaussian, f_height_gaussian;
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);

	f_width = f_width_gaussian - sobel_kernel_radius * 2;
	f_height = f_height_gaussian - sobel_kernel_radius * 2;

	size_t module_image_size = f_width * f_height;
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
		dim3 grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
		printf("Grid: %d, %d, %d\n", grid.x, grid.y, grid.z);
		printf("Blocks: %dx%d\n", block_sizes[i], block_sizes[i]);
		begin_timer();
		CHECK(cudaMalloc((void**)&d_image, image_size));
		CHECK(cudaMalloc((void**)&d_gaussian_image, gaussian_image_size));
		CHECK(cudaMalloc((void**)&d_module_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_non_max_image, filtered_image_size));
		CHECK(cudaMalloc((void**)&d_orientations, orientations_size));
		CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));

		//Gaussian filter
		kernel_gaussian_convolution << <block, grid >> > (d_image, d_gaussian_image, width, height, channels, kernel_size, kernel_radius);
		grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
		//Module and orientations
		kernel_module_orientation << <block, grid >> > (d_gaussian_image, d_module_image, d_orientations, f_width_gaussian, f_height_gaussian, 1, sobel_kernel_size, sobel_kernel_radius);
		grid = dim3((f_width + block.x - 3) / block.x, (f_height + block.y - 3) / block.y);
		//Non max suppression
		kernel_non_max_suppression << <block, grid >> > (d_module_image, d_non_max_image, d_orientations, f_width, f_height, weak_color, strong_color, low_threshold, high_threshold);
		//Hysteresis
		kernel_hysteresis << <block, grid >> > (d_non_max_image, d_module_image, f_width, f_height, weak_color, strong_color);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_module_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file((char*)output_filename_canny[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		CHECK(cudaFree(d_image));
		CHECK(cudaFree(d_gaussian_image));
		CHECK(cudaFree(d_module_image));
		CHECK(cudaFree(d_non_max_image));
		CHECK(cudaFree(d_orientations));
	}

	free(image);
	free(filtered_image);
}
