#include"gpu_imp.cuh"
#include "utils.h"
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STREAMS 2
#define BLOCK_SIZES 3
__constant__ int d_robert_kernel_3x3_h[3][3];
__constant__ int d_robert_kernel_3x3_v[3][3];

const int block_sizes[3] = { 8, 16, 32 };

char* output_filename_robert[] = { "Sample_Naive_Convolution_Robert_8x8_block.png",
									"Sample_Naive_Convolution_Robert_16x16_block.png",
									"Sample_Naive_Convolution_Robert_32x32_block.png" };

char* output_filename_robert_smem[] = { "Sample_Convolution_Robert_8x8_block_smem.png",
									"Sample_Convolution_Robert_16x16_block_smem.png",
									"Sample_Convolution_Robert_32x32_block_smem.png" };

char* output_filename_robert_stream[] = { "Sample_Convolution_Robert_8x8_block_stream.png",
									"Sample_Convolution_Robert_16x16_block_stream.png",
									"Sample_Convolution_Robert_32x32_block_stream.png" };

__device__ int grayscale(unsigned char* pixel, int channels)
{
	int color = 0;
	for (int j = 0; j < channels; j++)
		color += pixel[j];
	color /= channels;
	return color;
}

__global__ void kernel_robert_h_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int kernel_size, int kernel_radius)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (kernel_radius) * 2 <= col || height - (kernel_radius) * 2 <= row)
		return;

	int result = 0;
	int index = row * width + col - ((kernel_radius) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			result += grayscale(pixel, channels) * d_robert_kernel_3x3_h[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}
	if (result < 0)
		result = 0;
	(filtered_image + index)[0] = result;
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

	int result = 0;
	unsigned char* pixel = image + row * width * channels + col * channels + offset_input;

	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			result += grayscale(pixel, channels) * d_robert_kernel_3x3_h[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}
	if (result < 0)
		result = 0;
	int index = offset_output + (row * width + col - ((kernel_radius) * 2)*row);
	(filtered_image + index)[0] = result;
}

void load_constant_memory_robert_h(int* kernel, int kernel_size)
{
	CHECK(cudaMemcpyToSymbol(d_robert_kernel_3x3_h, kernel, kernel_size * kernel_size * sizeof(int)));
}

void naive_robert_gpu_convolution(char* filename, int* kernel, int kernel_size, int kernel_radius, bool output)
{
	unsigned char* image;
	unsigned char* filtered_image;
	unsigned char* d_image;
	unsigned char* d_filtered_image;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;

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
		kernel_robert_h_convolution << < grid, block>> > (d_image, d_filtered_image, width, height, channels, kernel_size, kernel_radius);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost));

		end_timer();

		if (output)
			save_file(output_filename_robert[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void smem_gpu_convolution(char* filename, int* kernel, int kernel_size, int kernel_radius, bool output)
{
	unsigned char* image;
	unsigned char* filtered_image;
	unsigned char* d_image;
	unsigned char* d_filtered_image;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;

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
			save_file(output_filename_robert_smem[i], filtered_image, f_width, f_height, 1);
		printf("Time elapsed for memory allocation, computation and memcpy H2D and D2H:%f seconds\n", time_elapsed());
		printf("Speedup: %f %\n\n", speedup());

		cudaFree(d_image);
		cudaFree(d_filtered_image);
	}
	free(image);
	free(filtered_image);
}

void stream_gpu_convolution(char* filename, int* kernel, int kernel_size, int kernel_radius, bool output)
{
	unsigned char* image;
	unsigned char* pinned_image;
	unsigned char* pinned_filtered_image;
	unsigned char* d_image;
	unsigned char* d_filtered_image;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;

	image=load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
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
			offset_input += (image_size / STREAMS) - width * channels;
			offset_output += chunk_size_result;
		}

		for(int j=0; j<STREAMS; j++)
			CHECK(cudaStreamSynchronize(stream[j]));
		end_timer();

		if (output)
			save_file(output_filename_robert_stream[i], pinned_filtered_image, f_width, f_height, 1);
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