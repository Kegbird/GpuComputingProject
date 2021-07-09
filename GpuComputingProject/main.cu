
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<math.h>
#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include <stb_image_write.h>
#include <time.h>
#include <cpu_imp.cuh>

#define KERNEL_SIZE 3
#define KERNEL_RADIUS KERNEL_SIZE/2
#define BLOCK_SIZES 3
#define STREAMS 2
#define OUTPUT true

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#pragma region GLOBAL_VARIABLES

__constant__ int d_robert_kernel_3x3_h[3][3];
__constant__ int d_robert_kernel_3x3_v[3][3];

int block_sizes[3] = { 8, 16, 32 };

unsigned char* d_image;
unsigned char* d_filtered_image;
unsigned char* pinned_image;
unsigned char* pinned_filtered_image;

cudaStream_t stream[STREAMS];

char filename[] = "Sample.png";
const char* output_filename_robert[] = { "Sample_Naive_Convolution_Robert_8x8_block.png",
									"Sample_Naive_Convolution_Robert_16x16_block.png",
									"Sample_Naive_Convolution_Robert_32x32_block.png" };

const char* output_filename_robert_smem[] = { "Sample_Convolution_Robert_8x8_block_smem.png",
									"Sample_Convolution_Robert_16x16_block_smem.png",
									"Sample_Convolution_Robert_32x32_block_smem.png" };

const char* output_filename_robert_stream[] = { "Sample_Convolution_Robert_8x8_block_stream.png",
									"Sample_Convolution_Robert_16x16_block_stream.png",
									"Sample_Convolution_Robert_32x32_block_stream.png" };
clock_t begin_cpu, end_cpu;
cudaEvent_t begin_gpu, end_gpu;

dim3 grid;
dim3 block;
int width;
int height;
int f_width;
int f_height;
int channels;
size_t image_size;
size_t filtered_image_size;
float cpu_time = 1;
float elapsed_time = 1;

#pragma endregion

#pragma region GPU_IMPLEMENTATIONS

__device__ int grayscale(unsigned char* pixel, int channels)
{
	int color = 0;
	for (int j = 0; j < channels; j++)
		color += pixel[j];
	color /= channels;
	return color;
}

__global__ void kernel_robert_h_convolution(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (KERNEL_RADIUS) * 2 <= col || height - (KERNEL_RADIUS) * 2 <= row)
		return;

	int result = 0;
	int index = row * width + col - ((KERNEL_RADIUS) * 2)*row;
	unsigned char* pixel = image + row * width * channels + col * channels;
	for (int i = 0; i < KERNEL_SIZE; i++)
	{
		for (int j = 0; j < KERNEL_SIZE; j++)
		{
			result += grayscale(pixel, channels) * d_robert_kernel_3x3_h[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (KERNEL_SIZE - 1) - channels;
	}
	if (result < 0)
		result = 0;
	(filtered_image + index)[0] = result;
}

__global__ void kernel_robert_h_convolution_smem(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int tile_side)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((width - (KERNEL_RADIUS) * 2 <= col || height - (KERNEL_RADIUS) * 2 <= row))
		return;

	extern __shared__ unsigned char image_tile[];

	unsigned char *pixel = image + row * width *channels + col * channels;

	int tile_index = threadIdx.y*tile_side + threadIdx.x;

	image_tile[tile_index] = grayscale(pixel, channels);

	if ((threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) || (row == height - (KERNEL_RADIUS) * 2 - 1) || (col == width - (KERNEL_RADIUS) * 2 - 1))
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
	else if (threadIdx.x == blockDim.x - 1 || (col == width - (KERNEL_RADIUS) * 2 - 1))
	{
		//Right edge thread
		image_tile[tile_index + 1] = grayscale(pixel + channels, channels);
		image_tile[tile_index + 2] = grayscale(pixel + channels * 2, channels);
	}
	else if (threadIdx.y == blockDim.y - 1 || (row == height - (KERNEL_RADIUS) * 2 - 1))
	{
		//Bottom left corner thread
		image_tile[tile_index + tile_side] = grayscale(pixel + width * channels, channels);
		image_tile[tile_index + tile_side * 2] = grayscale(pixel + (width*channels) * 2, channels);
	}

	__syncthreads();

	int result = 0;

	for (int i = 0; i < KERNEL_SIZE; i++)
	{
		for (int j = 0; j < KERNEL_SIZE; j++, tile_index++)
			result += image_tile[tile_index] * d_robert_kernel_3x3_h[i][j];
		tile_index += tile_side - KERNEL_RADIUS * 2 - 1;
	}
	if (result < 0)
		result = 0;

	int index = row * width + col - ((KERNEL_RADIUS) * 2)*row;
	(filtered_image + index)[0] = result;
}

__global__ void kernel_robert_h_convolution_stream(unsigned char* image, unsigned char* filtered_image, int width, int height, int channels, int offset_input, int offset_output)
{
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if (width - (KERNEL_RADIUS) * 2 <= col || height - (KERNEL_RADIUS) * 2 < row)
		return;

	int result = 0;
	unsigned char* pixel = image + row * width * channels + col * channels + offset_input;

	for (int i = 0; i < KERNEL_SIZE; i++)
	{
		for (int j = 0; j < KERNEL_SIZE; j++)
		{
			result += grayscale(pixel, channels) * d_robert_kernel_3x3_h[i][j];
			pixel += channels;
		}
		pixel += (width * channels) - channels * (KERNEL_SIZE - 1) - channels;
	}
	if (result < 0)
		result = 0;
	int index = offset_output + (row * width + col - ((KERNEL_RADIUS) * 2)*row);
	(filtered_image + index)[0] = result;
}

#pragma endregion

#pragma region WRAPPERS

void naive_gpu_convolution()
{
	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		printf("%dx%d blocks\n", block_sizes[i], block_sizes[i]);
		block = dim3(block_sizes[i], block_sizes[i]);
		grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);

		CHECK(cudaEventRecord(begin_gpu, 0));
		CHECK(cudaMemcpyAsync(d_image, image, image_size, cudaMemcpyHostToDevice, 0));

		kernel_robert_h_convolution << < grid, block, 0, 0 >> > (d_image, d_filtered_image, width, height, channels);

		CHECK(cudaMemcpyAsync(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost, 0));
		CHECK(cudaEventRecord(end_gpu, 0));
		CHECK(cudaEventSynchronize(end_gpu));
		CHECK(cudaEventElapsedTime(&elapsed_time, begin_gpu, end_gpu));

		if (OUTPUT)
			stbi_write_png(output_filename_robert[i], f_width, f_height, 1, filtered_image, f_width);
		elapsed_time = elapsed_time * pow(10.0, -3.0);
		printf("Time elapsed for computation and memcpy H2D and D2H:%f seconds\n", elapsed_time);
		printf("Speedup: %f %\n\n", (cpu_time / elapsed_time)*100.0);

		CHECK(cudaMemsetAsync(d_filtered_image, 0, filtered_image_size, 0));
	}
}

void smem_gpu_convolution()
{
	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		printf("%dx%d blocks\n", block_sizes[i], block_sizes[i]);
		block = dim3(block_sizes[i], block_sizes[i]);
		grid = dim3((f_width + block.x - 1) / block.x, (f_height + block.y - 1) / block.y);
		int tile_side = block_sizes[i] + KERNEL_RADIUS * 2;
		size_t tile_size = tile_side * tile_side;

		CHECK(cudaEventRecord(begin_gpu, 0));
		CHECK(cudaMemcpyAsync(d_image, image, image_size, cudaMemcpyHostToDevice, 0));

		kernel_robert_h_convolution_smem << < grid, block, tile_size >> > (d_image, d_filtered_image, width, height, channels, tile_side);

		CHECK(cudaMemcpyAsync(filtered_image, d_filtered_image, filtered_image_size, cudaMemcpyDeviceToHost, 0));
		CHECK(cudaEventRecord(end_gpu, 0));
		CHECK(cudaEventSynchronize(end_gpu));

		CHECK(cudaEventElapsedTime(&elapsed_time, begin_gpu, end_gpu));
		if (OUTPUT)
			stbi_write_png(output_filename_robert_smem[i], f_width, f_height, 1, filtered_image, f_width);

		elapsed_time = elapsed_time * pow(10.0, -3.0);
		printf("Time elapsed for computation and memcpy H2D and D2H:%f seconds\n", elapsed_time);
		printf("Speedup: %f %\n\n", (cpu_time / elapsed_time)*100.0);
	}
}

void stream_gpu_convolution()
{
	//Chunk_size is the chunk of the input image wich is elaborated by the stream
	size_t chunk_size = (image_size / STREAMS) + width * channels;
	//Chunk_size_result is the chunk of data written by kernels in the output
	size_t chunk_size_result = filtered_image_size / STREAMS;

	for (int i = 0; i < BLOCK_SIZES; i++)
	{
		printf("%dx%d blocks with %d streams\n", block_sizes[i], block_sizes[i], STREAMS);
		//Offset_input is the offset from which a kernel starts to read input image data
		int offset_input = 0;
		//Since the input potentially has more channels than the output(the output is always in grayscale), we need a different offset.
		int offset_output = 0;
		block = dim3(block_sizes[i], block_sizes[i]);
		grid = dim3((f_width + block.x - 1) / block.x, ((f_height / STREAMS) + block.y - 1) / block.y);
		CHECK(cudaEventRecord(begin_gpu));

		for (int j = 0; j < STREAMS; j++)
		{
			CHECK(cudaMemcpyAsync(&d_image[offset_input], &pinned_image[offset_input], chunk_size, cudaMemcpyHostToDevice, stream[j]));
			kernel_robert_h_convolution_stream << <grid, block, 0, stream[j] >> > (d_image, d_filtered_image, width, height / STREAMS, channels, offset_input, offset_output);
			CHECK(cudaMemcpyAsync(&pinned_filtered_image[offset_output], &d_filtered_image[offset_output], chunk_size_result, cudaMemcpyDeviceToHost, stream[j]));
			offset_input += (image_size / STREAMS) - width * channels;
			offset_output += chunk_size_result;
		}
		CHECK(cudaEventRecord(end_gpu));
		CHECK(cudaEventSynchronize(end_gpu));
		CHECK(cudaEventElapsedTime(&elapsed_time, begin_gpu, end_gpu));
		if (OUTPUT)
			stbi_write_png(output_filename_robert_stream[i], f_width, f_height, 1, pinned_filtered_image, f_width);
		elapsed_time = elapsed_time * pow(10.0, -3.0);
		printf("Time elapsed for computation and memcpy H2D and D2H:%f seconds\n", elapsed_time);
		printf("Speedup: %f %\n\n", (cpu_time / elapsed_time)*100.0);
	}
}

#pragma endregion

void freeHostMemory()
{
	free(image);
	free(filtered_image);
	cudaFreeHost(pinned_image);
	cudaFreeHost(pinned_filtered_image);
}

void freeDeviceMemory()
{
	cudaFree(d_image);
	cudaFree(d_filtered_image);

	for (int i = 0; i < STREAMS; i++)
	{
		cudaStreamDestroy(stream[i]);
	}

	cudaEventDestroy(begin_gpu);
	cudaEventDestroy(end_gpu);
}

int main()
{
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
	begin_cpu = clock();
	cpu_filter(image, width, height, channels, image_size, &(robert_kernel_3x3_h[0][0]), KERNEL_SIZE, filtered_image);
	end_cpu = clock();
	cpu_time = (float)(end_cpu - begin_cpu) / CLOCKS_PER_SEC;
	printf("CPU Convolution:%f seconds\n\n", cpu_time);
	if (OUTPUT)
		stbi_write_png("Sample_Convolution_Robert.png", f_width, f_height, 1, filtered_image, f_width);


	printf("Allocation of GPU gmem, streams and pinned memory...\n\n");

	//Creation timing events
	CHECK(cudaEventCreate(&begin_gpu));
	CHECK(cudaEventCreate(&end_gpu));

	//Allocation gmem
	CHECK(cudaMalloc((void**)&d_image, image_size));
	CHECK(cudaMalloc((void**)&d_filtered_image, filtered_image_size));

	//Allocation and initialization of pinned memory
	CHECK(cudaHostAlloc(&pinned_image, image_size, 0));
	CHECK(cudaHostAlloc(&pinned_filtered_image, filtered_image_size, 0));
	pinned_image = stbi_load(filename, &width, &height, &channels, 0);

	//Stream creation
	for (int i = 0; i < STREAMS; i++)
		CHECK(cudaStreamCreate(&stream[i]));

	printf("============================\n");
	printf("	GPU naive Convolution(Robert)	\n");
	printf("============================\n\n");

	naive_gpu_convolution();

	printf("============================\n");
	printf("	GPU Convolution(Smem)	\n");
	printf("============================\n\n");

	smem_gpu_convolution();

	printf("============================\n");
	printf("	GPU Convolution(Stream)	\n");
	printf("============================\n\n");

	stream_gpu_convolution();

	printf("============================\n");
	printf("	CPU Module(Sobel)	\n");
	printf("============================\n\n");
	begin_cpu = clock();
	cpu_module(image, width, height, channels, image_size, &(sobel_kernel_3x3_h[0][0]), &(sobel_kernel_3x3_v[0][0]), KERNEL_SIZE, filtered_image);
	end_cpu = clock();
	cpu_time = (float)(end_cpu - begin_cpu) / CLOCKS_PER_SEC;
	printf("CPU Module:%f seconds\n\n", elapsed_time);
	stbi_write_png("Sample_Module_Sobel.png", f_width, f_height, 1, filtered_image, f_width);

	freeHostMemory();
	freeDeviceMemory();
	return 0;
}

