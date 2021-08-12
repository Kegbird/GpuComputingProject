#include "utils.h"
#include <stdio.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include <stb_image_write.h>

clock_t begin, end;
float cpu_time = 1;


inline int convertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
			{0x20, 32},
			{0x30, 192},
			{0x32, 192},
			{0x35, 192},
			{0x37, 192},
			{0x50, 128},
			{0x52, 128},
			{0x53, 128},
			{0x60,  64},
			{0x61, 128},
			{0x62, 128},
			{0x70,  64},
			{0x72,  64},
			{0x75,  64},
			{-1, -1} };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

void print_device_props()
{
	printf("- Device Info -\n\n");
	int dev = 0, driverVersion = 0, runtimeVersion = 0;

	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	printf("Device: %s\n", deviceProp.name);

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10,
		runtimeVersion / 1000, (runtimeVersion % 100) / 10);

	printf("CUDA Capability Major/Minor version number: %d.%d\n",
		deviceProp.major, deviceProp.minor);

	printf("Total amount of global memory: %.0f MBytes (%llu bytes)\n",
		(float)deviceProp.totalGlobalMem / 1048576.0f,
		(unsigned long long) deviceProp.totalGlobalMem);

	printf("(%d) Multiprocessors, (%3d) CUDA Cores/MP: %d CUDA Cores\n",
		deviceProp.multiProcessorCount,
		convertSMVer2Cores(deviceProp.major, deviceProp.minor),
		convertSMVer2Cores(deviceProp.major, deviceProp.minor) *
		deviceProp.multiProcessorCount);

	printf("GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n",
		deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

	printf("Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
	printf("Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
	if (deviceProp.l2CacheSize)
		printf("L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);

	printf("Maximum Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
		deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
		deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
		deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

	printf("Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
		deviceProp.maxTexture1DLayered[0],
		deviceProp.maxTexture1DLayered[1]);

	printf("Maximum Layered 2D Texture Size, (num) layers 2D=(%d, %d), %d layers\n",
		deviceProp.maxTexture2DLayered[0],
		deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);

	printf("Total amount of constant memory: %zu bytes\n",
		deviceProp.totalConstMem);
	printf("Total amount of shared memory per block: %zu bytes\n",
		deviceProp.sharedMemPerBlock);
	printf("Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf("Warp size: %d\n",
		deviceProp.warpSize);
	printf("Maximum number of threads per multiprocessor: %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("Maximum number of threads per block: %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("Maximum memory pitch: %zu bytes\n",
		deviceProp.memPitch);
	printf("Texture alignment: %zu bytes\n",
		deviceProp.textureAlignment);
	printf("Concurrent copy and kernel execution: %s with %d copy engine(s)\n",
		(deviceProp.deviceOverlap ? "Yes" : "No"),
		deviceProp.asyncEngineCount);
	printf("Run time limit on kernels: %s\n",
		deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
	printf("Integrated GPU sharing Host Memory: %s\n",
		deviceProp.integrated ? "Yes" : "No");
	printf("Support host page-locked memory mapping: %s\n",
		deviceProp.canMapHostMemory ? "Yes" : "No");
	printf("Alignment requirement for Surfaces: %s\n",
		deviceProp.surfaceAlignment ? "Yes" : "No");
	printf("Device has ECC support: %s\n",
		deviceProp.ECCEnabled ? "Enabled" : "Disabled");

	printf("Device supports Unified Addressing (UVA): %s\n",
		deviceProp.unifiedAddressing ? "Yes" : "No");
	printf("Device PCI Domain ID / Bus ID / location ID: %d / %d / %d\n\n",
		deviceProp.pciDomainID, deviceProp.pciBusID,
		deviceProp.pciDeviceID);
}

bool check_input(const char* filename)
{
	unsigned char* image;
	int width, height, channels;
	image = stbi_load(filename, &width, &height, &channels, 0);
	if (image == NULL)
	{
		free(image);
		printf("No image provided!");
		return false;
	}
	else if (width < 3 || height < 3)
	{
		free(image);
		printf("The image provided is too small; the minimum resolution is 3x3 pixels.\n");
		return false;
	}
	free(image);
	return true;
}

unsigned char* load_file_details(const char* filename, int* width, int* height, int* channels, size_t* image_size, size_t* filtered_image_size, int* f_width, int* f_height, int kernel_radius)
{
	unsigned char* image = stbi_load(filename, width, height, channels, 0);
	*image_size = *width * *height * *channels;
	*f_width = *width - kernel_radius * 2;
	*f_height = *height - kernel_radius * 2;
	*filtered_image_size = (*f_width) * (*f_height);
	return image;
}

void print_file_details(const char* filename)
{
	unsigned char* image;
	int width, height, channels;
	image = stbi_load(filename, &width, &height, &channels, 0);
	printf("- Input Details -\n\n");
	printf("Width: %d\n", width);
	printf("Height: %d\n", height);
	printf("Channels: %d\n", channels);
	printf("Size: %d bytes\n\n", width*height*channels);
	free(image);
}

void begin_timer()
{
	begin = clock();
}

void end_timer()
{
	end = clock();
}

float time_elapsed()
{
	return (float)(end - begin) / CLOCKS_PER_SEC;
}

void set_cpu_time(float time)
{
	cpu_time = time;
}

float speedup()
{
	if (time_elapsed > 0)
		return cpu_time / time_elapsed();
	return 0;
}

void save_file(const char* filename, unsigned char* image, int width, int height, int channels)
{
	stbi_write_png(filename, width, height, 1, image, width*channels);
}

void calculate_gaussian_kernel(float* kernel, float sigma, int kernel_side, int kernel_radius)
{
	float sum = 0;
	float r, s = 2.0 * sigma * sigma;
	for (int i = -kernel_radius; i <= kernel_radius; i++)
	{
		for (int j = -kernel_radius; j <= kernel_radius; j++)
		{
			r = (float)sqrt(i*i + j * j);
			kernel[(i + kernel_radius)*kernel_side + j + kernel_radius] = (exp(-(r*r) / s)) / (M_PI*s);
			sum += kernel[(i + kernel_radius)*kernel_side + j + kernel_radius];
		}
	}

	for (int i = 0; i < kernel_side; i++)
	{
		for (int j = 0; j < kernel_side; j++)
			kernel[i*kernel_side + j] /= sum;
	}
}