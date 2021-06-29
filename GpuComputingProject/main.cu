
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION 
#include <stb_image_write.h>

int robert_kernel_3x3_h[3][3] = { {1, 0, 0}, {0, -1, 0}, {0, 0, 0} };
int robert_kernel_3x3_v[3][3] = { {0, 1, 0},{-1, 0, 0}, {0, 0, 0} };

int sobel_kernel_3x3_h[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
int sobel_kernel_3x3_v[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

int cpu_convolution(unsigned char* pixel, int channels, int* kernel, int width, int height, int kernel_size)
{
	int result = 0;
	int kernel_length = kernel_size * kernel_size;
	int color = 0;
	for (int j = 0; j < kernel_length; j++)
	{
		for (int i = 0; i < channels; i++)
			color += pixel[i];
		color /= channels;
		result += color*kernel[j];
		if ((j + 1) % kernel_size == 0)
			pixel += (width * channels) - channels * (kernel_size - 1);
		else
			pixel += channels;
		color = 0;
	}

	return result;
}

void cpu_module(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel_h, int* kernel_v, int kernel_size, unsigned char* result)
{
	int gh = 0;
	int gv = 0;
	int modulo = 0;
	int kernel_radius = kernel_size / 2;

	int index = 0;
	unsigned char* pixel = image;
	unsigned char* r = result;
	for (int i = 0; i < height - kernel_radius*2; i++)
	{
		for (int j = 0; j < width - kernel_radius*2; j++, pixel += channels)
		{
			gh = cpu_convolution(pixel, channels, kernel_h, width, height, kernel_size);
			gv = cpu_convolution(pixel, channels, kernel_v, width, height, kernel_size);
			r[0] = sqrt(gh*gh+gv*gv);
			r += 1;
		}
		pixel += (kernel_radius * channels)*2;
	}
}

int main()
{
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;
	int kernel_size = 3;
	char filename[] = "Prova.jpg";
	char filtered_filename[] = "prova_filtered.png";
	unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);

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
	f_width = width - (kernel_size / 2) * 2;
	f_height = height - (kernel_size / 2) * 2;
	unsigned char* filtered_image = (unsigned char*)malloc(f_width*f_height);
	cpu_module(image, width, height, channels, image_size, &(sobel_kernel_3x3_h[0][0]), &(sobel_kernel_3x3_v[0][0]), kernel_size, filtered_image);
	stbi_write_png(filtered_filename, f_width, f_height, 1, filtered_image, f_width);
	free(image);
	free(filtered_image);
}

