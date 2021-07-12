#include "cpu_imp.cuh"
#include "utils.h"
#include <stdio.h>

int cpu_convolution(unsigned char* pixel, int channels, int* kernel, int width, int height, int kernel_size)
{
	int result = 0;
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
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}

	return result;
}

void cpu_filter(char* filename, char* output_filename, int* kernel, int kernel_size, int kernel_radius, bool output)
{
	unsigned char* image;
	unsigned char* filtered_image;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;
	begin_timer();
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	unsigned char* pixel = image;
	unsigned char* res = filtered_image;
	int value = 0;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			value = cpu_convolution(pixel, channels, kernel, width, height, kernel_size);
			if (value < 0)
				res[0] = 0;
			else
				res[0] = value;
			res += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
	end_timer();
	set_cpu_time(time_elapsed());
	printf("Time elapsed:%f seconds\n\n", time_elapsed());
	if (output)
		save_file(output_filename, filtered_image, f_width, f_height, 1);
	free(image);
	free(filtered_image);
}

void cpu_module(char* filename, char* output_filename, int* kernel_h, int* kernel_v, int kernel_size, int kernel_radius, bool output)
{
	int gh = 0;
	int gv = 0;
	unsigned char* image;
	unsigned char* filtered_image;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	size_t image_size;
	size_t filtered_image_size;
	begin_timer();
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	unsigned char* pixel = image;
	unsigned char* res = filtered_image;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			gh = cpu_convolution(pixel, channels, kernel_h, width, height, kernel_size);
			gv = cpu_convolution(pixel, channels, kernel_v, width, height, kernel_size);
			res[0] = (unsigned char)sqrt(gh*gh + gv * gv);
			res += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
	end_timer();
	set_cpu_time(time_elapsed());
	printf("Time elapsed:%f seconds\n\n", time_elapsed());
	if (output)
		save_file(output_filename, filtered_image, f_width, f_height, 1);
	free(image);
	free(filtered_image);
}