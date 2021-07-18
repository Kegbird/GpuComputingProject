#include "cpu_imp.cuh"
#include "utils.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

bool strong_neighbour(unsigned char * pixel, float strong_value, int width, int height)
{
	//Inside the image
	if (*(pixel - width - 1) == strong_value || *(pixel - width) == strong_value || *(pixel - width + 1) == strong_value
		|| *(pixel - 1) == strong_value || *(pixel + 1) == strong_value ||
		*(pixel + width - 1) == strong_value || *(pixel + width) == strong_value || *(pixel + width + 1) == strong_value)
		return true;
	return false;
}

float cpu_convolution(unsigned char* pixel, int channels, float* kernel, int width, int height, int kernel_size)
{
	float result = 0;
	float color = 0;
	for (int j = 0; j < kernel_size; j++)
	{
		for (int k = 0; k < kernel_size; k++)
		{
			for (int i = 0; i < channels; i++)
				color += pixel[i] / channels;
			result += color * kernel[j*kernel_size + k];
			pixel += channels;
			color = 0.0;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}

	return result;
}

void cpu_convolution_module(unsigned char* pixel, int channels, float* kernel_h, float* kernel_v, int width, int height, int kernel_size, float* gh, float* gv)
{
	int result = 0;
	int color = 0;
	*gh = 0;
	*gv = 0;
	for (int j = 0; j < kernel_size; j++)
	{
		for (int k = 0; k < kernel_size; k++)
		{
			for (int i = 0; i < channels; i++)
				color += pixel[i] / channels;
			*gh += color * kernel_h[j*kernel_size + k];
			*gv += color * kernel_v[j*kernel_size + k];
			pixel += channels;
			color = 0;
		}
		pixel += (width * channels) - channels * (kernel_size - 1) - channels;
	}
}

void cpu_filter(char* filename, char* output_filename, float* kernel, int kernel_size, int kernel_radius, bool output)
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
	float value = 0;
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

void cpu_module(char* filename, char* output_filename, float* kernel_h, float* kernel_v, int kernel_size, int kernel_radius, bool output)
{
	float gh = 0;
	float gv = 0;
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
			cpu_convolution_module(pixel, channels, kernel_h, kernel_v, width, height, kernel_size, &gh, &gv);
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

void cpu_canny(char * filename, char * output_filename, float* kernel_h, float* kernel_v, float* gaussian_kernel, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	calculate_gaussian_kernel(gaussian_kernel, sigma, kernel_size, kernel_radius);

	unsigned char* image;
	unsigned char* gaussian_filtered_image;
	int width;
	int height;
	int f_width;
	int f_height;
	int channels;
	int sobel_kernel_size = 3;
	size_t image_size;
	size_t filtered_image_size;
	begin_timer();
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	gaussian_filtered_image = (unsigned char*)malloc(filtered_image_size);
	unsigned char* pixel = image;
	float value = 0;
	unsigned char* module = (unsigned char*)gaussian_filtered_image;
	begin_timer();
	//Gaussian filtering
	for (int i = 0; i < f_height; i++)
	{
		for (int j = 0; j < f_width; j++, pixel += channels)
		{
			value = cpu_convolution(pixel, channels, gaussian_kernel, width, height, kernel_size);
			if (value < 0)
				module[0] = 0;
			else
				module[0] = value;
			module += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}

	//Module and orientatio
	float gh = 0;
	float gv = 0;
	f_height = f_height - kernel_radius * 2;
	f_width = f_width - kernel_radius * 2;
	filtered_image_size = f_width * f_height;
	unsigned char* module_image = (unsigned char*)malloc(filtered_image_size);
	float* orientations = (float*)malloc(filtered_image_size * sizeof(float));
	pixel = gaussian_filtered_image;
	float deg = 180 / M_PI;
	float angle;
	for (int i = 0; i < f_height; i++)
	{
		for (int j = 0; j < f_width; j++, pixel++)
		{
			cpu_convolution_module(pixel, 1, &kernel_h[0], &kernel_v[0], f_width + kernel_radius * 2, f_height + kernel_radius * 2, sobel_kernel_size, &gh, &gv);
			module_image[i*f_width + j] = (unsigned char)sqrt(gh*gh + gv * gv);
			angle = (atan(gv / gh)*deg);
			if (angle < 0)
				angle += 180.0f;
			orientations[i*f_width + j] = angle;
		}
		pixel += kernel_radius * 2;
	}

	unsigned char* non_max_image = (unsigned char*)malloc(f_height*f_width);
	memset(non_max_image, 0, f_height*f_width);

	//Non maxima suppression
	int strong_color = 0, weak_color = 0;
	for (int i = 1; i < f_height - 1; i++)
	{
		for (int j = 1; j < f_width - 1; j++)
		{
			float angle = orientations[i*f_height + j];
			float r, q;
			if ((0.0 <= angle && angle <= 22.5) || (157.5 <= angle && angle <= 180))
			{
				r = module_image[i*f_width + j + 1];
				q = module_image[i*f_width + j - 1];
			}
			else if (22.5 < angle && angle <= 67.5)
			{
				r = module_image[i*f_width + j + 1 - f_width];
				q = module_image[i*f_width + j - 1 + f_width];
			}
			else if (67.5 < angle && angle <= 112.5)
			{
				r = module_image[i*f_width + j - f_width];
				q = module_image[i*f_width + j + f_width];
			}
			else
			{
				r = module_image[i*f_width + j - f_width - 1];
				q = module_image[i*f_width + j + f_width + 1];
			}

			if (module_image[i*f_width + j] >= r && module_image[i*f_width + j] >= q)
				non_max_image[i*f_width + j] = module_image[i*f_width + j];
			else
				non_max_image[i*f_width + j] = 0;

			if (strong_color < module_image[i*f_width + j])
				strong_color = module_image[i*f_width + j];
		}
	}

	//Double threshold
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;

	strong_color = 255;
	weak_color = 40;

	unsigned char* threshold_image = (unsigned char*)malloc(f_height*f_width);
	memset(threshold_image, 0, f_height*f_width);

	for (int i = 0; i < f_height; i++)
	{
		for (int j = 0; j < f_width; j++)
		{
			if (non_max_image[i*f_width + j] < low_threshold)
				threshold_image[i*f_width + j] = 0;
			else if (non_max_image[i*f_width + j] >= high_threshold)
				threshold_image[i*f_width + j] = strong_color;
			else if (low_threshold <= non_max_image[i*f_width + j] && non_max_image[i*f_width + j] <= high_threshold)
				threshold_image[i*f_width + j] = weak_color;
		}
	}

	unsigned char* filtered_image = (unsigned char*)malloc(f_height*f_width);
	memset(filtered_image, 0, f_height*f_width);

	//Hysteresis
	for (int i = 1; i < f_height - 1; i++)
	{
		for (int j = 1; j < f_width - 1; j++)
		{
			if (threshold_image[i*f_width + j] == weak_color && strong_neighbour(threshold_image + i * f_width + j, strong_color, f_width, f_height))
				filtered_image[i*f_width + j] = strong_color;
			else if (threshold_image[i*f_width + j] == strong_color)
				filtered_image[i*f_width + j] = strong_color;
			else
				filtered_image[i*f_width + j] = 0;
		}
	}
	end_timer();
	set_cpu_time(time_elapsed());

	if (output)
		save_file(output_filename, filtered_image, f_width, f_height, 1);

	free(image);
	free(gaussian_filtered_image);
	free(module_image);
	free(non_max_image);
	free(threshold_image);
	free(filtered_image);
	free(orientations);
}
