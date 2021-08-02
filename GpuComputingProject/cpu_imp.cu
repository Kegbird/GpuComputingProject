#include "cpu_imp.cuh"
#include "utils.h"
#include <math.h>
#include <stdio.h>

bool strong_neighbour(unsigned char * pixel, int strong_value, int width)
{
	if (*(pixel - width - 1) == strong_value || *(pixel - width) == strong_value || *(pixel - width + 1) == strong_value
		|| *(pixel - 1) == strong_value || *(pixel + 1) == strong_value ||
		*(pixel + width - 1) == strong_value || *(pixel + width) == strong_value || *(pixel + width + 1) == strong_value)
		return true;
	return false;
}

float convolution_cpu(unsigned char* pixel, int channels, float* kernel, int width, int height, int kernel_side)
{
	float result = 0;
	float color = 0;
	for (int j = 0; j < kernel_side; j++)
	{
		for (int k = 0; k < kernel_side; k++)
		{
			for (int i = 0; i < channels; i++)
				color += pixel[i] / channels;
			result += color * kernel[j*kernel_side + k];
			pixel += channels;
			color = 0.0;
		}
		pixel += (width * channels) - channels * (kernel_side - 1) - channels;
	}

	return result;
}

void convolution_module_cpu(unsigned char* pixel, int channels, float* kernel_h, float* kernel_v, int width, int height, int kernel_side, float* gh, float* gv)
{
	int color = 0;
	*gh = 0;
	*gv = 0;
	for (int j = 0; j < kernel_side; j++)
	{
		for (int k = 0; k < kernel_side; k++)
		{
			for (int i = 0; i < channels; i++)
				color += pixel[i] / channels;
			*gh += color * kernel_h[j*kernel_side + k];
			*gv += color * kernel_v[j*kernel_side + k];
			pixel += channels;
			color = 0;
		}
		pixel += (width * channels) - channels * (kernel_side - 1) - channels;
	}
}

void filter_cpu(const char* filename, const char* output_filename, float* kernel, int kernel_side, int kernel_radius, bool output)
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
	printf("Begin execution...\n");
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
			value = convolution_cpu(pixel, channels, kernel, width, height, kernel_side);
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
	printf("Execution ended.\n");
	printf("Time elapsed:%f seconds\n\n", time_elapsed());
	if (output)
	{
		printf("Saving saving the result...\n");
		printf("Result saved as %s.\n\n", output_filename);
		save_file(output_filename, filtered_image, f_width, f_height, 1);
	}
	free(image);
	free(filtered_image);
}

void module_cpu(const char* filename, const char* output_filename, float* kernel_h, float* kernel_v, int kernel_side, int kernel_radius, bool output)
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
	printf("Begin execution...\n");
	begin_timer();
	image = load_file_details(filename, &width, &height, &channels, &image_size, &filtered_image_size, &f_width, &f_height, kernel_radius);
	filtered_image = (unsigned char*)malloc(filtered_image_size);
	unsigned char* pixel = image;
	unsigned char* res = filtered_image;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			convolution_module_cpu(pixel, channels, kernel_h, kernel_v, width, height, kernel_side, &gh, &gv);
			res[0] = (unsigned char)sqrt(gh*gh + gv * gv);
			res += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
	end_timer();
	set_cpu_time(time_elapsed());
	printf("Execution ended.\n");
	printf("Time elapsed:%f seconds\n\n", time_elapsed());
	if (output)
	{
		printf("Saving saving the result...\n");
		printf("Result saved as %s.\n\n", output_filename);
		save_file(output_filename, filtered_image, f_width, f_height, 1);
	}
	free(image);
	free(filtered_image);
}

void canny_cpu(const char * filename, const char * output_filename, float* kernel_h, float* kernel_v, float* gaussian_kernel, float sigma, int kernel_side, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output)
{
	unsigned char* image;
	unsigned char* gaussian_filtered_image;
	int width, height;
	int f_width, f_height;
	int f_width_gaussian, f_height_gaussian;
	int channels;
	int sobel_kernel_side = 3;
	int sobel_kernel_radius = 1;
	size_t image_size, filtered_image_size, gaussian_image_size;
	printf("Begin execution...\n");
	begin_timer();
	image = load_file_details(filename, &width, &height, &channels, &image_size, &gaussian_image_size, &f_width_gaussian, &f_height_gaussian, kernel_radius);
	gaussian_filtered_image = (unsigned char*)malloc(gaussian_image_size);
	unsigned char* pixel = image;
	float gh = 0;
	float gv = 0;
	f_height = f_height_gaussian - sobel_kernel_radius * 2;
	f_width = f_width_gaussian - sobel_kernel_radius * 2;
	filtered_image_size = f_width * f_height;

	//Gaussian filtering
	for (int i = 0; i < f_height_gaussian; i++)
	{
		float value = 0;
		for (int j = 0; j < f_width_gaussian; j++, pixel += channels)
		{
			value = convolution_cpu(pixel, channels, gaussian_kernel, width, height, kernel_side);
			if (value < 0)
				*(gaussian_filtered_image+i* f_width_gaussian +j) = 0;
			else
				*(gaussian_filtered_image+i* f_width_gaussian +j) = value;
		}
		pixel += (kernel_radius * channels) * 2;
	}

	//Module and orientations
	unsigned char* module_image = (unsigned char*)malloc(filtered_image_size);
	float* orientations = (float*)malloc(f_height* f_width * sizeof(float));
	pixel = gaussian_filtered_image;
	for (int i = 0; i < f_height; i++)
	{
		for (int j = 0; j < f_width; j++, pixel++)
		{
			convolution_module_cpu(pixel, 1, &kernel_h[0], &kernel_v[0], f_width + sobel_kernel_radius * 2, f_height + sobel_kernel_radius * 2, sobel_kernel_side, &gh, &gv);
			module_image[i*f_width + j] = (unsigned char)sqrt(gh*gh + gv * gv);
			orientations[i*f_width + j] = atan2(gv, gh);
		}
		pixel += sobel_kernel_radius*2;
	}

	unsigned char* non_max_image = (unsigned char*)malloc(filtered_image_size);
	memset(non_max_image, 0, f_height*f_width);

	int strong_color = 255;
	int weak_color = 40;
	float high_threshold = high_threshold_ratio * strong_color;
	float low_threshold = low_threshold_ratio * high_threshold;

	//Non maxima suppression
	for (int i = 0; i < f_height; i++)
	{
		for (int j = 0; j < f_width; j++)
		{
			int index = i * f_width + j;
			
			if (i == 0 || j == 0 || i == f_height - 1 || j == f_width - 1)
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
					r = module_image[index + 1 - f_width];
					q = module_image[index - 1 + f_width];
				}
				else if (67.5 < angle && angle <= 112.5)
				{
					r = module_image[index - f_width];
					q = module_image[index + f_width];
				}
				else
				{
					r = module_image[index - f_width - 1];
					q = module_image[index + f_width + 1];
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
	}

	//Hysteresis
	for (int i = 0; i < f_height; i++)
	{
		for (int j = 0; j < f_width; j++)
		{
			if (i == 0 || j == 0 || i == f_height - 1 || j == f_width - 1)
				module_image[i*f_width + j] = 0;
			else if (non_max_image[i*f_width + j] == strong_color || ((non_max_image[i*f_width + j] == weak_color && strong_neighbour(non_max_image + i * f_width + j, strong_color, f_width))))
				module_image[i*f_width + j] = strong_color;
			else
				module_image[i*f_width + j] = 0;
		}
	}
	end_timer();
	set_cpu_time(time_elapsed());
	printf("Execution ended.\n");
	printf("Time elapsed:%f seconds\n\n", time_elapsed());
	if (output)
	{
		printf("Saving saving the result...\n");
		printf("Result saved as %s.\n\n", output_filename);
		save_file(output_filename, module_image, f_width, f_height, 1);
	}
	free(image);
	free(gaussian_filtered_image);
	free(module_image);
	free(non_max_image);
	free(orientations);
}
