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
	printf("============================\n");
	printf("	Input Details	\n");
	printf("============================\n\n");
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

void calculate_gaussian_kernel(float* kernel, float sigma, int kernel_size, int kernel_radius)
{
	float sum = 0;
	float r, s = 2.0 * sigma * sigma;
	for (int i = -kernel_radius; i <= kernel_radius; i++)
	{
		for (int j = -kernel_radius; j <= kernel_radius; j++)
		{
			r = (float)sqrt(i*i + j * j);
			kernel[(i + kernel_radius)*kernel_size + j + kernel_radius] = (exp(-(r*r) / s)) / (M_PI*s);
			sum += kernel[(i + kernel_radius)*kernel_size + j + kernel_radius];
		}
	}

	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
			kernel[i*kernel_size + j] /= sum;
	}
}