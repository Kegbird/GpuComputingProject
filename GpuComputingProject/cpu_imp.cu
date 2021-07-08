#include<math.h>

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


void cpu_filter(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel, int kernel_size, unsigned char* result)
{
	unsigned char* pixel = image;
	unsigned char* r = result;
	int kernel_radius = kernel_size / 2;
	int value = 0;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			value = cpu_convolution(pixel, channels, kernel, width, height, kernel_size);
			if (value < 0)
				r[0] = 0;
			else
				r[0] = value;
			r += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
}

void cpu_module(unsigned char* image, int width, int height, int channels, size_t image_size, int* kernel_h, int* kernel_v, int kernel_size, unsigned char* result)
{
	int gh = 0;
	int gv = 0;
	int kernel_radius = kernel_size / 2;

	unsigned char* pixel = image;
	unsigned char* r = result;
	for (int i = 0; i < height - kernel_radius * 2; i++)
	{
		for (int j = 0; j < width - kernel_radius * 2; j++, pixel += channels)
		{
			gh = cpu_convolution(pixel, channels, kernel_h, width, height, kernel_size);
			gv = cpu_convolution(pixel, channels, kernel_v, width, height, kernel_size);
			r[0] = (unsigned char)sqrt(gh*gh + gv * gv);
			r += 1;
		}
		pixel += (kernel_radius * channels) * 2;
	}
}