
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION 
#include <stb_image.h>

int main()
{
	int width;
	int height;
	int channels;
	size_t image_size;
	unsigned char* image = stbi_load("prova.png", &width, &height, &channels, 0);
	if (image == NULL)
	{
		printf("No image provided!");
		return 0;
	}
	image_size = width * height * channels;

	printf("Number of channels:%d\n", channels);
	for (unsigned char* pixel = image; pixel != image + image_size; pixel+=channels)
	{
		printf("%d ",(int)pixel[0]);
		printf("%d ", (int)pixel[1]);
		printf("%d ", (int)pixel[2]);
		printf("\n");
	}

	free(image);
}

