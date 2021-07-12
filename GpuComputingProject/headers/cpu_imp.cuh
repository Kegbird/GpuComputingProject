#ifndef CPU_IMP
#define CPU_IMP

int cpu_convolution(unsigned char* pixel, int channels, int* kernel, int width, int height, int kernel_size);
void cpu_filter(char* filename, char* output_filename, int* kernel, int kernel_size, int kernel_radius, bool output);
void cpu_module(char* filename, char* output_filename, int* kernel_h, int* kernel_v, int kernel_size, int kernel_radius, bool output);

#endif