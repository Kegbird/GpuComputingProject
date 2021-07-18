#ifndef CPU_IMP
#define CPU_IMP

bool strong_neighbour(unsigned char * pixel, float strong_value, int width, int height);
float cpu_convolution(unsigned char* pixel, int channels, float* kernel, int width, int height, int kernel_size);
void cpu_filter(char* filename, char* output_filename, float* kernel, int kernel_size, int kernel_radius, bool output);
void cpu_module(char* filename, char* output_filename, float* kernel_h, float* kernel_v, int kernel_size, int kernel_radius, bool output);
void cpu_convolution_module(unsigned char* pixel, int channels, float* kernel_h, float* kernel_v, int width, int height, int kernel_size, float* gh, float* gv);
void cpu_canny(char * filename, char * output_filename, float* kernel_h, float* kernel_v, float* gaussian_kernel, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float highthreshold_ratio, bool output);

#endif