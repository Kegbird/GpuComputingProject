#ifndef CPU_IMP
#define CPU_IMP

bool strong_neighbour(unsigned char * pixel, int strong_value, int width);
float convolution_cpu(unsigned char* pixel, int channels, float* kernel, int width, int height, int kernel_side);
void filter_cpu(const char* filename, const char* output_filename, float* kernel, int kernel_side, int kernel_radius, bool output);
void module_cpu(const char* filename, const char* output_filename, float* kernel_h, float* kernel_v, int kernel_side, int kernel_radius, bool output);
void convolution_module_cpu(unsigned char* pixel, int channels, float* kernel_h, float* kernel_v, int width, int height, int kernel_side, float* gh, float* gv);
void canny_cpu(const char * filename, const char * output_filename, float* kernel_h, float* kernel_v, float* gaussian_kernel, float sigma, int kernel_side, int kernel_radius, float low_threshold_ratio, float highthreshold_ratio, bool output);

#endif