#ifndef GPU_IMP
#define GPU_IMP

void load_constant_memory_robert_h(float* kernel, int kernel_size);
void load_constant_memory_robert_v(float* kernel, int kernel_size);
void load_constant_memory_sobel_h(float* kernel, int kernel_size);
void load_constant_memory_sobel_v(float* kernel, int kernel_size);
void load_constant_memory_gaussian(float* kernel, int kernel_size);
void naive_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void smem_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void stream_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void naive_sobel_module_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void smem_sobel_module_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void stream_sobel_module_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void naive_canny_gpu(char * filename, float * kernel_h, float * kernel_v, float * gaussian_kernel, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float highthreshold_ratio, bool output);

#endif