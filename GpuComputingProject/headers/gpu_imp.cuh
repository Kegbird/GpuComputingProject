#ifndef GPU_IMP
#define GPU_IMP

void load_constant_memory_robert_h(float* kernel, int kernel_size);
void load_constant_memory_robert_v(float* kernel, int kernel_size);
void load_constant_memory_sobel_h(float* kernel, int kernel_size);
void load_constant_memory_sobel_v(float* kernel, int kernel_size);
void load_constant_memory_gaussian(float* kernel, int kernel_size);
void naive_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void smem_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void stream_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void stream_smem_robert_convolution_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void naive_module_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void smem_module_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void stream_module_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void stream_smem_module_gpu(const char* filename, int kernel_size, int kernel_radius, bool output);
void naive_canny_gpu(const char * filename, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output);
void smem_canny_gpu(const char * filename, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output);
void stream_canny_gpu(const char * filename, float sigma, int kernel_size, int kernel_radius, float low_threshold_ratio, float high_threshold_ratio, bool output);

#endif