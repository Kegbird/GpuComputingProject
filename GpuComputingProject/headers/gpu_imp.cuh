#ifndef GPU_IMP
#define GPU_IMP

void load_constant_memory_robert_h(int* kernel, int kernel_size);
void load_constant_memory_robert_v(int* kernel, int kernel_size);
void load_constant_memory_sobel_h(int* kernel, int kernel_size);
void load_constant_memory_sobel_v(int* kernel, int kernel_size);
void naive_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void smem_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void stream_robert_convolution_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void naive_sobel_module_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void smem_sobel_module_gpu(char* filename, int kernel_size, int kernel_radius, bool output);
void stream_sobel_module_gpu(char* filename, int kernel_size, int kernel_radius, bool output);

#endif