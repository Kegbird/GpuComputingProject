#ifndef GPU_IMP
#define GPU_IMP

void load_constant_memory_robert_h(int* kernel, int kernel_size);
void naive_robert_gpu_convolution(char* filename, int* kernel, int kernel_size, int kernel_radius, bool output);
void smem_gpu_convolution(char* filename, int* kernel, int kernel_size, int kernel_radius, bool output);
void stream_gpu_convolution(char* filename, int* kernel, int kernel_size, int kernel_radius, bool output);

#endif