//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
//#define USE_MPIR
#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
//#include <cuSparse.h>
#elif(VKFFT_BACKEND==2)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND==3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#endif
#include "pfSolve.h"
#include "utils_VkFFT.h"
#ifdef USE_MPIR
#include <mpir.h>
#endif

pfLD mu2(pfLD n, pfLD alpha, pfLD beta) {
	return pfFPinit("1.0");//pfsqrt(pfFPinit("2.0") * (n + beta) * (n + alpha + beta) / (pfFPinit("2.0") * n + alpha + beta) / (pfFPinit("2.0") * n + alpha + beta + pfFPinit("1.0")));
}
pfLD nu2(pfLD n, pfLD alpha, pfLD beta) {
	return pfsqrt(pfFPinit("2.0") * (n + pfFPinit("1.0")) * (n + alpha + pfFPinit("1.0")) / (pfFPinit("2.0") * n + alpha + beta + 1) / (pfFPinit("2.0") * n + alpha + beta + pfFPinit("2.0")));
}
static inline void convToDoubleDouble(double* out, pfLD in) {
	double high, low;
	high = (double) in;
	if (isnan (high) || isinf (high)){
		low = 0.0;
	}else{
		low = (double) (in - (pfLD)high);
		double temp = high + low;
		low = (high - temp) + low;
		high = temp;
	}
	out[0] = high;
	out[1] = low;
	return;
}
PfSolveResult sample_0_benchmark_VkFFT_single(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
	PfSolveResult resFFT = PFSOLVE_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#endif
	if (file_output)
		fprintf(output, "0 - PfSolve PCR test\n");
	printf("0 - PfSolve PCR test\n");
	const int num_runs = 1;
	float benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	for (uint64_t n = 0; n < 1; n++) {
		float run_time[num_runs];
		for (uint64_t r = 0; r < 1; r++) {
			//Configuration + FFT application .
			PfSolveConfiguration configuration = {};
			PfSolveApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.M_size = 33;// 16 + 16 * (n % 128); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.M_size_pow2 = (int64_t)pow(2, (int)ceil(log2((double)configuration.M_size)));; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[0] = configuration.M_size; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
			configuration.size[1] = 1;
			configuration.size[2] = 1;
			configuration.scaleC = 1;
			configuration.jw_type = 10;
			configuration.quadDoubleDoublePrecision = 1;
			configuration.isOutputFormatted = 1;
			int* x;
			int** y;
			y = &x;
			//configuration.aimThreads = 32;
			configuration.jw_control_bitmask = (8+16+32+128+256);// (1 << 6);
			//configuration.JW_sequential = 1;
			//configuration.JW_parallel = 1;
			configuration.outputBufferStride[0] = configuration.size[0];
			//configuration.performWorland = 1;
			//configuration.upperBound = 1;
			configuration.offsetV = 2 * configuration.size[0];
			//CUstream hStream;
			//cudaStreamCreate(&hStream);
			//configuration.stream = &hStream;
			//configuration.num_streams = 1;
			configuration.keepShaderCode = 1;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before PfSolve plan creation. if not, PfSolve will create and destroy one after initialization
#elif(VKFFT_BACKEND==3)
			configuration.platform = &vkGPU->platform;
			configuration.context = &vkGPU->context;
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSolveSize;
			uint64_t bufferSolveResSize;

			bufferSolveSize = (uint64_t)2*sizeof(double) * 4 * configuration.size[0];
			bufferSolveResSize = (uint64_t)2*sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2];

#if(VKFFT_BACKEND==0)
			VkBuffer bufferSolve = {};
			VkDeviceMemory bufferSolveDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &bufferSolve, &bufferSolveDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSolveSize);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
			VkBuffer bufferSolveRes = {};
			VkDeviceMemory bufferSolveResDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &bufferSolveRes, &bufferSolveResDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSolveResSize);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			cuDoubleComplex* bufferSolve = 0;
			res = cudaMalloc((void**)&bufferSolve, bufferSolveSize);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
			cuDoubleComplex* bufferSolveRes = 0;
			res = cudaMalloc((void**)&bufferSolveRes, bufferSolveResSize);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
			hipDoubleComplex* bufferSolve = 0;
			res = hipMalloc((void**)&bufferSolve, bufferSolveSize);
			if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
			hipDoubleComplex* bufferSolveRes = 0;
			res = hipMalloc((void**)&bufferSolveRes, bufferSolveResSize);
			if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==3)
			cl_mem bufferSolve = 0;
			bufferSolve = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveSize, 0, &res);
			if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
			cl_mem bufferSolveRes = 0;
			bufferSolveRes = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveResSize, 0, &res);
			if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#endif

#if(VKFFT_BACKEND==0)
			configuration.buffer = &bufferSolve;
			configuration.outputBuffer = &bufferSolveRes;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&bufferSolve;
			configuration.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&bufferSolve;
			configuration.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==3)
			configuration.buffer = &bufferSolve;
			configuration.outputBuffer = &bufferSolveRes;
#endif
			configuration.bufferSize = &bufferSolveSize;
			configuration.outputBufferSize = &bufferSolveResSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			int l = 1;// 2 * configuration.size[0];
			pfLD* buffer_input_matrix = (pfLD*)calloc(bufferSolveSize,sizeof(char));
			double* buffer_input_matrix_gpu = (double*)calloc(bufferSolveSize,sizeof(char));
			
			if (configuration.upperBound != 1) {
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					buffer_input_matrix[i] = pfFPinit("1.0");
					pfLD in = pfFPinit("1.0") / mu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5"));
					convToDoubleDouble(&buffer_input_matrix_gpu[2*i], in);
							
					//printf("%f\n", buffer_input_matrix_gpu[i]);
				}
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					buffer_input_matrix[configuration.size[0] + i] = 0;
					buffer_input_matrix_gpu[2*(configuration.size[0] + i)] = 0;
					buffer_input_matrix_gpu[2*(configuration.size[0] + i)+1] = 0;
				}
			}
			else {
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					buffer_input_matrix[configuration.size[0] + i] = pfFPinit("1.0");
					pfLD in = pfFPinit("1.0") / mu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5"));
					convToDoubleDouble(&buffer_input_matrix_gpu[(2*(configuration.size[0] + i))], in);
					//printf("%f\n", buffer_input_matrix_gpu[i]);.
				}
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					buffer_input_matrix[i] = 0;
					buffer_input_matrix_gpu[2*i] = 0;
					buffer_input_matrix_gpu[2*i+1] = 0;
				}
			}
			if (configuration.upperBound != 1) {
				for (uint64_t i = 1; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[2 * configuration.size[0] + i] = nu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5"));// / mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					pfLD in = nu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5")) / mu2(i - 1, pfFPinit("-0.5"), l + pfFPinit("0.5"));// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					convToDoubleDouble(&buffer_input_matrix_gpu[2 * (2 * configuration.size[0] + i - 1)], in);
					
					//printf("%f %f\n", buffer_input_matrix[2 * configuration.size[0] + i], buffer_input_matrix_gpu[configuration.size[0] + i-1]);
				}
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[3 * configuration.size[0] + i] = mu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5"));// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					//buffer_input_matrix_gpu[3 * configuration.size[0] + i] = 1;// mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					//printf("%f\n", buffer_input_matrix[3 * configuration.size[0] + i]);
				}
			}
			else {
				for (uint64_t i = 0; i < 1 * configuration.size[0]-1; i++) {
					buffer_input_matrix[3 * configuration.size[0] + i] = nu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5"));// / mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					pfLD in = nu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5")) / mu2(i + 1, pfFPinit("-0.5"), l + pfFPinit("0.5"));// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					convToDoubleDouble(&buffer_input_matrix_gpu[2 * (i+1)], in);
					
					//printf("%f %f\n", buffer_input_matrix[3 * configuration.size[0] + i], buffer_input_matrix_gpu[1 + i]);
				}
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[2 * configuration.size[0] + i] = mu2(i, pfFPinit("-0.5"), l + pfFPinit("0.5"));// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					//buffer_input_matrix_gpu[2 * configuration.size[0] + i] = 1;// mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					//printf("%f\n", buffer_input_matrix[3 * configuration.size[0] + i]);
				}
			}

			/*for (uint64_t i = 0; i < configuration.size[0]; i++) {
					
				buffer_input_matrix_gpu[i] =  a0[i];
					
				buffer_input_matrix_gpu[i + configuration.size[0]] =  b0[i];
					
				buffer_input_matrix_gpu[i+ 2*configuration.size[0]] = -c0[i];
				buffer_input_matrix[i] = a0[i];
				buffer_input_matrix[i + configuration.size[0]] = b0[i];
				buffer_input_matrix[i + 2 * configuration.size[0]+1] =  -c0[i];
				buffer_input_matrix[i + 3 * configuration.size[0]] = 1;// c0[i];
					//printf("%f\n", buffer_input_matrix_gpu[i]);
			}
			
			buffer_input_matrix[ 3 * configuration.size[0]] = 1;*/
			for (uint64_t i = 100; i < 32; i++) {
				//buffer_input_matrix[2 * configuration.size[0] + i] = 0;// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				//printf("%f\n", buffer_input_matrix[2 * configuration.size[0] + i]);
			}
			//buffer_input_matrix[0] = 0;
			pfLD* buffer_input_systems = (pfLD*)calloc(bufferSolveResSize, sizeof(char));
			double* buffer_input_systems2 = (double*)calloc(bufferSolveResSize, sizeof(char));
			for (uint64_t j = 0; j < configuration.size[1]; j++) {
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					pfLD in = (pfFPinit("2.0") * ((pfLD)rand()) / RAND_MAX - pfFPinit("1.0"));
					convToDoubleDouble(&buffer_input_systems2[2*(i + j * configuration.size[0])], in);
					buffer_input_systems[(i + j * configuration.size[0])] = in;// (pfLD)buffer_input_systems2[2*(i + j * configuration.size[0])] + (pfLD)buffer_input_systems2[2*(i + j * configuration.size[0])+1];//x0[i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);// +i + j * configuration.size[0];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				}
			}
			
			//buffer_input_systems[0] = 0.69;
			//buffer_input_systems[1] = 0.23;
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			resFFT = transferDataFromCPU(vkGPU, buffer_input_matrix_gpu, &bufferSolve, bufferSolveSize);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
			resFFT = transferDataFromCPU(vkGPU, buffer_input_systems, &bufferSolveRes, bufferSolveResSize);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			res = cudaMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			res = cudaMemcpy(bufferSolveRes, buffer_input_systems2, bufferSolveResSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
			res = hipMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, hipMemcpyHostToDevice);
			if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			res = hipMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, hipMemcpyHostToDevice);
			if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
			res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolve, CL_TRUE, 0, bufferSolveSize, buffer_input_matrix_gpu, 0, NULL, NULL);
			if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
			res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, buffer_input_systems, 0, NULL, NULL);
			if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
#endif
			for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
				//buffer_input_matrix[2 * configuration.size[0] + i] = 1/ buffer_input_matrix[2 * configuration.size[0] + i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				//printf("%f\n", buffer_input_matrix[2 * configuration.size[0] + i]);
			}
#ifdef USE_MPIR
			pfLD prev0;
			pfLD prev = buffer_input_systems[0];
			buffer_input_systems[0] = buffer_input_systems[0] * buffer_input_matrix[configuration.size[0]];
			for (uint64_t j = 1; j < configuration.size[0]; j++) {
				prev0 = buffer_input_systems[j];
				buffer_input_systems[j] = buffer_input_systems[j] * buffer_input_matrix[configuration.size[0]+j] + prev * buffer_input_matrix[j] ;// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				prev = prev0;			
			}
			mpf_set_default_prec(128);
			mpf_t temp;
			mpf_init(temp);
			mpf_set_d(temp, 0);
			mpf_t temp0;
			mpf_init(temp0);
			mpf_set_d(temp0, 0);


			/*for (uint64_t i = 0; i < configuration.size[0]; i++) {
				mpf_set_d(temp, 1.0);
				mpf_set_d(temp0, mu(i, -0.5, l - 0.5 + 1));
				mpf_div(temp, temp, temp0);
				buffer_input_matrix_gpu[i] = mpf_get_d(temp);	
			}
			for (uint64_t i = 1; i < configuration.size[0]; i++) {	
				mpf_set_d(temp, nu(i, -0.5, l - 0.5 + 1));
				mpf_set_d(temp0, mu(i - 1, -0.5, l - 0.5 + 1));// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					
				mpf_div(temp, temp, temp0);
				buffer_input_matrix_gpu[2 * configuration.size[0] + i - 1] = mpf_get_d(temp);	
			}
			res = cudaMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;*/

			mpf_t *inp = (mpf_t * )malloc(sizeof(mpf_t) * configuration.size[0] * configuration.size[1]);
			//mpf_t* outp = (mpf_t*)malloc(sizeof(mpf_t)*configuration.size[0]);
			mpf_t* matrix = (mpf_t*)malloc(4* sizeof(mpf_t) * configuration.size[0]);
			for (uint64_t j = 0; j < configuration.size[0] * configuration.size[1]; j++) {
				mpf_init(inp[j]);
				mpf_set_d(inp[j], buffer_input_systems[j]);
			}
			for (uint64_t j = 0; j < 4 * configuration.size[0]; j++) {
				mpf_init(matrix[j]);
				mpf_set_d(matrix[j], buffer_input_matrix[j]);
			}
			double* ress = (double*)malloc(bufferSolveResSize);
			double* ress2 = (double*)malloc(bufferSolveResSize);
			
			for (uint64_t j = 0; j < configuration.size[1]; j++) {
				if (configuration.upperBound != 1) {
					ress2[configuration.size[0] - 1 + j * configuration.size[0]] = buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]] / buffer_input_matrix[4 * configuration.size[0] - 1];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					//ress[configuration.size[0] - 1 + j * configuration.size[0]] = ress2[configuration.size[0] - 1 + j * configuration.size[0]];
					//printf("%f\n", ress2[configuration.size[0] - 1 + j * configuration.size[0]]);
//printf("%f\n", buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]]);
//printf("%f\n", buffer_input_matrix[4 * configuration.size[0] - 1]);
mpf_div(temp, inp[configuration.size[0] - 1 + j * configuration.size[0]], matrix[4 * configuration.size[0] - 1]);
ress[configuration.size[0] - 1 + j * configuration.size[0]] = mpf_get_d(temp);
mpf_set(temp0, temp);
					for (uint64_t i = 2; i < configuration.size[0] + 1; i++) {
						mpf_mul(temp0, temp0, matrix[3 * configuration.size[0] - i + 1]);
						mpf_sub(temp, inp[configuration.size[0] - i + j * configuration.size[0]], temp0);
						mpf_div(temp, temp, matrix[4 * configuration.size[0] - i]);
						mpf_set(temp0, temp);
						ress2[configuration.size[0] - i + j * configuration.size[0]] = (buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - ress2[configuration.size[0] - i + 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] - i + 1]) / buffer_input_matrix[4 * configuration.size[0] - i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						ress[configuration.size[0] - i + j * configuration.size[0]] = mpf_get_d(temp);// buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - ress[configuration.size[0] - i + 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] - i + 1]) / buffer_input_matrix[4 * configuration.size[0] - i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						//ress[configuration.size[0] - i + j * configuration.size[0]] = ress2[configuration.size[0] - i + j * configuration.size[0]];
					}
				}
				else {
					ress2[j * configuration.size[0]] = buffer_input_systems[j * configuration.size[0]] / buffer_input_matrix[2 * configuration.size[0]];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					//ress[j * configuration.size[0]] = ress2[j * configuration.size[0]];
					//printf("%f\n", ress2[configuration.size[0] - 1 + j * configuration.size[0]]);
//printf("%f\n", buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]]);
//printf("%f\n", buffer_input_matrix[4 * configuration.size[0] - 1]);
mpf_div(temp, inp[j * configuration.size[0]], matrix[2 * configuration.size[0]]);
ress[j * configuration.size[0]] = mpf_get_d(temp);
mpf_set(temp0, temp);
					for (uint64_t i = 1; i < configuration.size[0]; i++) {
						mpf_mul(temp0, temp0, matrix[3 * configuration.size[0] + i - 1]);
						mpf_sub(temp, inp[i + j * configuration.size[0]], temp0);
						mpf_div(temp, temp, matrix[2 * configuration.size[0] + i]);
						mpf_set(temp0, temp);
						ress2[i + j * configuration.size[0]] = (buffer_input_systems[i + j * configuration.size[0]] - ress2[i - 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] + i-1]) / buffer_input_matrix[2 * configuration.size[0] + i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						ress[i + j * configuration.size[0]] = mpf_get_d(temp);// buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - ress[configuration.size[0] - i + 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] - i + 1]) / buffer_input_matrix[4 * configuration.size[0] - i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						//ress[i + j * configuration.size[0]] = ress2[i + j * configuration.size[0]];
					}
				}
			}
			mpf_clear(temp);
			for (uint64_t j = 0; j < configuration.size[0]; j++) {
				mpf_clear(inp[j]);
				//mpf_clear(outp[j]);
			}
			for (uint64_t j = 0; j < 4 * configuration.size[0]; j++) {
				mpf_clear(matrix[j]);
			}
			free(inp);
			//free(outp);
			free(matrix);
#else
			pfLD prev0;
			pfLD prev = buffer_input_systems[0];
			buffer_input_systems[0] = buffer_input_systems[0] * buffer_input_matrix[configuration.size[0]];
			for (uint64_t j = 1; j < configuration.size[0]; j++) {
				prev0 = buffer_input_systems[j];
				buffer_input_systems[j] = buffer_input_systems[j] * buffer_input_matrix[configuration.size[0]+j] + prev * buffer_input_matrix[j] ;// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				prev = prev0;			
			}
			double* ress = (double*)malloc(bufferSolveResSize);
			pfLD* ress2 = (pfLD*)malloc(bufferSolveResSize);
			for (uint64_t j = 0; j < configuration.size[1]; j++) {
				if (configuration.upperBound != 1) {
					ress2[configuration.size[0] - 1 + j * configuration.size[0]] = buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]] / buffer_input_matrix[4 * configuration.size[0] - 1];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					ress[configuration.size[0] - 1 + j * configuration.size[0]] = (double)buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]] / (double)buffer_input_matrix[4 * configuration.size[0] - 1];
					
					for (uint64_t i = 2; i < configuration.size[0] + 1; i++) {
						ress2[configuration.size[0] - i + j * configuration.size[0]] = (buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - ress2[configuration.size[0] - i + 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] - i + 1]) / buffer_input_matrix[4 * configuration.size[0] - i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						ress[configuration.size[0] - i + j * configuration.size[0]] = ((double)buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - (double)ress[configuration.size[0] - i + 1 + j * configuration.size[0]] * (double)buffer_input_matrix[3 * configuration.size[0] - i + 1]) / (double)buffer_input_matrix[4 * configuration.size[0] - i];
					}
				}
				else {
					ress2[j * configuration.size[0]] = buffer_input_systems[j * configuration.size[0]] / buffer_input_matrix[2 * configuration.size[0]];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					ress[j * configuration.size[0]] = (double)buffer_input_systems[j * configuration.size[0]] / (double)buffer_input_matrix[2 * configuration.size[0]];
					
					for (uint64_t i = 1; i < configuration.size[0]; i++) {
						ress2[i + j * configuration.size[0]] = (buffer_input_systems[i + j * configuration.size[0]] - ress2[i - 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] + i-1]) / buffer_input_matrix[2 * configuration.size[0] + i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
						ress[i + j * configuration.size[0]] = ((double)buffer_input_systems[i + j * configuration.size[0]] - (double)ress[i - 1 + j * configuration.size[0]] * (double)buffer_input_matrix[3 * configuration.size[0] + i-1]) / (double)buffer_input_matrix[2 * configuration.size[0] + i];
					}
				}
			}
			
#endif
			// 
			//PfSolve_AppLibrary appLibrary = {};
			PfSolveApplication* tempApp = 0;

			/*if (configuration.JW_sequential) {
				PfSolve_MapKey_JonesWorland_sequential mapKey = {};
				mapKey.size[0] = configuration.size[0];
				mapKey.size[1] = configuration.size[1];
				mapKey.outputBufferStride = configuration.outputBufferStride[0];
				mapKey.offsetSolution = configuration.offsetSolution;
				resFFT = checkLibrary_JonesWorland_sequential(&appLibrary, mapKey, &tempApp);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside PfSolve library.  
				if (!tempApp) {
					resFFT = initializePfSolve(&app, configuration);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
					resFFT = addToLibrary_JonesWorland_sequential(&appLibrary, mapKey, &app);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				}
				else {
					app = tempApp[0];
				}
			}*/
			/*PfSolve_MapKey_JonesWorland mapKey = {};
			mapKey.size[0] = configuration.size[0];
			mapKey.size[1] = configuration.size[1];
			mapKey.outputBufferStride = configuration.outputBufferStride[0];
			mapKey.offsetSolution = configuration.offsetSolution;
			resFFT = checkLibrary_JonesWorland(&appLibrary, mapKey, &tempApp);*/
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside PfSolve library.  
			if (!tempApp) {
				resFFT = initializePfSolve(&app, configuration);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				//resFFT = addToLibrary_JonesWorland(&appLibrary, mapKey, &app);
				//if (resFFT != PFSOLVE_SUCCESS) return resFFT;
			}
			else {
				app = tempApp[0];
			}
		
			//Submit FFT+iFFT.
			uint64_t num_iter = 1;
			double totTime = 0;
			//cusparseHandle_t handle;
			//cusparseStatus_t resS= CUSPARSE_STATUS_SUCCESS;
			//resS= cusparseCreate(&handle);
			float* asas = 0;
			//res = cudaMalloc((void**)&asas, 128*100000);
			//resS = cusparseSgtsv2_nopivot(handle, 111, 10000, bufferSolve, bufferSolve, bufferSolve, bufferSolveRes, 111, asas);
			PfSolveLaunchParams launchParams = {};
#if(VKFFT_BACKEND==0)
			launchParams.buffer = &bufferSolve;
			launchParams.outputBuffer = &bufferSolveRes;
#elif(VKFFT_BACKEND==1)
			launchParams.buffer = (void**)&bufferSolve;
			launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==2)
			launchParams.buffer = (void**)&bufferSolve;
			launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==3)
			launchParams.buffer = &bufferSolve;
			launchParams.outputBuffer = &bufferSolveRes;
#endif
			//launchParams.offsetV = 2 * configuration.size[0];
			launchParams.offsetM = 0;
            launchParams.offsetV = 0;
			launchParams.offsetSolution = 0;
			launchParams.inputZeropad[0]= 0;
            launchParams.inputZeropad[1]= configuration.M_size;
            //launchParams.inputZeropad[1]--;
            launchParams.outputZeropad[0]= 0;
            launchParams.outputZeropad[1]= configuration.M_size;
			launchParams.outputBufferStride = configuration.M_size;
			launchParams.inputBufferStride = configuration.M_size;;
			launchParams.scaleC = 1.0;
			//cudaGraph_t graph;
			//cudaGraphExec_t instance;

			//cudaStreamBeginCapture(hStream, cudaStreamCaptureModeGlobal);

			resFFT = performVulkanFFT(vkGPU, &app, &launchParams, 0, num_iter, &totTime);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
			//cudaStreamEndCapture(hStream, &graph);
			//cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			//cudaDeviceSynchronize();
			double totTime2 = 0;
			std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();

			//cudaGraphLaunch(instance, hStream);
			//cudaDeviceSynchronize();

			std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
			totTime2 = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;

			double* output_PfSolve = (double*)(malloc(2 * sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2]));
			if (!output_PfSolve) return PFSOLVE_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
			resFFT = transferDataToCPU(vkGPU, output_PfSolve, &bufferSolveRes, bufferSolveResSize);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			res = cudaMemcpy(output_PfSolve, bufferSolveRes, bufferSolveResSize, cudaMemcpyDeviceToHost);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
			res = hipMemcpy(output_PfSolve, bufferSolveRes, bufferSolveResSize, hipMemcpyDeviceToHost);
			if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
			res = clEnqueueReadBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, output_PfSolve, 0, NULL, NULL);
			if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
#endif
			num_iter = 1000;
			//resFFT = performVulkanFFT(vkGPU, &app, &launchParams, 0, num_iter, &totTime2);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
			/*if (configuration.upperBound != 1) {
				for (uint64_t i = 1; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[1 * configuration.size[0] + i] = nu(i, -0.5, l - 0.5 + 1);// / mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				}
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[2 * configuration.size[0] + i] = mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					
				}
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[3 * configuration.size[0] + i] = 0;// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					
				}
			}
			else {
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[3 * configuration.size[0] + i] = nu(i, -0.5, l - 0.5 + 1);// / mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					
				}
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[2 * configuration.size[0] + i] = mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					
				}
				for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
					buffer_input_matrix[1 * configuration.size[0] + i] = 0;// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					
				}
			}
			res = cudaMemcpy(bufferSolve, buffer_input_matrix, bufferSolveSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			res = cudaMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			cusparseStatus_t stat;
			cusparseHandle_t tt;
			stat = cusparseCreate(&tt);
			size_t tsize;
			stat = cusparseZgtsv2StridedBatch_bufferSizeExt(tt, configuration.M_size, bufferSolve+configuration.M_size,bufferSolve+2*configuration.M_size, bufferSolve+3*configuration.M_size, bufferSolveRes,configuration.size[1],configuration.M_size, &tsize);
			cuDoubleComplex* tbuff = 0;
			res = cudaMalloc((void**)&tbuff, tsize);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
			stat = cusparseZgtsv2StridedBatch(tt, configuration.M_size, bufferSolve+configuration.M_size,bufferSolve+2*configuration.M_size, bufferSolve+configuration.M_size, bufferSolveRes,configuration.size[1],configuration.M_size, tbuff);
			double* output_cuFFT = (double*)(malloc(sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2]));
			if (!output_PfSolve) return PFSOLVE_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
			cudaDeviceSynchronize();
			res = cudaMemcpy(output_cuFFT, bufferSolveRes, bufferSolveResSize, cudaMemcpyDeviceToHost);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			*/
			pfLD resCPUSUM = 0;
			pfLD resGPUSUM = 0;
			pfLD resCPUMAX = 0;
			//double resSUM2_cu = 0;
			//double resSUM3_cu = 0;
			pfLD resGPUMAX = 0;
			for (uint64_t l = 0; l < configuration.size[2]; l++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {
					for (uint64_t i = 0; i < configuration.size[0]; i++) {
						uint64_t loc_i = i;
						uint64_t loc_j = j;
						uint64_t loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_PfSolve[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_PfSolve[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						pfLD resCPU = 0;
						pfLD resGPU = 0;
						pfLD resGPU2 = 0;
						pfLD resMUL2_cu = 0;
						pfLD resMUL3_cu = 0;
						pfLD resCPU2 = 0;
						/*
						if (i > 0)
							resMUL += buffer_input_matrix[i] * output_PfSolve[(i + j * configuration.size[0]) - 1];
						resMUL += buffer_input_matrix[i + configuration.size[0]] * output_PfSolve[(i + j * configuration.size[0])];
						if (i < configuration.size[0] - 1)
							resMUL += buffer_input_matrix[i + 2 * configuration.size[0]] * output_PfSolve[(i + j * configuration.size[0]) + 1];
						*/
						if (configuration.upperBound != 1) {
							resCPU += buffer_input_matrix[i + 3 * configuration.size[0]] * (pfLD)ress[(i + j * configuration.size[0])];
							if (i < configuration.size[0] - 1)
								resCPU += buffer_input_matrix[i + 2 * configuration.size[0]+1] * (pfLD)ress[(i + j * configuration.size[0]) + 1];
						}
						else {
							if (i > 0)
								resCPU += buffer_input_matrix[i + 3 * configuration.size[0]-1] * (pfLD)ress[(i + j * configuration.size[0]) - 1];
							resCPU += buffer_input_matrix[i + 2 * configuration.size[0]] * (pfLD)ress[(i + j * configuration.size[0])];
						}
						if (configuration.upperBound != 1) {
							resGPU += buffer_input_matrix[i + 3 * configuration.size[0]] * ((pfLD)output_PfSolve[2*(i + j * configuration.size[0])] + (pfLD)output_PfSolve[2*(i + j * configuration.size[0])+1]);
							if (i < configuration.size[0] - 1)
								resGPU += buffer_input_matrix[i + 2 * configuration.size[0] + 1] * ((pfLD)output_PfSolve[2*(i + j * configuration.size[0]+1)] + (pfLD)output_PfSolve[2*(i + j * configuration.size[0]+1)+1]);
						}
						else {
							if (i > 0)
								resGPU += buffer_input_matrix[i + 3 * configuration.size[0]-1] * ((pfLD)output_PfSolve[2*((i + j * configuration.size[0]) - 1)] + (pfLD)output_PfSolve[2*((i + j * configuration.size[0]) - 1)+1]);
							resGPU += buffer_input_matrix[i + 2 * configuration.size[0]] * ((pfLD) output_PfSolve[2*(i + j * configuration.size[0])] + (pfLD) output_PfSolve[2*(i + j * configuration.size[0])+1]);
						}

						/*if (configuration.upperBound != 1) {
							resMUL2_cu += buffer_input_matrix[i + 3 * configuration.size[0]] * output_cuFFT[(i + j * configuration.size[0])];
							if (i < configuration.size[0] - 1)
								resMUL2_cu += buffer_input_matrix[i + 2 * configuration.size[0] + 1] * output_cuFFT[(i + j * configuration.size[0]) + 1];
						}
						else {
							if (i > 0)
								resMUL2_cu += buffer_input_matrix[i + 3 * configuration.size[0]-1] * output_cuFFT[(i + j * configuration.size[0]) - 1];
							resMUL2_cu += buffer_input_matrix[i + 2 * configuration.size[0]] * output_cuFFT[(i + j * configuration.size[0])];
						}*/
						resGPU2 = (ress2[(i + j * configuration.size[0])]- ((pfLD)output_PfSolve[2*(i + j * configuration.size[0])] + (pfLD)output_PfSolve[2*(i + j * configuration.size[0])+1]))* (ress2[(i + j * configuration.size[0])] - ((pfLD)output_PfSolve[2*(i + j * configuration.size[0])] + (pfLD)output_PfSolve[2*(i + j * configuration.size[0])+1]));
						//resMUL3_cu = (ress[(i + j * configuration.size[0])]- output_cuFFT[(i + j * configuration.size[0])])* (ress[(i + j * configuration.size[0])] - output_cuFFT[(i + j * configuration.size[0])]);
						resCPU2 = ((pfLD)ress[(i + j * configuration.size[0])] - ress2[(i + j * configuration.size[0])]) * ((pfLD)ress[(i + j * configuration.size[0])] - ress2[(i + j * configuration.size[0])]);
						//printf("%.17f %.17f %.17f\n", resMUL, resMUL2, buffer_input_systems[i + j * configuration.size[0]]);
						printf("(%.17e %.17e) %.17e %.17e - %.17e\n", output_PfSolve[2*(i + j * configuration.size[0])], output_PfSolve[2*(i + j * configuration.size[0])+1], (double)ress[(i + j * configuration.size[0])], (double)ress2[(i + j * configuration.size[0])], (double)pfsqrt(resGPU2));
						/*char buf[128];
						__float128 r;
						int n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", 46, ((pfLD)output_PfSolve[2*(i + j * configuration.size[0])] + (pfLD)output_PfSolve[2*(i + j * configuration.size[0])+1]));
						printf("%s ", buf);
						n = quadmath_snprintf (buf, sizeof buf, "%+-#*.20Qe", 46, ress2[(i + j * configuration.size[0])]);
						printf("%s\n", buf);*/
						resCPUSUM += pfsqrt((resCPU - buffer_input_systems[i + j * configuration.size[0]]) * (resCPU - buffer_input_systems[i + j * configuration.size[0]]));
						resGPUSUM += pfsqrt((resGPU - buffer_input_systems[i + j * configuration.size[0]]) * (resGPU - buffer_input_systems[i + j * configuration.size[0]]));
						resCPUMAX = (pfsqrt(resCPU2)/ ress2[(i + j * configuration.size[0])] > resCPUMAX) ? pfsqrt(resCPU2)/ ress2[(i + j * configuration.size[0])] : resCPUMAX;

						//resSUM2_cu += sqrt((resMUL2 - buffer_input_systems[i + j * configuration.size[0]]) * (resMUL2 - buffer_input_systems[i + j * configuration.size[0]]));
						//resSUM3_cu = (sqrt(resMUL3) > resSUM3) ? sqrt(resMUL3) : resSUM3;
						resGPUMAX = (pfsqrt(resGPU2)/ ress2[(i + j * configuration.size[0])] > resGPUMAX) ? pfsqrt(resGPU2)/ ress2[(i + j * configuration.size[0])] : resGPUMAX;
						//printf("%f \n", output_PfSolve[(i + j * configuration.size[0])]);

					}
					//printf("\n");

				}
			}
			//resSUM /= configuration.size[0] * configuration.size[1];
			//printf("res bs = %.17f\n", resSUM);
			//printf("res pcr vk = %.17f\n", resSUM2);
			//printf("res pcr cu = %.17f\n", resSUM2_cu);
			//printf("max res gpu - 128b = %.17f\n", resSUM3);
			//printf("max res cpu bs - 128b = %.17f\n", resSUM4);
			printf("%d %.3e %.3e - %.3e %.3e\n", configuration.size[0], (double)(resGPUSUM/ (configuration.size[0] * configuration.size[1])), (double)(resCPUSUM/ (configuration.size[0] * configuration.size[1])), (double)resGPUMAX, (double)resCPUMAX);
			//printf("time 1 = %.6f time %d = %.6f\n",totTime, num_iter, totTime2/num_iter);
			//printf("size  = %d MB, time at peak bw = %f ms\n", 2*bufferSolveResSize/1024/1024, 2*bufferSolveResSize/1024.0/1024.0/1024.0/1200.0*1000.0);
			free(buffer_input_systems);
			free(buffer_input_systems2);
			free(buffer_input_matrix);
			free(buffer_input_matrix_gpu);
			free(ress);
			free(ress2);
#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, bufferSolve, NULL);
			vkFreeMemory(vkGPU->device, bufferSolveDeviceMemory, NULL);
			vkDestroyBuffer(vkGPU->device, bufferSolveRes, NULL);
			vkFreeMemory(vkGPU->device, bufferSolveResDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			cudaFree(bufferSolve);
			cudaFree(bufferSolveRes);
#elif(VKFFT_BACKEND==2)
			hipFree(bufferSolve);
			hipFree(bufferSolveRes);
#elif(VKFFT_BACKEND==3)
			clReleaseMemObject(bufferSolve);
			clReleaseMemObject(bufferSolveRes);
#endif
			if (!tempApp) {
				deletePfSolve(&app);
			}

		}
	}
	return resFFT;
}
