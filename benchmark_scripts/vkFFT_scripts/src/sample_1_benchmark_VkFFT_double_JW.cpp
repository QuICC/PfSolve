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

#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
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
#include "vkSolve.hpp"
#include "utils_VkFFT.h"
#include <mpir.h>
//#include <mpir.h>
double mu(double n, double alpha, double beta) {
	return sqrt(2 * (n + beta) * (n + alpha + beta) / (2 * n + alpha + beta) / (2 * n + alpha + beta + 1));
}
double nu(double n, double alpha, double beta) {
	return sqrt(2 * (n + 1) * (n + alpha + 1) / (2 * n + alpha + beta+1) / (2 * n + alpha + beta + 2));
}
VkSolveResult sample_1_benchmark_VkFFT_double(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
	VkSolveResult resFFT = VKSOLVE_SUCCESS;
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
		fprintf(output, "0 - VkSolve PCR test\n");
	printf("0 - VkSolve PCR test\n");
	const int num_runs = 3;
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	for (uint64_t n = 0; n < 1; n++) {
		double run_time[num_runs];
		for (uint64_t r = 0; r < 1; r++) {
			//Configuration + FFT application .
			VkSolveConfiguration configuration = {};
			VkSolveApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.M_size = 111; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[0] = configuration.M_size; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
			configuration.size[1] =1;
			configuration.size[2] = 1;
			configuration.doublePrecision = 1;
			configuration.isOutputFormatted = 1;
			int* x;
			int** y;
			y = &x;
			configuration.aimThreads = 256;
			//configuration.JW_sequential = 1;
			configuration.JW_parallel = 1;
			configuration.outputBufferStride[0] = configuration.size[0];
			//configuration.performWorland = 1;
			//configuration.upperBanded = 1;
			configuration.keepShaderCode = 1;
			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
			configuration.device = &vkGPU->device;
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkSolve plan creation. if not, VkSolve will create and destroy one after initialization
#elif(VKFFT_BACKEND==3)
			configuration.platform = &vkGPU->platform;
			configuration.context = &vkGPU->context;
#endif
			//Allocate buffer for the input data.
			uint64_t bufferSolveSize;
			uint64_t bufferSolveResSize;

			bufferSolveSize = (uint64_t)sizeof(double) * 4 * configuration.size[0];
			bufferSolveResSize = (uint64_t)sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2];

#if(VKFFT_BACKEND==0)
			VkBuffer bufferSolve = {};
			VkDeviceMemory bufferSolveDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &bufferSolve, &bufferSolveDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSolveSize);
			if (resFFT != VKSOLVE_SUCCESS) return resFFT;
			VkBuffer bufferSolveRes = {};
			VkDeviceMemory bufferSolveResDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &bufferSolveRes, &bufferSolveResDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSolveResSize);
			if (resFFT != VKSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			cuDoubleComplex* bufferSolve = 0;
			res = cudaMalloc((void**)&bufferSolve, bufferSolveSize);
			if (res != cudaSuccess) return VKSOLVE_ERROR_FAILED_TO_ALLOCATE;
			cuDoubleComplex* bufferSolveRes = 0;
			res = cudaMalloc((void**)&bufferSolveRes, bufferSolveResSize);
			if (res != cudaSuccess) return VKSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
			hipDoubleComplex* bufferSolve = 0;
			res = hipMalloc((void**)&bufferSolve, bufferSolveSize);
			if (res != hipSuccess) return VKSOLVE_ERROR_FAILED_TO_ALLOCATE;
			hipDoubleComplex* bufferSolveRes = 0;
			res = hipMalloc((void**)&bufferSolveRes, bufferSolveResSize);
			if (res != hipSuccess) return VKSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==3)
			cl_mem bufferSolve = 0;
			bufferSolve = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveSize, 0, &res);
			if (res != CL_SUCCESS) return VKSOLVE_ERROR_FAILED_TO_ALLOCATE;
			cl_mem bufferSolveRes = 0;
			bufferSolveRes = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveResSize, 0, &res);
			if (res != CL_SUCCESS) return VKSOLVE_ERROR_FAILED_TO_ALLOCATE;
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
			int l = 2* configuration.size[0];
			double* buffer_input_matrix = (double*)malloc(bufferSolveSize);
			double* buffer_input_matrix_gpu = (double*)malloc(bufferSolveSize);
			for (uint64_t i = 0; i < configuration.size[0]; i++) {
				buffer_input_matrix[i] = 1;
				buffer_input_matrix_gpu[i] = 1.0 / mu(i, -0.5, l - 0.5 + 1);;
				//printf("%f\n", buffer_input_matrix_gpu[i]);
			}
			for (uint64_t i = 0; i < configuration.size[0]; i++) {
				buffer_input_matrix[configuration.size[0] + i] = 0;
				buffer_input_matrix_gpu[configuration.size[0] + i] = 0;
			}
			for (uint64_t i = 1; i < 1 * configuration.size[0]; i++) {
				buffer_input_matrix[2 * configuration.size[0] + i] = nu(i, -0.5, l - 0.5 + 1);// / mu(i, -0.5, l - 0.5 + 1);// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				buffer_input_matrix_gpu[2 * configuration.size[0] + i] = nu(i, -0.5, l - 0.5 + 1) / mu(i - 1, -0.5, l - 0.5 + 1);// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				//printf("%f %f\n", buffer_input_matrix[2 * configuration.size[0] + i], buffer_input_matrix_gpu[2 * configuration.size[0] + i]);
			}
			for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
				buffer_input_matrix[3 * configuration.size[0] + i] = mu(i, -0.5, l - 0.5 + 1);// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				buffer_input_matrix_gpu[3 * configuration.size[0] + i] = 1;// mu(i, -0.5, l - 0.5 + 1);// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				//printf("%f\n", buffer_input_matrix[3 * configuration.size[0] + i]);
			}
			for (uint64_t i = 100; i < 32; i++) {
				//buffer_input_matrix[2 * configuration.size[0] + i] = 0;// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				//printf("%f\n", buffer_input_matrix[2 * configuration.size[0] + i]);
			}
			//buffer_input_matrix[0] = 0;
			double* buffer_input_systems = (double*)malloc(bufferSolveResSize);

			for (uint64_t j = 0; j < configuration.size[1]; j++) {
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					buffer_input_systems[i + j * configuration.size[0]] = i;// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				}
			}
			//buffer_input_systems[0] = 0.69;
			//buffer_input_systems[1] = 0.23;
			//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
			resFFT = transferDataFromCPU(vkGPU, buffer_input_matrix_gpu, &bufferSolve, bufferSolveSize);
			if (resFFT != VKSOLVE_SUCCESS) return resFFT;
			resFFT = transferDataFromCPU(vkGPU, buffer_input_systems, &bufferSolveRes, bufferSolveResSize);
			if (resFFT != VKSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			res = cudaMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return VKSOLVE_ERROR_FAILED_TO_COPY;
			res = cudaMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return VKSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
			res = hipMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, hipMemcpyHostToDevice);
			if (res != hipSuccess) return VKSOLVE_ERROR_FAILED_TO_COPY;
			res = hipMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, hipMemcpyHostToDevice);
			if (res != hipSuccess) return VKSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
			res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolve, CL_TRUE, 0, bufferSolveSize, buffer_input_matrix_gpu, 0, NULL, NULL);
			if (res != CL_SUCCESS) return VKSOLVE_ERROR_FAILED_TO_COPY;
			res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, buffer_input_systems, 0, NULL, NULL);
			if (res != CL_SUCCESS) return VKSOLVE_ERROR_FAILED_TO_COPY;
#endif
			for (uint64_t i = 0; i < 1 * configuration.size[0]; i++) {
				//buffer_input_matrix[2 * configuration.size[0] + i] = 1/ buffer_input_matrix[2 * configuration.size[0] + i];// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				//printf("%f\n", buffer_input_matrix[2 * configuration.size[0] + i]);
			}
			/*mpf_set_default_prec(128);
			mpf_t temp;
			mpf_init(temp);
			mpf_set_d(temp, 0);
			mpf_t *inp = (mpf_t * )malloc(sizeof(mpf_t) * configuration.size[0]);
			mpf_t* outp = (mpf_t*)malloc(sizeof(mpf_t)*configuration.size[0]);
			mpf_t* matrix = (mpf_t*)malloc(4* sizeof(mpf_t) * configuration.size[0]);
			for (uint64_t j = 0; j < configuration.size[0]; j++) {
				mpf_init(inp[j]);
				mpf_set_d(inp[j], buffer_input_systems[j]);
				mpf_init(outp[j]);
				mpf_set_d(outp[j], 0);
			}
			for (uint64_t j = 0; j < 4 * configuration.size[0]; j++) {
				mpf_init(matrix[j]);
				mpf_set_d(matrix[j], buffer_input_matrix[j]);
			}*/
			double* ress = (double*)malloc(bufferSolveResSize);
			double* ress2 = (double*)malloc(bufferSolveResSize);
			for (uint64_t j = 0; j < 1; j++) {
				ress2[configuration.size[0] - 1 + j * configuration.size[0]] = buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]] / buffer_input_matrix[4 * configuration.size[0] - 1];// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
				ress[configuration.size[0] - 1 + j * configuration.size[0]] = ress2[configuration.size[0] - 1 + j * configuration.size[0]];
																																																				//printf("%f\n", ress2[configuration.size[0] - 1 + j * configuration.size[0]]);
				//printf("%f\n", buffer_input_systems[configuration.size[0] - 1 + j * configuration.size[0]]);
				//printf("%f\n", buffer_input_matrix[4 * configuration.size[0] - 1]);
				//mpf_div(outp[configuration.size[0] - 1 + j * configuration.size[0]], inp[configuration.size[0] - 1 + j * configuration.size[0]], matrix[4 * configuration.size[0] - 1]);
				//ress[configuration.size[0] - 1 + j * configuration.size[0]] = mpf_get_d(outp[configuration.size[0] - 1 + j * configuration.size[0]]);
				for (uint64_t i = 2; i < configuration.size[0] + 1; i++) {
					//mpf_mul(temp, outp[configuration.size[0] - i + 1 + j * configuration.size[0]], matrix[3 * configuration.size[0] - i + 1]);
					//mpf_sub(outp[configuration.size[0] - i + j * configuration.size[0]], inp[configuration.size[0] - i + j * configuration.size[0]], temp);
					//mpf_div(outp[configuration.size[0] - i + j * configuration.size[0]], outp[configuration.size[0] - i + j * configuration.size[0]], matrix[4 * configuration.size[0] - i]);
					ress2[configuration.size[0] - i + j * configuration.size[0]] = (buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - ress2[configuration.size[0] - i + 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] - i+1]) / buffer_input_matrix[4 * configuration.size[0] - i];// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
					//ress[configuration.size[0] - i + j * configuration.size[0]] = mpf_get_d(outp[configuration.size[0] - i + j * configuration.size[0]]);// buffer_input_systems[configuration.size[0] - i + j * configuration.size[0]] - ress[configuration.size[0] - i + 1 + j * configuration.size[0]] * buffer_input_matrix[3 * configuration.size[0] - i + 1]) / buffer_input_matrix[4 * configuration.size[0] - i];// (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
					ress[configuration.size[0] - i + j * configuration.size[0]] = ress2[configuration.size[0] - i + j * configuration.size[0]];
				}
			}
			//mpf_clear(temp);
			for (uint64_t j = 0; j < configuration.size[0]; j++) {
				//mpf_clear(inp[j]);
				//mpf_clear(outp[j]);
			}
			for (uint64_t j = 0; j < 4 * configuration.size[0]; j++) {
				//mpf_clear(matrix[j]);
			}
			//free(inp);
			//free(outp);
			//free(matrix);
			VkSolve_AppLibrary appLibrary = {};
			VkSolveApplication* tempApp = 0;
			if (configuration.JW_sequential) {
				VkSolve_MapKey_JonesWorland_sequential mapKey = {};
				mapKey.size[0] = configuration.size[0];
				mapKey.size[1] = configuration.size[1];
				mapKey.outputBufferStride = configuration.outputBufferStride[0];
				mapKey.offsetSolution = configuration.offsetSolution;
				resFFT = checkLibrary_JonesWorland_sequential(&appLibrary, mapKey, &tempApp);
				if (resFFT != VKSOLVE_SUCCESS) return resFFT;
				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkSolve library.  
				if (!tempApp) {
					resFFT = initializeVkSolve(&app, configuration);
					if (resFFT != VKSOLVE_SUCCESS) return resFFT;
					resFFT = addToLibrary_JonesWorland_sequential(&appLibrary, mapKey, &app);
					if (resFFT != VKSOLVE_SUCCESS) return resFFT;
				}
				else {
					app = tempApp[0];
				}
			}
			if (configuration.JW_parallel) {
				VkSolve_MapKey_JonesWorland mapKey = {};
				mapKey.size[0] = configuration.size[0];
				mapKey.size[1] = configuration.size[1];
				mapKey.outputBufferStride = configuration.outputBufferStride[0];
				mapKey.offsetSolution = configuration.offsetSolution;
				resFFT = checkLibrary_JonesWorland(&appLibrary, mapKey, &tempApp);
				//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkSolve library.  
				if (!tempApp) {
					resFFT = initializeVkSolve(&app, configuration);
					if (resFFT != VKSOLVE_SUCCESS) return resFFT;
					resFFT = addToLibrary_JonesWorland(&appLibrary, mapKey, &app);
					if (resFFT != VKSOLVE_SUCCESS) return resFFT;
				}
				else {
					app = tempApp[0];
				}
			}

			//Submit FFT+iFFT.
			uint64_t num_iter = 1;
			double totTime = 0;

			VkSolveLaunchParams launchParams = {};
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
			resFFT = performVulkanFFT(vkGPU, &app, &launchParams, &totTime, num_iter);
			if (resFFT != VKSOLVE_SUCCESS) return resFFT;

			double* output_VkSolve = (double*)(malloc(sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2]));
			if (!output_VkSolve) return VKSOLVE_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
			resFFT = transferDataToCPU(vkGPU, output_VkSolve, &bufferSolveRes, bufferSolveResSize);
			if (resFFT != VKSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			res = cudaMemcpy(output_VkSolve, bufferSolveRes, bufferSolveResSize, cudaMemcpyDeviceToHost);
			if (res != cudaSuccess) return VKSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
			res = hipMemcpy(output_VkSolve, bufferSolveRes, bufferSolveResSize, hipMemcpyDeviceToHost);
			if (res != hipSuccess) return VKSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
			res = clEnqueueReadBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, output_VkSolve, 0, NULL, NULL);
			if (res != CL_SUCCESS) return VKSOLVE_ERROR_FAILED_TO_COPY;
#endif
			double resSUM = 0;
			double resSUM2 = 0;
			double resSUM3 = 0;
			double resSUM4 = 0;
			for (uint64_t l = 0; l < configuration.size[2]; l++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {
					for (uint64_t i = 0; i < configuration.size[0]; i++) {
						uint64_t loc_i = i;
						uint64_t loc_j = j;
						uint64_t loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkSolve[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkSolve[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						double resMUL = 0;
						double resMUL2 = 0; 
						double resMUL3 = 0;
						double resMUL4 = 0;
						/*
						if (i > 0)
							resMUL += buffer_input_matrix[i] * output_VkSolve[(i + j * configuration.size[0]) - 1];
						resMUL += buffer_input_matrix[i + configuration.size[0]] * output_VkSolve[(i + j * configuration.size[0])];
						if (i < configuration.size[0] - 1)
							resMUL += buffer_input_matrix[i + 2 * configuration.size[0]] * output_VkSolve[(i + j * configuration.size[0]) + 1];
						*/
						if (app.configuration.upperBanded != 1) {
							resMUL += buffer_input_matrix[i + 3 * configuration.size[0]] * ress2[(i + j * configuration.size[0])];
							if (i < configuration.size[0] - 1)
								resMUL += buffer_input_matrix[i + 2 * configuration.size[0]+1] * ress2[(i + j * configuration.size[0]) + 1];
						}
						else {
							if (i > 0)
								resMUL += buffer_input_matrix[i + 3 * configuration.size[0]] * ress2[(i + j * configuration.size[0]) - 1];
							resMUL += buffer_input_matrix[i + 2 * configuration.size[0]] * ress2[(i + j * configuration.size[0])];
						}
						if (app.configuration.upperBanded != 1) {
							resMUL2 += buffer_input_matrix[i + 3 * configuration.size[0]] * output_VkSolve[(i + j * configuration.size[0])];
							if (i < configuration.size[0] - 1)
								resMUL2 += buffer_input_matrix[i + 2 * configuration.size[0] + 1] * output_VkSolve[(i + j * configuration.size[0]) + 1];
						}
						else {
							if (i > 0)
								resMUL2 += buffer_input_matrix[i + 3 * configuration.size[0]] * output_VkSolve[(i + j * configuration.size[0]) - 1];
							resMUL2 += buffer_input_matrix[i + 2 * configuration.size[0]] * output_VkSolve[(i + j * configuration.size[0])];
						}
						resMUL3 = (ress[(i + j * configuration.size[0])]- output_VkSolve[(i + j * configuration.size[0])])* (ress[(i + j * configuration.size[0])] - output_VkSolve[(i + j * configuration.size[0])]);
						resMUL4 = (ress[(i + j * configuration.size[0])] - ress2[(i + j * configuration.size[0])]) * (ress[(i + j * configuration.size[0])] - ress2[(i + j * configuration.size[0])]);
						//printf("%.17f %.17f %.17f\n", resMUL, resMUL2, buffer_input_systems[i + j * configuration.size[0]]);
						printf("%.17f %.17f %.17f\n", output_VkSolve[(i + j * configuration.size[0])], ress2[(i + j * configuration.size[0])], ress[(i + j * configuration.size[0])]);
						resSUM += sqrt((resMUL - buffer_input_systems[i + j * configuration.size[0]]) * (resMUL - buffer_input_systems[i + j * configuration.size[0]]));
						resSUM2 += sqrt((resMUL2 - buffer_input_systems[i + j * configuration.size[0]]) * (resMUL2 - buffer_input_systems[i + j * configuration.size[0]]));
						resSUM3 = (sqrt(resMUL3) > resSUM3) ? sqrt(resMUL3) : resSUM3;
						resSUM4 = (sqrt(resMUL4) > resSUM4) ? sqrt(resMUL4) : resSUM4;
						//printf("%f \n", output_VkSolve[(i + j * configuration.size[0])]);

					}
					//printf("\n");

				}
			}
			resSUM /= configuration.size[0] * configuration.size[1];
			//printf("res bs = %.17f\n", resSUM);
			//printf("res pcr = %.17f\n", resSUM2);
			//printf("max res gpu - 128b = %.17f\n", resSUM3);
			//printf("max res cpu bs - 128b = %.17f\n", resSUM4);
			printf("%d %.3e %.3e\n", configuration.size[0], resSUM3, resSUM4);
			//printf("time = %.6f\n", totTime/num_iter);
			//printf("size  = %d MB, time at peak bw = %f ms\n", 2*bufferSolveResSize/1024/1024, 2*bufferSolveResSize/1024.0/1024.0/1024.0/950.0*1000.0);
			free(buffer_input_systems);
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
				deleteVkSolve(&app);
			}

		}
	}
	return resFFT;
}
