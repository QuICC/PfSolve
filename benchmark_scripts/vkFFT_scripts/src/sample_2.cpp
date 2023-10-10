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
#include "pfSolve.h"
#include "utils_VkFFT.h"
//#include <mpir.h>

PfSolveResult sample_2(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized, uint64_t type, uint64_t size, uint64_t* logicBlock)
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
		fprintf(output, "generating CUDA kernels\n");
	printf("generating CUDA kernels\n");
	const int num_runs = 3;
	float benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * (uint64_t)pow(2, 27));
	if (!buffer_input) return PFSOLVE_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, 27); i++) {
		buffer_input[i] = (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
	}
	int warpSize = 32;
	if (size > 1) {
		for (uint64_t n = 0; n < 1; n++) {
			float run_time[num_runs];
			for (uint64_t r = 0; r < num_runs; r++) {
				//Configuration + FFT application .
				PfSolveConfiguration configuration = {};
				PfSolveApplication app = {};
				//FFT + iFFT sample code.
				//Setting up FFT configuration for forward and inverse FFT.
				configuration.FFTdim = 3; //FFT dimension, 1D, 2D or 3D (default 1).
				//configuration.M_size = 63; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
				configuration.size[2] = size; //configuration.M_size
				configuration.size[0] = size;// configuration.M_size; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
				configuration.size[1] = size; // configuration.M_size;

				configuration.logicBlock[0] = logicBlock[0];
				configuration.logicBlock[1] = logicBlock[1];
				configuration.logicBlock[2] = logicBlock[2];

				configuration.doublePrecision = 1;
				configuration.keepShaderCode = 1;
				//configuration.k_nf = 1;
				//configuration.arg = 1;
				configuration.s_dx = configuration.size[0] / 40.0;
				configuration.s_dy = configuration.size[1] / 40.0;
				configuration.s_dz = configuration.size[2] / 40.0;
				configuration.s_dt_D = 1.0 / (1.0 + 40.0 / (4 * 3.1415926535897932384626433832795 * 1.0 / sqrt(3.1)) * configuration.s_dx); // 5.32083411470606546e-02;//
				//configuration.s_dt_D = 1.0 / ((4 * 3.1415926535897932384626433832795 * 1.0) / (1.0/sqrt(3.1)*40.0)* configuration.s_dx); // 1.1478594718471122;//
				int* x;
				int** y;
				y = &x;
				configuration.aimThreads = 32;
				//configuration.JW_sequential = 1;
				configuration.finiteDifferences = 1;
				//configuration.compute_flux_D = (type == 0) ? 1 : 0;
				configuration.compute_Pf = (type == 1) ? 1 : 0;
				//configuration.outputBufferStride[0] = configuration.size[0];
				if (r == 0) configuration.saveApplicationToString = 1;
				if (r != 0) configuration.loadApplicationFromString = 1;
				//configuration.keepShaderCode = 1;
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

				bufferSolveSize = (uint64_t)sizeof(double) * (configuration.size[0]) * (configuration.size[1]) * (configuration.size[2]);

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
				double* Pf = 0;
				res = cudaMalloc((void**)&Pf, bufferSolveSize);
				if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* qDx = 0;
				res = cudaMalloc((void**)&qDx, bufferSolveSize);
				if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* qDy = 0;
				res = cudaMalloc((void**)&qDy, bufferSolveSize);
				if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* qDz = 0;
				res = cudaMalloc((void**)&qDz, bufferSolveSize);
				if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* T = 0;
				//res = cudaMalloc((void**)&T, bufferSolveSize);
				//if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* buffer[20];
				buffer[0] = qDx;
				buffer[1] = qDy;
				buffer[2] = qDz;
				buffer[3] = Pf;
				//buffer[4] = T;

				/*if (app.configuration.compute_flux_D) {
					buffer[5] = &configuration.k_nf;
					buffer[6] = &configuration.s_dt_D;
					buffer[7] = &configuration.arg;
					buffer[8] = &configuration.s_dx;
					buffer[9] = &configuration.s_dy;
					buffer[10] = &configuration.s_dz;
				}*/
				if (app.configuration.compute_Pf) {
					//buffer[5] = &configuration.s_dt_D;
					//buffer[6] = &configuration.s_dx;
					//buffer[7] = &configuration.s_dy;
					//buffer[8] = &configuration.s_dz;
				}
#elif(VKFFT_BACKEND==2)
				double* Pf = 0;
				res = hipMalloc((void**)&Pf, bufferSolveSize);
				if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* qDx = 0;
				res = hipMalloc((void**)&qDx, bufferSolveSize);
				if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* qDy = 0;
				res = hipMalloc((void**)&qDy, bufferSolveSize);
				if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* qDz = 0;
				res = hipMalloc((void**)&qDz, bufferSolveSize);
				if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* T = 0;
				//res = hipMalloc((void**)&T, bufferSolveSize);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				double* buffer[20];
				buffer[0] = qDx;
				buffer[1] = qDy;
				buffer[2] = qDz;
				buffer[3] = Pf;
#elif(VKFFT_BACKEND==3)
				cl_mem bufferSolve = 0;
				bufferSolve = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveSize, 0, &res);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				cl_mem bufferSolveRes = 0;
				bufferSolveRes = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveResSize, 0, &res);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#endif
				resFFT = transferDataFromCPU(vkGPU, buffer_input, &Pf, bufferSolveSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				resFFT = transferDataFromCPU(vkGPU, buffer_input, &qDx, bufferSolveSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				resFFT = transferDataFromCPU(vkGPU, buffer_input, &qDy, bufferSolveSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				resFFT = transferDataFromCPU(vkGPU, buffer_input, &qDz, bufferSolveSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
				configuration.buffer = &bufferSolve;
				configuration.outputBuffer = &bufferSolveRes;
#elif(VKFFT_BACKEND==1)
				configuration.buffer = (void**)buffer;
#elif(VKFFT_BACKEND==2)
				configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
				configuration.buffer = &bufferSolve;
				configuration.outputBuffer = &bufferSolveRes;
#endif
				configuration.bufferSize = &bufferSolveSize;

				//Fill data on CPU. It is best to perform all operations on GPU after initial upload

				//buffer_input_systems[0] = 0.69;
				//buffer_input_systems[1] = 0.23;
				//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
				resFFT = transferDataFromCPU(vkGPU, buffer_input_matrix_gpu, &bufferSolve, bufferSolveSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
				resFFT = transferDataFromCPU(vkGPU, buffer_input_systems, &bufferSolveRes, bufferSolveResSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			//res = cudaMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, cudaMemcpyHostToDevice);
			//if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
				//res = hipMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, hipMemcpyHostToDevice);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
				//res = hipMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, hipMemcpyHostToDevice);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
				res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolve, CL_TRUE, 0, bufferSolveSize, buffer_input_matrix_gpu, 0, NULL, NULL);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
				res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, buffer_input_systems, 0, NULL, NULL);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
#endif
				//float* ress = (float*)malloc(bufferSolveResSize);
				//float* ress2 = (float*)malloc(bufferSolveResSize);

				//free(inp);
				//free(outp);
				//free(matrix);
				if (configuration.loadApplicationFromString) {
					FILE* kernelCache;
					uint64_t str_len;
					char fname[500];
					sprintf(fname, "compute_Pf_size_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "_logicblock_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "", configuration.size[0], configuration.size[1], configuration.size[2], configuration.logicBlock[0], configuration.logicBlock[1], configuration.logicBlock[2]);
					kernelCache = fopen(fname, "rb");
					if (!kernelCache) return PFSOLVE_ERROR_EMPTY_FILE;
					fseek(kernelCache, 0, SEEK_END);
					str_len = ftell(kernelCache);
					fseek(kernelCache, 0, SEEK_SET);
					configuration.loadApplicationString = malloc(str_len);
					fread(configuration.loadApplicationString, str_len, 1, kernelCache);
					fclose(kernelCache);
				}
				resFFT = initializePfSolve(&app, configuration);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;

				if (configuration.loadApplicationFromString)
					free(configuration.loadApplicationString);

				if (configuration.saveApplicationToString != 0) {
					FILE* kernelCache;
					char fname[500];
					sprintf(fname, "compute_Pf_size_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "_logicblock_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "", configuration.size[0], configuration.size[1], configuration.size[2], configuration.logicBlock[0], configuration.logicBlock[1], configuration.logicBlock[2]);
					kernelCache = fopen(fname, "wb");
					fwrite(app.saveApplicationString, app.applicationStringSize, 1, kernelCache);
					fclose(kernelCache);
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
				//launchParams.buffer = (void**)&bufferSolve;
				//launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==2)
				//launchParams.buffer = (void**)&bufferSolve;
				//launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==3)
				launchParams.buffer = &bufferSolve;
				launchParams.outputBuffer = &bufferSolveRes;
#endif

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

				//float* output_PfSolve = (float*)(malloc(sizeof(float) * configuration.size[0] * configuration.size[1] * configuration.size[2]));
				//if (!output_PfSolve) return PFSOLVE_ERROR_MALLOC_FAILED;
				//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
				resFFT = transferDataToCPU(vkGPU, output_PfSolve, &bufferSolveRes, bufferSolveResSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			//res = cudaMemcpy(output_PfSolve, bufferSolveRes, bufferSolveResSize, cudaMemcpyDeviceToHost);
			//if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
				//res = hipMemcpy(output_PfSolve, bufferSolveRes, bufferSolveResSize, hipMemcpyDeviceToHost);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
				res = clEnqueueReadBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, output_PfSolve, 0, NULL, NULL);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
#endif
				float resSUM = 0;
				float resSUM2 = 0;
				float resSUM3 = 0;
				float resSUM4 = 0;
				//printf("res bs = %.17f\n", resSUM);
				//printf("res pcr = %.17f\n", resSUM2);
				//printf("max res gpu - 128b = %.17f\n", resSUM3);
				//printf("max res cpu bs - 128b = %.17f\n", resSUM4);
				//printf("%d %.3e %.3e\n", configuration.size[0], resSUM3, resSUM4);
				//printf("time = %.6f ms\n", totTime/num_iter);
				if (r == 2)
					printf("size = %d, lb %d %d %d, time = %.6f ms, buffer  = %d MB, acheived bw = %f GB/s\n", configuration.size[0], configuration.logicBlock[0], configuration.logicBlock[1], configuration.logicBlock[2], totTime / num_iter, bufferSolveSize / 1024 / 1024, 4 * bufferSolveSize / 1024.0 / 1024.0 / 1024.0 * 1000.0 / (totTime / num_iter));

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, bufferSolve, NULL);
				vkFreeMemory(vkGPU->device, bufferSolveDeviceMemory, NULL);
				vkDestroyBuffer(vkGPU->device, bufferSolveRes, NULL);
				vkFreeMemory(vkGPU->device, bufferSolveResDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(Pf);
				cudaFree(qDx);
				cudaFree(qDy);
				cudaFree(qDz);
#elif(VKFFT_BACKEND==2)
				hipFree(Pf);
				hipFree(qDx);
				hipFree(qDy);
				hipFree(qDz);
#elif(VKFFT_BACKEND==3)
				clReleaseMemObject(bufferSolve);
				clReleaseMemObject(bufferSolveRes);
#endif
				deletePfSolve(&app);
			}
		}
	}
	else {
		for (uint64_t k = 0; k < 4; k++) {
			int maxx = 1 + (int)log2(64 / warpSize * pow(2, k));
			for (uint64_t n = 0; n < maxx * 4 * 4; n++) {
				float run_time[num_runs];
				for (uint64_t r = 0; r < num_runs; r++) {
					//Configuration + FFT application .
					PfSolveConfiguration configuration = {};
					PfSolveApplication app = {};
					//FFT + iFFT sample code.
					//Setting up FFT configuration for forward and inverse FFT.
					configuration.FFTdim = 3; //FFT dimension, 1D, 2D or 3D (default 1).
					//configuration.M_size = 63; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
					configuration.size[2] = 64 * pow(2, k); //configuration.M_size
					configuration.size[0] = 64 * pow(2, k);// configuration.M_size; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
					configuration.size[1] = 64 * pow(2, k); // configuration.M_size;

					//configuration.logicBlock[0] = logicBlock[0];
					//configuration.logicBlock[1] = logicBlock[1];
					//configuration.logicBlock[2] = logicBlock[2];

					configuration.logicBlock[0] = warpSize * pow(2, n % (1 + (int)log2(configuration.size[0] / warpSize)));
					configuration.logicBlock[1] = pow(2, (n / (1 + (int)log2(configuration.size[0] / warpSize))) % 4);
					configuration.logicBlock[2] = pow(2, (n / (1 + (int)log2(configuration.size[0] / warpSize))) / 4);
					configuration.doublePrecision = 1;
					//configuration.keepShaderCode = 1;
					//configuration.k_nf = 1;
					//configuration.arg = 1;
					configuration.s_dx = configuration.size[0] / 40.0;
					configuration.s_dy = configuration.size[1] / 40.0;
					configuration.s_dz = configuration.size[2] / 40.0;
					configuration.s_dt_D = 1.0 / (1.0 + 40.0 / (4 * 3.1415926535897932384626433832795 * 1.0 / sqrt(3.1)) * configuration.s_dx); // 5.32083411470606546e-02;//
					//configuration.s_dt_D = 1.0 / ((4 * 3.1415926535897932384626433832795 * 1.0) / (1.0/sqrt(3.1)*40.0)* configuration.s_dx); // 1.1478594718471122;//
					int* x;
					int** y;
					y = &x;
					configuration.aimThreads = 32;
					//configuration.JW_sequential = 1;
					configuration.finiteDifferences = 1;
					//configuration.compute_flux_D = (type == 0) ? 1 : 0;
					configuration.compute_Pf = (type == 1) ? 1 : 0;
					//configuration.outputBufferStride[0] = configuration.size[0];
					if (r == 0) configuration.saveApplicationToString = 1;
					if (r != 0) configuration.loadApplicationFromString = 1;
					//configuration.keepShaderCode = 1;
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

					bufferSolveSize = (uint64_t)sizeof(double) * (configuration.size[0]) * (configuration.size[1]) * (configuration.size[2]);

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
					double* Pf = 0;
					res = cudaMalloc((void**)&Pf, bufferSolveSize);
					if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* qDx = 0;
					res = cudaMalloc((void**)&qDx, bufferSolveSize);
					if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* qDy = 0;
					res = cudaMalloc((void**)&qDy, bufferSolveSize);
					if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* qDz = 0;
					res = cudaMalloc((void**)&qDz, bufferSolveSize);
					if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* T = 0;
					//res = cudaMalloc((void**)&T, bufferSolveSize);
					//if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* buffer[20];
					buffer[0] = qDx;
					buffer[1] = qDy;
					buffer[2] = qDz;
					buffer[3] = Pf;
					//buffer[4] = T;

					/*if (app.configuration.compute_flux_D) {
						buffer[5] = &configuration.k_nf;
						buffer[6] = &configuration.s_dt_D;
						buffer[7] = &configuration.arg;
						buffer[8] = &configuration.s_dx;
						buffer[9] = &configuration.s_dy;
						buffer[10] = &configuration.s_dz;
					}*/
					if (app.configuration.compute_Pf) {
						//buffer[5] = &configuration.s_dt_D;
						//buffer[6] = &configuration.s_dx;
						//buffer[7] = &configuration.s_dy;
						//buffer[8] = &configuration.s_dz;
					}
#elif(VKFFT_BACKEND==2)
					double* Pf = 0;
					res = hipMalloc((void**)&Pf, bufferSolveSize);
					if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* qDx = 0;
					res = hipMalloc((void**)&qDx, bufferSolveSize);
					if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* qDy = 0;
					res = hipMalloc((void**)&qDy, bufferSolveSize);
					if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* qDz = 0;
					res = hipMalloc((void**)&qDz, bufferSolveSize);
					if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* T = 0;
					//res = hipMalloc((void**)&T, bufferSolveSize);
					//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					double* buffer[20];
					buffer[0] = qDx;
					buffer[1] = qDy;
					buffer[2] = qDz;
					buffer[3] = Pf;
#elif(VKFFT_BACKEND==3)
					cl_mem bufferSolve = 0;
					bufferSolve = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveSize, 0, &res);
					if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
					cl_mem bufferSolveRes = 0;
					bufferSolveRes = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSolveResSize, 0, &res);
					if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#endif
					resFFT = transferDataFromCPU(vkGPU, buffer_input, &Pf, bufferSolveSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
					resFFT = transferDataFromCPU(vkGPU, buffer_input, &qDx, bufferSolveSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
					resFFT = transferDataFromCPU(vkGPU, buffer_input, &qDy, bufferSolveSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
					resFFT = transferDataFromCPU(vkGPU, buffer_input, &qDz, bufferSolveSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#if(VKFFT_BACKEND==0)
					configuration.buffer = &bufferSolve;
					configuration.outputBuffer = &bufferSolveRes;
#elif(VKFFT_BACKEND==1)
					configuration.buffer = (void**)buffer;
#elif(VKFFT_BACKEND==2)
					configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
					configuration.buffer = &bufferSolve;
					configuration.outputBuffer = &bufferSolveRes;
#endif
					configuration.bufferSize = &bufferSolveSize;

					//Fill data on CPU. It is best to perform all operations on GPU after initial upload

					//buffer_input_systems[0] = 0.69;
					//buffer_input_systems[1] = 0.23;
					//Sample buffer transfer tool. Uses staging buffer of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
#if(VKFFT_BACKEND==0)
					resFFT = transferDataFromCPU(vkGPU, buffer_input_matrix_gpu, &bufferSolve, bufferSolveSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
					resFFT = transferDataFromCPU(vkGPU, buffer_input_systems, &bufferSolveRes, bufferSolveResSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			//res = cudaMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, cudaMemcpyHostToDevice);
			//if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
				//res = hipMemcpy(bufferSolve, buffer_input_matrix_gpu, bufferSolveSize, hipMemcpyHostToDevice);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
				//res = hipMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, hipMemcpyHostToDevice);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
					res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolve, CL_TRUE, 0, bufferSolveSize, buffer_input_matrix_gpu, 0, NULL, NULL);
					if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
					res = clEnqueueWriteBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, buffer_input_systems, 0, NULL, NULL);
					if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
#endif
					//float* ress = (float*)malloc(bufferSolveResSize);
					//float* ress2 = (float*)malloc(bufferSolveResSize);

					//free(inp);
					//free(outp);
					//free(matrix);
					if (configuration.loadApplicationFromString) {
						FILE* kernelCache;
						uint64_t str_len;
						char fname[500];
						sprintf(fname, "compute_Pf_size_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "_logicblock_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "", configuration.size[0], configuration.size[1], configuration.size[2], configuration.logicBlock[0], configuration.logicBlock[1], configuration.logicBlock[2]);
						kernelCache = fopen(fname, "rb");
						if (!kernelCache) return PFSOLVE_ERROR_EMPTY_FILE;
						fseek(kernelCache, 0, SEEK_END);
						str_len = ftell(kernelCache);
						fseek(kernelCache, 0, SEEK_SET);
						configuration.loadApplicationString = malloc(str_len);
						fread(configuration.loadApplicationString, str_len, 1, kernelCache);
						fclose(kernelCache);
					}
					resFFT = initializePfSolve(&app, configuration);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;

					if (configuration.loadApplicationFromString)
						free(configuration.loadApplicationString);

					if (configuration.saveApplicationToString != 0) {
						FILE* kernelCache;
						char fname[500];
						sprintf(fname, "compute_Pf_size_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "_logicblock_%" PRIu64 "_%" PRIu64 "_%" PRIu64 "", configuration.size[0], configuration.size[1], configuration.size[2], configuration.logicBlock[0], configuration.logicBlock[1], configuration.logicBlock[2]);
						kernelCache = fopen(fname, "wb");
						fwrite(app.saveApplicationString, app.applicationStringSize, 1, kernelCache);
						fclose(kernelCache);
					}
					//Submit FFT+iFFT.
					uint64_t num_iter = 100;
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
					//launchParams.buffer = (void**)&bufferSolve;
					//launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==2)
					//launchParams.buffer = (void**)&bufferSolve;
					//launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==3)
					launchParams.buffer = &bufferSolve;
					launchParams.outputBuffer = &bufferSolveRes;
#endif

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

					//float* output_PfSolve = (float*)(malloc(sizeof(float) * configuration.size[0] * configuration.size[1] * configuration.size[2]));
					//if (!output_PfSolve) return PFSOLVE_ERROR_MALLOC_FAILED;
					//Transfer data from GPU using staging buffer.
#if(VKFFT_BACKEND==0)
					resFFT = transferDataToCPU(vkGPU, output_PfSolve, &bufferSolveRes, bufferSolveResSize);
					if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
			//res = cudaMemcpy(output_PfSolve, bufferSolveRes, bufferSolveResSize, cudaMemcpyDeviceToHost);
			//if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
				//res = hipMemcpy(output_PfSolve, bufferSolveRes, bufferSolveResSize, hipMemcpyDeviceToHost);
				//if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==3)
					res = clEnqueueReadBuffer(vkGPU->commandQueue, bufferSolveRes, CL_TRUE, 0, bufferSolveResSize, output_PfSolve, 0, NULL, NULL);
					if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_COPY;
#endif
					float resSUM = 0;
					float resSUM2 = 0;
					float resSUM3 = 0;
					float resSUM4 = 0;
					//printf("res bs = %.17f\n", resSUM);
					//printf("res pcr = %.17f\n", resSUM2);
					//printf("max res gpu - 128b = %.17f\n", resSUM3);
					//printf("max res cpu bs - 128b = %.17f\n", resSUM4);
					//printf("%d %.3e %.3e\n", configuration.size[0], resSUM3, resSUM4);
					//printf("time = %.6f ms\n", totTime/num_iter);
					if (r == 2)
						printf("size = %d, lb %d %d %d, time = %.6f ms, buffer  = %d MB, acheived bw = %f GB/s\n", configuration.size[0], configuration.logicBlock[0], configuration.logicBlock[1], configuration.logicBlock[2], totTime / num_iter, bufferSolveSize / 1024 / 1024, 4 * bufferSolveSize / 1024.0 / 1024.0 / 1024.0 * 1000.0 / (totTime / num_iter));

#if(VKFFT_BACKEND==0)
					vkDestroyBuffer(vkGPU->device, bufferSolve, NULL);
					vkFreeMemory(vkGPU->device, bufferSolveDeviceMemory, NULL);
					vkDestroyBuffer(vkGPU->device, bufferSolveRes, NULL);
					vkFreeMemory(vkGPU->device, bufferSolveResDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
					cudaFree(Pf);
					cudaFree(qDx);
					cudaFree(qDy);
					cudaFree(qDz);
#elif(VKFFT_BACKEND==2)
					hipFree(Pf);
					hipFree(qDx);
					hipFree(qDy);
					hipFree(qDz);
#elif(VKFFT_BACKEND==3)
					clReleaseMemObject(bufferSolve);
					clReleaseMemObject(bufferSolveRes);
#endif
					deletePfSolve(&app);
				}
			}
		}
	}
	return resFFT;
}
