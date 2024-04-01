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
#define USE_MPIR
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

int reorder_i0(int i, int M_size, int warpSize, int used_registers) {
	int ret = i / used_registers;
	if ((i % used_registers) < (M_size% used_registers)) {
		ret += (i % used_registers) * ((M_size + used_registers - 1) / used_registers);
	}
	else
	{
		ret += (M_size% used_registers) * ((M_size + used_registers - 1) / used_registers);
		ret += ((i % used_registers) - (M_size % used_registers)) * (M_size / used_registers);
	}
	return ret;
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
	printf("size GPU_L2 CPU_L2 - GPU_MAX CPU_MAX\n");
			
	for (uint64_t n = 1; n < 2; n++) {
		printf("%d size GPU_L2 CPU_L2 - GPU_MAX CPU_MAX\n", n);
		float run_time[num_runs];
		for (uint64_t r = 0; r < 1; r++) {
			//Configuration + FFT application .
			PfSolveConfiguration configuration = {};
			PfSolveApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.M_size = 250;// 1 + r;// 16 + 16 * (n % 128); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.M_size_pow2 = (int64_t)pow(2, (int)ceil(log2((double)configuration.M_size)));; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[0] = configuration.M_size; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.  
			configuration.size[1] = 1;
			configuration.size[2] = 1;
			configuration.scaleC = 1;
			configuration.jw_type = 13;
			configuration.doublePrecision = 1;
			configuration.isOutputFormatted = 1;
			configuration.keepShaderCode = 1;
			int* x;
			int** y;
			y = &x;
			//configuration.aimThreads = 32;
			configuration.numConsecutiveJWIterations = 1;
			configuration.useMultipleInputBuffers = 1;
			configuration.jw_control_bitmask = (RUNTIME_OFFSETSOLUTION + RUNTIME_INPUTZEROPAD + RUNTIME_OUTPUTZEROPAD + RUNTIME_INPUTBUFFERSTRIDE + RUNTIME_OUTPUTBUFFERSTRIDE);// (RUNTIME_SCALEC);
			//configuration.JW_sequential = 1;
			//configuration.JW_parallel = 1;
			configuration.outputBufferStride[0] = configuration.size[0];
			//configuration.performWorland = 1;
			configuration.upperBanded = 2;
			//configuration.offsetV = 2 * configuration.size[0];
			//CUstream hStream;
			//cudaStreamCreate(&hStream);
			//configuration.stream = &hStream;
			//configuration.num_streams = 1;
			configuration.disableCaching = 1;
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

			bufferSolveSize = (uint64_t)sizeof(double) * 3 * configuration.size[0];
			bufferSolveResSize = (uint64_t)sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2];

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
			cuDoubleComplex** bufferSolve = (cuDoubleComplex**)calloc(configuration.numConsecutiveJWIterations, sizeof(cuDoubleComplex*));
			for (int i = 0; i < configuration.numConsecutiveJWIterations; i++)
				res = cudaMalloc((void**)&bufferSolve[i], bufferSolveSize);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
			cuDoubleComplex* bufferSolveRes = 0;
			res = cudaMalloc((void**)&bufferSolveRes, bufferSolveResSize);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
			hipDoubleComplex** bufferSolve = (hipDoubleComplex**)calloc(configuration.numConsecutiveJWIterations, sizeof(hipDoubleComplex*));
			for (int i = 0; i < configuration.numConsecutiveJWIterations; i++)
				res = hipMalloc((void**)&bufferSolve[i], bufferSolveSize);
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
			configuration.buffer = (void**)bufferSolve;
			configuration.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)bufferSolve;
			configuration.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==3)
			configuration.buffer = &bufferSolve;
			configuration.outputBuffer = &bufferSolveRes;
#endif
			configuration.bufferSize = &bufferSolveSize;
			configuration.outputBufferSize = &bufferSolveResSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			int l = 100;// 2 * configuration.size[0];
			double* buffer_input_matrix[100];

			double* buffer_input_matrix_gpu[100];
			for (int t = 0; t < configuration.numConsecutiveJWIterations; t++) {
				buffer_input_matrix[t] = (double*)calloc(bufferSolveSize, 1);
				buffer_input_matrix_gpu[t] = (double*)calloc(bufferSolveSize, 1);
			}

			int64_t tempM = configuration.size[0];
			if (!configuration.upperBanded) tempM += (configuration.numConsecutiveJWIterations - 1);
			int warpSize = 64;
			int used_registers = (uint64_t)ceil(tempM / (double)warpSize);

			tempM = configuration.size[0];

			/*for (int t = 0; t < configuration.numConsecutiveJWIterations; t++) {
				for (uint64_t i = 0; i < tempM; i++) {
					buffer_input_matrix[t][tempM + i] = mu(i, -0.5, l - 0.5 + 1);
					buffer_input_matrix[t][i] = nu(i, -0.5, l - 0.5 + 1);// / mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
					buffer_input_matrix[t][2 * tempM + i] = nu(i, -0.5, l - 0.5 + 1);// / mu(i, -0.5, l - 0.5 + 1);// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
				}
			}*/
			double scale = 0.05*n;
			for (int t = 0; t < configuration.numConsecutiveJWIterations; t++) {
				for (uint64_t i = 0; i < tempM; i++) {
					buffer_input_matrix[t][tempM + i] = 1.0 + scale * (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
					buffer_input_matrix[t][i] = scale* (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
					buffer_input_matrix[t][2 * tempM + i] = scale * (double)(2 * ((double)rand()) / RAND_MAX - 1.0);
					//printf("%.17e %.17e %.17e\n", buffer_input_matrix[t][i], buffer_input_matrix[t][tempM + i], buffer_input_matrix[t][2 * tempM + i]);
				}
			}
			buffer_input_matrix[0][0] = 0;// need to add conditional there
			buffer_input_matrix[0][3 * tempM-1] = 0;// need to add conditional there

#ifdef USE_MPIR
			mpf_set_default_prec(128);
			mpf_t temp_0;
			mpf_init(temp_0);
			mpf_t temp_1;
			mpf_init(temp_1);
			for (uint64_t i = 0; i < configuration.size[0]; i++) {
				mpf_set_d(temp_0, 1);
				mpf_set_d(temp_1, buffer_input_matrix[0][tempM + i]);
				mpf_div(temp_0, temp_0, temp_1);
				buffer_input_matrix_gpu[0][tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)] = mpf_get_d(temp_0);

				mpf_set_d(temp_0, buffer_input_matrix[0][i]);
				mpf_set_d(temp_1, buffer_input_matrix_gpu[0][tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)]);
				mpf_mul(temp_0, temp_0, temp_1);
				buffer_input_matrix_gpu[0][reorder_i0(i, configuration.size[0], warpSize, used_registers)] = mpf_get_d(temp_0);

				mpf_set_d(temp_0, buffer_input_matrix[0][2 * tempM + i]);
				mpf_set_d(temp_1, buffer_input_matrix_gpu[0][tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)]);
				mpf_mul(temp_0, temp_0, temp_1);
				buffer_input_matrix_gpu[0][2 * tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)] = mpf_get_d(temp_0);
			}
			mpf_clear(temp_0);
			mpf_clear(temp_1);
#else
			for (uint64_t i = 0; i < configuration.size[0]; i++) {
				buffer_input_matrix_gpu[0][tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)] = 1.0 / buffer_input_matrix[0][tempM + i];// buffer_input_matrix_gpu[t][reorder_i(i, tempM, 32, used_registers)];
				buffer_input_matrix_gpu[0][reorder_i0(i, configuration.size[0], warpSize, used_registers)] = buffer_input_matrix[0][i] * buffer_input_matrix_gpu[0][tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)];// buffer_input_matrix_gpu[t][reorder_i(i, tempM, 32, used_registers)];
				buffer_input_matrix_gpu[0][2 * tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)] = buffer_input_matrix[0][2 * tempM + i] * buffer_input_matrix_gpu[0][tempM + reorder_i0(i, configuration.size[0], warpSize, used_registers)];// buffer_input_matrix_gpu[t][reorder_i(i, tempM, 32, used_registers)];
			}
#endif
			double* buffer_input_systems = (double*)malloc(bufferSolveResSize);
			double* buffer_input_systems2 = (double*)malloc(bufferSolveResSize);
			for (uint64_t j = 0; j < configuration.size[1]; j++) {
				for (uint64_t i = 0; i < configuration.size[0]; i++) {
					buffer_input_systems2[i + j * configuration.size[0]] = (double)(2 * ((double)rand()) / RAND_MAX - 1.0);//x0[i];
					buffer_input_systems[i + j * configuration.size[0]] = buffer_input_systems2[i + j * configuration.size[0]];//x0[i];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);// +i + j * configuration.size[0];// (float)(2 * ((float)rand()) / RAND_MAX - 1.0);
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
			for (int t = 0; t < configuration.numConsecutiveJWIterations; t++) {
				res = cudaMemcpy(bufferSolve[t], buffer_input_matrix_gpu[t], bufferSolveSize, cudaMemcpyHostToDevice);
				if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			}
			res = cudaMemcpy(bufferSolveRes, buffer_input_systems, bufferSolveResSize, cudaMemcpyHostToDevice);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
#elif(VKFFT_BACKEND==2)
			for (int t = 0; t < configuration.numConsecutiveJWIterations; t++) {
				res = hipMemcpy(bufferSolve[t], buffer_input_matrix_gpu[t], bufferSolveSize, hipMemcpyHostToDevice);
				if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_COPY;
			}
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
			mpf_set_default_prec(128);
			mpf_t temp;
			mpf_init(temp);
			mpf_set_d(temp, 0);
			mpf_t temp0;
			mpf_init(temp0);
			mpf_set_d(temp0, 0);
			mpf_t temp1;
			mpf_init(temp1);
			mpf_set_d(temp1, 0);

			mpf_t* ress0 = (mpf_t*)malloc(sizeof(mpf_t) * configuration.size[0] * configuration.size[1]);
			mpf_t* input0 = (mpf_t*)malloc(sizeof(mpf_t) * configuration.size[0] * configuration.size[1]);
			mpf_t* buffer_input_matrix0 = (mpf_t*)malloc(4 * sizeof(mpf_t) * configuration.size[0]);
			mpf_t* temp_matrix0 = (mpf_t*)malloc(4 * sizeof(mpf_t) * configuration.size[0]);
			for (uint64_t j = 0; j < configuration.size[0] * configuration.size[1]; j++) {
				mpf_init(ress0[j]);
				mpf_init(input0[j]);
				mpf_set_d(input0[j], buffer_input_systems[j]);
			}
			for (uint64_t j = 0; j < 3 * configuration.size[0]; j++) {
				mpf_init(buffer_input_matrix0[j]);
				mpf_set_d(buffer_input_matrix0[j], buffer_input_matrix[0][j]);
			}
			for (uint64_t j = 0; j < 3 * configuration.size[0]; j++) {
				mpf_init(temp_matrix0[j]);
			}
			mpf_div(temp_matrix0[2 * tempM], buffer_input_matrix0[2 * tempM], buffer_input_matrix0[tempM]);
			mpf_div(ress0[0], input0[0], buffer_input_matrix0[tempM]);
			for (uint64_t j = 1; j < configuration.size[0]; j++) {
				mpf_mul(temp, buffer_input_matrix0[j], temp_matrix0[2 * tempM + j - 1]);
				mpf_sub(temp, buffer_input_matrix0[tempM + j], temp);
				mpf_div(temp_matrix0[2 * tempM + j], buffer_input_matrix0[2 * tempM + j], temp);

				mpf_mul(temp, buffer_input_matrix0[j], temp_matrix0[2 * tempM + j - 1]);
				mpf_sub(temp, buffer_input_matrix0[tempM + j], temp);
				mpf_mul(temp0, buffer_input_matrix0[j], ress0[j-1]);
				mpf_sub(temp0, input0[j], temp0);
				mpf_div(ress0[j], temp0, temp);
			}

			for (int64_t j = tempM-2; j >= 0; j--) {
				mpf_mul(temp, temp_matrix0[2 * tempM + j], ress0[j+1]);
				mpf_sub(ress0[j], ress0[j], temp);
			}
			tempM = configuration.size[0];
			double* temp_matrix = (double*)malloc(bufferSolveSize);
			double* ress = (double*)malloc(bufferSolveResSize);
			double* ress2 = (double*)malloc(bufferSolveResSize);
			double* input = buffer_input_systems;

			temp_matrix[2 * tempM] = buffer_input_matrix[0][2 * tempM] / buffer_input_matrix[0][tempM];
			ress2[0] = input[0] / buffer_input_matrix[0][tempM];
			for (int64_t j = 1; j < tempM; j++) {
				temp_matrix[2 * tempM + j] = buffer_input_matrix[0][2 * tempM + j] / (buffer_input_matrix[0][tempM + j] - buffer_input_matrix[0][j] * temp_matrix[2 * tempM + j - 1]);
				ress2[j] = (input[j] - buffer_input_matrix[0][j] * ress2[j-1]) / (buffer_input_matrix[0][tempM + j] - buffer_input_matrix[0][j] * temp_matrix[2 * tempM + j - 1]);
			}

			for (int64_t j = tempM-2; j >= 0; j--) {
				ress2[j] = ress2[j] - temp_matrix[2 * tempM + j] * ress2[j+1];
			}

			for (uint64_t j = 0; j < configuration.size[0]; j++) {
				ress[j] = mpf_get_d(ress0[j]);
			}
			mpf_clear(temp);
			mpf_clear(temp0);
			mpf_clear(temp1);
			for (uint64_t j = 0; j < configuration.size[0]; j++) {
				mpf_clear(ress0[j]);
				mpf_clear(input0[j]);
			}
			for (uint64_t j = 0; j < 3 * configuration.size[0]; j++) {
				mpf_clear(buffer_input_matrix0[j]);
				mpf_clear(temp_matrix0[j]);
			}
			free(ress0);
			free(input0);
			free(buffer_input_matrix0);
			free(temp_matrix0);
#else
			tempM = configuration.size[0];
			double* temp_matrix = (double*)malloc(bufferSolveSize);
			double* ress = (double*)malloc(bufferSolveResSize);
			double* ress2 = (double*)malloc(bufferSolveResSize);
			double* input = buffer_input_systems;

			temp_matrix[2 * tempM] = buffer_input_matrix[0][2 * tempM] / buffer_input_matrix[0][tempM];
			ress[0] = input[0] / buffer_input_matrix[0][tempM];
			for (int64_t j = 1; j < tempM; j++) {
				temp_matrix[2 * tempM + j] = buffer_input_matrix[0][2 * tempM + j] / (buffer_input_matrix[0][tempM + j] - buffer_input_matrix[0][j] * temp_matrix[2 * tempM + j - 1]);
				ress[j] = (input[j] - buffer_input_matrix[0][j] * ress[j-1]) / (buffer_input_matrix[0][tempM + j] - buffer_input_matrix[0][j] * temp_matrix[2 * tempM + j - 1]);
			}

			for (int64_t j = tempM-2; j >= 0; j--) {
				ress[j] = ress[j] - temp_matrix[2 * tempM + j] * ress[j+1];
			}
			for (int64_t j = 0; j < tempM; j++) {
				ress2[j] = ress[j];
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
			launchParams.buffer = (void**)bufferSolve;
			launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==2)
			launchParams.buffer = (void**)bufferSolve;
			launchParams.outputBuffer = (void**)&bufferSolveRes;
#elif(VKFFT_BACKEND==3)
			launchParams.buffer = &bufferSolve;
			launchParams.outputBuffer = &bufferSolveRes;
#endif
			//launchParams.offsetV = 2 * configuration.size[0];
			launchParams.offsetM = 0;
			launchParams.offsetV = 0;
			launchParams.offsetSolution = 0;
			launchParams.inputZeropad[0] = 0;
			launchParams.inputZeropad[1] = configuration.M_size;
			//launchParams.inputZeropad[1]--;
			launchParams.outputZeropad[0] = 0;
			launchParams.outputZeropad[1] = configuration.M_size;
			launchParams.outputBufferStride = configuration.M_size;
			launchParams.inputBufferStride = configuration.M_size;
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

			double* output_PfSolve = (double*)(malloc(sizeof(double) * configuration.size[0] * configuration.size[1] * configuration.size[2]));
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
			num_iter = 1;
			//resFFT = performVulkanFFT(vkGPU, &app, &launchParams, 0, num_iter, &totTime2);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
			/*if (configuration.upperBanded != 1) {
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
			double resCPUSUM = 0;
			double resGPUSUM = 0;
			double resCPUMAX = 0;
			//double resSUM2_cu = 0;
			//double resSUM3_cu = 0;
			double resGPUMAX = 0;
			for (uint64_t l = 0; l < configuration.size[2]; l++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {
					for (uint64_t i = 0; i < configuration.size[0]; i++) {
						uint64_t loc_i = i;
						uint64_t loc_j = j;
						uint64_t loc_l = l;

						//if (file_output) fprintf(output, "%f %f - %f %f \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_PfSolve[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_PfSolve[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
						double resCPU = 0;
						double resCPU_mpir = 0;
						double resGPU = 0;
						double resGPU2 = 0;
						double resMUL2_cu = 0;
						double resMUL3_cu = 0;
						double resCPU2 = 0;
						/*
						if (i > 0)
							resMUL += buffer_input_matrix[i] * output_PfSolve[(i + j * configuration.size[0]) - 1];
						resMUL += buffer_input_matrix[i + configuration.size[0]] * output_PfSolve[(i + j * configuration.size[0])];
						if (i < configuration.size[0] - 1)
							resMUL += buffer_input_matrix[i + 2 * configuration.size[0]] * output_PfSolve[(i + j * configuration.size[0]) + 1];
						*/
						resCPU_mpir += buffer_input_matrix[0][i + configuration.size[0]] * ress[(i + j * configuration.size[0])];
						if (i < configuration.size[0] - 1)
							resCPU_mpir += buffer_input_matrix[0][i + 2 * configuration.size[0]] * ress[(i + 1 + j * configuration.size[0])];
						if (i > 0)
							resCPU_mpir += buffer_input_matrix[0][i] * ress[(i - 1 + j * configuration.size[0])];

						resCPU += buffer_input_matrix[0][i + configuration.size[0]] * ress2[(i + j * configuration.size[0])];
						if (i < configuration.size[0] - 1)
							resCPU += buffer_input_matrix[0][i + 2 * configuration.size[0]] * ress2[(i + 1 + j * configuration.size[0])];
						if (i > 0)
							resCPU += buffer_input_matrix[0][i] * ress2[(i - 1 + j * configuration.size[0])];
						
						resGPU += buffer_input_matrix[0][i + configuration.size[0]] * output_PfSolve[(i + j * configuration.size[0])];
						if (i < configuration.size[0] - 1)
							resGPU += buffer_input_matrix[0][i + 2 * configuration.size[0]] * output_PfSolve[(i + 1 + j * configuration.size[0])];
						if (i > 0)
							resGPU += buffer_input_matrix[0][i] * output_PfSolve[(i - 1 + j * configuration.size[0])];
						
						/*if (configuration.upperBanded != 1) {
							resMUL2_cu += buffer_input_matrix[i + 3 * configuration.size[0]] * output_cuFFT[(i + j * configuration.size[0])];
							if (i < configuration.size[0] - 1)
								resMUL2_cu += buffer_input_matrix[i + 2 * configuration.size[0] + 1] * output_cuFFT[(i + j * configuration.size[0]) + 1];
						}
						else {
							if (i > 0)
								resMUL2_cu += buffer_input_matrix[i + 3 * configuration.size[0]-1] * output_cuFFT[(i + j * configuration.size[0]) - 1];
							resMUL2_cu += buffer_input_matrix[i + 2 * configuration.size[0]] * output_cuFFT[(i + j * configuration.size[0])];
						}*/
						resGPU2 = (ress[(i + j * configuration.size[0])] - output_PfSolve[(i + j * configuration.size[0])]) * (ress[(i + j * configuration.size[0])] - output_PfSolve[(i + j * configuration.size[0])]);
						//resMUL3_cu = (ress[(i + j * configuration.size[0])]- output_cuFFT[(i + j * configuration.size[0])])* (ress[(i + j * configuration.size[0])] - output_cuFFT[(i + j * configuration.size[0])]);
						resCPU2 = (ress[(i + j * configuration.size[0])] - ress2[(i + j * configuration.size[0])]) * (ress[(i + j * configuration.size[0])] - ress2[(i + j * configuration.size[0])]);
						//printf("%.17e %.17e %.17e %.17e\n", resGPU, resCPU, resCPU_mpir, buffer_input_systems[i + j * configuration.size[0]]);
						printf("%.17e %.17e %.17e\n", output_PfSolve[(i + j * configuration.size[0])], ress2[(i + j * configuration.size[0])], ress[(i + j * configuration.size[0])]);
						if (i > 0) {
							//resCPUSUM += sqrt((resCPU - buffer_input_systems[i + j * configuration.size[0]]) * (resCPU - buffer_input_systems[i + j * configuration.size[0]])) / abs(buffer_input_systems[i + j * configuration.size[0]]);
							//resGPUSUM += sqrt((resGPU - buffer_input_systems[i + j * configuration.size[0]]) * (resGPU - buffer_input_systems[i + j * configuration.size[0]])) / abs(buffer_input_systems[i + j * configuration.size[0]]);

							//resCPUSUM += sqrt((resCPU - buffer_input_systems[i + j * configuration.size[0]]) * (resCPU - buffer_input_systems[i + j * configuration.size[0]])) / abs(buffer_input_systems[i + j * configuration.size[0]]);
							//resGPUSUM += sqrt((resGPU - buffer_input_systems[i + j * configuration.size[0]]) * (resGPU - buffer_input_systems[i + j * configuration.size[0]])) / abs(buffer_input_systems[i + j * configuration.size[0]]);
						}
						resCPUSUM += (sqrt(resCPU2) / abs(ress[(i + j * configuration.size[0])]));
						resGPUSUM += (sqrt(resGPU2) / abs(ress[(i + j * configuration.size[0])]));

						resCPUMAX = ((sqrt(resCPU2) / abs(ress[(i + j * configuration.size[0])])) > resCPUMAX) ? (sqrt(resCPU2) / abs(ress[(i + j * configuration.size[0])])) : resCPUMAX;

						//resSUM2_cu += sqrt((resMUL2 - buffer_input_systems[i + j * configuration.size[0]]) * (resMUL2 - buffer_input_systems[i + j * configuration.size[0]]));
						//resSUM3_cu = (sqrt(resMUL3) > resSUM3) ? sqrt(resMUL3) : resSUM3;
						resGPUMAX = ((sqrt(resGPU2) / ress[(i + j * configuration.size[0])]) > resGPUMAX) ? (sqrt(resGPU2) / ress[(i + j * configuration.size[0])]) : resGPUMAX;
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
			
			printf("%d %.3e %.3e - %.3e %.3e\n", configuration.size[0], (double)(resGPUSUM / (configuration.size[0] * configuration.size[1])), (double)(resCPUSUM / (configuration.size[0] * configuration.size[1])), (double)resGPUMAX, (double)resCPUMAX);
			//printf("time 1 = %.6f time %d = %.6f\n",totTime, num_iter, totTime2/num_iter);
			//printf("size  = %d MB, time at peak bw = %f ms\n", 2*bufferSolveResSize/1024/1024, 2*bufferSolveResSize/1024.0/1024.0/1024.0/1200.0*1000.0);
			free(buffer_input_systems);
			free(buffer_input_systems2);
			for (int i = 0; i < configuration.numConsecutiveJWIterations; i++) {
				free(buffer_input_matrix[i]);
				free(buffer_input_matrix_gpu[i]);
			}
			free(ress);
			free(ress2);
#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, bufferSolve, NULL);
			vkFreeMemory(vkGPU->device, bufferSolveDeviceMemory, NULL);
			vkDestroyBuffer(vkGPU->device, bufferSolveRes, NULL);
			vkFreeMemory(vkGPU->device, bufferSolveResDeviceMemory, NULL);
#elif(VKFFT_BACKEND==1)
			for (int i = 0; i < configuration.numConsecutiveJWIterations; i++){
				cudaFree(bufferSolve[i]);
			}
			cudaFree(bufferSolveRes);
#elif(VKFFT_BACKEND==2)
			for (int i = 0; i < configuration.numConsecutiveJWIterations; i++){
				hipFree(bufferSolve[i]);
			}
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
