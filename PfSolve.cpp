#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#define __STDC_FORMAT_MACROS
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
#include "user_benchmark_VkFFT.h"
#include "sample_0_benchmark_VkFFT_single.h"
#include "sample_1_benchmark_VkFFT_double.h"
#include "sample_2.h"
#ifdef USE_FFTW
#include "fftw3.h"
#endif

PfSolveResult launchPfSolve(VkGPU* vkGPU, uint64_t sample_id, bool file_output, FILE* output, PfSolveUserSystemParameters* userParams) {
	//Sample Vulkan project GPU initialization.
	PfSolveResult resFFT = PFSOLVE_SUCCESS;

#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	//create instance - a connection between the application and the Vulkan library 
	res = createInstance(vkGPU, sample_id);
	if (res != 0) {
		//printf("Instance creation failed, error code: %" PRIu64 "\n", res);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_INSTANCE;
	}
	//set up the debugging messenger 
	res = setupDebugMessenger(vkGPU);
	if (res != 0) {
		//printf("Debug messenger creation failed, error code: %" PRIu64 "\n", res);
		return PFSOLVE_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;
	}
	//check if there are GPUs that support Vulkan and select one
	res = findPhysicalDevice(vkGPU);
	if (res != 0) {
		//printf("Physical device not found, error code: %" PRIu64 "\n", res);
		return PFSOLVE_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
	}
	//create logical device representation
	res = createDevice(vkGPU, sample_id);
	if (res != 0) {
		//printf("Device creation failed, error code: %" PRIu64 "\n", res);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_DEVICE;
	}
	//create fence for synchronization 
	res = createFence(vkGPU);
	if (res != 0) {
		//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_FENCE;
	}
	//create a place, command buffer memory is allocated from
	res = createCommandPool(vkGPU);
	if (res != 0) {
		//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
	}
	vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);

	glslang_initialize_process();//compiler can be initialized before PfSolve
#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	cudaError_t res2 = cudaSuccess;
	res = cuInit(0);
	if (res != CUDA_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_INITIALIZE;
	res2 = cudaSetDevice((int)vkGPU->device_id);
	if (res2 != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = cuDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
	if (res != CUDA_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_GET_DEVICE;
	res = cuCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
	if (res != CUDA_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	res = hipInit(0);
	if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_INITIALIZE;
	res = hipSetDevice((int)vkGPU->device_id);
	if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = hipDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
	if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_GET_DEVICE;
	res = hipCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
	if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	cl_uint numPlatforms;
	res = clGetPlatformIDs(0, 0, &numPlatforms);
	if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_INITIALIZE;
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	if (!platforms) return PFSOLVE_ERROR_MALLOC_FAILED;
	res = clGetPlatformIDs(numPlatforms, platforms, 0);
	if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_INITIALIZE;
	uint64_t k = 0;
	for (uint64_t j = 0; j < numPlatforms; j++) {
		cl_uint numDevices;
		res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
		cl_device_id* deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
		if (!deviceList) return PFSOLVE_ERROR_MALLOC_FAILED;
		res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
		if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_GET_DEVICE;
		for (uint64_t i = 0; i < numDevices; i++) {
			if (k == vkGPU->device_id) {
				vkGPU->platform = platforms[j];
				vkGPU->device = deviceList[i];
				vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_CONTEXT;
				cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				vkGPU->commandQueue = commandQueue;
				k++;
			}
			else {
				k++;
			}
		}
		free(deviceList);
	}
	free(platforms);
#endif

	uint64_t isCompilerInitialized = 1;

	switch (sample_id) {
	case 0:
	{
		resFFT = sample_0_benchmark_VkFFT_single(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 1:
	{
		resFFT = sample_1_benchmark_VkFFT_double(vkGPU, file_output, output, isCompilerInitialized);
		break;
	}
	case 2:
	{
		resFFT = sample_2(vkGPU, file_output, output, isCompilerInitialized, 1, userParams->size, userParams->logicBlock);
		break;
	}
	}
#if(VKFFT_BACKEND==0)
	vkDestroyFence(vkGPU->device, vkGPU->fence, NULL);
	vkDestroyCommandPool(vkGPU->device, vkGPU->commandPool, NULL);
	vkDestroyDevice(vkGPU->device, NULL);
	DestroyDebugUtilsMessengerEXT(vkGPU, NULL);
	vkDestroyInstance(vkGPU->instance, NULL);
	glslang_finalize_process();//destroy compiler after use
#elif(VKFFT_BACKEND==1)
	res = cuCtxDestroy(vkGPU->context);
#elif(VKFFT_BACKEND==2)
	res = hipCtxDestroy(vkGPU->context);
#elif(VKFFT_BACKEND==3)
	res = clReleaseCommandQueue(vkGPU->commandQueue);
	if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
	clReleaseContext(vkGPU->context);
#endif

	return resFFT;
}

bool findFlag(char** start, char** end, const std::string& flag) {
	return (std::find(start, end, flag) != end);
}
char* getFlagValue(char** start, char** end, const std::string& flag)
{
	char** value = std::find(start, end, flag);
	value++;
	if (value != end)
	{
		return *value;
	}
	return 0;
}
int main(int argc, char* argv[])
{
	VkGPU vkGPU = {};
#if(VKFFT_BACKEND==0)
	vkGPU.enableValidationLayers = 0;
#endif
	bool file_output = false;
	FILE* output = NULL;
	int sscanf_res = 0;
	if (findFlag(argv, argv + argc, "-h"))
	{
		//print help
		int version = PfSolveGetVersion();
		int version_decomposed[3];
		version_decomposed[0] = version / 10000;
		version_decomposed[1] = (version - version_decomposed[0] * 10000) / 100;
		version_decomposed[2] = (version - version_decomposed[0] * 10000 - version_decomposed[1] * 100);
		printf("PfSolve v%d.%d.%d. Author: Tolmachev Dmitrii\n", version_decomposed[0], version_decomposed[1], version_decomposed[2]);
#if (VKFFT_BACKEND==0)
		printf("Vulkan backend\n");
#elif (VKFFT_BACKEND==1)
		printf("CUDA backend\n");
#elif (VKFFT_BACKEND==2)
		printf("HIP backend\n");
#elif (VKFFT_BACKEND==3)
		printf("OpenCL backend\n");
#endif
		printf("	-h: print help\n");
		printf("	-devices: print the list of available device ids, used as -d argument\n");
		printf("	-d X: select device (default 0)\n");
		printf("	-o NAME: specify output file path\n");
		printf("	-generate -size S -lbx X -lby Y -lbz Z: generate kernels with nz = S and logic block (X,Y,Z):\n");
		
		return 0;
	}
	if (findFlag(argv, argv + argc, "-devices"))
	{
		//print device list
		PfSolveResult resFFT = devices_list();
		return resFFT;
	}
	if (findFlag(argv, argv + argc, "-d"))
	{
		//select device_id
		char* value = getFlagValue(argv, argv + argc, "-d");
		if (value != 0) {
			sscanf_res = sscanf(value, "%" PRIu64 "", &vkGPU.device_id);
			if (sscanf_res <= 0) {
				printf("sscanf failed\n");
				return 1;
			}
		}
		else {
			printf("No device is selected with -d flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-o"))
	{
		//specify output file
		char* value = getFlagValue(argv, argv + argc, "-o");
		if (value != 0) {
			file_output = true;
			output = fopen(value, "a");
		}
		else {
			printf("No output file is selected with -o flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-pfsolve"))
	{
		//select sample_id
		char* value = getFlagValue(argv, argv + argc, "-pfsolve");
		if (value != 0) {
			uint64_t sample_id = 0;
			sscanf_res = sscanf(value, "%" PRIu64 "", &sample_id);
			if (sscanf_res <= 0) {
				printf("sscanf failed\n");
				return 1;
			}
			PfSolveResult resFFT = launchPfSolve(&vkGPU, sample_id, file_output, output, 0);
			if (resFFT != PFSOLVE_SUCCESS) return resFFT;
		}
		else {
			printf("No sample is selected with -PfSolve flag\n");
			return 1;
		}
	}
	if (findFlag(argv, argv + argc, "-generate"))
	{
		//select sample_id
		PfSolveUserSystemParameters userParams = {};

		if (findFlag(argv, argv + argc, "-size"))
		{
			char* value = getFlagValue(argv, argv + argc, "-size");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.size);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No size is selected with -size flag\n");
				return 1;
			}
		}
		else {
			printf("No -size flag is selected\n");
			return 1;
		}
		if (findFlag(argv, argv + argc, "-lbx"))
		{
			char* value = getFlagValue(argv, argv + argc, "-lbx");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.logicBlock[0]);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No logicblock is selected with -lbx flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-lby"))
		{
			char* value = getFlagValue(argv, argv + argc, "-lby");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.logicBlock[1]);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No logicblock y is selected with -lby flag\n");
				return 1;
			}
		}
		if (findFlag(argv, argv + argc, "-lbz"))
		{
			char* value = getFlagValue(argv, argv + argc, "-lbz");
			if (value != 0) {
				sscanf_res = sscanf(value, "%" PRIu64 "", &userParams.logicBlock[2]);
				if (sscanf_res <= 0) {
					printf("sscanf failed\n");
					return 1;
				}
			}
			else {
				printf("No logicblock z is selected with -lbz flag\n");
				return 1;
			}
		}
		PfSolveResult resFFT = launchPfSolve(&vkGPU, 2, file_output, output, &userParams);
		if (resFFT != PFSOLVE_SUCCESS) return resFFT;

		return 0;
	}
	return 0;
}
