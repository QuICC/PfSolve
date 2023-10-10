// This file is part of PfSolve
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef PFSOLVE_RECURSIVEFFTGENERATORS_H
#define PFSOLVE_RECURSIVEFFTGENERATORS_H
#include "pfSolve_Structs/pfSolve_Structs.h"

#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_ManageMemory.h"
#include "pfSolve_AppManagement/pfSolve_InitializeApp.h"
static inline PfSolveResult initializePfSolve(PfSolveApplication* app, PfSolveConfiguration inputLaunchConfiguration);

static inline PfSolveResult PfSolveGeneratePhaseVectors(PfSolveApplication* app, PfSolvePlan* FFTPlan, uint64_t axis_id) {
	//generate two arrays used for Blueestein convolution and post-convolution multiplication
	PfSolveResult resFFT = PFSOLVE_SUCCESS;
	uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) bufferSize *= sizeof(double) / sizeof(float);
	app->bufferBluesteinSize[axis_id] = bufferSize;
#if(VKFFT_BACKEND==0)
	PfResult res = VK_SUCCESS;
	resFFT = allocateBufferVulkan(app, &app->bufferBluestein[axis_id], &app->bufferBluesteinDeviceMemory[axis_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
	if (resFFT != PFSOLVE_SUCCESS) return resFFT;
	if (!app->configuration.makeInversePlanOnly) {
		resFFT = allocateBufferVulkan(app, &app->bufferBluesteinFFT[axis_id], &app->bufferBluesteinFFTDeviceMemory[axis_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
		if (resFFT != PFSOLVE_SUCCESS) return resFFT;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		resFFT = allocateBufferVulkan(app, &app->bufferBluesteinIFFT[axis_id], &app->bufferBluesteinIFFTDeviceMemory[axis_id], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
		if (resFFT != PFSOLVE_SUCCESS) return resFFT;
	}
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	res = cudaMalloc((void**)&app->bufferBluestein[axis_id], bufferSize);
	if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	if (!app->configuration.makeInversePlanOnly) {
		res = cudaMalloc((void**)&app->bufferBluesteinFFT[axis_id], bufferSize);
		if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		res = cudaMalloc((void**)&app->bufferBluesteinIFFT[axis_id], bufferSize);
		if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	res = hipMalloc((void**)&app->bufferBluestein[axis_id], bufferSize);
	if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	if (!app->configuration.makeInversePlanOnly) {
		res = hipMalloc((void**)&app->bufferBluesteinFFT[axis_id], bufferSize);
		if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		res = hipMalloc((void**)&app->bufferBluesteinIFFT[axis_id], bufferSize);
		if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	app->bufferBluestein[axis_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
	if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	if (!app->configuration.makeInversePlanOnly) {
		app->bufferBluesteinFFT[axis_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
		if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		app->bufferBluesteinIFFT[axis_id] = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
		if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
	cl_command_queue commandQueue = clCreateCommandQueue(app->configuration.context[0], app->configuration.device[0], 0, &res);
	if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;

	ze_device_mem_alloc_desc_t device_desc = {};
	device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
	res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &app->bufferBluestein[axis_id]);
	if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;

	if (!app->configuration.makeInversePlanOnly) {
		res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &app->bufferBluesteinFFT[axis_id]);
		if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
	if (!app->configuration.makeForwardPlanOnly) {
		res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &app->bufferBluesteinIFFT[axis_id]);
		if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
	}
#elif(VKFFT_BACKEND==5)
	app->bufferBluestein[axis_id] = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);

	if (!app->configuration.makeInversePlanOnly) {
		app->bufferBluesteinFFT[axis_id] = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
	}
	if (!app->configuration.makeForwardPlanOnly) {
		app->bufferBluesteinIFFT[axis_id] = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
	}
#endif
#ifdef PfSolve_use_FP128_Bluestein_RaderFFT
	if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
		double* phaseVectors_fp64 = (double*)malloc(bufferSize);
		if (!phaseVectors_fp64) {
			return PFSOLVE_ERROR_MALLOC_FAILED;
		}
		long double* phaseVectors_fp128 = (long double*)malloc(2 * bufferSize);
		if (!phaseVectors_fp128) {
			free(phaseVectors_fp64);
			return PFSOLVE_ERROR_MALLOC_FAILED;
		}
		long double* phaseVectors_fp128_out = (long double*)malloc(2 * bufferSize);
		if (!phaseVectors_fp128) {
			free(phaseVectors_fp64);
			free(phaseVectors_fp128);
			return PFSOLVE_ERROR_MALLOC_FAILED;
		}
		uint64_t phaseVectorsNonZeroSize = (((app->configuration.performDCT == 4) && (app->configuration.size[axis_id] % 2 == 0)) || ((FFTPlan->multiUploadR2C) && (axis_id == 0))) ? app->configuration.size[axis_id] / 2 : app->configuration.size[axis_id];
		if (app->configuration.performDCT == 1) phaseVectorsNonZeroSize = 2 * app->configuration.size[axis_id] - 2;
		long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
		for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
			uint64_t rm = (i * i) % (2 * phaseVectorsNonZeroSize);
			long double angle = double_PI * rm / phaseVectorsNonZeroSize;
			phaseVectors_fp128[2 * i] = (i < phaseVectorsNonZeroSize) ? cos(angle) : 0;
			phaseVectors_fp128[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? -sin(angle) : 0;
		}
		for (uint64_t i = 1; i < phaseVectorsNonZeroSize; i++) {
			phaseVectors_fp128[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_fp128[2 * i];
			phaseVectors_fp128[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_fp128[2 * i + 1];
		}
		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->configuration.makeForwardPlanOnly)) {
			fftwl_plan p;
			p = fftwl_plan_dft_1d((int)(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]), (fftwl_complex*)phaseVectors_fp128, (fftwl_complex*)phaseVectors_fp128_out, -1, FFTW_ESTIMATE);
			fftwl_execute(p);
			fftwl_destroy_plan(p);
			for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
				uint64_t out = 0;
				if (FFTPlan->numAxisUploads[axis_id] == 1) {
					out = i;
				}
				else if (FFTPlan->numAxisUploads[axis_id] == 2) {
					out = i / FFTPlan->axisSplit[axis_id][1] + (i % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0];
				}
				else {
					out = (i / FFTPlan->axisSplit[axis_id][2]) / FFTPlan->axisSplit[axis_id][1] + ((i / FFTPlan->axisSplit[axis_id][2]) % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0] + (i % FFTPlan->axisSplit[axis_id][2]) * FFTPlan->axisSplit[axis_id][1] * FFTPlan->axisSplit[axis_id][0];
				}
				phaseVectors_fp64[2 * out] = (double)phaseVectors_fp128_out[2 * i];
				phaseVectors_fp64[2 * out + 1] = (double)phaseVectors_fp128_out[2 * i + 1];
			}
			resFFT = PfSolve_transferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluesteinIFFT[axis_id], bufferSize);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors_fp64);
				free(phaseVectors_fp128);
				free(phaseVectors_fp128_out);
				return resFFT;
			}
		}
		for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
			phaseVectors_fp128[2 * i + 1] = -phaseVectors_fp128[2 * i + 1];
		}
		for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
			phaseVectors_fp64[2 * i] = (double)phaseVectors_fp128[2 * i];
			phaseVectors_fp64[2 * i + 1] = (double)phaseVectors_fp128[2 * i + 1];
		}
		resFFT = PfSolve_transferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluestein[axis_id], bufferSize);
		if (resFFT != PFSOLVE_SUCCESS) {
			free(phaseVectors_fp64);
			free(phaseVectors_fp128);
			free(phaseVectors_fp128_out);
			return resFFT;
		}
		if (!app->configuration.makeInversePlanOnly) {
			fftwl_plan p;
			p = fftwl_plan_dft_1d((int)(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]), (fftwl_complex*)phaseVectors_fp128, (fftwl_complex*)phaseVectors_fp128_out, -1, FFTW_ESTIMATE);
			fftwl_execute(p);
			fftwl_destroy_plan(p);
			for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
				uint64_t out = 0;
				if (FFTPlan->numAxisUploads[axis_id] == 1) {
					out = i;
				}
				else if (FFTPlan->numAxisUploads[axis_id] == 2) {
					out = i / FFTPlan->axisSplit[axis_id][1] + (i % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0];
				}
				else {
					out = (i / FFTPlan->axisSplit[axis_id][2]) / FFTPlan->axisSplit[axis_id][1] + ((i / FFTPlan->axisSplit[axis_id][2]) % FFTPlan->axisSplit[axis_id][1]) * FFTPlan->axisSplit[axis_id][0] + (i % FFTPlan->axisSplit[axis_id][2]) * FFTPlan->axisSplit[axis_id][1] * FFTPlan->axisSplit[axis_id][0];
				}
				phaseVectors_fp64[2 * out] = (double)phaseVectors_fp128_out[2 * i];
				phaseVectors_fp64[2 * out + 1] = (double)phaseVectors_fp128_out[2 * i + 1];
			}
			resFFT = PfSolve_transferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluesteinFFT[axis_id], bufferSize);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors_fp64);
				free(phaseVectors_fp128);
				free(phaseVectors_fp128_out);
				return resFFT;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			fftwl_plan p;
			p = fftwl_plan_dft_1d((int)(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]), (fftwl_complex*)phaseVectors_fp128, (fftwl_complex*)phaseVectors_fp128_out, 1, FFTW_ESTIMATE);
			fftwl_execute(p);
			fftwl_destroy_plan(p);

			for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
				phaseVectors_fp64[2 * i] = (double)phaseVectors_fp128_out[2 * i];
				phaseVectors_fp64[2 * i + 1] = (double)phaseVectors_fp128_out[2 * i + 1];
			}
			resFFT = PfSolve_transferDataFromCPU(app, phaseVectors_fp64, &app->bufferBluesteinIFFT[axis_id], bufferSize);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors_fp64);
				free(phaseVectors_fp128);
				free(phaseVectors_fp128_out);
				return resFFT;
			}
		}
		free(phaseVectors_fp64);
		free(phaseVectors_fp128);
		free(phaseVectors_fp128_out);
	}
	else {
#endif
		PfSolveApplication kernelPreparationApplication = {};
		PfSolveConfiguration kernelPreparationConfiguration = {};
		kernelPreparationConfiguration.FFTdim = 1;
		kernelPreparationConfiguration.size[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
		kernelPreparationConfiguration.size[1] = 1;
		kernelPreparationConfiguration.size[2] = 1;
		kernelPreparationConfiguration.doublePrecision = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory);
		kernelPreparationConfiguration.useLUT = 1;
		kernelPreparationConfiguration.useLUT_4step = 1;
		kernelPreparationConfiguration.registerBoost = 1;
		kernelPreparationConfiguration.disableReorderFourStep = 1;
		kernelPreparationConfiguration.fixMinRaderPrimeFFT = 17;
		kernelPreparationConfiguration.fixMinRaderPrimeMult = 17;
		kernelPreparationConfiguration.fixMaxRaderPrimeFFT = 17;
		kernelPreparationConfiguration.fixMaxRaderPrimeMult = 17;
		kernelPreparationConfiguration.saveApplicationToString = app->configuration.saveApplicationToString;
		kernelPreparationConfiguration.loadApplicationFromString = app->configuration.loadApplicationFromString;
		if (kernelPreparationConfiguration.loadApplicationFromString) {
			kernelPreparationConfiguration.loadApplicationString = (void*)((char*)app->configuration.loadApplicationString + app->currentApplicationStringPos);
		}
		kernelPreparationConfiguration.performBandwidthBoost = (app->configuration.performBandwidthBoost > 0) ? app->configuration.performBandwidthBoost : 1;
		if (axis_id == 0) kernelPreparationConfiguration.performBandwidthBoost = 0;
		if (axis_id > 0) kernelPreparationConfiguration.considerAllAxesStrided = 1;
		if (app->configuration.tempBuffer) {
			kernelPreparationConfiguration.userTempBuffer = 1;
			kernelPreparationConfiguration.tempBuffer = app->configuration.tempBuffer;
			kernelPreparationConfiguration.tempBufferSize = app->configuration.tempBufferSize;
			kernelPreparationConfiguration.tempBufferNum = app->configuration.tempBufferNum;
		}
		kernelPreparationConfiguration.device = app->configuration.device;
#if(VKFFT_BACKEND==0)
		kernelPreparationConfiguration.queue = app->configuration.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
		kernelPreparationConfiguration.fence = app->configuration.fence;
		kernelPreparationConfiguration.commandPool = app->configuration.commandPool;
		kernelPreparationConfiguration.physicalDevice = app->configuration.physicalDevice;
		kernelPreparationConfiguration.isCompilerInitialized = 1;//compiler can be initialized before PfSolve plan creation. if not, PfSolve will create and destroy one after initialization
		kernelPreparationConfiguration.tempBufferDeviceMemory = app->configuration.tempBufferDeviceMemory;
#elif(VKFFT_BACKEND==3)
		kernelPreparationConfiguration.context = app->configuration.context;
#elif(VKFFT_BACKEND==4)
		kernelPreparationConfiguration.context = app->configuration.context;
		kernelPreparationConfiguration.commandQueue = app->configuration.commandQueue;
		kernelPreparationConfiguration.commandQueueID = app->configuration.commandQueueID;
#elif(VKFFT_BACKEND==5)
		kernelPreparationConfiguration.device = app->configuration.device;
		kernelPreparationConfiguration.queue = app->configuration.queue;
#endif			

		kernelPreparationConfiguration.inputBufferSize = &app->bufferBluesteinSize[axis_id];
		kernelPreparationConfiguration.bufferSize = &app->bufferBluesteinSize[axis_id];
		kernelPreparationConfiguration.isInputFormatted = 1;
		resFFT = initializePfSolve(&kernelPreparationApplication, kernelPreparationConfiguration);
		if (resFFT != PFSOLVE_SUCCESS) return resFFT;
		if (kernelPreparationConfiguration.loadApplicationFromString) {
			app->currentApplicationStringPos += kernelPreparationApplication.currentApplicationStringPos;
		}
		void* phaseVectors = malloc(bufferSize);
		if (!phaseVectors) {
			deletePfSolve(&kernelPreparationApplication);
			return PFSOLVE_ERROR_MALLOC_FAILED;
		}
		uint64_t phaseVectorsNonZeroSize = (((app->configuration.performDCT == 4) && (app->configuration.size[axis_id] % 2 == 0)) || ((FFTPlan->multiUploadR2C) && (axis_id == 0))) ? app->configuration.size[axis_id] / 2 : app->configuration.size[axis_id];
		if (app->configuration.performDCT == 1) phaseVectorsNonZeroSize = 2 * app->configuration.size[axis_id] - 2;

		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->configuration.makeForwardPlanOnly)) {
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
				long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
				double* phaseVectors_cast = (double*)phaseVectors;
				for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					uint64_t rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					long double angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (double)cos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (double)-sin(angle) : 0;
				}
				for (uint64_t i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
			else {
				double double_PI = 3.14159265358979323846264338327950288419716939937510;
				float* phaseVectors_cast = (float*)phaseVectors;
				for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					uint64_t rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					double angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (float)cos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (float)-sin(angle) : 0;
				}
				for (uint64_t i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
			resFFT = PfSolve_TransferDataFromCPU(app, phaseVectors, &app->bufferBluestein[axis_id], bufferSize);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
#if(VKFFT_BACKEND==0)
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
				commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VkCommandBuffer commandBuffer = {};
				res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
				if (res != 0) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
				}
				VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
				commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
				res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
				if (res != 0) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
				}
				PfSolveLaunchParams launchParams = {};
				launchParams.commandBuffer = &commandBuffer;
				launchParams.inputBuffer = &app->bufferBluestein[axis_id];
				launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
				//Record commands
				resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
				res = vkEndCommandBuffer(commandBuffer);
				if (res != 0) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
				}
				VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &commandBuffer;
				res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
				if (res != 0) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
				}
				res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
				if (res != 0) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES;
				}
				res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
				if (res != 0) {
					free(phaseVectors);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_RESET_FENCES;
				}
				vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
			}
#elif(VKFFT_BACKEND==1)
			PfSolveLaunchParams launchParams = {};
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = cudaDeviceSynchronize();
			if (res != cudaSuccess) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
#elif(VKFFT_BACKEND==2)
			PfSolveLaunchParams launchParams = {};
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = hipDeviceSynchronize();
			if (res != hipSuccess) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
#elif(VKFFT_BACKEND==3)
			PfSolveLaunchParams launchParams = {};
			launchParams.commandQueue = &commandQueue;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = clFinish(commandQueue);
			if (res != CL_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
#elif(VKFFT_BACKEND==4)
			ze_command_list_desc_t commandListDescription = {};
			commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
			ze_command_list_handle_t commandList = {};
			res = zeCommandListCreate(app->configuration.context[0], app->configuration.device[0], &commandListDescription, &commandList);
			if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			PfSolveLaunchParams launchParams = {};
			launchParams.commandList = &commandList;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = zeCommandListClose(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
			res = zeCommandListDestroy(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
			}
#elif(VKFFT_BACKEND==5)
			PfSolveLaunchParams launchParams = {};
			MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
			if (commandBuffer == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
			if (commandEncoder == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

			launchParams.commandBuffer = commandBuffer;
			launchParams.commandEncoder = commandEncoder;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			commandEncoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			commandEncoder->release();
			commandBuffer->release();
#endif
		}
		if ((FFTPlan->numAxisUploads[axis_id] > 1) && (!app->configuration.makeForwardPlanOnly)) {
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
				double* phaseVectors_cast = (double*)phaseVectors;
				for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					phaseVectors_cast[2 * i + 1] = -phaseVectors_cast[2 * i + 1];
				}

			}
			else {
				float* phaseVectors_cast = (float*)phaseVectors;
				for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					phaseVectors_cast[2 * i + 1] = -phaseVectors_cast[2 * i + 1];
				}
			}
		}
		else {
			if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
				long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
				double* phaseVectors_cast = (double*)phaseVectors;
				for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					uint64_t rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					long double angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (double)cos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (double)sin(angle) : 0;
				}
				for (uint64_t i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
			else {
				double double_PI = 3.14159265358979323846264338327950288419716939937510;
				float* phaseVectors_cast = (float*)phaseVectors;
				for (uint64_t i = 0; i < FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]; i++) {
					uint64_t rm = (i * i) % (2 * phaseVectorsNonZeroSize);
					double angle = double_PI * rm / phaseVectorsNonZeroSize;
					phaseVectors_cast[2 * i] = (i < phaseVectorsNonZeroSize) ? (float)cos(angle) : 0;
					phaseVectors_cast[2 * i + 1] = (i < phaseVectorsNonZeroSize) ? (float)sin(angle) : 0;
				}
				for (uint64_t i = 1; i < phaseVectorsNonZeroSize; i++) {
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i)] = phaseVectors_cast[2 * i];
					phaseVectors_cast[2 * (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - i) + 1] = phaseVectors_cast[2 * i + 1];
				}
			}
		}
		resFFT = PfSolve_TransferDataFromCPU(app, phaseVectors, &app->bufferBluestein[axis_id], bufferSize);
		if (resFFT != PFSOLVE_SUCCESS) {
			free(phaseVectors);
			deletePfSolve(&kernelPreparationApplication);
			return resFFT;
		}
#if(VKFFT_BACKEND==0)
		if (!app->configuration.makeInversePlanOnly) {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = {};
			res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
			}
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
			}
			PfSolveLaunchParams launchParams = {};
			launchParams.commandBuffer = &commandBuffer;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			//Record commands
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = vkEndCommandBuffer(commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES;
			}
			res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_RESET_FENCES;
			}
			vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = {};
			res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
			}
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
			}
			PfSolveLaunchParams launchParams = {};
			launchParams.commandBuffer = &commandBuffer;
			launchParams.inputBuffer = &app->bufferBluestein[axis_id];
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			//Record commands
			resFFT = PfSolveAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = vkEndCommandBuffer(commandBuffer);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES;
			}
			res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
			if (res != 0) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_RESET_FENCES;
			}
			vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
		}
#elif(VKFFT_BACKEND==1)
		PfSolveLaunchParams launchParams = {};
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = cudaDeviceSynchronize();
			if (res != cudaSuccess) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = cudaDeviceSynchronize();
			if (res != cudaSuccess) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
#elif(VKFFT_BACKEND==2)
		PfSolveLaunchParams launchParams = {};
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = hipDeviceSynchronize();
			if (res != hipSuccess) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = hipDeviceSynchronize();
			if (res != hipSuccess) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
#elif(VKFFT_BACKEND==3)
		PfSolveLaunchParams launchParams = {};
		launchParams.commandQueue = &commandQueue;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = clFinish(commandQueue);
			if (res != CL_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = clFinish(commandQueue);
			if (res != CL_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
#elif(VKFFT_BACKEND==4)
		ze_command_list_desc_t commandListDescription = {};
		commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		ze_command_list_handle_t commandList = {};
		res = zeCommandListCreate(app->configuration.context[0], app->configuration.device[0], &commandListDescription, &commandList);
		if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
		PfSolveLaunchParams launchParams = {};
		launchParams.commandList = &commandList;
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];

		if (!app->configuration.makeInversePlanOnly) {
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}

			res = zeCommandListClose(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
			res = zeCommandListReset(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
			}
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			res = zeCommandListClose(commandList);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
			}
			res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
			}
			res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
			if (res != ZE_RESULT_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
			}
		}
		res = zeCommandListDestroy(commandList);
		if (res != ZE_RESULT_SUCCESS) {
			free(phaseVectors);
			deletePfSolve(&kernelPreparationApplication);
			return PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
		}
#elif(VKFFT_BACKEND==5)
		PfSolveLaunchParams launchParams = {};
		launchParams.inputBuffer = &app->bufferBluestein[axis_id];
		if (!app->configuration.makeInversePlanOnly) {
			MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
			if (commandBuffer == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
			if (commandEncoder == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

			launchParams.commandBuffer = commandBuffer;
			launchParams.commandEncoder = commandEncoder;
			launchParams.buffer = &app->bufferBluesteinFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			commandEncoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			commandEncoder->release();
			commandBuffer->release();
		}
		if ((FFTPlan->numAxisUploads[axis_id] == 1) && (!app->configuration.makeForwardPlanOnly)) {
			MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
			if (commandBuffer == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
			MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
			if (commandEncoder == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

			launchParams.commandBuffer = commandBuffer;
			launchParams.commandEncoder = commandEncoder;
			launchParams.buffer = &app->bufferBluesteinIFFT[axis_id];
			resFFT = PfSolveAppend(&kernelPreparationApplication, 1, &launchParams);
			if (resFFT != PFSOLVE_SUCCESS) {
				free(phaseVectors);
				deletePfSolve(&kernelPreparationApplication);
				return resFFT;
			}
			commandEncoder->endEncoding();
			commandBuffer->commit();
			commandBuffer->waitUntilCompleted();
			commandEncoder->release();
			commandBuffer->release();
		}
#endif
#if(VKFFT_BACKEND==0)
		kernelPreparationApplication.configuration.isCompilerInitialized = 0;
#elif(VKFFT_BACKEND==3)
		res = clReleaseCommandQueue(commandQueue);
		if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
#endif
		if (kernelPreparationConfiguration.saveApplicationToString) {
			app->applicationBluesteinStringSize[axis_id] = kernelPreparationApplication.applicationStringSize;
			app->applicationBluesteinString[axis_id] = calloc(app->applicationBluesteinStringSize[axis_id], 1);
			if (!app->applicationBluesteinString[axis_id]) {
				deletePfSolve(&kernelPreparationApplication);
				return PFSOLVE_ERROR_MALLOC_FAILED;
			}
			memcpy(app->applicationBluesteinString[axis_id], kernelPreparationApplication.saveApplicationString, app->applicationBluesteinStringSize[axis_id]);
		}
		deletePfSolve(&kernelPreparationApplication);
		free(phaseVectors);
#ifdef PfSolve_use_FP128_Bluestein_RaderFFT
	}
#endif
	return resFFT;
}
static inline PfSolveResult PfSolveGenerateRaderFFTKernel(PfSolveApplication* app, PfSolveAxis* axis) {
	//generate Rader FFTKernel
	PfSolveResult resFFT = PFSOLVE_SUCCESS;
	if (axis->specializationConstants.useRader) {
		for (uint64_t i = 0; i < axis->specializationConstants.numRaderPrimes; i++) {
			if (axis->specializationConstants.raderContainer[i].type == 0) {
				for (uint64_t j = 0; j < app->numRaderFFTPrimes; j++) {
					if (app->rader_primes[j] == axis->specializationConstants.raderContainer[i].prime) {
						axis->specializationConstants.raderContainer[i].raderFFTkernel = app->raderFFTkernel[j];
					}
				}
				if (axis->specializationConstants.raderContainer[i].raderFFTkernel) continue;

				uint64_t write_id = app->numRaderFFTPrimes;
				app->rader_primes[write_id] = axis->specializationConstants.raderContainer[i].prime;
				app->numRaderFFTPrimes++;

				if (app->configuration.loadApplicationFromString) continue;

#ifdef PfSolve_use_FP128_Bluestein_RaderFFT
				if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
					long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
					double* raderFFTkernel = (double*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2);
					if (!raderFFTkernel) return PFSOLVE_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2;

					long double* raderFFTkernel_temp = (long double*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(long double) * 2);
					if (!raderFFTkernel_temp) return PFSOLVE_ERROR_MALLOC_FAILED;
					for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						uint64_t g_pow = 1;
						for (uint64_t t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						raderFFTkernel_temp[2 * j] = cos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
						raderFFTkernel_temp[2 * j + 1] = -sin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
					}
					fftwl_plan p;
					p = fftwl_plan_dft_1d((int)(axis->specializationConstants.raderContainer[i].prime - 1), (fftwl_complex*)raderFFTkernel_temp, (fftwl_complex*)raderFFTkernel_temp, -1, FFTW_ESTIMATE);
					fftwl_execute(p);
					fftwl_destroy_plan(p);
					for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						raderFFTkernel[2 * j] = (double)raderFFTkernel_temp[2 * j];
						raderFFTkernel[2 * j + 1] = (double)raderFFTkernel_temp[2 * j + 1];
					}
					free(raderFFTkernel_temp);
					continue;
				}
#endif
				if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
					long double double_PI = 3.14159265358979323846264338327950288419716939937510L;
					double* raderFFTkernel = (double*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2);
					if (!raderFFTkernel) return PFSOLVE_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(double) * 2;
					for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						uint64_t g_pow = 1;
						for (uint64_t t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						raderFFTkernel[2 * j] = (double)cos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
						raderFFTkernel[2 * j + 1] = (double)-sin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
					}
				}
				else {
					double double_PI = 3.14159265358979323846264338327950288419716939937510;
					float* raderFFTkernel = (float*)malloc((axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(float) * 2);
					if (!raderFFTkernel) return PFSOLVE_ERROR_MALLOC_FAILED;
					axis->specializationConstants.raderContainer[i].raderFFTkernel = (void*)raderFFTkernel;
					app->raderFFTkernel[write_id] = (void*)raderFFTkernel;
					app->rader_buffer_size[write_id] = (axis->specializationConstants.raderContainer[i].prime - 1) * sizeof(float) * 2;
					for (uint64_t j = 0; j < (axis->specializationConstants.raderContainer[i].prime - 1); j++) {//fix later
						uint64_t g_pow = 1;
						for (uint64_t t = 0; t < axis->specializationConstants.raderContainer[i].prime - 1 - j; t++) {
							g_pow = (g_pow * axis->specializationConstants.raderContainer[i].generator) % axis->specializationConstants.raderContainer[i].prime;
						}
						raderFFTkernel[2 * j] = (float)cos(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime);
						raderFFTkernel[2 * j + 1] = (float)(-sin(2.0 * g_pow * double_PI / axis->specializationConstants.raderContainer[i].prime));
					}
				}

				PfSolveApplication kernelPreparationApplication = {};
				PfSolveConfiguration kernelPreparationConfiguration = {};

				kernelPreparationConfiguration.FFTdim = 1;
				kernelPreparationConfiguration.size[0] = axis->specializationConstants.raderContainer[i].prime - 1;
				kernelPreparationConfiguration.size[1] = 1;
				kernelPreparationConfiguration.size[2] = 1;
				kernelPreparationConfiguration.doublePrecision = (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory);
				kernelPreparationConfiguration.useLUT = 1;
				kernelPreparationConfiguration.fixMinRaderPrimeFFT = 17;
				kernelPreparationConfiguration.fixMinRaderPrimeMult = 17;
				kernelPreparationConfiguration.fixMaxRaderPrimeFFT = 17;
				kernelPreparationConfiguration.fixMaxRaderPrimeMult = 17;

				kernelPreparationConfiguration.device = app->configuration.device;
#if(VKFFT_BACKEND==0)
				kernelPreparationConfiguration.queue = app->configuration.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
				kernelPreparationConfiguration.fence = app->configuration.fence;
				kernelPreparationConfiguration.commandPool = app->configuration.commandPool;
				kernelPreparationConfiguration.physicalDevice = app->configuration.physicalDevice;
				kernelPreparationConfiguration.isCompilerInitialized = 1;//compiler can be initialized before PfSolve plan creation. if not, PfSolve will create and destroy one after initialization
				kernelPreparationConfiguration.tempBufferDeviceMemory = app->configuration.tempBufferDeviceMemory;
#elif(VKFFT_BACKEND==3)
				kernelPreparationConfiguration.context = app->configuration.context;
#elif(VKFFT_BACKEND==4)
				kernelPreparationConfiguration.context = app->configuration.context;
				kernelPreparationConfiguration.commandQueue = app->configuration.commandQueue;
				kernelPreparationConfiguration.commandQueueID = app->configuration.commandQueueID;
#elif(VKFFT_BACKEND==5)
				kernelPreparationConfiguration.device = app->configuration.device;
				kernelPreparationConfiguration.queue = app->configuration.queue;
#endif			

				uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * kernelPreparationConfiguration.size[0] * kernelPreparationConfiguration.size[1] * kernelPreparationConfiguration.size[2];
				if (kernelPreparationConfiguration.doublePrecision) bufferSize *= sizeof(double) / sizeof(float);

				kernelPreparationConfiguration.bufferSize = &bufferSize;
				resFFT = initializePfSolve(&kernelPreparationApplication, kernelPreparationConfiguration);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;

#if(VKFFT_BACKEND==0)
				VkDeviceMemory bufferRaderFFTDeviceMemory;
				VkBuffer bufferRaderFFT;
#elif(VKFFT_BACKEND==1)
				void* bufferRaderFFT;
#elif(VKFFT_BACKEND==2)
				void* bufferRaderFFT;
#elif(VKFFT_BACKEND==3)
				cl_mem bufferRaderFFT;
#elif(VKFFT_BACKEND==4)
				void* bufferRaderFFT;
#elif(VKFFT_BACKEND==5)
				MTL::Buffer* bufferRaderFFT;
#endif
#if(VKFFT_BACKEND==0)
				PfResult res = VK_SUCCESS;
				resFFT = allocateBufferVulkan(app, &bufferRaderFFT, &bufferRaderFFTDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
				if (resFFT != PFSOLVE_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
				cudaError_t res = cudaSuccess;
				res = cudaMalloc(&bufferRaderFFT, bufferSize);
				if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
				hipError_t res = hipSuccess;
				res = hipMalloc(&bufferRaderFFT, bufferSize);
				if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==3)
				cl_int res = CL_SUCCESS;
				bufferRaderFFT = clCreateBuffer(app->configuration.context[0], CL_MEM_READ_WRITE, bufferSize, 0, &res);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
				cl_command_queue commandQueue = clCreateCommandQueue(app->configuration.context[0], app->configuration.device[0], 0, &res);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
#elif(VKFFT_BACKEND==4)
				ze_result_t res = ZE_RESULT_SUCCESS;
				ze_device_mem_alloc_desc_t device_desc = {};
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(app->configuration.context[0], &device_desc, bufferSize, sizeof(float), app->configuration.device[0], &bufferRaderFFT);
				if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==5)
				bufferRaderFFT = app->configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
#endif

				resFFT = PfSolve_TransferDataFromCPU(app, axis->specializationConstants.raderContainer[i].raderFFTkernel, &bufferRaderFFT, bufferSize);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
#if(VKFFT_BACKEND==0)
				{
					VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
					commandBufferAllocateInfo.commandPool = kernelPreparationApplication.configuration.commandPool[0];
					commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
					commandBufferAllocateInfo.commandBufferCount = 1;
					VkCommandBuffer commandBuffer = {};
					res = vkAllocateCommandBuffers(kernelPreparationApplication.configuration.device[0], &commandBufferAllocateInfo, &commandBuffer);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
					}
					VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
					commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
					res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
					}
					PfSolveLaunchParams launchParams = {};
					launchParams.commandBuffer = &commandBuffer;
					launchParams.buffer = &bufferRaderFFT;
					//Record commands
					resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
					if (resFFT != PFSOLVE_SUCCESS) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return resFFT;
					}
					res = vkEndCommandBuffer(commandBuffer);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
					}
					VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
					submitInfo.commandBufferCount = 1;
					submitInfo.pCommandBuffers = &commandBuffer;
					res = vkQueueSubmit(kernelPreparationApplication.configuration.queue[0], 1, &submitInfo, kernelPreparationApplication.configuration.fence[0]);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
					}
					res = vkWaitForFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence, VK_TRUE, 100000000000);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES;
					}
					res = vkResetFences(kernelPreparationApplication.configuration.device[0], 1, kernelPreparationApplication.configuration.fence);
					if (res != 0) {
						free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
						deletePfSolve(&kernelPreparationApplication);
						return PFSOLVE_ERROR_FAILED_TO_RESET_FENCES;
					}
					vkFreeCommandBuffers(kernelPreparationApplication.configuration.device[0], kernelPreparationApplication.configuration.commandPool[0], 1, &commandBuffer);
				}
#elif(VKFFT_BACKEND==1)
				PfSolveLaunchParams launchParams = {};
				launchParams.buffer = &bufferRaderFFT;
				resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
				res = cudaDeviceSynchronize();
				if (res != cudaSuccess) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
				}
#elif(VKFFT_BACKEND==2)
				PfSolveLaunchParams launchParams = {};
				launchParams.buffer = &bufferRaderFFT;
				resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
				res = hipDeviceSynchronize();
				if (res != hipSuccess) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
				}
#elif(VKFFT_BACKEND==3)
				PfSolveLaunchParams launchParams = {};
				launchParams.commandQueue = &commandQueue;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
				res = clFinish(commandQueue);
				if (res != CL_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
				}
#elif(VKFFT_BACKEND==4)
				ze_command_list_desc_t commandListDescription = {};
				commandListDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
				ze_command_list_handle_t commandList = {};
				res = zeCommandListCreate(app->configuration.context[0], app->configuration.device[0], &commandListDescription, &commandList);
				if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
				PfSolveLaunchParams launchParams = {};
				launchParams.commandList = &commandList;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
				res = zeCommandListClose(commandList);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER;
				}
				res = zeCommandQueueExecuteCommandLists(app->configuration.commandQueue[0], 1, &commandList, 0);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE;
				}
				res = zeCommandQueueSynchronize(app->configuration.commandQueue[0], UINT32_MAX);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
				}
				res = zeCommandListDestroy(commandList);
				if (res != ZE_RESULT_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST;
				}
#elif(VKFFT_BACKEND==5)
				PfSolveLaunchParams launchParams = {};
				MTL::CommandBuffer* commandBuffer = app->configuration.queue->commandBuffer();
				if (commandBuffer == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;
				MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
				if (commandEncoder == 0) return PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST;

				launchParams.commandBuffer = commandBuffer;
				launchParams.commandEncoder = commandEncoder;
				launchParams.buffer = &bufferRaderFFT;
				resFFT = PfSolveAppend(&kernelPreparationApplication, -1, &launchParams);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}
				commandEncoder->endEncoding();
				commandBuffer->commit();
				commandBuffer->waitUntilCompleted();
				commandEncoder->release();
				commandBuffer->release();
#endif
				resFFT = PfSolve_TransferDataToCPU(&kernelPreparationApplication, axis->specializationConstants.raderContainer[i].raderFFTkernel, &bufferRaderFFT, bufferSize);
				if (resFFT != PFSOLVE_SUCCESS) {
					free(axis->specializationConstants.raderContainer[i].raderFFTkernel);
					deletePfSolve(&kernelPreparationApplication);
					return resFFT;
				}

#if(VKFFT_BACKEND==0)
				kernelPreparationApplication.configuration.isCompilerInitialized = 0;
#elif(VKFFT_BACKEND==3)
				res = clReleaseCommandQueue(commandQueue);
				if (res != CL_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
#endif
#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(app->configuration.device[0], bufferRaderFFT, 0);
				vkFreeMemory(app->configuration.device[0], bufferRaderFFTDeviceMemory, 0);
#elif(VKFFT_BACKEND==1)
				cudaFree(bufferRaderFFT);
#elif(VKFFT_BACKEND==2)
				hipFree(bufferRaderFFT);
#elif(VKFFT_BACKEND==3)
				clReleaseMemObject(bufferRaderFFT);
#elif(VKFFT_BACKEND==4)
				zeMemFree(app->configuration.context[0], bufferRaderFFT);
#elif(VKFFT_BACKEND==5)
				bufferRaderFFT->release();
#endif
				deletePfSolve(&kernelPreparationApplication);
			}
		}
		if (app->configuration.loadApplicationFromString) {
			uint64_t offset = 0;
			for (uint64_t i = 0; i < app->numRaderFFTPrimes; i++) {
				uint64_t current_size = 0;
				if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
					current_size = (app->rader_primes[i] - 1) * sizeof(double) * 2;
				}
				else {
					current_size = (app->rader_primes[i] - 1) * sizeof(float) * 2;
				}
				if (!app->raderFFTkernel[i]) {
					app->raderFFTkernel[i] = (void*)malloc(current_size);
					if (!app->raderFFTkernel[i]) return PFSOLVE_ERROR_MALLOC_FAILED;
					memcpy(app->raderFFTkernel[i], (char*)app->configuration.loadApplicationString + app->applicationStringOffsetRader + offset, current_size);
				}
				for (uint64_t j = 0; j < axis->specializationConstants.numRaderPrimes; j++) {
					if ((app->rader_primes[i] == axis->specializationConstants.raderContainer[j].prime) && (axis->specializationConstants.raderContainer[j].type == 0))
						axis->specializationConstants.raderContainer[j].raderFFTkernel = app->raderFFTkernel[i];
				}
				offset += current_size;
			}
		}
	}
	return resFFT;
}

#endif
