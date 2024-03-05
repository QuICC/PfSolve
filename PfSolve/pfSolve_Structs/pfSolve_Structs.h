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

#ifndef PFSOLVE_STRUCTS_H
#define PFSOLVE_STRUCTS_H

#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
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
#elif(VKFFT_BACKEND==4)
#include <ze_api.h>
#elif(VKFFT_BACKEND==5)
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#endif

//unified PfSolve container
typedef struct PfContainer PfContainer;
typedef union PfData PfData;

#define RUNTIME_MSIZE 1
#define RUNTIME_OFFSETM 2
#define RUNTIME_OFFSETV 4
#define RUNTIME_OFFSETSOLUTION 8
#define RUNTIME_INPUTZEROPAD 16
#define RUNTIME_OUTPUTZEROPAD 32
#define RUNTIME_SCALEC 64
#define RUNTIME_INPUTBUFFERSTRIDE 128
#define RUNTIME_OUTPUTBUFFERSTRIDE 256

#define BLOCK_READ_REAL_WRITE_REAL 1
#define BLOCK_READ_COMPLEX_R_WRITE_REAL 2
#define BLOCK_READ_COMPLEX_I_WRITE_REAL 3
#define BLOCK_READ_COMPLEX_STRIDED_WRITE_COMPLEX_PACKED 4
#define BLOCK_READ_REAL_WRITE_COMPLEX_R 5
#define BLOCK_READ_REAL_WRITE_COMPLEX_I 6 
#define BLOCK_READ_COMPLEX_PACKED_WRITE_COMPLEX_STRIDED 7
#define BLOCK_ADD_OUTPUTBUFFER_VALUES 20
#define BLOCK_SCALEC 100
#define BLOCK_SCALED 200
#define BLOCK_SCALESPHLAPLA 300
#define BLOCK_SCALESPHLAPLB 400

#define JWT_DISABLE_TRIDIAGONAL_SOLVE 1
#define JWT_DISABLE_MATVECMUL_SOLVE 2
#define JWT_DIAGONAL_MATVECMUL 3
#define JWT_USE_PARALLEL_THOMAS 10

typedef union PfData {
	pfINT i; // int
	pfLD d; // long double
	PfContainer* c; // [2] complex
	PfContainer* dd; // [2] double-double __ibm128
} PfData;
struct PfContainer{
	int type; // 0 - uninitialized
			  // 1 - int, 2 - float, 3 - complex float; 
			  // + X0: 0 - half, 1 - float, 2 - double, 3 - double-double, 4 - fp128 (if available): - precision identifiers (only for strings now, all number values are in max pfLD precision for simplicity)
			  // 100 + X - variable name, containing same type as in X
			  
	PfData data; // memory contents of the container
	char* name; // name of the container
	int size; //  bytes allcoated in name
};


typedef struct {
	//WHDCN layout

	//required parameters:
	uint64_t FFTdim; //FFT dimensionality (1, 2 or 3)
	uint64_t size[3]; // WHD -system dimensions

#if(VKFFT_BACKEND==0)
	VkPhysicalDevice* physicalDevice;//pointer to Vulkan physical device, obtained from vkEnumeratePhysicalDevices
	VkDevice* device;//pointer to Vulkan device, created with vkCreateDevice
	VkQueue* queue;//pointer to Vulkan queue, created with vkGetDeviceQueue
	VkCommandPool* commandPool;//pointer to Vulkan command pool, created with vkCreateCommandPool
	VkFence* fence;//pointer to Vulkan fence, created with vkCreateFence
	uint64_t isCompilerInitialized;//specify if glslang compiler has been intialized before (0 - off, 1 - on). Default 0
#elif(VKFFT_BACKEND==1)
	CUdevice* device;//pointer to CUDA device, obtained from cuDeviceGet
	//CUcontext* context;//pointer to CUDA context, obtained from cuDeviceGet
	cudaStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
	uint64_t num_streams;//try to submit CUDA kernels in multiple streams for asynchronous execution. Default 1
#elif(VKFFT_BACKEND==2)
	hipDevice_t* device;//pointer to HIP device, obtained from hipDeviceGet
	//hipCtx_t* context;//pointer to HIP context, obtained from hipDeviceGet
	hipStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
	uint64_t num_streams;//try to submit HIP kernels in multiple streams for asynchronous execution. Default 1
#elif(VKFFT_BACKEND==3)
	cl_platform_id* platform;//not required
	cl_device_id* device;
	cl_context* context;
#elif(VKFFT_BACKEND==4)
	ze_device_handle_t* device;
	ze_context_handle_t* context;
	ze_command_queue_handle_t* commandQueue;
	uint32_t commandQueueID;
#elif(VKFFT_BACKEND==5)
	MTL::Device* device;
	MTL::CommandQueue* queue;
#endif

	//data parameters:
	uint64_t userTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation (0 - off, 1 - on)

	uint64_t bufferNum;//multiple buffer sequence storage is Vulkan only. Default 1
	uint64_t tempBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation
	uint64_t inputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isInputFormatted is enabled
	uint64_t outputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isOutputFormatted is enabled
	uint64_t kernelNum;//multiple buffer sequence storage is Vulkan only. Default 1, if performConvolution is enabled

	//sizes are obligatory in Vulkan backend, optional in others
	uint64_t* bufferSize;//array of buffers sizes in bytes
	uint64_t* tempBufferSize;//array of temp buffers sizes in bytes. Default set to bufferSize sum, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation
	uint64_t* inputBufferSize;//array of input buffers sizes in bytes, if isInputFormatted is enabled
	uint64_t* outputBufferSize;//array of output buffers sizes in bytes, if isOutputFormatted is enabled
	uint64_t* kernelSize;//array of kernel buffers sizes in bytes, if performConvolution is enabled

#if(VKFFT_BACKEND==0)
	VkBuffer* buffer;//pointer to array of buffers (or one buffer) used for computations
	VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same sum size or bigger as buffer (can be split in multiple). Default 0. Setting to non zero value enables manual user allocation
	VkBuffer* inputBuffer;//pointer to array of input buffers (or one buffer) used to read data from if isInputFormatted is enabled
	VkBuffer* outputBuffer;//pointer to array of output buffers (or one buffer) used for write data to if isOutputFormatted is enabled
	VkBuffer* kernel;//pointer to array of kernel buffers (or one buffer) used for read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==1)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==2)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==3)
	cl_mem* buffer;//pointer to device buffer used for computations
	cl_mem* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	cl_mem* inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	cl_mem* outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	cl_mem* kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==4)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==5)
	MTL::Buffer** buffer;//pointer to device buffer used for computations
	MTL::Buffer** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	MTL::Buffer** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	MTL::Buffer** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	MTL::Buffer** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#endif
	uint64_t bufferOffset;//specify if PfSolve has to offset the first element position inside the buffer. In bytes. Default 0 
	uint64_t tempBufferOffset;//specify if PfSolve has to offset the first element position inside the temp buffer. In bytes. Default 0 
	uint64_t inputBufferOffset;//specify if PfSolve has to offset the first element position inside the input buffer. In bytes. Default 0 
	uint64_t outputBufferOffset;//specify if PfSolve has to offset the first element position inside the output buffer. In bytes. Default 0
	uint64_t kernelOffset;//specify if PfSolve has to offset the first element position inside the kernel. In bytes. Default 0
	uint64_t specifyOffsetsAtLaunch;//specify if offsets will be selected with launch parameters PfSolveLaunchParams (0 - off, 1 - on). Default 0

	//optional: (default 0 if not stated otherwise)
#if(VKFFT_BACKEND==0)
	VkPipelineCache* pipelineCache;//pointer to Vulkan pipeline cache
#endif
	double s_dx;
	double s_dy;
	double s_dz;
	double s_dt_D;
	int64_t compute_Pf;
	int64_t finiteDifferences;
	uint64_t logicBlock[3]; // logic block per warp

	
	int upperBanded;
	int64_t M_size_pow2;

	int jw_type;
	int block;
	int lshift;
	int LDA; 
	int KU;
	int KL;

	int64_t jw_control_bitmask;

	int64_t M_size; //0
	int64_t offsetM; //1
	int64_t offsetV; //2
	int64_t offsetSolution; //3
	int64_t inputZeropad[2]; //4
	int64_t outputZeropad[2]; //5
	double scaleC; //6
	//inputbufferstride 7
	//outputbufferstride 8

	uint64_t coalescedMemory;//in bytes, for Nvidia and AMD is equal to 32, Intel is equal 64, scaled for half precision. Gonna work regardles, but if specified by user correctly, the performance will be higher.
	uint64_t aimThreads;//aim at this many threads per block. Default 128
	uint64_t numSharedBanks;//how many banks shared memory has. Default 32
	uint64_t inverseReturnToInputBuffer;//return data to the input buffer in inverse transform (0 - off, 1 - on). isInputFormatted must be enabled
	uint64_t numberBatches;// N - used to perform multiple batches of initial data. Default 1
	uint64_t useUint64;// use 64-bit addressing mode in generated kernels
	uint64_t omitDimension[3];//disable FFT for this dimension (0 - FFT enabled, 1 - FFT disabled). Default 0. Doesn't work for R2C dimension 0 for now. Doesn't work with convolutions.
	uint64_t performBandwidthBoost;//try to reduce coalsesced number by a factor of X to get bigger sequence in one upload for strided axes. Default: -1 for DCT, 2 for Bluestein's algorithm (or -1 if DCT), 0 otherwise 
	uint64_t groupedBatch[3];

	uint64_t doublePrecision; //perform calculations in double precision (0 - off, 1 - on).
	pfUINT quadDoubleDoublePrecision; //perform calculations in double-double quad precision (0 - off, 1 - on).
	pfUINT quadDoubleDoublePrecisionDoubleMemory; //perform calculations in double-double quad precision, while all memory storage is done in FP64.
	uint64_t halfPrecision; //perform calculations in half precision (0 - off, 1 - on)
	uint64_t halfPrecisionMemoryOnly; //use half precision only as input/output buffer. Input/Output have to be allocated as half, buffer/tempBuffer have to be allocated as float (out of place mode only). Specify isInputFormatted and isOutputFormatted to use (0 - off, 1 - on)
	uint64_t doublePrecisionFloatMemory; //use FP64 precision for all calculations, while all memory storage is done in FP32.

	uint64_t performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)
	uint64_t performDCT; //perform DCT transformation (X - DCT type, 1-4)
	uint64_t disableMergeSequencesR2C; //disable merging of two real sequences to reduce calculations (0 - off, 1 - on)
	uint64_t normalize; //normalize inverse transform (0 - off, 1 - on)
	uint64_t disableReorderFourStep; // disables unshuffling of Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on)
	int64_t useLUT; //switches from calculating sincos to using precomputed LUT tables (-1 - off, 0 - auto, 1 - on). Configured by initialization routine
	int64_t useLUT_4step; //switches from calculating sincos to using precomputed LUT tables for intermediate roots of 1 in the Four-step FFT algorithm. (-1 - off, 0 - auto, 1 - on). Configured by initialization routine
	uint64_t makeForwardPlanOnly; //generate code only for forward FFT (0 - off, 1 - on)
	uint64_t makeInversePlanOnly; //generate code only for inverse FFT (0 - off, 1 - on)

	uint64_t bufferStride[3];//buffer strides - default set to x - x*y - x*y*z values
	uint64_t isInputFormatted; //specify if input buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)
	uint64_t isOutputFormatted; //specify if output buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)
	uint64_t inputBufferStride[3];//input buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values
	uint64_t outputBufferStride[3];//output buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values

	uint64_t considerAllAxesStrided;//will create plan for nonstrided axis similar as a strided axis - used with disableReorderFourStep to get the same layout for Bluestein kernel (0 - off, 1 - on)
	uint64_t keepShaderCode;//will keep shader code and print all executed shaders during the plan execution in order (0 - off, 1 - on)
	uint64_t printMemoryLayout;//will print order of buffers used in shaders (0 - off, 1 - on)

	int disableCaching;
	uint64_t saveApplicationToString;//will save all compiled binaries to PfSolveApplication.saveApplicationString (will be allocated by PfSolve, deallocated with deletePfSolve call). Currently disabled in Metal backend. (0 - off, 1 - on)

	uint64_t loadApplicationFromString;//will load all binaries from loadApplicationString instead of recompiling them (must be allocated by user, must contain what saveApplicationToString call generated previously in PfSolveApplication.saveApplicationString). Currently disabled in Metal backend. (0 - off, 1 - on). Mutually exclusive with saveApplicationToString
	void* loadApplicationString;//memory binary array through which user can load PfSolve binaries, must be provided by user if loadApplicationFromString = 1. Use rb/wb flags to load/save.

	uint64_t disableSetLocale;//disables all PfSolve attempts to set locale to C - user must ensure that PfSolve has C locale during the plan initialization. This option is needed for multithreading. Default 0.

	//optional Bluestein optimizations: (default 0 if not stated otherwise)
	uint64_t fixMaxRadixBluestein;//controls the padding of sequences in Bluestein convolution. If specified, padded sequence will be made of up to fixMaxRadixBluestein primes. Default: 2 for CUDA and Vulkan/OpenCL/HIP up to 1048576 combined dimension FFT system, 7 for Vulkan/OpenCL/HIP past after. Min = 2, Max = 13.
	uint64_t forceBluesteinSequenceSize;// force the sequence size to pad to in Bluestein's algorithm. Must be at least 2*N-1 and decomposable with primes 2-13.
	uint64_t useCustomBluesteinPaddingPattern;// force the sequence sizes to pad to in Bluestein's algorithm, but on a range. This number specifies the number of elements in primeSizes and in paddedSizes arrays. primeSizes - array of non-decomposable as radix scheme sizes - 17, 23, 31 etc. 
											  // paddedSizes - array of lengths to pad to. paddedSizes[i] will be the padding size for all non-decomposable sequences from primeSizes[i] to primeSizes[i+1] (will use default scheme after last one) - 42, 60, 64 for primeSizes before and 37+ will use default scheme (for example). Default is vendor and API-based specified in autoCustomBluesteinPaddingPattern.
	uint64_t* primeSizes; // described in useCustomBluesteinPaddingPattern
	uint64_t* paddedSizes; // described in useCustomBluesteinPaddingPattern

	uint64_t fixMinRaderPrimeMult;//start direct multiplication Rader's algorithm for radix primes from this number. This means that PfSolve will inline custom Rader kernels if sequence is divisible by these primes. Default is 17, as PfSolve has kernels for 2-13. If you make it less than 13, PfSolve will switch from these kernels to Rader.
	uint64_t fixMaxRaderPrimeMult;//switch from Mult Rader's algorithm for radix primes from this number. Current limitation for Rader is maxThreadNum/2+1, realistically you would want to switch somewhere on 30-100 range. Default is vendor-specific (currently ~40)

	uint64_t fixMinRaderPrimeFFT;//start FFT convolution version of Rader for radix primes from this number. Better than direct multiplication version for almost all primes (except small ones, like 17-23 on some GPUs). Must be bigger or equal to fixMinRaderPrimeMult. Deafult 29 on AMD and 17 on other GPUs. 
	uint64_t fixMaxRaderPrimeFFT;//switch to Bluestein's algorithm for radix primes from this number. Switch may happen earlier if prime can't fit in shared memory. Default is 16384, which is bigger than most current GPU's shared memory.

	int numConsecutiveJWIterations;
	int useMultipleInputBuffers;

	//optional zero padding control parameters: (default 0 if not stated otherwise)
	uint64_t performZeropadding[3]; // don't read some data/perform computations if some input sequences are zeropadded for each axis (0 - off, 1 - on)
	uint64_t fft_zeropad_left[3];//specify start boundary of zero block in the system for each axis
	uint64_t fft_zeropad_right[3];//specify end boundary of zero block in the system for each axis
	uint64_t frequencyZeroPadding; //set to 1 if zeropadding of frequency domain, default 0 - spatial zeropadding

	//optional convolution control parameters: (default 0 if not stated otherwise)
	uint64_t performConvolution; //perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter
	uint64_t conjugateConvolution;//0 off, 1 - conjugation of the sequence FFT is currently done on, 2 - conjugation of the convolution kernel
	uint64_t crossPowerSpectrumNormalization;//normalize the FFT x kernel multiplication in frequency domain
	uint64_t coordinateFeatures; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
	uint64_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
	uint64_t symmetricKernel; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
	uint64_t numberKernels;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
	uint64_t kernelConvolution;// specify if this application is used to create kernel for convolution, so it has the same properties. performConvolution has to be set to 0 for kernel creation

	//register overutilization (experimental): (default 0 if not stated otherwise)
	uint64_t registerBoost; //specify if register file size is bigger than shared memory and can be used to extend it X times (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4 to emulate 128KB of shared memory). Default 1
	uint64_t registerBoostNonPow2; //specify if register overutilization should be used on non power of 2 sequences (0 - off, 1 - on)
	uint64_t registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same definition as registerBoost. Default 1

	//not used techniques:
	uint64_t swapTo3Stage4Step; //specify at which number to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled. Must be at least 131072
	uint64_t devicePageSize;//in KB, the size of a page on the GPU. Setting to 0 disables local buffer split in pages
	uint64_t localPageSize;//in KB, the size to split page into if sequence spans multiple devicePageSize pages

	//automatically filled based on device info (still can be reconfigured by user):
	uint64_t computeCapabilityMajor; // CUDA/HIP compute capability of the device
	uint64_t computeCapabilityMinor; // CUDA/HIP compute capability of the device
	uint64_t maxComputeWorkGroupCount[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
	uint64_t maxComputeWorkGroupSize[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
	uint64_t maxThreadsNum; //max number of threads from VkPhysicalDeviceLimits
	uint64_t sharedMemorySizeStatic; //available for static allocation shared memory size, in bytes
	uint64_t sharedMemorySize; //available for allocation shared memory size, in bytes
	uint64_t sharedMemorySizePow2; //power of 2 which is less or equal to sharedMemorySize, in bytes
	uint64_t warpSize; //number of threads per warp/wavefront.
	uint64_t halfThreads;//Intel fix
	uint64_t allocateTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Parameter to check if it has been allocated
	uint64_t reorderFourStep; // unshuffle Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on). Default 1.
	int64_t maxCodeLength; //specify how big can be buffer used for code generation (in char). Default 4000000 chars.
	int64_t maxTempLength; //specify how big can be buffer used for intermediate string sprintfs be (in char). Default 5000 chars. If code segfaults for some reason - try increasing this number.
	uint64_t autoCustomBluesteinPaddingPattern; // default value for useCustomBluesteinPaddingPattern
	uint64_t useRaderUintLUT; // allocate additional LUT to store g_pow
	uint64_t vendorID; // vendorID 0x10DE - NVIDIA, 0x8086 - Intel, 0x1002 - AMD, etc.
#if(VKFFT_BACKEND==0)
	VkDeviceMemory tempBufferDeviceMemory;//Filled at app creation
	VkCommandBuffer* commandBuffer;//Filled at app execution
	VkMemoryBarrier* memory_barrier;//Filled at app creation
#elif(VKFFT_BACKEND==1)
	cudaEvent_t* stream_event;//Filled at app creation
	uint64_t streamCounter;//Filled at app creation
	uint64_t streamID;//Filled at app creation
#elif(VKFFT_BACKEND==2)
	hipEvent_t* stream_event;//Filled at app creation
	uint64_t streamCounter;//Filled at app creation
	uint64_t streamID;//Filled at app creation
	int64_t  useStrict32BitAddress; // guarantee 32 bit addresses in bytes instead of number of elements. This results in fewer instructions generated. -1: Disable, 0: Infer based on size, 1: enable. Has no effect with useUint64.
#elif(VKFFT_BACKEND==3)
	cl_command_queue* commandQueue;
#elif(VKFFT_BACKEND==4)
	ze_command_list_handle_t* commandList;//Filled at app execution
#elif(VKFFT_BACKEND==5)
	MTL::CommandBuffer* commandBuffer;//Filled at app execution
	MTL::ComputeCommandEncoder* commandEncoder;//Filled at app execution
#endif
} PfSolveConfiguration;//parameters specified at plan creation

typedef struct {
#if(VKFFT_BACKEND==0)
	VkCommandBuffer* commandBuffer;//commandBuffer to which FFT is appended

	VkBuffer* buffer;//pointer to array of buffers (or one buffer) used for computations
	VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same sum size or bigger as buffer (can be split in multiple). Default 0. Setting to non zero value enables manual user allocation
	VkBuffer* inputBuffer;//pointer to array of input buffers (or one buffer) used to read data from if isInputFormatted is enabled
	VkBuffer* outputBuffer;//pointer to array of output buffers (or one buffer) used for write data to if isOutputFormatted is enabled
	VkBuffer* kernel;//pointer to array of kernel buffers (or one buffer) used for read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==1)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==2)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==3)
	cl_command_queue* commandQueue;//commandBuffer to which FFT is appended

	cl_mem* buffer;//pointer to device buffer used for computations
	cl_mem* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	cl_mem* inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	cl_mem* outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	cl_mem* kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==4)
	ze_command_list_handle_t* commandList;//commandList to which FFT is appended

	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==5)
	MTL::CommandBuffer* commandBuffer;//commandBuffer to which FFT is appended
	MTL::ComputeCommandEncoder* commandEncoder;//encoder associated with commandBuffer

	MTL::Buffer** buffer;//pointer to array of buffers (or one buffer) used for computations
	MTL::Buffer** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same sum size or bigger as buffer (can be split in multiple). Default 0. Setting to non zero value enables manual user allocation
	MTL::Buffer** inputBuffer;//pointer to array of input buffers (or one buffer) used to read data from if isInputFormatted is enabled
	MTL::Buffer** outputBuffer;//pointer to array of output buffers (or one buffer) used for write data to if isOutputFormatted is enabled
	MTL::Buffer** kernel;//pointer to array of kernel buffers (or one buffer) used for read kernel data from if performConvolution is enabled
#endif
	//following parameters can be specified during kernels launch, if specifyOffsetsAtLaunch parameter was enabled during the initializePfSolve call
	uint64_t bufferOffset;//specify if PfSolve has to offset the first element position inside the buffer. In bytes. Default 0 
	uint64_t tempBufferOffset;//specify if PfSolve has to offset the first element position inside the temp buffer. In bytes. Default 0 
	uint64_t inputBufferOffset;//specify if PfSolve has to offset the first element position inside the input buffer. In bytes. Default 0 
	uint64_t outputBufferOffset;//specify if PfSolve has to offset the first element position inside the output buffer. In bytes. Default 0
	uint64_t kernelOffset;//specify if PfSolve has to offset the first element position inside the kernel. In bytes. Default 0

	uint64_t M_size; // RUNTIME_MSIZE - not implemented 
	uint64_t offsetM; // RUNTIME_OFFSETM
	uint64_t offsetV; // RUNTIME_OFFSETV
	uint64_t offsetSolution; // RUNTIME_OFFSETSOLUTION
	uint64_t inputZeropad[2]; // RUNTIME_INPUTZEROPAD
	uint64_t outputZeropad[2]; // RUNTIME_OUTPUTZEROPAD
	long double scaleC; // RUNTIME_SCALEC
	uint64_t inputBufferStride; // RUNTIME_INPUTBUFFERSTRIDE
	uint64_t outputBufferStride; // RUNTIME_OUTPUTBUFFERSTRIDE

	uint64_t batchSize;

} PfSolveLaunchParams;//parameters specified at plan execution
typedef enum PfSolveResult {
	PFSOLVE_SUCCESS = 0,
	PFSOLVE_ERROR_MALLOC_FAILED = 1,
	PFSOLVE_ERROR_INSUFFICIENT_CODE_BUFFER = 2,
	PFSOLVE_ERROR_INSUFFICIENT_TEMP_BUFFER = 3,
	PFSOLVE_ERROR_PLAN_NOT_INITIALIZED = 4,
	PFSOLVE_ERROR_NULL_TEMP_PASSED = 5,
	PFSOLVE_ERROR_MATH_FAILED = 6,
	PFSOLVE_ERROR_INVALID_PHYSICAL_DEVICE = 1001,
	PFSOLVE_ERROR_INVALID_DEVICE = 1002,
	PFSOLVE_ERROR_INVALID_QUEUE = 1003,
	PFSOLVE_ERROR_INVALID_COMMAND_POOL = 1004,
	PFSOLVE_ERROR_INVALID_FENCE = 1005,
	PFSOLVE_ERROR_ONLY_FORWARD_FFT_INITIALIZED = 1006,
	PFSOLVE_ERROR_ONLY_INVERSE_FFT_INITIALIZED = 1007,
	PFSOLVE_ERROR_INVALID_CONTEXT = 1008,
	PFSOLVE_ERROR_INVALID_PLATFORM = 1009,
	PFSOLVE_ERROR_ENABLED_saveApplicationToString = 1010,
	PFSOLVE_ERROR_EMPTY_FILE = 1011,
	PFSOLVE_ERROR_EMPTY_FFTdim = 2001,
	PFSOLVE_ERROR_EMPTY_size = 2002,
	PFSOLVE_ERROR_EMPTY_bufferSize = 2003,
	PFSOLVE_ERROR_EMPTY_buffer = 2004,
	PFSOLVE_ERROR_EMPTY_tempBufferSize = 2005,
	PFSOLVE_ERROR_EMPTY_tempBuffer = 2006,
	PFSOLVE_ERROR_EMPTY_inputBufferSize = 2007,
	PFSOLVE_ERROR_EMPTY_inputBuffer = 2008,
	PFSOLVE_ERROR_EMPTY_outputBufferSize = 2009,
	PFSOLVE_ERROR_EMPTY_outputBuffer = 2010,
	PFSOLVE_ERROR_EMPTY_kernelSize = 2011,
	PFSOLVE_ERROR_EMPTY_kernel = 2012,
	PFSOLVE_ERROR_EMPTY_applicationString = 2013,
	PFSOLVE_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays = 2014,
	PFSOLVE_ERROR_UNSUPPORTED_RADIX = 3001,
	PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH = 3002,
	PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH_R2C = 3003,
	PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH_DCT = 3004,
	PFSOLVE_ERROR_UNSUPPORTED_FFT_OMIT = 3005,
	PFSOLVE_ERROR_FAILED_TO_ALLOCATE = 4001,
	PFSOLVE_ERROR_FAILED_TO_MAP_MEMORY = 4002,
	PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS = 4003,
	PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER = 4004,
	PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER = 4005,
	PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE = 4006,
	PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES = 4007,
	PFSOLVE_ERROR_FAILED_TO_RESET_FENCES = 4008,
	PFSOLVE_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL = 4009,
	PFSOLVE_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT = 4010,
	PFSOLVE_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS = 4011,
	PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT = 4012,
	PFSOLVE_ERROR_FAILED_SHADER_PREPROCESS = 4013,
	PFSOLVE_ERROR_FAILED_SHADER_PARSE = 4014,
	PFSOLVE_ERROR_FAILED_SHADER_LINK = 4015,
	PFSOLVE_ERROR_FAILED_SPIRV_GENERATE = 4016,
	PFSOLVE_ERROR_FAILED_TO_CREATE_SHADER_MODULE = 4017,
	PFSOLVE_ERROR_FAILED_TO_CREATE_INSTANCE = 4018,
	PFSOLVE_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER = 4019,
	PFSOLVE_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE = 4020,
	PFSOLVE_ERROR_FAILED_TO_CREATE_DEVICE = 4021,
	PFSOLVE_ERROR_FAILED_TO_CREATE_FENCE = 4022,
	PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_POOL = 4023,
	PFSOLVE_ERROR_FAILED_TO_CREATE_BUFFER = 4024,
	PFSOLVE_ERROR_FAILED_TO_ALLOCATE_MEMORY = 4025,
	PFSOLVE_ERROR_FAILED_TO_BIND_BUFFER_MEMORY = 4026,
	PFSOLVE_ERROR_FAILED_TO_FIND_MEMORY = 4027,
	PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE = 4028,
	PFSOLVE_ERROR_FAILED_TO_COPY = 4029,
	PFSOLVE_ERROR_FAILED_TO_CREATE_PROGRAM = 4030,
	PFSOLVE_ERROR_FAILED_TO_COMPILE_PROGRAM = 4031,
	PFSOLVE_ERROR_FAILED_TO_GET_CODE_SIZE = 4032,
	PFSOLVE_ERROR_FAILED_TO_GET_CODE = 4033,
	PFSOLVE_ERROR_FAILED_TO_DESTROY_PROGRAM = 4034,
	PFSOLVE_ERROR_FAILED_TO_LOAD_MODULE = 4035,
	PFSOLVE_ERROR_FAILED_TO_GET_FUNCTION = 4036,
	PFSOLVE_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY = 4037,
	PFSOLVE_ERROR_FAILED_TO_MODULE_GET_GLOBAL = 4038,
	PFSOLVE_ERROR_FAILED_TO_LAUNCH_KERNEL = 4039,
	PFSOLVE_ERROR_FAILED_TO_EVENT_RECORD = 4040,
	PFSOLVE_ERROR_FAILED_TO_ADD_NAME_EXPRESSION = 4041,
	PFSOLVE_ERROR_FAILED_TO_INITIALIZE = 4042,
	PFSOLVE_ERROR_FAILED_TO_SET_DEVICE_ID = 4043,
	PFSOLVE_ERROR_FAILED_TO_GET_DEVICE = 4044,
	PFSOLVE_ERROR_FAILED_TO_CREATE_CONTEXT = 4045,
	PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE = 4046,
	PFSOLVE_ERROR_FAILED_TO_SET_KERNEL_ARG = 4047,
	PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE = 4048,
	PFSOLVE_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE = 4049,
	PFSOLVE_ERROR_FAILED_TO_ENUMERATE_DEVICES = 4050,
	PFSOLVE_ERROR_FAILED_TO_GET_ATTRIBUTE = 4051,
	PFSOLVE_ERROR_FAILED_TO_CREATE_EVENT = 4052,
	PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST = 4053,
	PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST = 4054,
	PFSOLVE_ERROR_FAILED_TO_SUBMIT_BARRIER = 4055
} PfSolveResult;

static inline const char* getPfSolveErrorString(PfSolveResult result)
{
	switch (result)
	{
	case PFSOLVE_SUCCESS:
		return "PFSOLVE_SUCCESS";
	case PFSOLVE_ERROR_MALLOC_FAILED:
		return "PFSOLVE_ERROR_MALLOC_FAILED";
	case PFSOLVE_ERROR_INSUFFICIENT_CODE_BUFFER:
		return "PFSOLVE_ERROR_INSUFFICIENT_CODE_BUFFER";
	case PFSOLVE_ERROR_INSUFFICIENT_TEMP_BUFFER:
		return "PFSOLVE_ERROR_INSUFFICIENT_TEMP_BUFFER";
	case PFSOLVE_ERROR_PLAN_NOT_INITIALIZED:
		return "PFSOLVE_ERROR_PLAN_NOT_INITIALIZED";
	case PFSOLVE_ERROR_NULL_TEMP_PASSED:
		return "PFSOLVE_ERROR_NULL_TEMP_PASSED";
	case PFSOLVE_ERROR_INVALID_PHYSICAL_DEVICE:
		return "PFSOLVE_ERROR_INVALID_PHYSICAL_DEVICE";
	case PFSOLVE_ERROR_INVALID_DEVICE:
		return "PFSOLVE_ERROR_INVALID_DEVICE";
	case PFSOLVE_ERROR_INVALID_QUEUE:
		return "PFSOLVE_ERROR_INVALID_QUEUE";
	case PFSOLVE_ERROR_INVALID_COMMAND_POOL:
		return "PFSOLVE_ERROR_INVALID_COMMAND_POOL";
	case PFSOLVE_ERROR_INVALID_FENCE:
		return "PFSOLVE_ERROR_INVALID_FENCE";
	case PFSOLVE_ERROR_ONLY_FORWARD_FFT_INITIALIZED:
		return "PFSOLVE_ERROR_ONLY_FORWARD_FFT_INITIALIZED";
	case PFSOLVE_ERROR_ONLY_INVERSE_FFT_INITIALIZED:
		return "PFSOLVE_ERROR_ONLY_INVERSE_FFT_INITIALIZED";
	case PFSOLVE_ERROR_INVALID_CONTEXT:
		return "PFSOLVE_ERROR_INVALID_CONTEXT";
	case PFSOLVE_ERROR_INVALID_PLATFORM:
		return "PFSOLVE_ERROR_INVALID_PLATFORM";
	case PFSOLVE_ERROR_ENABLED_saveApplicationToString:
		return "PFSOLVE_ERROR_ENABLED_saveApplicationToString";
	case PFSOLVE_ERROR_EMPTY_FILE:
		return "PFSOLVE_ERROR_EMPTY_FILE";
	case PFSOLVE_ERROR_EMPTY_FFTdim:
		return "PFSOLVE_ERROR_EMPTY_FFTdim";
	case PFSOLVE_ERROR_EMPTY_size:
		return "PFSOLVE_ERROR_EMPTY_size";
	case PFSOLVE_ERROR_EMPTY_bufferSize:
		return "PFSOLVE_ERROR_EMPTY_bufferSize";
	case PFSOLVE_ERROR_EMPTY_buffer:
		return "PFSOLVE_ERROR_EMPTY_buffer";
	case PFSOLVE_ERROR_EMPTY_tempBufferSize:
		return "PFSOLVE_ERROR_EMPTY_tempBufferSize";
	case PFSOLVE_ERROR_EMPTY_tempBuffer:
		return "PFSOLVE_ERROR_EMPTY_tempBuffer";
	case PFSOLVE_ERROR_EMPTY_inputBufferSize:
		return "PFSOLVE_ERROR_EMPTY_inputBufferSize";
	case PFSOLVE_ERROR_EMPTY_inputBuffer:
		return "PFSOLVE_ERROR_EMPTY_inputBuffer";
	case PFSOLVE_ERROR_EMPTY_outputBufferSize:
		return "PFSOLVE_ERROR_EMPTY_outputBufferSize";
	case PFSOLVE_ERROR_EMPTY_outputBuffer:
		return "PFSOLVE_ERROR_EMPTY_outputBuffer";
	case PFSOLVE_ERROR_EMPTY_kernelSize:
		return "PFSOLVE_ERROR_EMPTY_kernelSize";
	case PFSOLVE_ERROR_EMPTY_kernel:
		return "PFSOLVE_ERROR_EMPTY_kernel";
	case PFSOLVE_ERROR_EMPTY_applicationString:
		return "PFSOLVE_ERROR_EMPTY_applicationString";
	case PFSOLVE_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays:
		return "PFSOLVE_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays";
	case PFSOLVE_ERROR_UNSUPPORTED_RADIX:
		return "PFSOLVE_ERROR_UNSUPPORTED_RADIX";
	case PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH:
		return "PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH";
	case PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH_R2C:
		return "PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH_R2C";
	case PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH_DCT:
		return "PFSOLVE_ERROR_UNSUPPORTED_FFT_LENGTH_DCT";
	case PFSOLVE_ERROR_UNSUPPORTED_FFT_OMIT:
		return "PFSOLVE_ERROR_UNSUPPORTED_FFT_OMIT";
	case PFSOLVE_ERROR_FAILED_TO_ALLOCATE:
		return "PFSOLVE_ERROR_FAILED_TO_ALLOCATE";
	case PFSOLVE_ERROR_FAILED_TO_MAP_MEMORY:
		return "PFSOLVE_ERROR_FAILED_TO_MAP_MEMORY";
	case PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS:
		return "PFSOLVE_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS";
	case PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER:
		return "PFSOLVE_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER";
	case PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER:
		return "PFSOLVE_ERROR_FAILED_TO_END_COMMAND_BUFFER";
	case PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE:
		return "PFSOLVE_ERROR_FAILED_TO_SUBMIT_QUEUE";
	case PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES:
		return "PFSOLVE_ERROR_FAILED_TO_WAIT_FOR_FENCES";
	case PFSOLVE_ERROR_FAILED_TO_RESET_FENCES:
		return "PFSOLVE_ERROR_FAILED_TO_RESET_FENCES";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT";
	case PFSOLVE_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS:
		return "PFSOLVE_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT";
	case PFSOLVE_ERROR_FAILED_SHADER_PREPROCESS:
		return "PFSOLVE_ERROR_FAILED_SHADER_PREPROCESS";
	case PFSOLVE_ERROR_FAILED_SHADER_PARSE:
		return "PFSOLVE_ERROR_FAILED_SHADER_PARSE";
	case PFSOLVE_ERROR_FAILED_SHADER_LINK:
		return "PFSOLVE_ERROR_FAILED_SHADER_LINK";
	case PFSOLVE_ERROR_FAILED_SPIRV_GENERATE:
		return "PFSOLVE_ERROR_FAILED_SPIRV_GENERATE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_SHADER_MODULE:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_SHADER_MODULE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_INSTANCE:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_INSTANCE";
	case PFSOLVE_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER:
		return "PFSOLVE_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER";
	case PFSOLVE_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE:
		return "PFSOLVE_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_DEVICE:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_DEVICE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_FENCE:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_FENCE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_POOL:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_POOL";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_BUFFER:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_BUFFER";
	case PFSOLVE_ERROR_FAILED_TO_ALLOCATE_MEMORY:
		return "PFSOLVE_ERROR_FAILED_TO_ALLOCATE_MEMORY";
	case PFSOLVE_ERROR_FAILED_TO_BIND_BUFFER_MEMORY:
		return "PFSOLVE_ERROR_FAILED_TO_BIND_BUFFER_MEMORY";
	case PFSOLVE_ERROR_FAILED_TO_FIND_MEMORY:
		return "PFSOLVE_ERROR_FAILED_TO_FIND_MEMORY";
	case PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE:
		return "PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE";
	case PFSOLVE_ERROR_FAILED_TO_COPY:
		return "PFSOLVE_ERROR_FAILED_TO_COPY";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_PROGRAM:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_PROGRAM";
	case PFSOLVE_ERROR_FAILED_TO_COMPILE_PROGRAM:
		return "PFSOLVE_ERROR_FAILED_TO_COMPILE_PROGRAM";
	case PFSOLVE_ERROR_FAILED_TO_GET_CODE_SIZE:
		return "PFSOLVE_ERROR_FAILED_TO_GET_CODE_SIZE";
	case PFSOLVE_ERROR_FAILED_TO_GET_CODE:
		return "PFSOLVE_ERROR_FAILED_TO_GET_CODE";
	case PFSOLVE_ERROR_FAILED_TO_DESTROY_PROGRAM:
		return "PFSOLVE_ERROR_FAILED_TO_DESTROY_PROGRAM";
	case PFSOLVE_ERROR_FAILED_TO_LOAD_MODULE:
		return "PFSOLVE_ERROR_FAILED_TO_LOAD_MODULE";
	case PFSOLVE_ERROR_FAILED_TO_GET_FUNCTION:
		return "PFSOLVE_ERROR_FAILED_TO_GET_FUNCTION";
	case PFSOLVE_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY:
		return "PFSOLVE_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY";
	case PFSOLVE_ERROR_FAILED_TO_MODULE_GET_GLOBAL:
		return "PFSOLVE_ERROR_FAILED_TO_MODULE_GET_GLOBAL";
	case PFSOLVE_ERROR_FAILED_TO_LAUNCH_KERNEL:
		return "PFSOLVE_ERROR_FAILED_TO_LAUNCH_KERNEL";
	case PFSOLVE_ERROR_FAILED_TO_EVENT_RECORD:
		return "PFSOLVE_ERROR_FAILED_TO_EVENT_RECORD";
	case PFSOLVE_ERROR_FAILED_TO_ADD_NAME_EXPRESSION:
		return "PFSOLVE_ERROR_FAILED_TO_ADD_NAME_EXPRESSION";
	case PFSOLVE_ERROR_FAILED_TO_INITIALIZE:
		return "PFSOLVE_ERROR_FAILED_TO_INITIALIZE";
	case PFSOLVE_ERROR_FAILED_TO_SET_DEVICE_ID:
		return "PFSOLVE_ERROR_FAILED_TO_SET_DEVICE_ID";
	case PFSOLVE_ERROR_FAILED_TO_GET_DEVICE:
		return "PFSOLVE_ERROR_FAILED_TO_GET_DEVICE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_CONTEXT:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_CONTEXT";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE";
	case PFSOLVE_ERROR_FAILED_TO_SET_KERNEL_ARG:
		return "PFSOLVE_ERROR_FAILED_TO_SET_KERNEL_ARG";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE";
	case PFSOLVE_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE:
		return "PFSOLVE_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE";
	case PFSOLVE_ERROR_FAILED_TO_ENUMERATE_DEVICES:
		return "PFSOLVE_ERROR_FAILED_TO_ENUMERATE_DEVICES";
	case PFSOLVE_ERROR_FAILED_TO_GET_ATTRIBUTE:
		return "PFSOLVE_ERROR_FAILED_TO_GET_ATTRIBUTE";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_EVENT:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_EVENT";
	case PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST:
		return "PFSOLVE_ERROR_FAILED_TO_CREATE_COMMAND_LIST";
	case PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST:
		return "PFSOLVE_ERROR_FAILED_TO_DESTROY_COMMAND_LIST";
	case PFSOLVE_ERROR_FAILED_TO_SUBMIT_BARRIER:
		return "PFSOLVE_ERROR_FAILED_TO_SUBMIT_BARRIER";
	}
	return "Unknown PfSolve error";
}


typedef struct PfSolveRaderContainer PfSolveRaderContainer;

struct PfSolveRaderContainer {
	int prime;
	int generator;
	int multiplier;
	int inline_rader_g_pow;
	int raderUintLUToffset;

	int type; //0 - FFT, 1 - Direct multiplication

	int raderRegisters;
	int rader_min_registers;

	//Direct multiplication parameters

	//FFT parameters
	int registers_per_thread;
	int min_registers_per_thread;
	int loc_multipliers[33];
	int registers_per_thread_per_radix[33];
	int stageRadix[20];
	int numStages;
	int numSubPrimes;
	int stage_rader_generator[20];
	int containerFFTDim;
	int containerFFTNum;
	int subLogicalGroupSizeMax;//how many threads are needed per Rader transform
	int64_t RaderKernelOffsetLUT;
	int64_t RaderRadixOffsetLUT;
	int64_t RaderRadixOffsetLUTiFFT;
	PfContainer g_powConstantStruct;
	PfContainer r_rader_kernelConstantStruct;
	PfContainer i_rader_kernelConstantStruct;
	void* raderFFTkernel;

	struct PfSolveRaderContainer* container;
};

typedef struct {
	PfSolveResult res;
	long double double_PI; 

	PfContainer size[3];
	PfContainer localSize[3];
	int numSubgroups;
	PfContainer sourceFFTSize;
	PfContainer fftDim;
	int precision;
	int inverse;
	int actualInverse;
	int inverseBluestein;
	int zeropad[2];
	int zeropadBluestein[2];
	int axis_id;
	int axis_upload_id;
	int numAxisUploads;
	int registers_per_thread;
	int registers_per_thread_per_radix[33];
	int min_registers_per_thread;
	int maxNonPow2Radix;
	int usedLocRegs;
	int readToRegisters;
	int writeFromRegisters;
	int LUT;
	int LUT_4step;
	int raderUintLUT;
	int useCoalescedLUTUploadToSM;
	int useBluesteinFFT;
	int reverseBluesteinMultiUpload;
	int BluesteinConvolutionStep;
	int BluesteinPreMultiplication;
	int BluesteinPostMultiplication;
	PfContainer startDCT3LUT;
	PfContainer startDCT4LUT;
	int performR2C;
	int performR2CmultiUpload;
	int performDCT;
	int performBandwidthBoost;
	int frequencyZeropadding;
	int performZeropaddingFull[3]; // don't do read/write if full sequence is omitted
	int performZeropaddingInput[3]; // don't read if input is zeropadded (0 - off, 1 - on)
	int performZeropaddingOutput[3]; // don't write if output is zeropadded (0 - off, 1 - on)
	PfContainer fft_zeropad_left_full[3];
	PfContainer fft_zeropad_left_read[3];
	PfContainer fft_zeropad_left_write[3];
	PfContainer fft_zeropad_right_full[3];
	PfContainer fft_zeropad_right_read[3];
	PfContainer fft_zeropad_right_write[3];
	PfContainer fft_zeropad_Bluestein_left_read[3];
	PfContainer fft_zeropad_Bluestein_left_write[3];
	PfContainer fft_zeropad_Bluestein_right_read[3];
	PfContainer fft_zeropad_Bluestein_right_write[3];
	PfContainer inputStride[5];
	PfContainer outputStride[5];
	PfContainer fft_dim_full;
	PfContainer stageStartSize;
	PfContainer firstStageStartSize;
	PfContainer fft_dim_x;
	PfContainer dispatchZactualFFTSize;
	int numStages;
	int stageRadix[33];
	PfContainer inputOffset;
	PfContainer kernelOffset;
	PfContainer outputOffset;
	int reorderFourStep;
	int pushConstantsStructSize;
	int performWorkGroupShift[3];
	int performPostCompilationInputOffset;
	int performPostCompilationOutputOffset;
	int performPostCompilationKernelOffset;
	uint64_t inputBufferBlockNum;
	uint64_t inputBufferBlockSize;
	uint64_t outputBufferBlockNum;
	uint64_t outputBufferBlockSize;
	uint64_t kernelBlockNum;
	uint64_t kernelBlockSize;
	int numCoordinates;
	int matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
	PfContainer numBatches;
	PfContainer numKernels;
	int conjugateConvolution;
	int crossPowerSpectrumNormalization;
	PfContainer usedSharedMemory;
	int sharedMemSize;
	int sharedMemSizePow2;
	int normalize;
	int complexSize;
	int inputNumberByteSize;
	int outputNumberByteSize;
	int kernelNumberByteSize;
	int maxStageSumLUT;


	int64_t finiteDifferences;

	int64_t compute_flux_D;
	int64_t compute_Pf;
	int64_t compute_flux_T;
	int64_t compute_T;
	int64_t compute_r;

	PfContainer scaleC;
	//int64_t scaleType;
	int64_t lshift;
	//int64_t copy; // 1 real, -1 imag
	int64_t block; // 1 real, -1 imag
	int64_t LDA;
	int64_t KU;
	int64_t KL;
	int64_t testWarpReduce;

	PfContainer s_dx;
	PfContainer s_dy;
	PfContainer s_dz;
	PfContainer s_dt_D;

	double windowSize;
	double* grid_x;


	PfContainer logicBlock[3]; // logic block per warp
	PfContainer M_size; // matrix size
	PfContainer M_size_pow2; // matrix size rounded up to next power of 2

	PfContainer* regIDs_x;
	PfContainer* regIDs_y;
	PfContainer* regIDs_z;

	PfContainer qDxStruct;
	PfContainer qDyStruct;
	PfContainer qDzStruct;
	PfContainer PfStruct;

	int performMatVecMul;
	int performTriSolve;
	int useParallelThomas;
	PfContainer* rd;
	PfContainer* rd_copy;
	PfContainer* ld;
	PfContainer* ld_copy;
	PfContainer* ud;
	PfContainer* ud_copy;
	PfContainer* md;
	
	pfUINT inputBufferId;

	int upperBanded;
	int md_zero;
	int ud_zero;
	int ld_zero;
	int num_warps_data_parallel;
	int sharedMatricesUpload;
	int num_threads;
	int numConsecutiveJWIterations;
	int useMultipleInputBuffers;
	int64_t jw_control_bitmask;

	PfContainer offset_md; //0
	PfContainer offset_ld; //1
	PfContainer offset_ud; //2
	PfContainer offset_res; //3
	PfContainer offset_md_global; //4
	PfContainer offset_ld_global; //5
	PfContainer offset_ud_global; //6
	PfContainer offset_res_global; //7
	PfContainer offsetSolution; //8
	PfContainer offsetM; //9
	PfContainer offsetV; //10

	PfContainer inputZeropad[2]; //11
	PfContainer outputZeropad[2]; //12

	int swapComputeWorkGroupID;
	int convolutionStep;
	int symmetricKernel;
	int supportAxis;
	int cacheShuffle;
	int registerBoost;
	int warpSize;
	int numSharedBanks;
	int resolveBankConflictFirstStages;
	int storeSharedComplexComponentsSeparately;
	PfContainer offsetImaginaryShared;
	PfContainer sharedStrideBankConflictFirstStages;
	PfContainer sharedStrideReadWriteConflict;

	PfContainer sharedStrideRaderFFT;
	PfContainer sharedShiftRaderFFT;

	PfContainer maxSharedStride;
	PfContainer maxSingleSizeStrided;
	int axisSwapped;
	int stridedSharedLayout;
	int mergeSequencesR2C;

	int numBuffersBound[10];
	int convolutionBindingID;
	int LUTBindingID;
	int BluesteinConvolutionBindingID;
	int BluesteinMultiplicationBindingID;

	int useRader;
	int numRaderPrimes;
	int minRaderFFTThreadNum;
	PfSolveRaderContainer* raderContainer;
	PfSolveRaderContainer* currentRaderContainer;
	int RaderUintLUTBindingID;

	int useRaderMult;
	PfContainer additionalRaderSharedSize;
	PfContainer RaderKernelOffsetShared[33];
	PfContainer RaderKernelOffsetLUT[33];
	int rader_generator[33];
	int fixMinRaderPrimeMult;//start Rader algorithm for primes from this number
	int fixMaxRaderPrimeMult;//switch from Rader to Bluestein algorithm for primes from this number
	int fixMinRaderPrimeFFT;//start Rader algorithm for primes from this number
	int fixMaxRaderPrimeFFT;//switch from Rader to Bluestein algorithm for primes from this number

	int inline_rader_g_pow;
	int inline_rader_kernel;

	int raderRegisters;
	int rader_min_registers;

	int useRaderFFT;

	int performOffsetUpdate;
	int performBufferSetUpdate;
	int useUint64;
#if(VKFFT_BACKEND==2)
	int64_t  useStrict32BitAddress;
#endif
	int disableSetLocale;

	PfContainer* regIDs;
	PfContainer* regIDs_copy; //only for convolutions
	PfContainer* temp_conv; //only for convolutions
	PfContainer* disableThreadsStart;
	PfContainer* disableThreadsEnd;
	PfContainer sdataID;
	PfContainer inoutID;
	PfContainer inoutID_x;
	PfContainer inoutID_y;
	PfContainer inoutID_z;
	PfContainer combinedID;
	PfContainer LUTId;
	PfContainer raderIDx;
	PfContainer raderIDx2;
	PfContainer gl_LocalInvocationID_x;
	PfContainer gl_LocalInvocationID_y;
	PfContainer gl_LocalInvocationID_z;
	PfContainer gl_GlobalInvocationID_x;
	PfContainer gl_GlobalInvocationID_y;
	PfContainer gl_GlobalInvocationID_z;
	PfContainer gl_SubgroupInvocationID;
	PfContainer gl_SubgroupID;
	PfContainer tshuffle;
	PfContainer sharedStride;
	PfContainer gl_WorkGroupSize_x;
	PfContainer gl_WorkGroupSize_y;
	PfContainer gl_WorkGroupSize_z;

	PfContainer halfDef;
	PfContainer floatDef;
	PfContainer doubleDef;
	PfContainer quadDef;

	PfContainer half2Def;
	PfContainer float2Def;
	PfContainer double2Def;
	PfContainer quad2Def;

	PfContainer halfLiteral;
	PfContainer floatLiteral;
	PfContainer doubleLiteral;
	PfContainer quadLiteral;

	PfContainer intDef;
	PfContainer uintDef;

	PfContainer int64Def;
	PfContainer uint64Def;

	PfContainer constDef;

	PfContainer functionDef;

	PfContainer gl_WorkGroupID_x;
	PfContainer gl_WorkGroupID_y;
	PfContainer gl_WorkGroupID_z;

	PfContainer workGroupShiftX;
	PfContainer workGroupShiftY;
	PfContainer workGroupShiftZ;

	PfContainer shiftX;
	PfContainer shiftY;
	PfContainer shiftZ;

	int useDisableThreads;
	PfContainer disableThreads;

	PfContainer tempReg;
	
	PfContainer coordinate;
	PfContainer batchID;
	PfContainer stageInvocationID;
	PfContainer blockInvocationID;
	PfContainer warpInvocationID;
	PfContainer warpID;
	PfContainer temp;
	PfContainer temp1;
	PfContainer temp2;
	PfContainer tempInt;
	PfContainer tempInt2;
	PfContainer tempFloat;
	PfContainer tempFloat2;
	PfContainer tempQuad;
	PfContainer tempQuad2;
	PfContainer tempQuad3;
	PfContainer tempIntQuad;
	PfContainer w;
	PfContainer iw;
	PfContainer angle;
	PfContainer mult;
	PfContainer x0[33];
	PfContainer locID[33];
	char* code0;
	char* tempStr;
	int64_t tempLen;
	int64_t currentLen;
	int64_t currentTempLen;
	int64_t maxCodeLength;
	int64_t maxTempLength;

	int dataTypeSize;
	PfContainer LFending;
	int complexDataType;

	// 0 - float, 1 - double, 2 - half
	int floatTypeCode;
	int floatTypeKernelMemoryCode;
	int floatTypeInputMemoryCode;
	int floatTypeOutputMemoryCode;

	int vecTypeCode;
	int vecTypeKernelMemoryCode;
	int vecTypeInputMemoryCode;
	int vecTypeOutputMemoryCode;

	int intTypeCode;
	int uintTypeCode;
	int uintType32Code;

	int inputMemoryCode;
	int outputMemoryCode;
	//int inputType;
	//int outputType;
	PfContainer inputsStruct;
	PfContainer outputsStruct;
	PfContainer kernelStruct;
	PfContainer sdataStruct;
	PfContainer LUTStruct;
	PfContainer BluesteinStruct;
	PfContainer BluesteinConvolutionKernelStruct;
	PfContainer g_powStruct;

	char PfSolveFunctionName[200];
	//PfContainer cosDef;
	//PfContainer sinDef;

	PfContainer oldLocale;

	int64_t id;
} PfSolveSpecializationConstantsLayout;

typedef struct {
	char data[128];
#if(VKFFT_BACKEND == 5)
	MTL::Buffer* dataUintBuffer;
#endif
	//specify what can be in layout
	uint64_t performWorkGroupShift[3];
	uint64_t workGroupShift[3];

	uint64_t performPostCompilationInputOffset;
	uint64_t inputOffset;

	uint64_t performPostCompilationOutputOffset;
	uint64_t outputOffset;

	uint64_t performPostCompilationKernelOffset;
	uint64_t kernelOffset;

	uint64_t M_size;
	uint64_t offsetM;
	uint64_t offsetV;
	uint64_t offsetSolution;
	uint64_t inputZeropad[2]; //11
	uint64_t outputZeropad[2]; //12
	uint64_t inputBufferStride; 
	uint64_t outputBufferStride;
	uint64_t batch;
	long double scaleC;

	uint64_t structSize;
} PfSolvePushConstantsLayout;

typedef struct {
	uint64_t numBindings;
	uint64_t axisBlock[4];
	uint64_t groupedBatch;
	PfSolveSpecializationConstantsLayout specializationConstants;
	PfSolvePushConstantsLayout pushConstants;
	uint64_t updatePushConstants;
#if(VKFFT_BACKEND==0)
	VkBuffer* inputBuffer;
	VkBuffer* outputBuffer;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkDeviceMemory bufferLUTDeviceMemory;
	VkBuffer bufferLUT;
	VkDeviceMemory bufferRaderUintLUTDeviceMemory;
	VkBuffer bufferRaderUintLUT;
	VkDeviceMemory* bufferBluesteinDeviceMemory;
	VkDeviceMemory* bufferBluesteinFFTDeviceMemory;
	VkBuffer* bufferBluestein;
	VkBuffer* bufferBluesteinFFT;
#elif(VKFFT_BACKEND==1)
	void** inputBuffer;
	void** outputBuffer;
	CUmodule PfSolveModule;
	CUfunction PfSolveKernel;
	void* bufferLUT;
	void* bufferRaderUintLUT;
	CUdeviceptr consts_addr;
	void** bufferBluestein;
	void** bufferBluesteinFFT;
#elif(VKFFT_BACKEND==2)
	void** inputBuffer;
	void** outputBuffer;
	hipModule_t PfSolveModule;
	hipFunction_t PfSolveKernel;
	void* bufferLUT;
	void* bufferRaderUintLUT;
	hipDeviceptr_t consts_addr;
	void** bufferBluestein;
	void** bufferBluesteinFFT;
#elif(VKFFT_BACKEND==3)
	cl_mem* inputBuffer;
	cl_mem* outputBuffer;
	cl_program  program;
	cl_kernel kernel;
	cl_mem bufferLUT;
	cl_mem bufferRaderUintLUT;
	cl_mem* bufferBluestein;
	cl_mem* bufferBluesteinFFT;
#elif(VKFFT_BACKEND==4)
	void** inputBuffer;
	void** outputBuffer;
	ze_module_handle_t PfSolveModule;
	ze_kernel_handle_t PfSolveKernel;
	void* bufferLUT;
	void* bufferRaderUintLUT;
	void** bufferBluestein;
	void** bufferBluesteinFFT;
#elif(VKFFT_BACKEND==5)
	MTL::Buffer** inputBuffer;
	MTL::Buffer** outputBuffer;
	MTL::Library* library;
	MTL::ComputePipelineState* pipeline;
	MTL::Buffer* bufferLUT;
	MTL::Buffer* bufferRaderUintLUT;
	MTL::Buffer** bufferBluestein;
	MTL::Buffer** bufferBluesteinFFT;
#endif

	void* binary;
	uint64_t binarySize;

	uint64_t bufferLUTSize;
	uint64_t bufferRaderUintLUTSize;
	uint64_t referenceLUT;
} PfSolveAxis;

typedef struct {
	uint64_t actualFFTSizePerAxis[3][3];
	uint64_t numAxisUploads[3];
	uint64_t axisSplit[3][4];
	PfSolveAxis axes[3][4];

	uint64_t multiUploadR2C;
	uint64_t actualPerformR2CPerAxis[3]; // automatically specified, shows if R2C is actually performed or inside FFT or as a separate step
	PfSolveAxis R2Cdecomposition;
	PfSolveAxis inverseBluesteinAxes[3][4];
} PfSolvePlan;
typedef struct {
	PfSolveConfiguration configuration;
	PfSolvePlan* localFFTPlan;
	PfSolvePlan* localFFTPlan_inverse; //additional inverse plan
	char kernelName[200];
	uint64_t actualNumBatches;
	uint64_t firstAxis;
	uint64_t lastAxis;
	//Bluestein buffers reused among plans
	uint64_t useBluesteinFFT[3];
#if(VKFFT_BACKEND==0)
	VkDeviceMemory bufferRaderUintLUTDeviceMemory[3][4];
	VkBuffer bufferRaderUintLUT[3][4];
	VkDeviceMemory bufferBluesteinDeviceMemory[3];
	VkDeviceMemory bufferBluesteinFFTDeviceMemory[3];
	VkDeviceMemory bufferBluesteinIFFTDeviceMemory[3];
	VkBuffer bufferBluestein[3];
	VkBuffer bufferBluesteinFFT[3];
	VkBuffer bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==1)
	void* bufferRaderUintLUT[3][4];
	void* bufferBluestein[3];
	void* bufferBluesteinFFT[3];
	void* bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==2)
	void* bufferRaderUintLUT[3][4];
	void* bufferBluestein[3];
	void* bufferBluesteinFFT[3];
	void* bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==3)
	cl_mem bufferRaderUintLUT[3][4];
	cl_mem bufferBluestein[3];
	cl_mem bufferBluesteinFFT[3];
	cl_mem bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==4)
	void* bufferRaderUintLUT[3][4];
	void* bufferBluestein[3];
	void* bufferBluesteinFFT[3];
	void* bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==5)
	MTL::Buffer* bufferRaderUintLUT[3][4];
	MTL::Buffer* bufferBluestein[3];
	MTL::Buffer* bufferBluesteinFFT[3];
	MTL::Buffer* bufferBluesteinIFFT[3];
#endif
	uint64_t bufferRaderUintLUTSize[3][4];
	uint64_t bufferBluesteinSize[3];
	void* applicationBluesteinString[3];
	uint64_t applicationBluesteinStringSize[3];

	uint64_t numRaderFFTPrimes;
	uint64_t rader_primes[30];
	uint64_t rader_buffer_size[30];
	void* raderFFTkernel[30];
	uint64_t applicationStringOffsetRader;

	uint64_t currentApplicationStringPos;

	uint64_t applicationStringSize;//size of saveApplicationString in bytes
	void* saveApplicationString;//memory array(uint32_t* for Vulkan, char* for CUDA/HIP/OpenCL) through which user can access PfSolve generated binaries. (will be allocated by PfSolve, deallocated with deletePfSolve call)
} PfSolveApplication;

#endif