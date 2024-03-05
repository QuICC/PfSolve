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
#ifndef PFSOLVE_KERNELSTARTEND_H
#define PFSOLVE_KERNELSTARTEND_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_SharedMemory.h"
static inline void appendKernelStart(PfSolveSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType); 
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	appendSharedMemoryPfSolve(sc, locType);
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ") %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);

	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* kernel_obj", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* twiddleLUT", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* g_pow", uintType32->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* BluesteinConvolutionKernel", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* BluesteinMultiplication", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
	
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> kernel_obj", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> twiddleLUT", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> g_pow", uintType32->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> BluesteinConvolutionKernel", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> BluesteinMultiplication", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void %s ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* kernel_obj", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* twiddleLUT", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* g_pow", uintType32->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* BluesteinConvolutionKernel", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* BluesteinMultiplication", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
		args_id++;
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void %s ", sc->PfSolveFunctionName);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->name);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "threadgroup %s* sdata [[threadgroup(0)]], ", vecType->name);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* kernel_obj[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* twiddleLUT[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* g_pow[[buffer(%d)]]", uintType32->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* BluesteinConvolutionKernel[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* BluesteinMultiplication[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", constant PushConsts& consts[[buffer(%d)]]", args_id);
		PfAppendLine(sc);
		
		args_id++;
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
#endif
	return;
}

static inline void appendKernelStart_R2C(PfSolveSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ") PfSolve_main_R2C ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
	PfAppendLine(sc);

	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* twiddleLUT", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#elif(VKFFT_BACKEND==2)
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void PfSolve_main_R2C ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i);
	PfAppendLine(sc);
	
	sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
	PfAppendLine(sc);
	
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> twiddleLUT", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void PfSolve_main_R2C ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* twiddleLUT", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void PfSolve_main_R2C ");
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->name, vecTypeOutputMemory->name);
	PfAppendLine(sc);
	int args_id = 2;
	
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* twiddleLUT[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", constant PushConsts& consts[[buffer(%d)]]", args_id);
		PfAppendLine(sc);

		args_id++;
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);

	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
#endif
	return;
}

static inline void appendKernelStart_compute_Pf(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "layout (local_size_x = %" PRIi64 ", local_size_y = %" PRIi64 ", local_size_z = %" PRIi64 ") in;\n", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
#elif(VKFFT_BACKEND==1)
	//sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	//res = PfAppendLine(sc);
	//if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ __launch_bounds__(%" PRIi64 ") void %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, "(%s* Pf, %s* qDx, %s* qDy, %s* qDz", floatType->name, floatType->name, floatType->name, floatType->name);
	sc->tempLen = sprintf(sc->tempStr, "(%s* Pf, %s* qDx, %s* qDy, %s* qDz", floatType->name, floatType->name, floatType->name, floatType->name);
	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIu64 ") __global__ void %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(%s* Pf, %s* qDx, %s* qDy, %s* qDz", floatType->name, floatType->name, floatType->name, floatType->name);
	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIu64 ", %" PRIu64 ", %" PRIu64 "))) void %s ", sc->localSize[0], sc->localSize[1], sc->localSize[2], sc->PfSolveFunctionName);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatType->name, sc->dataTypeOutput);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
#endif
	return;
}

static inline void appendKernelStart_solve(PfSolveSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	appendSharedMemoryPfSolve(sc, locType);
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ") %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);

	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* kernel_obj", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* twiddleLUT", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* g_pow", uintType32->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* BluesteinConvolutionKernel", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* BluesteinMultiplication", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);

#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> kernel_obj", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> twiddleLUT", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> g_pow", uintType32->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> BluesteinConvolutionKernel", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", const Inputs<%s> BluesteinMultiplication", vecType->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void %s ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "(__global %s* inputs, __global %s* outputs", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* kernel_obj", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* twiddleLUT", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* g_pow", uintType32->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* BluesteinConvolutionKernel", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", __global %s* BluesteinMultiplication", vecType->name);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void PfSolve_main ");
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "threadgroup %s* sdata [[threadgroup(0)]], ", vecType->name);
	PfAppendLine(sc);
	switch (type) {
	case 5:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	case 6:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	case 110:case 111:case 120:case 121:case 130:case 131:case 140:case 141:case 142:case 143:case 144:case 145:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		break;
	}
	default:
	{
		sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", vecTypeInputMemory->name, vecTypeOutputMemory->name);
		break;
	}
	}
	PfAppendLine(sc);
	int args_id = 2;
	if (sc->convolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* kernel_obj[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* twiddleLUT[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->raderUintLUT) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* g_pow[[buffer(%d)]]", uintType32->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinConvolutionStep) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* BluesteinConvolutionKernel[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->BluesteinPreMultiplication || sc->BluesteinPostMultiplication) {
		sc->tempLen = sprintf(sc->tempStr, ", constant %s* BluesteinMultiplication[[buffer(%d)]]", vecType->name, args_id);
		PfAppendLine(sc);
		args_id++;
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", constant PushConsts& consts[[buffer(%d)]]", args_id);
		PfAppendLine(sc);

		args_id++;
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);

	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	appendSharedMemoryPfSolve(sc, locType);
#endif
	return;
}

static inline void appendKernelStart_jw(PfSolveSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* uintType32;
	PfGetTypeFromCode(sc, sc->uintType32Code, &uintType32);
#if(VKFFT_BACKEND==0)
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	if (sc->usedSharedMemory.data.i)
	{
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	}
	int64_t estimateMinBlocksPerSM = 1;
	/*if (sc->registers_per_thread<48){//this is kind of an estimate numbers with not much testing
		if (sc->registers_per_thread>24)
			estimateMinBlocksPerSM = ((16+sc->num_warps_data_parallel-1)/sc->num_warps_data_parallel);//128 registers per thread
		else
			estimateMinBlocksPerSM = ((24+sc->num_warps_data_parallel-1)/sc->num_warps_data_parallel);//80 registers per thread
	}*/
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ", %" PRIi64 ") %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, estimateMinBlocksPerSM, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	if (sc->useMultipleInputBuffers) {
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs0", floatTypeInputMemory->name);
		PfAppendLine(sc);
		for (int i = 1; i < sc->useMultipleInputBuffers; i++) {
			sc->tempLen = sprintf(sc->tempStr, ", %s* inputs%d", floatTypeInputMemory->name, i);
			PfAppendLine(sc);
		}
		sc->tempLen = sprintf(sc->tempStr, ", %s* outputs", floatTypeOutputMemory->name);
		PfAppendLine(sc);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);

#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	if (sc->useMultipleInputBuffers) {
		sc->tempLen = sprintf(sc->tempStr, "(%s* inputs0", floatTypeInputMemory->name);
		PfAppendLine(sc);
		for (int i = 1; i < sc->useMultipleInputBuffers; i++) {
			sc->tempLen = sprintf(sc->tempStr, ", %s* inputs%d", floatTypeInputMemory->name, i);
			PfAppendLine(sc);
		}
		sc->tempLen = sprintf(sc->tempStr, ", %s* outputs", floatTypeOutputMemory->name);
		PfAppendLine(sc);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
		PfAppendLine(sc);
	}
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void %s ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void %s ", sc->PfSolveFunctionName);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "threadgroup %s* sdata [[threadgroup(0)]], ", vecType->name);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->name, floatTypeOutputMemory->name);

	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);

	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
#endif
	return;
}
static inline void appendKernelStart_block(PfSolveSpecializationConstantsLayout* sc, int64_t type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 140) || (type == 142) || (type == 144)) && (sc->axisSwapped)) ? 1 : type;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	PfContainer* floatTypeInputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeInputMemoryCode, &floatTypeInputMemory);
	PfContainer* floatTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->floatTypeOutputMemoryCode, &floatTypeOutputMemory);
	PfContainer* floatTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->floatTypeKernelMemoryCode, &floatTypeKernelMemory);

	PfContainer* vecType;
	PfGetTypeFromCode(sc, sc->vecTypeCode, &vecType);
	PfContainer* vecTypeInputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeInputMemoryCode, &vecTypeInputMemory);
	PfContainer* vecTypeOutputMemory;
	PfGetTypeFromCode(sc, sc->vecTypeOutputMemoryCode, &vecTypeOutputMemory);
	PfContainer* vecTypeKernelMemory;
	PfGetTypeFromCode(sc, sc->vecTypeKernelMemoryCode, &vecTypeKernelMemory);

	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);

	PfContainer* typeInputMemory;
	
	PfContainer* typeOutputMemory;
	
	if ((sc->block%10 == 2) || (sc->block%10 == 3) || (sc->block%10 == 4))
		typeInputMemory = vecTypeInputMemory;
    else 
		typeInputMemory = floatTypeInputMemory;

	PfAllocateContainerFlexible(sc, &sc->outputsStruct, 50);

	if ((sc->block%10 == 5) || (sc->block%10 == 6) || (sc->block%10 == 7))
		typeOutputMemory = vecTypeOutputMemory;
    else
		typeOutputMemory = floatTypeOutputMemory;
#if(VKFFT_BACKEND==0)
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
	sc->tempLen = sprintf(sc->tempStr, "void main() {\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	if (sc->usedSharedMemory.data.i)
	{
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	}

	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __global__ void __launch_bounds__(%" PRIi64 ") %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", typeInputMemory->name, typeOutputMemory->name);
	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);

#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	PfAppendLine(sc);
	if (!sc->useUint64 && sc->useStrict32BitAddress > 0) {
		// These wrappers help hipcc to generate faster code for load and store operations where
		// 64-bit scalar + 32-bit vector registers are used instead of 64-bit vector saving a few
		// instructions for computing 64-bit vector addresses.
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"struct Inputs\n"
			"{\n"
			"	const T* buffer;\n"
			"	inline __device__ Inputs(const T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ const T& operator[](unsigned int idx) const { return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
			"template<typename T>\n"
			"struct Outputs\n"
			"{\n"
			"	T* buffer;\n"
			"	inline __device__ Outputs(T* buffer) : buffer(buffer) {}\n"
			"	inline __device__ T& operator[](unsigned int idx) const { return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) + idx * static_cast<unsigned int>(sizeof(T))); }\n"
			"};\n"
		);
	}
	else {
		sc->tempLen = sprintf(sc->tempStr,
			"template<typename T>\n"
			"using Inputs = const T*;\n"
			"template<typename T>\n"
			"using Outputs = T*;\n"
		);
	}
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIi64 ") __global__ void %s ", sc->localSize[0].data.i * sc->localSize[1].data.i * sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", typeInputMemory->name, typeOutputMemory->name);
	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "__kernel __attribute__((reqd_work_group_size(%" PRIi64 ", %" PRIi64 ", %" PRIi64 "))) void %s ", sc->localSize[0].data.i, sc->localSize[1].data.i, sc->localSize[2].data.i, sc->PfSolveFunctionName);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "(const Inputs<%s> inputs, Outputs<%s> outputs", floatTypeInputMemory->name, floatTypeOutputMemory->name);
	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "kernel void %s ", sc->PfSolveFunctionName);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "(%s3 thread_position_in_grid [[thread_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "%s3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], ", uintType->name);
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "threadgroup %s* sdata [[threadgroup(0)]], ", vecType->name);
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "device %s* inputs[[buffer(0)]], device %s* outputs[[buffer(1)]]", floatTypeInputMemory->name, floatTypeOutputMemory->name);

	PfAppendLine(sc);
	if (sc->pushConstantsStructSize > 0) {
		sc->tempLen = sprintf(sc->tempStr, ", PushConsts consts");
		PfAppendLine(sc);
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	PfAppendLine(sc);

	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	if (sc->usedSharedMemory.data.i) appendSharedMemoryPfSolve(sc, locType);
#endif
	return;
}

static inline void appendKernelEnd(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "}\n");
	PfAppendLine(sc);
	return;
}
#endif
