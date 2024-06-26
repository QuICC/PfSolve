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
#ifndef PFSOLVE_PLAN_JONESWORLAND_H
#define PFSOLVE_PLAN_JONESWORLAND_H
#include "pfSolve_Structs/pfSolve_Structs.h"
//#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel2/pfSolve_JonesWorland.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_InitAPIParameters.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_CompileKernel.h"
#include "pfSolve_AppManagement/pfSolve_DeleteApp.h"
static inline int get_ALT_stride(int M_size_align, int Msplit0) {
	if (Msplit0 == 0){
#if(VKFFT_BACKEND==1)
		Msplit0 = 256;
#elif(VKFFT_BACKEND==2)
		Msplit0 = 704;
#endif
	}
	int size_1 = M_size_align;
	int used_registers = 0;
#if(VKFFT_BACKEND==1)
	if (M_size_align <= 4352) {
#elif(VKFFT_BACKEND==2)
	if (M_size_align <= 3840) {
#endif
	}
	else {
		size_1 = Msplit0;
	}
#if(VKFFT_BACKEND==1)
	int warpSize0 = 32;
#elif(VKFFT_BACKEND==2)
	int warpSize0 = 64;
#endif
	int warpSize = ((uint64_t)ceil(size_1 / (double)(1024))) * 64;
	if (size_1 < 512) warpSize = warpSize0;
	if (warpSize < warpSize0) warpSize = warpSize0;
	if (warpSize % warpSize0) warpSize = warpSize0 * ((warpSize + warpSize0 - 1)/warpSize0);
	
	used_registers = (size_1 + warpSize - 1)/ warpSize; //adjust
#if(VKFFT_BACKEND==2)
	if (used_registers % 2 == 0) used_registers++;
#endif
	return used_registers * warpSize;
}

static inline int reorder_i0(int i, int M_size, int M_size_align, int Msplit0) {
	if (Msplit0 == 0){
#if(VKFFT_BACKEND==1)
		Msplit0 = 256;
		/*if (M_size < 3840 * 320) Msplit0 = 320;
		else if(M_size < 3840*384) Msplit0 = 384;
		else if(M_size < 3840*448) Msplit0 = 448;
		else if(M_size < 3840*576) Msplit0 = 576;
		else if(M_size < 3840*704) Msplit0 = 704;
		else if(M_size < 3840*832) Msplit0 = 832;*/
#elif(VKFFT_BACKEND==2)
		Msplit0 = 704;
		//if(M_size < 3840*832) Msplit0 = 832;
#endif
		/*else if (M_size < 3840 * 960) Msplit0 = 960;
		else if (M_size < 3840*1152) Msplit0 = 1152;
		else if (M_size < 3840*1408) Msplit0 = 1408;
		else if (M_size < 3840*1920) Msplit0 = 1920;
		else if (M_size < 3840*2496) Msplit0 = 2496;
		else if (M_size < 3840*3840) Msplit0 = 3840;
		else if (M_size < 3840*4672) Msplit0 = 4672;
		else if (M_size < 3840*5632) Msplit0 = 5632;
		else if (M_size < 3840*6720) Msplit0 = 6720;
		else if (M_size < 3840*7680) Msplit0 = 7680;*/
	}
	int size_0 = M_size;
	int size_1 = (M_size_align) ? M_size_align : M_size;
	int used_registers = 0;
#if(VKFFT_BACKEND==1)
	if (M_size <= 4352) {
#elif(VKFFT_BACKEND==2)
	if (M_size <= 3840) {
#endif
	}
	else {
		size_0 = Msplit0;
		size_1 = Msplit0;
	}
#if(VKFFT_BACKEND==1)
	int warpSize0 = 32;
#elif(VKFFT_BACKEND==2)
	int warpSize0 = 64;
#endif
	int warpSize = ((uint64_t)ceil(size_1 / (double)(1024))) * 64;
	if (size_1 < 512) warpSize = warpSize0;
	if (warpSize < warpSize0) warpSize = warpSize0;
	if (warpSize % warpSize0) warpSize = warpSize0 * ((warpSize + warpSize0 - 1)/warpSize0);
	
	used_registers = (size_1 + warpSize - 1)/ warpSize; //adjust
#if(VKFFT_BACKEND==2)
	if (used_registers % 2 == 0) used_registers++;
#endif
	
	int ret_offset = (i / size_0) * size_0;
	int local_i = (i % size_0);

	if ((((i / size_0) + 1) * size_0) > M_size) {
		size_0 = M_size % size_0;
	}
	int ret;
	ret = local_i / used_registers;
	if ((local_i % used_registers) < (size_0 % used_registers)) {
		ret += (local_i % used_registers) * ((size_0 + used_registers - 1) / used_registers);
	}
	else
	{
		ret += (size_0 % used_registers) * ((size_0 + used_registers - 1) / used_registers);
		ret += ((local_i % used_registers) - (size_0 % used_registers)) * (size_0 / used_registers);
	}
	ret += ret_offset;

	//alt hack warpSize
	
	//ret = i / (2 * used_registers) + (i % (2 * used_registers)) * ((M_size + (2*used_registers) - 1) / (2*used_registers));
	//printf("%d %d\n", i, ret);
	return ret;
}

static inline int reorder_i_alt(int i, int M_size, int M_size_align, int Msplit0) {
	if (Msplit0 == 0){
#if(VKFFT_BACKEND==1)
		Msplit0 = 256;
		/*if (M_size < 3840 * 320) Msplit0 = 320;
		else if(M_size < 3840*384) Msplit0 = 384;
		else if(M_size < 3840*448) Msplit0 = 448;
		else if(M_size < 3840*576) Msplit0 = 576;
		else if(M_size < 3840*704) Msplit0 = 704;
		else if(M_size < 3840*832) Msplit0 = 832;*/
#elif(VKFFT_BACKEND==2)
		Msplit0 = 704;
		//if(M_size < 3840*832) Msplit0 = 832;
#endif
		/*else if (M_size < 3840 * 960) Msplit0 = 960;
		else if (M_size < 3840*1152) Msplit0 = 1152;
		else if (M_size < 3840*1408) Msplit0 = 1408;
		else if (M_size < 3840*1920) Msplit0 = 1920;
		else if (M_size < 3840*2496) Msplit0 = 2496;
		else if (M_size < 3840*3840) Msplit0 = 3840;
		else if (M_size < 3840*4672) Msplit0 = 4672;
		else if (M_size < 3840*5632) Msplit0 = 5632;
		else if (M_size < 3840*6720) Msplit0 = 6720;
		else if (M_size < 3840*7680) Msplit0 = 7680;*/
	}
	int size_0 = M_size;
	int size_1 = (M_size_align) ? M_size_align : M_size;
	int used_registers = 0;
#if(VKFFT_BACKEND==1)
	if (M_size <= 4352) {
#elif(VKFFT_BACKEND==2)
	if (M_size <= 3840) {
#endif
	}
	else {
		size_0 = Msplit0;
		size_1 = Msplit0;
	}
#if(VKFFT_BACKEND==1)
	int warpSize0 = 32;
#elif(VKFFT_BACKEND==2)
	int warpSize0 = 64;
#endif
	int warpSize = ((uint64_t)ceil(size_1 / (double)(1024))) * 64;
	if (size_1 < 512) warpSize = warpSize0;
	if (warpSize < warpSize0) warpSize = warpSize0;
	if (warpSize % warpSize0) warpSize = warpSize0 * ((warpSize + warpSize0 - 1)/warpSize0);
	
	used_registers = (size_1 + warpSize - 1)/ warpSize; //adjust
#if(VKFFT_BACKEND==2)
	if (used_registers % 2 == 0) used_registers++;
#endif
	
	int ret_offset = (i / size_0) * size_0;
	int local_i = (i % size_0);

	if ((((i / size_0) + 1) * size_0) > M_size) {
		size_0 = M_size % size_0;
	}
	int ret;
	int temp = local_i % 2;
	temp *= warpSize / 2;
		
	ret = (local_i/2) % used_registers;
	ret *= warpSize;
	ret += temp;
	temp = (local_i/2) / used_registers;
	ret += temp;
	ret += ret_offset;

	//alt hack warpSize
	
	//ret = i / (2 * used_registers) + (i % (2 * used_registers)) * ((M_size + (2*used_registers) - 1) / (2*used_registers));
	//printf("%d %d\n", i, ret);
	return ret;
}

static inline PfSolveResult PfSolve_Plan_JonesWorland(PfSolveApplication* app, PfSolvePlan* plan, pfUINT axis_upload_id) {
	//get radix stages
	PfSolveResult res = PFSOLVE_SUCCESS;
#if(VKFFT_BACKEND==0)
	PfResult result = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t result = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t result = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int result = CL_SUCCESS;
#endif

	PfSolveAxis* axis = &plan->axes[0][axis_upload_id];

	axis->specializationConstants.warpSize = app->configuration.warpSize;
	axis->specializationConstants.numSharedBanks = app->configuration.numSharedBanks;
	axis->specializationConstants.useUint64 = app->configuration.useUint64;
	axis->specializationConstants.maxCodeLength = app->configuration.maxCodeLength;
	axis->specializationConstants.maxTempLength = app->configuration.maxTempLength;
	axis->specializationConstants.sharedMemSize = app->configuration.sharedMemorySize;
	axis->specializationConstants.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;

	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
		axis->specializationConstants.precision = 3;
		axis->specializationConstants.complexSize = 32;
		//axis->specializationConstants.storeSharedComplexComponentsSeparately = 1;
	}
	else {
		if (app->configuration.doublePrecision || app->configuration.doublePrecisionFloatMemory) {
			axis->specializationConstants.precision = 1;
			axis->specializationConstants.complexSize = 16;
		}
		else {
			if (app->configuration.halfPrecision) {
				axis->specializationConstants.precision = 0;
				axis->specializationConstants.complexSize = 8;
			}
			else {
				axis->specializationConstants.precision = 0;
				axis->specializationConstants.complexSize = 8;
			}
		}
	}
	axis->specializationConstants.performALT = ((app->configuration.jw_type/100)%10 == 1)? 1 : 0;
	//axis->specializationConstants.doublePrecision = app->configuration.doublePrecision;
	axis->specializationConstants.numCoordinates = app->configuration.coordinateFeatures;
	axis->specializationConstants.normalize = app->configuration.normalize;
	axis->specializationConstants.axis_id = 0;
	axis->specializationConstants.axis_upload_id = axis_upload_id;
	axis->specializationConstants.LUT = 0;// app->configuration.useLUT;
	//axis->specializationConstants.pushConstants = &axis->pushConstants;
	axis->specializationConstants.M_size.type = 31;
	axis->specializationConstants.M_size.data.i = app->localFFTPlan->axisSplit[0][axis_upload_id];
	axis->specializationConstants.M_size_pow2.type = 31;
	int64_t tempM = axis->specializationConstants.M_size.data.i;
	if (!app->configuration.upperBanded)
	{
		if (axis->specializationConstants.performALT)
			tempM += 2*(app->configuration.numConsecutiveJWIterations - 1);
		else
			tempM += (app->configuration.numConsecutiveJWIterations - 1);
	}
	axis->specializationConstants.M_size_pow2.data.i = (int64_t)pow(2, (int)ceil(log2((double)tempM)));

	axis->specializationConstants.size[0].type = 31;
	axis->specializationConstants.size[0].data.i = app->configuration.M_size;
	if ((axis_upload_id>0) && (axis_upload_id <= (app->localFFTPlan->numAxisUploads[0] / 2))) {
		axis->specializationConstants.size[0].data.i = 2 * ((app->localFFTPlan->axes[0][axis_upload_id - 1].specializationConstants.size[0].data.i + app->localFFTPlan->axisSplit[0][axis_upload_id - 1] - 1) / app->localFFTPlan->axisSplit[0][axis_upload_id - 1]);
	}
	if (axis_upload_id > (app->localFFTPlan->numAxisUploads[0] / 2)) {
		axis->specializationConstants.size[0].data.i = app->localFFTPlan->axes[0][app->localFFTPlan->numAxisUploads[0] - axis_upload_id - 1].specializationConstants.size[0].data.i;
	}
	axis->specializationConstants.size[1].type = 31;
	axis->specializationConstants.size[1].data.i = app->configuration.size[1];
	axis->specializationConstants.size[2].type = 31;
	axis->specializationConstants.size[2].data.i = app->configuration.size[2];
	axis->specializationConstants.numAxisUploads = app->localFFTPlan->numAxisUploads[0];

	axis->specializationConstants.jw_control_bitmask = app->configuration.jw_control_bitmask;
	axis->specializationConstants.performMatVecMul = (((app->configuration.jw_type % 10) != 2) && ((axis_upload_id == 0) || (axis_upload_id == (app->localFFTPlan->numAxisUploads[0] - 1)))) ? 1 : 0;
	axis->specializationConstants.performTriSolve = ((app->configuration.jw_type%10)!=1) ? 1 : 0;
	if((app->configuration.jw_type % 10) == 3) axis->specializationConstants.performTriSolve = 2;

	axis->specializationConstants.useParallelThomas = 1;
	if ((app->configuration.jw_type > 1000) || ((axis->specializationConstants.M_size.data.i <= app->configuration.warpSize) && (app->configuration.numConsecutiveJWIterations == 1))) axis->specializationConstants.useParallelThomas = 0;
	
	axis->specializationConstants.sharedMatricesUpload = 0;
	axis->specializationConstants.num_warps_data_parallel = 1;//(axis->specializationConstants.useParallelThomas && (axis->specializationConstants.size[1].data.i > 100)) ? 8 : 1;
	
	axis->specializationConstants.logicalWarpSize = ((uint64_t)ceil(tempM / (double)(1024))) * 64;
	if (tempM < 512) axis->specializationConstants.logicalWarpSize = app->configuration.warpSize;
	if (axis->specializationConstants.logicalWarpSize < app->configuration.warpSize) axis->specializationConstants.logicalWarpSize = app->configuration.warpSize;
	if (axis->specializationConstants.logicalWarpSize % app->configuration.warpSize) axis->specializationConstants.logicalWarpSize = app->configuration.warpSize * ((axis->specializationConstants.logicalWarpSize + app->configuration.warpSize - 1)/app->configuration.warpSize);
	
	axis->specializationConstants.useUncoalescedJWTnoSharedMemory = 0;//(axis->specializationConstants.sharedMatricesUpload || axis->specializationConstants.num_warps_data_parallel || (axis->specializationConstants.useParallelThomas!)) 0 : 0;
	
	axis->specializationConstants.useMultipleInputBuffers = app->configuration.useMultipleInputBuffers;
	axis->specializationConstants.numConsecutiveJWIterations = app->configuration.numConsecutiveJWIterations;
	//axis->specializationConstants.GivensSteps = 1;
	res = initMemoryParametersAPI(app, &axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	if (app->configuration.quadDoubleDoublePrecision) {
		axis->specializationConstants.inputNumberByteSize = 16;
		axis->specializationConstants.outputNumberByteSize = 16;
	}
	else if (app->configuration.doublePrecision) {
		axis->specializationConstants.inputNumberByteSize = 8;
		axis->specializationConstants.outputNumberByteSize = 8;
	}
	else {
		axis->specializationConstants.inputNumberByteSize = 4;
		axis->specializationConstants.outputNumberByteSize = 4;
	}


	axis->specializationConstants.registers_per_thread = 1;
	int numWarps = (uint64_t)ceil((axis->specializationConstants.M_size.data.i / (double)app->configuration.warpSize) / 4.0);
	if (axis->specializationConstants.useParallelThomas) numWarps = axis->specializationConstants.logicalWarpSize / app->configuration.warpSize;// (uint64_t)ceil((app->configuration.M_size_pow2 / (double)app->configuration.warpSize) / 8.0);
	if ((numWarps * app->configuration.warpSize) > app->configuration.maxThreadsNum) numWarps = app->configuration.maxThreadsNum / app->configuration.warpSize;
	
	if (axis->specializationConstants.useParallelThomas) {
		axis->specializationConstants.registers_per_thread = (uint64_t)ceil(tempM / (double)(app->configuration.warpSize*numWarps));
		if ((app->configuration.vendorID == 0x1002) && (axis->specializationConstants.registers_per_thread % 2 == 0)) axis->specializationConstants.registers_per_thread++;
	}
	else {
		axis->axisBlock[0] = axis->specializationConstants.M_size.data.i / numWarps;
		while (axis->axisBlock[0] > app->configuration.warpSize) {
			axis->axisBlock[0] = (uint64_t)ceil(axis->axisBlock[0] / 2.0);
			axis->specializationConstants.registers_per_thread *= 2;
		}
	}
	axis->axisBlock[0] = axis->specializationConstants.num_warps_data_parallel * numWarps * axis->specializationConstants.warpSize;// ((uint64_t)ceil(axis->axisBlock[0] / (double)axis->specializationConstants.warpSize))* axis->specializationConstants.warpSize;
	axis->axisBlock[1] = 1;
	axis->axisBlock[2] = 1;
	axis->specializationConstants.num_threads = (axis->axisBlock[0] / axis->specializationConstants.num_warps_data_parallel) * axis->axisBlock[1] * axis->axisBlock[2];

	axis->specializationConstants.localSize[0].type = 31;
	axis->specializationConstants.localSize[0].data.i = axis->axisBlock[0];
	axis->specializationConstants.localSize[1].type = 31;
	axis->specializationConstants.localSize[1].data.i = axis->axisBlock[1];
	axis->specializationConstants.localSize[2].type = 31;
	axis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];

	res = initParametersAPI_JW(app, &axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}

	axis->specializationConstants.usedSharedMemory.type = 31;
	if ((axis->specializationConstants.useParallelThomas) || (axis->specializationConstants.performALT) || (axis->specializationConstants.numAxisUploads > 1)){
		int64_t shared_stride = 2*(axis->specializationConstants.registers_per_thread / 2) + 1;
		int64_t temp_scale = 1;
		if (axis->specializationConstants.num_warps_data_parallel > 1) temp_scale = ((axis->specializationConstants.num_warps_data_parallel > 3) || (!axis->specializationConstants.sharedMatricesUpload)) ? axis->specializationConstants.num_warps_data_parallel : 3;
		if (axis->specializationConstants.useUncoalescedJWTnoSharedMemory) temp_scale = 0;
		axis->specializationConstants.usedSharedMemory.data.i = temp_scale*(shared_stride * axis->specializationConstants.warpSize * numWarps) * (axis->specializationConstants.complexSize/2);// 4 * axis->specializationConstants.M_size * axis->specializationConstants.dataTypeSize;
	}
	else
		axis->specializationConstants.usedSharedMemory.data.i = (axis->specializationConstants.num_threads == axis->specializationConstants.warpSize) ? 0 : ((numWarps+1) * axis->specializationConstants.warpSize) * axis->specializationConstants.registers_per_thread * (axis->specializationConstants.complexSize/2);// 4 * axis->specializationConstants.M_size * axis->specializationConstants.dataTypeSize;
	//configure strides


	PfContainer* axisStride = axis->specializationConstants.inputStride;

	axisStride[0].type = 31;
	axisStride[0].data.i = 1;

	if (app->configuration.jw_control_bitmask & (RUNTIME_INPUTBUFFERSTRIDE)) {
		axisStride[1].type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axisStride[1], 50);
		sprintf(axisStride[1].name, "inputBufferStride");
	}
	else
	{
		axisStride[1].type = 31;
		axisStride[1].data.i = app->configuration.inputBufferStride[0];
	}

	/*axisStride[1].type = 31;
	axisStride[1].data.i = app->configuration.inputBufferStride[0];
	axisStride[2].type = 31;
	axisStride[2].data.i = axisStride[1].data.i * axis->specializationConstants.size[1].data.i;

	axisStride[3].type = 31;
	axisStride[3].data.i = axisStride[2].data.i * axis->specializationConstants.size[2].data.i;

	axisStride[4].type = 31;
	axisStride[4].data.i = axisStride[3].data.i;*/

	axisStride = axis->specializationConstants.outputStride;

	axisStride[0].type = 31;
	axisStride[0].data.i = 1;


	if (app->configuration.jw_control_bitmask & (RUNTIME_OUTPUTBUFFERSTRIDE)) {
		axisStride[1].type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axisStride[1], 50);
		sprintf(axisStride[1].name, "outputBufferStride");
	}
	else
	{
		axisStride[1].type = 31;
		axisStride[1].data.i = app->configuration.outputBufferStride[0];
	}
	/*axisStride[1].type = 31;
	axisStride[1].data.i = app->configuration.outputBufferStride[0];
	axisStride[2].type = 31;
	axisStride[2].data.i = axisStride[1].data.i * axis->specializationConstants.size[1].data.i;

	axisStride[3].type = 31;
	axisStride[3].data.i = axisStride[2].data.i * axis->specializationConstants.size[2].data.i;

	axisStride[4].type = 31;
	axisStride[4].data.i = axisStride[3].data.i;*/
	//axisStride[1] = app->configuration.outputBufferStride[0];
	//axisStride[2] = app->configuration.outputBufferStride[0] * axis->specializationConstants.size[1];
	//axisStride[3] = axisStride[2];
	//axisStride[4] = axisStride[3];

	/*if ((axis->specializationConstants.inputStride[1] == axis->specializationConstants.M_size - 1) && (!axis->specializationConstants.upperBanded)) {
		axis->specializationConstants.M_size-=1;

	}
	axis->specializationConstants.offsetM = app->configuration.offsetMV;
	axis->specializationConstants.offsetV = ((app->configuration.jw_type%10)!=2) ? app->configuration.offsetMV + 2 * axis->specializationConstants.inputStride[1] : app->configuration.offsetMV;
	axis->specializationConstants.offsetSolution = app->configuration.offsetSolution;*/
	if (app->configuration.jw_control_bitmask & (RUNTIME_OFFSETM)) {
		axis->specializationConstants.offsetM.type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.offsetM, 50);
		sprintf(axis->specializationConstants.offsetM.name, "offsetM");
	}
	else
	{
		axis->specializationConstants.offsetM.type = 31;
		axis->specializationConstants.offsetM.data.i = app->configuration.offsetM;
	}

	if (app->configuration.jw_control_bitmask & (RUNTIME_OFFSETV)) {
		axis->specializationConstants.offsetV.type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.offsetV, 50);
		sprintf(axis->specializationConstants.offsetV.name, "offsetV");
	}
	else
	{
		axis->specializationConstants.offsetV.type = 31;
		axis->specializationConstants.offsetV.data.i = app->configuration.offsetV;
	}

	if (app->configuration.jw_control_bitmask & (RUNTIME_OFFSETSOLUTION)) {
		axis->specializationConstants.offsetSolution.type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.offsetSolution, 50);
		sprintf(axis->specializationConstants.offsetSolution.name, "offsetSolution");
	}
	else
	{
		axis->specializationConstants.offsetSolution.type = 31;
		axis->specializationConstants.offsetSolution.data.i = app->configuration.offsetSolution;
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_INPUTZEROPAD)) {
		axis->specializationConstants.inputZeropad[0].type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.inputZeropad[0], 50);
		sprintf(axis->specializationConstants.inputZeropad[0].name, "inputZeropad_0");

		axis->specializationConstants.inputZeropad[1].type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.inputZeropad[1], 50);
		sprintf(axis->specializationConstants.inputZeropad[1].name, "inputZeropad_1");
	}
	else
	{
		axis->specializationConstants.inputZeropad[0].type = 31;
		axis->specializationConstants.inputZeropad[0].data.i = app->configuration.inputZeropad[0];
		axis->specializationConstants.inputZeropad[1].type = 31;
		axis->specializationConstants.inputZeropad[1].data.i = app->configuration.inputZeropad[1];
	}

	if (app->configuration.jw_control_bitmask & (RUNTIME_OUTPUTZEROPAD)) {
		axis->specializationConstants.outputZeropad[0].type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.outputZeropad[0], 50);
		sprintf(axis->specializationConstants.outputZeropad[0].name, "outputZeropad_0");

		axis->specializationConstants.outputZeropad[1].type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.outputZeropad[1], 50);
		sprintf(axis->specializationConstants.outputZeropad[1].name, "outputZeropad_1");
	}
	else
	{
		axis->specializationConstants.outputZeropad[0].type = 31;
		axis->specializationConstants.outputZeropad[0].data.i = app->configuration.outputZeropad[0];
		axis->specializationConstants.outputZeropad[1].type = 31;
		axis->specializationConstants.outputZeropad[1].data.i = app->configuration.outputZeropad[1];
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_SCALEC)) {
		axis->specializationConstants.scaleC.type = 100 + axis->specializationConstants.floatTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.scaleC, 50);
		sprintf(axis->specializationConstants.scaleC.name, "scaleC");
	}
	else
	{
		axis->specializationConstants.scaleC.type = 32;
		axis->specializationConstants.scaleC.data.d = app->configuration.scaleC;
	}
	axis->specializationConstants.upperBanded = app->configuration.upperBanded;

	axis->specializationConstants.inputOffset.type = 31;
	axis->specializationConstants.inputOffset.data.i = 0;
	if ((axis_upload_id > 1) && (axis_upload_id <= (app->localFFTPlan->numAxisUploads[0] / 2))) {
		axis->specializationConstants.inputOffset.data.i =  app->localFFTPlan->axes[0][axis_upload_id - 1].specializationConstants.outputOffset.data.i;// app->localFFTPlan->axes[0][axis_upload_id - 1].specializationConstants.inputOffset.data.i + 3 * app->localFFTPlan->axes[0][axis_upload_id - 1].specializationConstants.size[0].data.i * app->localFFTPlan->axes[0][axis_upload_id - 1].specializationConstants.size[1].data.i;
	}
	if (axis_upload_id > (app->localFFTPlan->numAxisUploads[0] / 2)) {
		axis->specializationConstants.inputOffset.data.i = app->localFFTPlan->axes[0][app->localFFTPlan->numAxisUploads[0] - axis_upload_id - 1].specializationConstants.inputOffset.data.i;
	}
	axis->specializationConstants.outputOffset.type = 31;
	axis->specializationConstants.outputOffset.data.i = 0;
	if ((axis_upload_id >= 1) && (axis_upload_id < (app->localFFTPlan->numAxisUploads[0] / 2))) {
		axis->specializationConstants.outputOffset.data.i = app->localFFTPlan->axes[0][axis_upload_id].specializationConstants.inputOffset.data.i + 3 * app->localFFTPlan->axes[0][axis_upload_id].specializationConstants.size[0].data.i * app->localFFTPlan->axes[0][axis_upload_id].specializationConstants.size[1].data.i;
	}
	if (axis_upload_id >= (app->localFFTPlan->numAxisUploads[0] / 2)) {
		axis->specializationConstants.outputOffset.data.i = app->localFFTPlan->axes[0][axis_upload_id].specializationConstants.inputOffset.data.i;
	}
	axis->specializationConstants.kernelOffset.type = 31;
	axis->specializationConstants.kernelOffset.data.i = 0;
	if (axis_upload_id > (app->localFFTPlan->numAxisUploads[0] / 2)) {
		axis->specializationConstants.kernelOffset.data.i = app->localFFTPlan->axes[0][axis_upload_id-1].specializationConstants.outputOffset.data.i;
	}
	res = PfSolveConfigureDescriptors(app, plan, axis, 0, 0, 0);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = PfSolveCheckUpdateBufferSet(app, axis, 1, 0);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}

	if ((axis_upload_id > 0) && (axis_upload_id < (app->localFFTPlan->numAxisUploads[0]-1))) axis->specializationConstants.performOffsetUpdate = 0;
	
	res = PfSolveUpdateBufferSet(app, plan, axis, 0,0,0);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	/*
#if(VKFFT_BACKEND==0)
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;
	if (axis->pushConstants.structSize > 0) {
		VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
		pushConstantRange.offset = 0;
		pushConstantRange.size = axis->pushConstants.structSize;
		// Push constant ranges are part of the pipeline layout
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
	}
	result = vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, 0, &axis->pipelineLayout);
	if (result != VK_SUCCESS) {
		deletePfSolve(app);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT;
	}
#endif*/
	
	uint64_t tempSize[3] = { (uint64_t)ceil((((uint64_t)ceil(axis->specializationConstants.size[1].data.i / (double)axis->specializationConstants.num_warps_data_parallel)) * app->configuration.size[2]) / (double)(axis->axisBlock[1])), 1, 1 };


	if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
	else  axis->specializationConstants.performWorkGroupShift[0] = 0;
	if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
	else  axis->specializationConstants.performWorkGroupShift[1] = 0;
	if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
	else  axis->specializationConstants.performWorkGroupShift[2] = 0;

    //app->localFFTPlan->numAxisUploads[0] = 1;
	//res = PfSolve_getPushConstantsSize(&axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}

	axis->specializationConstants.code0 = (char*)malloc(sizeof(char) * app->configuration.maxCodeLength);
	//axis->specializationConstants.output = axis->specializationConstants.code0;
	if (!axis->specializationConstants.code0) {
		deletePfSolve(app);
		return PFSOLVE_ERROR_MALLOC_FAILED;
	}
#if(VKFFT_BACKEND==0)
	sprintf(axis->specializationConstants.PfSolveFunctionName, "main");
#else
	sprintf(axis->specializationConstants.PfSolveFunctionName, "%s", app->kernelName);
#endif
	res = PfSolve_shaderGen_JonesWorlandMV(&axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}

	axis->pushConstants.structSize = axis->specializationConstants.pushConstantsStructSize;

	res = PfSolve_CompileKernel(app, axis);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_OFFSETM)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.offsetM);
	}
	
	if (app->configuration.jw_control_bitmask & (RUNTIME_OFFSETV)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.offsetV);
	}

	if (app->configuration.jw_control_bitmask & (RUNTIME_OFFSETSOLUTION)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.offsetSolution);
	}
	
	if (app->configuration.jw_control_bitmask & (RUNTIME_INPUTZEROPAD)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.inputZeropad[0]);
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.inputZeropad[1]);
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_OUTPUTZEROPAD)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.outputZeropad[0]);
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.outputZeropad[1]);
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_SCALEC)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.scaleC);
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_INPUTBUFFERSTRIDE)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.inputStride[1]);
	}
	if (app->configuration.jw_control_bitmask & (RUNTIME_OUTPUTBUFFERSTRIDE)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.outputStride[1]);
	}
	if (!app->configuration.keepShaderCode) {
		free(axis->specializationConstants.code0);
		axis->specializationConstants.code0 = 0;
	}
	res = freeParametersAPI_JW(app, &axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = freeMemoryParametersAPI(app, &axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	return res;
}

#endif
