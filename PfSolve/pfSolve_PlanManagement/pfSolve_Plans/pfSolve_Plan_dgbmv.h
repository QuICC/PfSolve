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
#ifndef PFSOLVE_PLAN_DGBMV_H
#define PFSOLVE_PLAN_DGBMV_H
#include "pfSolve_Structs/pfSolve_Structs.h"
//#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel2/pfSolve_dgbmv.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_InitAPIParameters.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_CompileKernel.h"
#include "pfSolve_AppManagement/pfSolve_DeleteApp.h"
static inline PfSolveResult PfSolve_Plan_dgbmv(PfSolveApplication* app, PfSolvePlan* plan) {
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
	PfSolveAxis* axis = plan->axes[0];

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
		axis->specializationConstants.storeSharedComplexComponentsSeparately = 1;
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
	
	//axis->specializationConstants.doublePrecision = app->configuration.doublePrecision;
	axis->specializationConstants.numCoordinates = app->configuration.coordinateFeatures;
	axis->specializationConstants.normalize = app->configuration.normalize;
	axis->specializationConstants.axis_id = 0;
	axis->specializationConstants.axis_upload_id = 0;
	axis->specializationConstants.LUT = 0;// app->configuration.useLUT;
	//axis->specializationConstants.pushConstants = &axis->pushConstants;
	axis->specializationConstants.M_size.type = 31;
	axis->specializationConstants.M_size.data.i = app->configuration.M_size;
	axis->specializationConstants.M_size_pow2.type = 31;
	axis->specializationConstants.M_size_pow2.data.i = app->configuration.M_size_pow2;

	axis->specializationConstants.size[0].type = 31;
	axis->specializationConstants.size[0].data.i = app->configuration.M_size;
	axis->specializationConstants.size[1].type = 31;
	axis->specializationConstants.size[1].data.i = 1;//app->configuration.size[1];
	axis->specializationConstants.size[2].type = 31;
	axis->specializationConstants.size[2].data.i = 1;
	
	axis->specializationConstants.jw_control_bitmask = app->configuration.jw_control_bitmask;

	
	axis->specializationConstants.LDA = app->configuration.LDA;
	axis->specializationConstants.KU = app->configuration.KU;
	axis->specializationConstants.KL = app->configuration.KL;

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

	res = initParametersAPI_JW(app, &axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}


	axis->specializationConstants.usedSharedMemory.type = 31;
	axis->specializationConstants.usedSharedMemory.data.i = axis->specializationConstants.size[0].data.i * axis->specializationConstants.size[1].data.i * sizeof(float);
    if(app->configuration.doublePrecision)
            axis->specializationConstants.usedSharedMemory.data.i *= 2;
	//axis->specializationConstants.registers_per_thread = 1;

	//axis->axisBlock[0] = axis->specializationConstants.size[0].data.i;// (axis->specializationConstants.size[1] > app->configuration.aimThreads) ? app->configuration.aimThreads : axis->specializationConstants.size[1];
	axis->specializationConstants.registers_per_thread = (int)ceil(app->configuration.size[0] / (double)app->configuration.aimThreads);

	axis->axisBlock[0] = (uint64_t)ceil(app->configuration.size[0]/ (double)axis->specializationConstants.registers_per_thread);
	
	axis->axisBlock[1] = 1;
	axis->axisBlock[2] = 1;
	axis->specializationConstants.num_threads = axis->axisBlock[0] * axis->axisBlock[1] * axis->axisBlock[2];
	uint64_t tempSize[3] = { (uint64_t)ceil((app->configuration.size[1] * app->configuration.size[2])), 1, 1 };


	if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
	else  axis->specializationConstants.performWorkGroupShift[0] = 0;
	if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
	else  axis->specializationConstants.performWorkGroupShift[1] = 0;
	if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
	else  axis->specializationConstants.performWorkGroupShift[2] = 0;

	axis->specializationConstants.localSize[0].type = 31;
	axis->specializationConstants.localSize[0].data.i = axis->axisBlock[0];
	axis->specializationConstants.localSize[1].type = 31;
	axis->specializationConstants.localSize[1].data.i = axis->axisBlock[1];
	axis->specializationConstants.localSize[2].type = 31;
	axis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];
	app->localFFTPlan->numAxisUploads[0] = 1;
	//configure strides

	PfContainer* axisStride = axis->specializationConstants.inputStride;

	axisStride[0].type = 31;
	axisStride[0].data.i = 1;

	if (app->configuration.jw_control_bitmask & (1 << 7)) {
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


	if (app->configuration.jw_control_bitmask & (1 << 8)) {
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

	if (app->configuration.jw_control_bitmask & (1 << 1)) {
		axis->specializationConstants.offsetM.type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.offsetM, 50);
		sprintf(axis->specializationConstants.offsetM.name, "offsetM");
	}
	else
	{
		axis->specializationConstants.offsetM.type = 31;
		axis->specializationConstants.offsetM.data.i = app->configuration.offsetM;
	}

	if (app->configuration.jw_control_bitmask & (1 << 2)) {
		axis->specializationConstants.offsetV.type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.offsetV, 50);
		sprintf(axis->specializationConstants.offsetV.name, "offsetV");
	}
	else
	{
		axis->specializationConstants.offsetV.type = 31;
		axis->specializationConstants.offsetV.data.i = app->configuration.offsetV;
	}

	if (app->configuration.jw_control_bitmask & (1 << 3)) {
		axis->specializationConstants.offsetSolution.type = 100 + axis->specializationConstants.uintTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.offsetSolution, 50);
		sprintf(axis->specializationConstants.offsetSolution.name, "offsetSolution");
	}
	else
	{
		axis->specializationConstants.offsetSolution.type = 31;
		axis->specializationConstants.offsetSolution.data.i = app->configuration.offsetSolution;
	}
	if (app->configuration.jw_control_bitmask & (1 << 4)) {
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

	if (app->configuration.jw_control_bitmask & (1 << 5)) {
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

	if (app->configuration.jw_control_bitmask & (1 << 6)) {
		axis->specializationConstants.scaleC.type = 100 + axis->specializationConstants.floatTypeCode;
		PfAllocateContainerFlexible(&axis->specializationConstants, &axis->specializationConstants.scaleC, 50);
		sprintf(axis->specializationConstants.scaleC.name, "scaleC");
	}
	else
	{
		axis->specializationConstants.scaleC.type = 32;
		axis->specializationConstants.scaleC.data.d = app->configuration.scaleC;
	}
	//axis->specializationConstants.offsetM = app->configuration.offsetMV;
	//axis->specializationConstants.offsetV = ((app->configuration.jw_type%10)!=2) ? app->configuration.offsetMV + 2 * axis->specializationConstants.inputStride[1] : app->configuration.offsetMV;

	axis->specializationConstants.upperBound = app->configuration.upperBound;

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
	res = PfSolveUpdateBufferSet(app, plan, axis, 0, 0, 0);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
#if(VKFFT_BACKEND==0)
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &axis->descriptorSetLayout;

	VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
	pushConstantRange.offset = 0;
	pushConstantRange.size = axis->pushConstants.structSize;
	// Push constant ranges are part of the pipeline layout
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

	result = vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, 0, &axis->pipelineLayout);
	if (result != VK_SUCCESS) {
		deletePfSolve(app);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT;
	}
#endif


	axis->specializationConstants.code0 = (char*)malloc(sizeof(char) * app->configuration.maxCodeLength);
	//axis->specializationConstants.output = axis->specializationConstants.code0;
	if (!axis->specializationConstants.code0) {
		deletePfSolve(app);
		return PFSOLVE_ERROR_MALLOC_FAILED;
	}
	#if(VKFFT_BACKEND==0)
	sprintf(axis->PfSolveFunctionName, "main");
#else
	sprintf(axis->PfSolveFunctionName, "PfSolve_main");
#endif
	res = PfSolve_shaderGen_dgbmv(&axis->specializationConstants);
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
	if (app->configuration.jw_control_bitmask & (1 << 1)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.offsetM);
	}
	
	if (app->configuration.jw_control_bitmask & (1 << 2)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.offsetV);
	}

	if (app->configuration.jw_control_bitmask & (1 << 3)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.offsetSolution);
	}
	if (app->configuration.jw_control_bitmask & (1 << 4)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.inputZeropad[0]);
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.inputZeropad[1]);
	}
	if (app->configuration.jw_control_bitmask & (1 << 5)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.outputZeropad[0]);
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.outputZeropad[1]);
	}
	if (app->configuration.jw_control_bitmask & (1 << 6)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.scaleC);
	}
	if (app->configuration.jw_control_bitmask & (1 << 7)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.inputStride[1]);
	}
	if (app->configuration.jw_control_bitmask & (1 << 8)) {
		PfDeallocateContainer(&axis->specializationConstants, &axis->specializationConstants.outputStride[1]);
	}
	return res;
}

#endif
