// This file is part of PfSolve
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This file is provided for informational purposes only. Redistribution without permission is not allowed.
#ifndef PFSOLVE_PLAN_FINITEDIFFERENCES_H
#define PFSOLVE_PLAN_FINITEDIFFERENCES_H
#include "pfSolve_Structs/pfSolve_Structs.h"
//#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel2/pfSolve_FiniteDifferences.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_InitAPIParameters.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_CompileKernel.h"
#include "pfSolve_AppManagement/pfSolve_DeleteApp.h"
static inline PfSolveResult PfSolve_Plan_FiniteDifferences(PfSolveApplication* app, PfSolvePlan* plan) {
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

	//axis->specializationConstants.doublePrecision = app->configuration.doublePrecision;
	axis->specializationConstants.numCoordinates = app->configuration.coordinateFeatures;
	axis->specializationConstants.normalize = app->configuration.normalize;
	axis->specializationConstants.axis_id = 0;
	axis->specializationConstants.axis_upload_id = 0;
	axis->specializationConstants.LUT = 0;// app->configuration.useLUT;
	//axis->specializationConstants.pushConstants = &axis->pushConstants;
	
	//axis->specializationConstants.M_size.type = 31;
	//axis->specializationConstants.M_size.data.i = app->configuration.M_size;// (uint64_t)ceil(pow(2, ceil(log2((double)app->configuration.M_size))));

	axis->specializationConstants.size[0].type = 31;
	axis->specializationConstants.size[0].data.i = app->configuration.size[0];
	//axis->specializationConstants.M_size_pow2 = (uint64_t)ceil(pow(2, ceil(log2((double)app->configuration.M_size))));

	axis->specializationConstants.size[1].type = 31;
	axis->specializationConstants.size[1].data.i = app->configuration.size[1];

	axis->specializationConstants.size[2].type = 31;
	axis->specializationConstants.size[2].data.i = app->configuration.size[2];

	axis->specializationConstants.logicBlock[0].type = 31;
	axis->specializationConstants.logicBlock[0].data.i = app->configuration.logicBlock[0];
	axis->specializationConstants.logicBlock[1].type = 31;
	axis->specializationConstants.logicBlock[1].data.i = app->configuration.logicBlock[1];
	axis->specializationConstants.logicBlock[2].type = 31;
	axis->specializationConstants.logicBlock[2].data.i = app->configuration.logicBlock[2];

	/*axis->specializationConstants.inputZeropad[0] = app->configuration.inputZeropad[0];
	axis->specializationConstants.inputZeropad[1] = app->configuration.inputZeropad[1];
	axis->specializationConstants.outputZeropad[0] = app->configuration.outputZeropad[0];
	axis->specializationConstants.outputZeropad[1] = app->configuration.outputZeropad[1];
	
	axis->specializationConstants.GivensSteps = 1;*/
	axis->specializationConstants.s_dx.type = 32;
	axis->specializationConstants.s_dx.data.d = app->configuration.s_dx;
	axis->specializationConstants.s_dy.type = 32;
	axis->specializationConstants.s_dy.data.d = app->configuration.s_dy;
	axis->specializationConstants.s_dz.type = 32;
	axis->specializationConstants.s_dz.data.d = app->configuration.s_dz;
	axis->specializationConstants.s_dt_D.type = 32;
	axis->specializationConstants.s_dt_D.data.d = app->configuration.s_dt_D;

	axis->specializationConstants.usedSharedMemory.type = 31;
	axis->specializationConstants.usedSharedMemory.data.i = 0;// 4 * axis->specializationConstants.M_size * axis->specializationConstants.dataTypeSize;
	//configure strides

	PfContainer* axisStride = axis->specializationConstants.inputStride;

	axisStride[0].type = 31;
	axisStride[0].data.i = 1;

	axisStride[1].type = 31;
	axisStride[1].data.i = axis->specializationConstants.size[0].data.i;
	axisStride[2].type = 31;
	axisStride[2].data.i = axisStride[1].data.i * axis->specializationConstants.size[1].data.i;

	axisStride[3].type = 31;
	axisStride[3].data.i = axisStride[2].data.i * axis->specializationConstants.size[2].data.i;

	axisStride[4].type = 31;
	axisStride[4].data.i = axisStride[3].data.i;

	axisStride = axis->specializationConstants.outputStride;

	axisStride[0].type = 31;
	axisStride[0].data.i = 1;

	axisStride[1].type = 31;
	axisStride[1].data.i = axis->specializationConstants.size[0].data.i;
	axisStride[2].type = 31;
	axisStride[2].data.i = axisStride[1].data.i * axis->specializationConstants.size[1].data.i;

	axisStride[3].type = 31;
	axisStride[3].data.i = axisStride[2].data.i * axis->specializationConstants.size[2].data.i;

	axisStride[4].type = 31;
	axisStride[4].data.i = axisStride[3].data.i;

	/*if ((axis->specializationConstants.inputStride[1] == axis->specializationConstants.M_size - 1) && (!axis->specializationConstants.upperBanded)) {
		axis->specializationConstants.M_size-=1;

	}
	axis->specializationConstants.offsetM = app->configuration.offsetMV;
	axis->specializationConstants.offsetV = ((app->configuration.jw_type%10)!=2) ? app->configuration.offsetMV + 2 * axis->specializationConstants.inputStride[1] : app->configuration.offsetMV;
	axis->specializationConstants.offsetSolution = app->configuration.offsetSolution;*/
	axis->specializationConstants.inputOffset.type = 31;
	axis->specializationConstants.inputOffset.data.i = 0;
	axis->specializationConstants.outputOffset.type = 31;
	axis->specializationConstants.outputOffset.data.i = 0;

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
	axis->specializationConstants.registers_per_thread = axis->specializationConstants.logicBlock[1].data.i* axis->specializationConstants.logicBlock[2].data.i * (uint64_t)ceil(axis->specializationConstants.logicBlock[0].data.i / (double)axis->specializationConstants.warpSize);

	axis->axisBlock[0] = axis->specializationConstants.warpSize;// ((uint64_t)ceil(axis->axisBlock[0] / (double)axis->specializationConstants.warpSize))* axis->specializationConstants.warpSize;
	axis->axisBlock[1] = 1;
	axis->axisBlock[2] = 1;
	//axis->specializationConstants.num_threads = axis->axisBlock[0] * axis->axisBlock[1] * axis->axisBlock[2];
	uint64_t tempSize[3] = { (uint64_t)ceil((app->configuration.size[0]) / (double)(axis->specializationConstants.logicBlock[0].data.i)), (uint64_t)ceil((app->configuration.size[1]) / (double)(axis->specializationConstants.logicBlock[1].data.i)), (uint64_t)ceil((app->configuration.size[2]) / (double)(axis->specializationConstants.logicBlock[2].data.i)) };


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

	/*res = PfSolve_getPushConstantsSize(&axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}*/

	res = initMemoryParametersAPI(app, &axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	axis->specializationConstants.inputMemoryCode = axis->specializationConstants.floatTypeInputMemoryCode;
	switch ((axis->specializationConstants.inputMemoryCode % 100) / 10) {
	case 0:
		axis->specializationConstants.inputNumberByteSize = 2;
		break;
	case 1:
		axis->specializationConstants.inputNumberByteSize = 4;
		break;
	case 2:
		axis->specializationConstants.inputNumberByteSize = 8;
		break;
	}
	axis->specializationConstants.outputMemoryCode = axis->specializationConstants.floatTypeOutputMemoryCode;
	switch ((axis->specializationConstants.outputMemoryCode % 100) / 10) {
	case 0:
		axis->specializationConstants.outputNumberByteSize = 2;
		break;
	case 1:
		axis->specializationConstants.outputNumberByteSize = 4;
		break;
	case 2:
		axis->specializationConstants.outputNumberByteSize = 8;
		break;
	}

	res = initParametersAPI_Pf(app, &axis->specializationConstants);
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
	//if (app->configuration.compute_flux_D)
	//	res = PfSolve_shaderGen_compute_fluxD(&axis->specializationConstants);
#if(VKFFT_BACKEND==0)
	sprintf(axis->specializationConstants.PfSolveFunctionName, "main");
#else
	sprintf(axis->specializationConstants.PfSolveFunctionName, "%s", app->kernelName);
#endif
	res = PfSolve_shaderGen_compute_Pf(&axis->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = PfSolve_CompileKernel(app, axis);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	freeMemoryParametersAPI(app, &axis->specializationConstants);
	freeParametersAPI_Pf(app, &axis->specializationConstants);
	return res;
}

#endif
