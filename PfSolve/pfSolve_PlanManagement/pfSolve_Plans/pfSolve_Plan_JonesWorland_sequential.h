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
#ifndef PFSOLVE_PLAN_JONESWORLAND_SEQUENTIAL_H
#define PFSOLVE_PLAN_JONESWORLAND_SEQUENTIAL_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel2/pfSolve_JonesWorland_sequential.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_InitAPIParameters.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_CompileKernel.h"
#include "pfSolve_AppManagement/pfSolve_DeleteApp.h"
static inline PfSolveResult PfSolve_Plan_JonesWorland_sequential(PfSolveApplication* app, PfSolvePlan* plan) {
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
	plan->specializationConstants.warpSize = app->configuration.warpSize;
	plan->specializationConstants.numSharedBanks = app->configuration.numSharedBanks;
	plan->specializationConstants.useUint64 = app->configuration.useUint64;
	plan->specializationConstants.maxCodeLength = app->configuration.maxCodeLength;
	plan->specializationConstants.maxTempLength = app->configuration.maxTempLength;
	plan->specializationConstants.sharedMemSize = app->configuration.sharedMemorySize;
	plan->specializationConstants.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;

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
	
	plan->specializationConstants.numCoordinates = app->configuration.coordinateFeatures;
	plan->specializationConstants.normalize = app->configuration.normalize;
	plan->specializationConstants.axis_id = 0;
	plan->specializationConstants.axis_upload_id = 0;
	plan->specializationConstants.LUT = 0;// app->configuration.useLUT;
	plan->specializationConstants.pushConstants = &plan->pushConstants;
	plan->specializationConstants.M_size.mode = 0;
	plan->specializationConstants.M_size.x_num = app->configuration.M_size;// (uint64_t)ceil(pow(2, ceil(log2((double)app->configuration.M_size))));
	set_x_str_uint(&plan->specializationConstants.M_size);
	plan->specializationConstants.M_size_pow2 = (uint64_t)ceil(pow(2, ceil(log2((double)app->configuration.M_size))));

	plan->specializationConstants.size[0] = (app->configuration.M_size>32) ? 32 : app->configuration.M_size;
	plan->specializationConstants.size[1] = app->configuration.size[1];
	plan->specializationConstants.size[2] = 1;
	plan->specializationConstants.inputZeropad[0]= app->configuration.inputZeropad[0];
	plan->specializationConstants.inputZeropad[1]= app->configuration.inputZeropad[1];
	plan->specializationConstants.outputZeropad[0]= app->configuration.outputZeropad[0];
	plan->specializationConstants.outputZeropad[1]= app->configuration.outputZeropad[1];
	plan->specializationConstants.performMatVecMul = ((app->configuration.jw_type%10)!=2) ? 1 : 0;
	plan->specializationConstants.performTriSolve = ((app->configuration.jw_type%10)!=1) ? 1 : 0;

	plan->specializationConstants.scaleC.mode = 0;
	plan->specializationConstants.scaleC.x_num = (app->configuration.scaleC!=1) ? app->configuration.scaleC : 0;
	set_x_str_float(&plan->specializationConstants.scaleC);

	plan->specializationConstants.GivensSteps = 1;
	uint64_t size1 = (app->configuration.size[1] > app->configuration.aimThreads) ? app->configuration.aimThreads : app->configuration.size[1];
	plan->specializationConstants.sharedStrideReadWriteConflict = size1;
	res = initMemoryParametersAPI(app, &plan->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = initParametersAPI(app, &plan->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}

	plan->specializationConstants.usedSharedMemory = plan->specializationConstants.size[0] * plan->specializationConstants.sharedStrideReadWriteConflict * plan->specializationConstants.dataTypeSize;
	//configure strides

	PfSolve_uint* axisStride = plan->specializationConstants.inputStride;

	axisStride[0].x_num = 1;
	set_x_str_uint(&axisStride[0]);
	axisStride[1].x_num = (app->configuration.inputBufferStride[0]) ? app->configuration.inputBufferStride[0] : plan->specializationConstants.M_size.x_num;
	set_x_str_uint(&axisStride[1]);
	axisStride[2].x_num = axisStride[1].x_num * size1;
	set_x_str_uint(&axisStride[2]);
	axisStride[3].x_num = 4 * axisStride[2].x_num;
	set_x_str_uint(&axisStride[3]);
	axisStride[4].x_num = axisStride[3].x_num;
	set_x_str_uint(&axisStride[4]);

	axisStride = plan->specializationConstants.outputStride;

	axisStride[0].x_num = 1;
	set_x_str_uint(&axisStride[0]);
	axisStride[1].x_num = app->configuration.outputBufferStride[0];
	set_x_str_uint(&axisStride[1]);
	axisStride[2].x_num = axisStride[1].x_num * size1;
	set_x_str_uint(&axisStride[2]);
	axisStride[3].x_num = axisStride[2].x_num;
	set_x_str_uint(&axisStride[3]);
	axisStride[4].x_num = axisStride[3].x_num;
	set_x_str_uint(&axisStride[4]);

	plan->specializationConstants.offsetM.x_num = app->configuration.offsetMV;
	set_x_str_uint(&plan->specializationConstants.offsetM);
	plan->specializationConstants.offsetV.x_num = ((app->configuration.jw_type%10)!=2) ? app->configuration.offsetMV + 2 * plan->specializationConstants.inputStride[1].x_num : app->configuration.offsetMV;
	set_x_str_uint(&plan->specializationConstants.offsetV);
	plan->specializationConstants.offsetSolution.x_num = app->configuration.offsetSolution;
	set_x_str_uint(&plan->specializationConstants.offsetSolution);
	plan->specializationConstants.upperBanded = app->configuration.upperBanded;

	plan->specializationConstants.inputOffset = 0;
	plan->specializationConstants.outputOffset = 0;

	res = PfSolveConfigureDescriptors(app, plan);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = PfSolveCheckUpdateBufferSet(app, plan, 1, 0);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = PfSolveUpdateBufferSet(app, plan);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
#if(VKFFT_BACKEND==0)
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &plan->descriptorSetLayout;

	VkPushConstantRange pushConstantRange = { VK_SHADER_STAGE_COMPUTE_BIT };
	pushConstantRange.offset = 0;
	pushConstantRange.size = plan->pushConstants.structSize;
	// Push constant ranges are part of the pipeline layout
	pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
	pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

	result = vkCreatePipelineLayout(app->configuration.device[0], &pipelineLayoutCreateInfo, 0, &plan->pipelineLayout);
	if (result != VK_SUCCESS) {
		deletePfSolve(app);
		return PFSOLVE_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT;
	}
#endif
	plan->specializationConstants.registers_per_thread = 1;

	plan->axisBlock[0] = size1;// (plan->specializationConstants.size[1] > app->configuration.aimThreads) ? app->configuration.aimThreads : plan->specializationConstants.size[1];
	plan->specializationConstants.registers_per_thread = plan->specializationConstants.size[0];
	plan->axisBlock[1] = 1;
	plan->axisBlock[2] = 1;
	plan->specializationConstants.num_threads = plan->axisBlock[0] * plan->axisBlock[1] * plan->axisBlock[2];
	uint64_t tempSize[3] = { (uint64_t)ceil((app->configuration.size[1] * app->configuration.size[2]) / (double)(plan->axisBlock[0])), 1, 1 };


	if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) plan->specializationConstants.performWorkGroupShift[0] = 1;
	else  plan->specializationConstants.performWorkGroupShift[0] = 0;
	if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) plan->specializationConstants.performWorkGroupShift[1] = 1;
	else  plan->specializationConstants.performWorkGroupShift[1] = 0;
	if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) plan->specializationConstants.performWorkGroupShift[2] = 1;
	else  plan->specializationConstants.performWorkGroupShift[2] = 0;

	plan->specializationConstants.localSize[0] = plan->axisBlock[0];
	plan->specializationConstants.localSize[1] = plan->axisBlock[1];
	plan->specializationConstants.localSize[2] = plan->axisBlock[2];
		

	plan->specializationConstants.code0 = (char*)malloc(sizeof(char) * app->configuration.maxCodeLength);
	plan->specializationConstants.output = plan->specializationConstants.code0;
	if (!plan->specializationConstants.code0) {
		deletePfSolve(app);
		return PFSOLVE_ERROR_MALLOC_FAILED;
	}
	res = PfSolve_getPushConstantsSize(&plan->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = PfSolve_shaderGen_JonesWorlandMV_sequential(&plan->specializationConstants);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	res = PfSolve_CompileKernel(app, plan);
	if (res != PFSOLVE_SUCCESS) {
		deletePfSolve(app);
		return res;
	}
	return res;
}

#endif
