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
#ifndef PFSOLVE_RUNAPP_H
#define PFSOLVE_RUNAPP_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_DispatchPlan.h"
#include "pfSolve_PlanManagement/pfSolve_API_handles/pfSolve_UpdateBuffers.h"

static inline PfSolveResult PfSolveSync(PfSolveApplication* app) {
#if(VKFFT_BACKEND==0)
	vkCmdPipelineBarrier(app->configuration.commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, app->configuration.memory_barrier, 0, 0, 0, 0);
#elif(VKFFT_BACKEND==1)
	if (app->configuration.num_streams > 1) {
		cudaError_t res = cudaSuccess;
		for (uint64_t s = 0; s < app->configuration.num_streams; s++) {
			res = cudaEventSynchronize(app->configuration.stream_event[s]);
			if (res != cudaSuccess) return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
		}
		app->configuration.streamCounter = 0;
	}
#elif(VKFFT_BACKEND==2)
	if (app->configuration.num_streams > 1) {
		hipError_t res = hipSuccess;
		for (uint64_t s = 0; s < app->configuration.num_streams; s++) {
			res = hipEventSynchronize(app->configuration.stream_event[s]);
			if (res != hipSuccess) return PFSOLVE_ERROR_FAILED_TO_SYNCHRONIZE;
		}
		app->configuration.streamCounter = 0;
	}
#elif(VKFFT_BACKEND==3)
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	res = zeCommandListAppendBarrier(app->configuration.commandList[0], nullptr, 0, nullptr);
	if (res != ZE_RESULT_SUCCESS) return PFSOLVE_ERROR_FAILED_TO_SUBMIT_BARRIER;
#elif(VKFFT_BACKEND==5)
#endif
	return PFSOLVE_SUCCESS;
}
static inline void printDebugInformation(PfSolveApplication* app, PfSolveAxis* axis) {
	if (app->configuration.keepShaderCode) printf("%s\n", axis->specializationConstants.code0);
	if (app->configuration.printMemoryLayout) {
		if ((axis->inputBuffer == app->configuration.inputBuffer) && (app->configuration.inputBuffer != app->configuration.buffer))
			printf("read: inputBuffer\n");
		if (axis->inputBuffer == app->configuration.buffer)
			printf("read: buffer\n");
		if (axis->inputBuffer == app->configuration.tempBuffer)
			printf("read: tempBuffer\n");
		if ((axis->inputBuffer == app->configuration.outputBuffer) && (app->configuration.outputBuffer != app->configuration.buffer))
			printf("read: outputBuffer\n");
		if ((axis->outputBuffer == app->configuration.inputBuffer) && (app->configuration.inputBuffer != app->configuration.buffer))
			printf("write: inputBuffer\n");
		if (axis->outputBuffer == app->configuration.buffer)
			printf("write: buffer\n");
		if (axis->outputBuffer == app->configuration.tempBuffer)
			printf("write: tempBuffer\n");
		if ((axis->outputBuffer == app->configuration.outputBuffer) && (app->configuration.outputBuffer != app->configuration.buffer))
			printf("write: outputBuffer\n");
	}
}

static inline PfSolveResult PfSolveAppend(PfSolveApplication* app, int inverse, PfSolveLaunchParams* launchParams) {
	return;
	PfSolveResult res = PFSOLVE_SUCCESS;
#if(VKFFT_BACKEND==0)
	app->configuration.commandBuffer = launchParams->commandBuffer;
	VkMemoryBarrier memory_barrier = {
			VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			0,
			VK_ACCESS_SHADER_WRITE_BIT,
			VK_ACCESS_SHADER_READ_BIT,
	};
	app->configuration.memory_barrier = &memory_barrier;
#elif(VKFFT_BACKEND==1)
	app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==2)
	app->configuration.streamCounter = 0;
#elif(VKFFT_BACKEND==3)
	app->configuration.commandQueue = launchParams->commandQueue;
#endif

	res = PfSolveCheckUpdateBufferSet(app, 0, 0, launchParams);
	if (res != PFSOLVE_SUCCESS) { return res; }
	res = PfSolveUpdateBufferSet(app, app->localFFTPlan, app->localFFTPlan->axes[0], 0, 0, inverse);
	if (res != PFSOLVE_SUCCESS) { return res; }
	res = PfSolveUpdatePushConstants(app, &app->localFFTPlan->axes[0]->pushConstants, launchParams);
	if (res != PFSOLVE_SUCCESS) { return res; }
	uint64_t dispatchBlock[3];
	dispatchBlock[1] = 1;
	dispatchBlock[2] = 1;
	if (app->configuration.finiteDifferences) {
		dispatchBlock[0] = (uint64_t)ceil(app->configuration.size[0] / (double)app->localFFTPlan->axes[0]->specializationConstants.logicBlock[0].data.i);
		dispatchBlock[1] = (uint64_t)ceil(app->configuration.size[1] / (double)app->localFFTPlan->axes[0]->specializationConstants.logicBlock[1].data.i);
		dispatchBlock[2] = (uint64_t)ceil(app->configuration.size[2] / (double)app->localFFTPlan->axes[0]->specializationConstants.logicBlock[2].data.i);
	}
	if (app->configuration.jw_type) {
		dispatchBlock[0] = app->configuration.size[1]* app->configuration.size[2];
		dispatchBlock[1] = 1;// (uint64_t)ceil(app->configuration.size[1] / (double)app->localFFTPlan->axes[0]->specializationConstants.localSize[1].data.i);
		dispatchBlock[2] = 1;// (uint64_t)ceil(app->configuration.size[2] / (double)app->localFFTPlan->axes[0]->specializationConstants.localSize[2].data.i);
	}
	if(app->configuration.LDA)
		dispatchBlock[0] = app->configuration.size[1];
	if (app->configuration.block)
		dispatchBlock[0] = (uint64_t)ceil(app->configuration.size[0] * app->configuration.size[1] / (double)(app->localFFTPlan->axes[0]->axisBlock[0] * app->localFFTPlan->axes[0]->specializationConstants.registers_per_thread));
	/*if(app->configuration.JW_sequential)
		std::cerr << "JW "<< std::endl;
	if(app->configuration.scaleType)
		std::cerr << "scale  "<< std::endl;
	if(app->configuration.copy)
		std::cerr << "copy  "<< std::endl;
	*/
	res = PfSolve_DispatchPlan(app, app->localFFTPlan->axes[0], dispatchBlock);
	if (app->configuration.keepShaderCode) printf("%s\n", app->localFFTPlan->axes[0]->specializationConstants.code0);
	if (res != PFSOLVE_SUCCESS) { return res; }
	return res;
}
#endif
