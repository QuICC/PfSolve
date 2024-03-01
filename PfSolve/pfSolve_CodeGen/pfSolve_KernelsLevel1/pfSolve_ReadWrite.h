// This file is part of PfSolve
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, &including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, &iNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef PFSOLVE_READWRITE_H
#define PFSOLVE_READWRITE_H

#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelUtils.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_Zeropad.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_Transfers.h"
static inline void setReadToRegisters(PfSolveSpecializationConstantsLayout* sc, int readType) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	switch (readType) {
	case 0: //single_c2c
	{
		if ((sc->localSize[1].data.i > 1) || ((sc->performR2C) && (sc->actualInverse)) || (sc->localSize[0].data.i * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim.data.i) || (sc->rader_generator[0] > 0))
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		break;
	}
	case 1: //grouped_c2c
	{
		if ((sc->localSize[1].data.i * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim.data.i) || (sc->rader_generator[0] > 0))
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		break;
	}
	case 2: //single_c2c_strided
	{
		if ((sc->localSize[1].data.i * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim.data.i) || (sc->rader_generator[0] > 0))
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		break;
	}
	case 5://single_r2c
	{
		if ((sc->stridedSharedLayout) || (sc->localSize[1].data.i > 1) || (sc->localSize[0].data.i * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim.data.i) || (sc->rader_generator[0] > 0))
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		break;
	}
	case 6: //single_c2r
	{
		sc->readToRegisters = 0;
		/*if ((sc->rader_generator[0] > 0) || ((sc->fftDim.data.i % sc->localSize[0].data.i) && (!sc->stridedSharedLayout)) || ((sc->fftDim.data.i % sc->localSize[1].data.i) && (sc->stridedSharedLayout)))
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;*/
		break;
	}
	case 110: case 111: case 130: case 131: case 140: case 141: case 144: case 145:
	{
		sc->readToRegisters = 0;
		break;
	}
	case 142: case 143:
	{
#if(((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))
		sc->readToRegisters = 1;
#else
		sc->readToRegisters = 0;
#endif
		break;
	}
	case 120: case 121:
	{
		sc->readToRegisters = 1;
		break;
	}
	}
	return;
}
static inline void setWriteFromRegisters(PfSolveSpecializationConstantsLayout* sc, int writeType) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	switch (writeType) {
	case 0: //single_c2c
	{
		if ((sc->localSize[1].data.i > 1) || (sc->localSize[0].data.i * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim.data.i) || (sc->rader_generator[sc->numStages - 1] > 0)) {
			sc->writeFromRegisters = 0;
		}
		else
			sc->writeFromRegisters = 1;
		break;
	}
	case 1: //grouped_c2c
	{
		if ((sc->localSize[1].data.i * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim.data.i) || (sc->rader_generator[sc->numStages - 1] > 0)) {
			sc->writeFromRegisters = 0;
		}
		else
			sc->writeFromRegisters = 1;
		break;
	}
	case 2: //single_c2c_strided
	{
		if ((sc->localSize[1].data.i * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim.data.i) || (sc->rader_generator[sc->numStages - 1] > 0)) {
			sc->writeFromRegisters = 0;
		}
		else
			sc->writeFromRegisters = 1;
		break;
	}
	case 5://single_r2c
	{
		sc->writeFromRegisters = 0;
		break;
	}
	case 6: //single_c2r
	{
		if ((sc->stridedSharedLayout) || (sc->localSize[1].data.i > 1) || (sc->localSize[0].data.i * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim.data.i) || (sc->rader_generator[sc->numStages - 1] > 0)) {
			sc->writeFromRegisters = 0;
		}
		else
			sc->writeFromRegisters = 1;
		break;
	}
	case 110: case 111: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143: case 144: case 145:
	{
		sc->writeFromRegisters = 0;
		break;
	}
	}
	return;
}
static inline void appendOffset(PfSolveSpecializationConstantsLayout* sc, int readWrite, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	PfContainer* bufferStride = (readWrite) ? sc->outputStride : sc->inputStride;

	if (sc->size[2].data.i > 1) {
		if (sc->numCoordinates * sc->matrixConvolution * sc->numBatches.data.i > 1) {
			if (sc->performWorkGroupShift[2]) {
				PfMul(sc, &sc->tempInt, &sc->workGroupShiftZ, &sc->gl_WorkGroupSize_z, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_GlobalInvocationID_z);
				PfMod(sc, &sc->tempInt, &sc->tempInt, &sc->dispatchZactualFFTSize);
				if (sc->axis_id == 2)
					PfCheckZeropad(sc, &sc->tempInt, 1);
				else
					PfCheckZeropad(sc, &sc->tempInt, 2);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[2], 0);
				PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &sc->dispatchZactualFFTSize);
				if (sc->axis_id == 2)
					PfCheckZeropad(sc, &sc->tempInt, 1);
				else
					PfCheckZeropad(sc, &sc->tempInt, 2);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[2], 0);
				PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);
			}
		}
		else {
			if (sc->performWorkGroupShift[2]) {
				PfMul(sc, &sc->tempInt, &sc->workGroupShiftZ, &sc->gl_WorkGroupSize_z, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_GlobalInvocationID_z);
				if (sc->axis_id == 2)
					PfCheckZeropad(sc, &sc->tempInt, 1);
				else
					PfCheckZeropad(sc, &sc->tempInt, 2);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[2], 0);
				PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);
			}
			else {
				if (sc->axis_id == 2)
					PfCheckZeropad(sc, &sc->gl_GlobalInvocationID_z, 1);
				else
					PfCheckZeropad(sc, &sc->gl_GlobalInvocationID_z, 2);
				PfMul(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &bufferStride[2], 0);
				PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);
			}
		}
	}
	int64_t maxCoordinate = sc->numCoordinates * sc->matrixConvolution;
	if (sc->numCoordinates * sc->matrixConvolution > 1) {
		PfDiv(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &sc->dispatchZactualFFTSize);
		temp_int.data.i = maxCoordinate;
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[3], 0);
		PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);
	}
	if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
		maxCoordinate = 1;
		PfMul(sc, &temp_int, &sc->coordinate, &bufferStride[3], 0);
		PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &temp_int);
	}
	if ((sc->numBatches.data.i > 1) || (sc->numKernels.data.i > 1)) {
		if (sc->convolutionStep && (sc->numKernels.data.i > 1)) {
			PfMul(sc, &sc->tempInt, &sc->batchID, &bufferStride[4], 0);
			PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);
		}
		else {
			temp_int.data.i = sc->dispatchZactualFFTSize.data.i * maxCoordinate;
			PfDiv(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &temp_int);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[4], 0);
			PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->tempInt);

		}
	}
	if (readWrite) {
		if (sc->outputOffset.type < 100) {
			temp_int.data.i = sc->outputOffset.data.i / sc->outputNumberByteSize;
			PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &temp_int);
		}
		else {
			if (sc->outputOffset.type == 101) {
				if (sc->performPostCompilationOutputOffset) {
					PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->outputOffset);
				}
			}
		}
	}
	else {
		if (sc->inputOffset.type < 100) {
			temp_int.data.i = sc->inputOffset.data.i / sc->inputNumberByteSize;
			PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &temp_int);
		}
		else {
			if (sc->inputOffset.type == 101) {
				if (sc->performPostCompilationInputOffset) {
					PfAdd(sc, &sc->shiftZ, &sc->shiftZ, &sc->inputOffset);
				}
			}
		}
	}
	return;
}

static inline void appendKernelOffset(PfSolveSpecializationConstantsLayout* sc, int readWrite, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	PfContainer* bufferStride = sc->inputStride;
	PfContainer batching_localSize = {};
	batching_localSize.type = 31;

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
	}

	if (type == 1) {
		if (sc->axis_id == 0) {
			if (sc->size[1].data.i > 1) {
				if (sc->performWorkGroupShift[1]) {
					PfAdd(sc, &sc->blockInvocationID, &sc->gl_WorkGroupID_y, &sc->workGroupShiftY);
					PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->inputStride[1], 0);
				}
				else
				{
					PfMov(sc, &sc->blockInvocationID, &sc->gl_WorkGroupID_y);
					PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->inputStride[1], 0);
				}
			}
		}
		else {
			PfSetToZero(sc, &sc->blockInvocationID);
		}
	}
	else {
		if (sc->size[1].data.i > 1) {
			if (sc->numAxisUploads != 1) {
				if (sc->performWorkGroupShift[1]) {
					PfAdd(sc, &sc->blockInvocationID, &sc->gl_WorkGroupID_y, &sc->workGroupShiftY);
					PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->inputStride[1], 0);
				}
				else
				{
					PfMov(sc, &sc->blockInvocationID, &sc->gl_WorkGroupID_y);
					PfMul(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->inputStride[1], 0);
				}
			}
		}
		else {
			PfSetToZero(sc, &sc->blockInvocationID);
		}
	}
	if (sc->size[2].data.i > 1) {
		if (sc->numCoordinates * sc->matrixConvolution * sc->numBatches.data.i > 1) {
			if (sc->performWorkGroupShift[2]) {
				PfMul(sc, &sc->tempInt, &sc->workGroupShiftZ, &sc->gl_WorkGroupSize_z, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_GlobalInvocationID_z);
				PfMod(sc, &sc->tempInt, &sc->tempInt, &sc->dispatchZactualFFTSize);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[2], 0);
				PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);
			}
			else {
				PfMod(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &sc->dispatchZactualFFTSize);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[2], 0);
				PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);
			}
		}
		else {
			if (sc->performWorkGroupShift[2]) {
				PfMul(sc, &sc->tempInt, &sc->workGroupShiftZ, &sc->gl_WorkGroupSize_z, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_GlobalInvocationID_z);
				PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[2], 0);
				PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);
			}
			else {
				PfMul(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &bufferStride[2], 0);
				PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);
			}
		}
	}
	int64_t maxCoordinate = sc->numCoordinates * sc->matrixConvolution;
	if (sc->numCoordinates * sc->matrixConvolution > 1) {
		PfDiv(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &sc->dispatchZactualFFTSize);
		temp_int.data.i = maxCoordinate;
		PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[3], 0);
		PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);
	}
	if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
		maxCoordinate = 1;
		PfMul(sc, &temp_int, &sc->coordinate, &bufferStride[3], 0);
		PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &temp_int);
	}
	if ((sc->numBatches.data.i > 1) || (sc->numKernels.data.i > 1)) {
		if (sc->convolutionStep && (sc->numKernels.data.i > 1)) {
			PfMul(sc, &sc->tempInt, &sc->batchID, &sc->inputStride[4], 0);
			PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);
		}
		else {
			temp_int.data.i = sc->dispatchZactualFFTSize.data.i * maxCoordinate;
			PfDiv(sc, &sc->tempInt, &sc->gl_GlobalInvocationID_z, &temp_int);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[4], 0);
			PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->tempInt);

		}
	}
	if (sc->kernelOffset.type < 100) {
		temp_int.data.i = sc->kernelOffset.data.i / sc->kernelNumberByteSize;
		PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &temp_int);
	}
	else {
		if (sc->kernelOffset.type == 101) {
			if (sc->performPostCompilationKernelOffset) {
				PfAdd(sc, &sc->blockInvocationID, &sc->blockInvocationID, &sc->kernelOffset);
			}
		}
	}
	return;
}
static inline void appendReadWriteDataPfSolve_nonstrided(PfSolveSpecializationConstantsLayout* sc, int readWrite, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	//&sc->tempIntLen = sprintf(&sc->tempIntStr, "	return;\n");
	//char shiftX[500] = "";
	//if (&sc->performWorkGroupShift[0])
	//	sprintf(shiftX, " + consts.workGroupShiftX ");
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	if ((!sc->writeFromRegisters) && (readWrite == 1))
		appendBarrierPfSolve(sc);
	//move to initialization
	//char shiftY[100] = "";
	//if (&sc->performWorkGroupShift[1])
	//	sprintf(shiftY, " + consts.workGroupShiftY ");

	//&sc->shiftY = &sc->workGroupShiftX;
	PfContainer localSize = {};
	localSize.type = 31;

	PfContainer batching_localSize = {};
	batching_localSize.type = 31;

	PfContainer* localInvocationID = {};
	PfContainer* batchingInvocationID = {};

	if (sc->stridedSharedLayout) {
		batching_localSize.data.i = sc->localSize[0].data.i;
		localSize.data.i = sc->localSize[1].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_y;
		batchingInvocationID = &sc->gl_LocalInvocationID_x;
	}
	else {
		batching_localSize.data.i = sc->localSize[1].data.i;
		localSize.data.i = sc->localSize[0].data.i;
		localInvocationID = &sc->gl_LocalInvocationID_x;
		batchingInvocationID = &sc->gl_LocalInvocationID_y;
	}

	PfContainer used_registers = {};
	used_registers.type = 31;

	PfContainer* bufferStride = (readWrite) ? sc->outputStride : sc->inputStride;

	PfContainer mult = {};
	mult.type = 31;

	PfContainer fftDim = {};
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (sc->numAxisUploads == 1) {
			if (readWrite) {
				fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
			}
			else {
				if (sc->readToRegisters == 0) {
					appendSetSMToZero(sc);
					appendBarrierPfSolve(sc);
				}
				fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
			}
		}
		else {
			fftDim.data.i = sc->fftDim.data.i;
		}
	}
	else
		fftDim.data.i = sc->fftDim.data.i;

	if (((type == 6) && (readWrite == 0)) || ((type == 5) && (readWrite == 1))) {
		temp_int.data.i = 2;
		PfDiv(sc, &fftDim, &fftDim, &temp_int);
		PfInc(sc, &fftDim);
	}
	else if (type == 110) {
		fftDim.data.i = (fftDim.data.i + 2) / 2;
	}
	else if ((type == 142) && (readWrite == 0)) {
		fftDim.data.i = 2 * fftDim.data.i;
	}
	if (sc->mergeSequencesR2C)
		mult.data.i = 2;
	else
		mult.data.i = 1;

	//prepare offsets
	if (readWrite == 0) {
		if (sc->performWorkGroupShift[0]) {
			PfAdd(sc, &sc->shiftX, &sc->gl_WorkGroupID_x, &sc->workGroupShiftX);
		}
		else {
			PfMov(sc, &sc->shiftX, &sc->gl_WorkGroupID_x);
		}
		if (sc->size[1].data.i > 1) {
			if (sc->numAxisUploads == 1) {
				if (sc->performWorkGroupShift[1]) {
					PfAdd(sc, &sc->shiftY, &sc->gl_WorkGroupID_y, &sc->workGroupShiftY);
					temp_int.data.i = mult.data.i * batching_localSize.data.i;
					PfMul(sc, &sc->shiftY, &sc->shiftY, &temp_int, 0);
					PfCheckZeropad(sc, &sc->shiftY, 1);
				}
				else {
					PfMov(sc, &sc->shiftY, &sc->gl_WorkGroupID_y);
					temp_int.data.i = mult.data.i * batching_localSize.data.i;
					PfMul(sc, &sc->shiftY, &sc->shiftY, &temp_int, 0);
					PfCheckZeropad(sc, &sc->shiftY, 1);
				}
				PfSetToZero(sc, &sc->shiftZ);
			}
			else {
				if (sc->performWorkGroupShift[1]) {
					PfAdd(sc, &sc->shiftY, &sc->gl_WorkGroupID_y, &sc->workGroupShiftY);
					PfCheckZeropad(sc, &sc->shiftY, 1);
					PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->inputStride[1], 0);
				}
				else
				{
					PfMov(sc, &sc->shiftY, &sc->gl_WorkGroupID_y);
					PfCheckZeropad(sc, &sc->shiftY, 1);
					PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->inputStride[1], 0);
				}
			}
		}
		else {
			PfSetToZero(sc, &sc->shiftZ);
		}
		appendOffset(sc, readWrite, type);

	}
	else {
		if ((sc->inputStride[1].type>100) || (sc->outputStride[2].type>100) ||  (sc->inputStride[1].data.i != sc->outputStride[1].data.i) || (sc->performPostCompilationInputOffset) || (sc->performPostCompilationOutputOffset) || ((sc->inputOffset.data.i != sc->outputOffset.data.i) && (sc->inputOffset.type < 100) && (sc->outputOffset.type < 100)) || ((sc->convolutionStep) && (sc->matrixConvolution > 1)) || (sc->batchID.data.i > 0)) {
			if ((sc->size[1].data.i > 1) && (sc->numAxisUploads != 1)) {
				if (sc->performWorkGroupShift[1]) {
					PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->outputStride[1], 0);
				}
				else
				{
					PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->outputStride[1], 0);
				}
			}
			else {
				PfSetToZero(sc, &sc->shiftZ);
			}
			appendOffset(sc, readWrite, type);
		}
	}

	if (((type == 6) && (readWrite == 0)) || ((type == 5) && (readWrite == 1))) {
		PfMul(sc, &used_registers, &fftDim, &mult, 0);
		mult.data.i = 1;
	}
	else {
		PfMov(sc, &used_registers, &fftDim);
	}

	PfDivCeil(sc, &used_registers, &used_registers, &localSize);

	PfContainer size1 = {};
	size1.type = 31;
	PfDivCeil(sc, &size1, &sc->size[1], &mult);

	if (sc->registerBoost > 1) {
		temp_int.data.i = sc->registerBoost;
		PfDiv(sc, &used_registers, &used_registers, &temp_int);
	}

	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}
	if (sc->fftDim.data.i != sc->fft_dim_full.data.i) {
		if ((sc->reorderFourStep) && (readWrite == 1)) {
			//sc->tempLen = sprintf(sc->tempStr, "		if (((%s + %" PRIu64 " * %s) %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " < %" PRIu64 ")){\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->fft_dim_full / sc->firstStageStartSize);
			PfMul(sc, &sc->tempInt2, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
			PfAdd(sc, &sc->tempInt2, &sc->tempInt2, &sc->gl_LocalInvocationID_x);
			PfMod(sc, &sc->tempInt2, &sc->tempInt2, &batching_localSize);

			PfDiv(sc, &temp_int, &sc->firstStageStartSize, &sc->fftDim);
			PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &batching_localSize, 0);

			PfAdd(sc, &sc->tempInt2, &sc->tempInt2, &sc->tempInt);
			PfDiv(sc, &temp_int, &sc->fft_dim_full, &sc->firstStageStartSize);
			PfIf_lt_start(sc, &sc->tempInt2, &temp_int);
		}
		else {
			PfDiv(sc, &temp_int, &sc->firstStageStartSize, &sc->fftDim);
			PfMod(sc, &sc->tempInt, &sc->shiftX, &temp_int);
			PfMul(sc, &sc->tempInt2, &sc->tempInt, &sc->fftDim, 0);

			PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
			PfMul(sc, &temp_int, &batching_localSize, &sc->firstStageStartSize, 0);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt2, &sc->tempInt2, &sc->tempInt);

			//sc->tempLen = sprintf(sc->tempStr, "		%s numActiveThreads = ((%s/%" PRIu64 ")==%" PRIu64 ") ? %" PRIu64 " : %" PRIu64 ";\n", uintType, sc->gl_WorkGroupID_x, sc->firstStageStartSize / sc->fftDim, ((uint64_t)pffloor(sc->fft_dim_full / ((double)sc->localSize[0] * sc->fftDim))) / (sc->firstStageStartSize / sc->fftDim), (uint64_t)ceil(((sc->fft_dim_full - (sc->firstStageStartSize / sc->fftDim) * ((((uint64_t)pffloor(sc->fft_dim_full / ((double)sc->localSize[0] * sc->fftDim))) / (sc->firstStageStartSize / sc->fftDim)) * sc->localSize[0] * sc->fftDim)) / (sc->firstStageStartSize / sc->fftDim)) / (double)used_registers_read), sc->localSize[0] * sc->localSize[1]);// sc->fft_dim_full, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0] * sc->firstStageStartSize, sc->fft_dim_full / (sc->localSize[0] * sc->fftDim));
			temp_int.data.i = sc->firstStageStartSize.data.i / sc->fftDim.data.i;
			PfDiv(sc, &sc->tempInt, &sc->gl_WorkGroupID_x, &temp_int);
			temp_int1.data.i = ((int64_t)pffloor(sc->fft_dim_full.data.i / ((pfLD)batching_localSize.data.i * sc->fftDim.data.i))) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i);
			PfIf_eq_start(sc, &sc->tempInt, &temp_int1);
			temp_int.data.i = (int64_t)pfceil(((sc->fft_dim_full.data.i - (sc->firstStageStartSize.data.i / sc->fftDim.data.i) * ((((int64_t)pffloor(sc->fft_dim_full.data.i / ((pfLD)batching_localSize.data.i * sc->fftDim.data.i))) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i)) * batching_localSize.data.i * sc->fftDim.data.i)) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i)) / (pfLD)used_registers.data.i);
			PfMov(sc, &sc->blockInvocationID, &temp_int);
			PfIf_else(sc);
			temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
			PfMov(sc, &sc->blockInvocationID, &temp_int);
			PfIf_end(sc);

			if (sc->stridedSharedLayout) {

				PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &sc->firstStageStartSize, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->tempInt2);
			}
			else {
				PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->firstStageStartSize, 0);
				PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->tempInt2);
			}

			/*sc->useDisableThreads = 1;
			PfIf_ge_start(sc, &sc->tempInt, &sc->fft_dim_full);
			temp_int.data.i = 0;
			PfMov(sc, &sc->disableThreads, &temp_int);
			PfIf_end(sc);*/

			PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
			PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
			PfIf_lt_start(sc, &sc->combinedID, &sc->blockInvocationID);
		}
	}
	if (bufferStride[1].data.i == fftDim.data.i) {
		PfMul(sc, &sc->inoutID, &bufferStride[1], &sc->shiftY, 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->shiftZ);
		if (sc->localSize[1].data.i == 1) {
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
		}
		else {
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);
			PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->combinedID);
		}
	}
	//for (uint64_t k = 0; k < &sc->registerBoost; k++) {
	for (int k = 0; k < sc->registerBoost; k++) {
		//for (uint64_t i = 0; i < used_registers; i++) {
		for (int i = 0; i < used_registers.data.i; i++) {
			//combined thread numeration
			if ((sc->fftDim.data.i != sc->fft_dim_full.data.i) && (!((sc->reorderFourStep) && (readWrite == 1)))) {
				if ((sc->fft_dim_full.data.i - (sc->firstStageStartSize.data.i / sc->fftDim.data.i) * ((((int64_t)pffloor(sc->fft_dim_full.data.i / ((pfLD)batching_localSize.data.i * sc->fftDim.data.i))) / (sc->firstStageStartSize.data.i / sc->fftDim.data.i)) * batching_localSize.data.i * sc->fftDim.data.i)) / used_registers.data.i / (sc->firstStageStartSize.data.i / sc->fftDim.data.i) > batching_localSize.data.i) {
					if (sc->localSize[1].data.i == 1) {
						//sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * used_registers_read) * sc->localSize[0]);
						temp_int.data.i = (k * used_registers.data.i + i) * sc->localSize[0].data.i;
						PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
					}
					else {
						//sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 "*numActiveThreads;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * used_registers_read));
						temp_int.data.i = (k * used_registers.data.i + i);
						PfMul(sc, &sc->combinedID, &sc->blockInvocationID, &temp_int, 0);

						PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->localSize[0], 0);
						PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_LocalInvocationID_x);

						PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->tempInt);
					}
				}
				else {
					if (sc->localSize[1].data.i == 1) {
						//sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 "*numActiveThreads;\n", sc->gl_LocalInvocationID_x, (i + k * used_registers_read));
						temp_int.data.i = (k * used_registers.data.i + i);
						PfMul(sc, &sc->combinedID, &sc->blockInvocationID, &temp_int, 0);
						PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);
					}
					else {
						//sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 "*numActiveThreads;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * used_registers_read));
						temp_int.data.i = (k * used_registers.data.i + i);
						PfMul(sc, &sc->combinedID, &sc->blockInvocationID, &temp_int, 0);

						PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->localSize[0], 0);
						PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->gl_LocalInvocationID_x);

						PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->tempInt);
					}
				}
			}
			else {
				if (sc->localSize[1].data.i == 1) {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		combinedID = %s + %" PRIu64 ";\n", &sc->gl_LocalInvocationID_x, (i + k * used_registers) * &sc->localSize[0]);
					temp_int.data.i = (k * used_registers.data.i + i) * sc->localSize[0].data.i;

					PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				else {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", &sc->gl_LocalInvocationID_x, &sc->localSize[0], &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[0] * &sc->localSize[1]);
					PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
					PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);

					temp_int.data.i = (k * used_registers.data.i + i) * sc->localSize[0].data.i * sc->localSize[1].data.i;

					PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
				}
			}
			//set inoutID - global array index. Two batching options - in consecutive x (if multi-upload), &in y if multidimensional.
			if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", &sc->fftDim, &sc->fftDim, &sc->inputStride[1]);
				PfMod(sc, &sc->inoutID_x, &sc->combinedID, &fftDim);
				PfDiv(sc, &sc->inoutID_y, &sc->combinedID, &fftDim);
				if (mult.data.i > 1) {
					PfMul(sc, &sc->inoutID_y, &sc->inoutID_y, &mult, 0);
				}
				PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &sc->shiftY);
				PfCheckZeropadStart(sc, &sc->inoutID_y, 1);
				//PfMul(sc, &sc->tempInt, &batching_localSize, &sc->shiftY,0);
				//PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &sc->tempInt);
				temp_int.data.i = batching_localSize.data.i;
				//we switched to reading 2x more data, but we still need to check out of bounds for odd size1
				if ((sc->mergeSequencesR2C) && (mult.data.i == 1))
					temp_int.data.i *= 2;

				if ((size1.data.i % temp_int.data.i) != 0) {
#if (VKFFT_BACKEND!=2) //AMD compiler fix
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){\n", &sc->fftDim, &sc->gl_WorkGroupID_y, shiftY, &sc->localSize[0], &sc->size[&sc->axis_id + 1]);
					if ((sc->mergeSequencesR2C) && (sc->size[1].data.i % 2) && (readWrite == 0)) {
						PfIf_ge_start(sc, &sc->inoutID_y, &sc->size[1]);
						PfSetToZero(sc, &sc->inoutID_x);
						PfSetToZero(sc, &sc->inoutID_y);
						PfIf_end(sc);
					}
					else {
						PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
					}
#else
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(!(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 ")) %s = 0; {\n", &sc->fftDim, &sc->gl_WorkGroupID_y, shiftY, &sc->localSize[0], &sc->size[&sc->axis_id + 1], &sc->inoutID);
					if (readWrite == 0) {
						PfIf_ge_start(sc, &sc->inoutID_y, &sc->size[1]);
						PfSetToZero(sc, &sc->inoutID_x);
						PfSetToZero(sc, &sc->inoutID_y);
						PfIf_end(sc);
					}
					else {
						PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
					}
#endif
				}
			}
			else {
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 " + ((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ";\n", &sc->fftDim, &sc->fftDim, &sc->firstStageStartSize, &sc->gl_WorkGroupID_x, shiftX, &sc->firstStageStartSize / &sc->fftDim, &sc->fftDim, &sc->gl_WorkGroupID_x, shiftX, &sc->firstStageStartSize / &sc->fftDim, &sc->localSize[0] * &sc->firstStageStartSize);
				if ((sc->reorderFourStep) && (readWrite == 1)) {
					PfMod(sc, &sc->inoutID_x, &sc->combinedID, &batching_localSize);

					temp_int.data.i = sc->firstStageStartSize.data.i / sc->fftDim.data.i;
					PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &batching_localSize, 0);

					PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt);

					PfDiv(sc, &sc->tempInt2, &sc->combinedID, &batching_localSize);
					temp_int.data.i = sc->fft_dim_full.data.i / sc->fftDim.data.i;
					PfMul(sc, &sc->tempInt2, &sc->tempInt2, &temp_int, 0);
					temp_int.data.i = sc->firstStageStartSize.data.i / sc->fftDim.data.i;
					PfMod(sc, &sc->tempInt, &sc->shiftX, &temp_int);
					temp_int.data.i = sc->fft_dim_full.data.i / sc->firstStageStartSize.data.i;
					PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
					PfAdd(sc, &sc->tempInt2, &sc->tempInt2, &sc->tempInt);

					PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt2);
				}
				else {
					PfMod(sc, &sc->inoutID_x, &sc->combinedID, &sc->fftDim);

					PfDiv(sc, &sc->tempInt, &sc->combinedID, &sc->fftDim);
					PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->firstStageStartSize, 0);
					PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt);

					PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt2);

					//PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[sc->axis_id]);
				}
			}

			temp_int.data.i = (k * used_registers.data.i + i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;

			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;

			if ((sc->mergeSequencesR2C) && (mult.data.i == 1))
				temp_int1.data.i *= 2;

			if (temp_int.data.i > temp_int1.data.i) {
				//check that we only read fftDim * local batch data
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(combinedID < %" PRIu64 "){\n", &sc->fftDim * &sc->localSize[0]);
				PfIf_lt_start(sc, &sc->combinedID, &temp_int1);
			}

			if (bufferStride[1].data.i == fftDim.data.i) {
				if (k * used_registers.data.i + i > 0) {
					temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
				}
			}
			else {
				if (bufferStride[0].data.i != 1)
					PfMul(sc, &sc->inoutID, &sc->inoutID_x, &bufferStride[0], 0);
				else
					PfMov(sc, &sc->inoutID, &sc->inoutID_x);

				if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
					PfMul(sc, &sc->tempInt, &sc->inoutID_y, &bufferStride[1], 0);
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				}
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->shiftZ);
			}
			if ((sc->readToRegisters && (readWrite == 0)) || (sc->writeFromRegisters && (readWrite == 1))) {
				//no need to calculate register addresses
			}
			else {
				if (sc->stridedSharedLayout) {
					if ((sc->reorderFourStep) && (readWrite == 1)) {
						//sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %ssdata[(combinedID %% %s)+(combinedID/%s)*sharedStride]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_WorkGroupSize_x, convTypeRight);
						PfMod(sc, &sc->sdataID, &sc->combinedID, &sc->localSize[0]);
						PfDiv(sc, &sc->tempInt, &sc->combinedID, &sc->localSize[0]);
						PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
					}
					else {
						//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")] = %s%s[%s]%s;\n", &sc->fftDim, &sc->fftDim, convTypeLeft, &inputsStruct, &sc->inoutID, convTypeRight);
						temp_int.data.i = fftDim.data.i;
						if ((sc->mergeSequencesR2C) && (mult.data.i == 1))
							temp_int.data.i *= 2;
						PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

						if ((type == 142) && (!sc->readToRegisters) && (readWrite == 0)) {
							temp_int1.data.i = 2;
							PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);
						}

						PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
					}
				}
				else {
					if ((sc->reorderFourStep) && (readWrite == 1)) {
						//sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);
						PfMod(sc, &sc->sdataID, &sc->combinedID, &sc->localSize[1]);
						PfMul(sc, &sc->sdataID, &sc->sdataID, &sc->sharedStride, 0);
						PfDiv(sc, &sc->tempInt, &sc->combinedID, &sc->localSize[1]);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
					}
					else {
						//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride] = %s%s[%s]%s;\n", &sc->fftDim, &sc->fftDim, convTypeLeft, &inputsStruct, &sc->inoutID, convTypeRight);
						temp_int.data.i = fftDim.data.i;
						if ((sc->mergeSequencesR2C) && (mult.data.i == 1))
							temp_int.data.i *= 2;
						PfMod(sc, &sc->sdataID, &sc->combinedID, &temp_int);

						if ((type == 142) && (!sc->readToRegisters) && (readWrite == 0)) {
							temp_int1.data.i = 2;
							PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);
						}

						PfDiv(sc, &sc->tempInt, &sc->combinedID, &temp_int);
						PfMul(sc, &sc->tempInt, &sc->tempInt, &sc->sharedStride, 0);
						PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->tempInt);
					}
				}
			}
			if ((sc->zeropad[readWrite]) || ((sc->numAxisUploads > 1) && (sc->zeropadBluestein[readWrite]))) {
				//sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
				PfSetToZero(sc, &sc->tempInt);

				//PfMod(sc, &sc->combinedID, &sc->inoutID_x, &sc->fft_dim_full);

				if (sc->zeropad[readWrite]) {
					if (readWrite)
						PfIf_lt_start(sc, &sc->inoutID_x, &sc->fft_zeropad_left_write[sc->axis_id]);
					else
						PfIf_lt_start(sc, &sc->inoutID_x, &sc->fft_zeropad_left_read[sc->axis_id]);
					temp_int.data.i = 1;
					PfMov(sc, &sc->tempInt, &temp_int);
					PfIf_else(sc);

					if (readWrite) {
						PfIf_ge_start(sc, &sc->inoutID_x, &sc->fft_zeropad_right_write[sc->axis_id]);
					}
					else {
						PfIf_ge_start(sc, &sc->inoutID_x, &sc->fft_zeropad_right_read[sc->axis_id]);
					}
					temp_int.data.i = 1;
					PfMov(sc, &sc->tempInt, &temp_int);
					PfIf_end(sc);

					PfIf_end(sc);
				}

				if (sc->numAxisUploads > 1) {
					if (sc->zeropadBluestein[readWrite]) {
						if (readWrite)
							PfIf_lt_start(sc, &sc->inoutID_x, &sc->fft_zeropad_Bluestein_left_write[sc->axis_id]);
						else
							PfIf_lt_start(sc, &sc->inoutID_x, &sc->fft_zeropad_Bluestein_left_read[sc->axis_id]);
						temp_int.data.i = 1;
						PfMov(sc, &sc->tempInt, &temp_int);
						PfIf_end(sc);
					}
				}
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			}
			if (readWrite == 0) {
				if ((type == 5) || (type == 110) || (type == 120) || (type == 130) || (type == 144)) {
					if (sc->readToRegisters) {
						appendGlobalToRegisters_x(sc, &sc->regIDs[k * sc->registers_per_thread + i], &sc->inputsStruct, &sc->inoutID);
						if (sc->mergeSequencesR2C) {
							if ((sc->size[1].data.i % 2) != 0) {
								temp_int.data.i = sc->size[1].data.i - 1;
								PfIf_lt_start(sc, &sc->inoutID_y, &temp_int);
							}
							PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);
							if ((sc->size[1].data.i % 2) != 0) {
								PfIf_end(sc);
							}
							appendGlobalToRegisters_y(sc, &sc->regIDs[k * sc->registers_per_thread + i], &sc->inputsStruct, &sc->inoutID);
						}
					}
					else {
						appendGlobalToRegisters_x(sc, &sc->temp, &sc->inputsStruct, &sc->inoutID);
						if (sc->mergeSequencesR2C) {
							if ((sc->size[1].data.i % 2) != 0) {
								temp_int.data.i = sc->size[1].data.i - 1;
								PfIf_lt_start(sc, &sc->inoutID_y, &temp_int);
							}
							PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);
							if ((sc->size[1].data.i % 2) != 0) {
								PfIf_end(sc);
							}
							appendGlobalToRegisters_y(sc, &sc->temp, &sc->inputsStruct, &sc->inoutID);
						}
						appendRegistersToShared(sc, &sc->sdataID, &sc->temp);
					}
				}
				else  if (type == 142) {
					if (sc->readToRegisters) {
						if (i < used_registers.data.i / 2) {
							appendGlobalToRegisters_x(sc, &sc->regIDs[k * sc->registers_per_thread + i], &sc->inputsStruct, &sc->inoutID);
						}
						else {
							appendGlobalToRegisters_y(sc, &sc->regIDs[k * sc->registers_per_thread + i - used_registers.data.i / 2], &sc->inputsStruct, &sc->inoutID);
						}
					}
					else {
						PfMod(sc, &sc->tempInt, &sc->combinedID, &fftDim);
						temp_int.data.i = 2;
						PfMod(sc, &sc->tempInt, &sc->tempInt, &temp_int);

						appendGlobalToRegisters_x(sc, &sc->temp, &sc->inputsStruct, &sc->inoutID);

						temp_int.data.i = 0;
						PfIf_eq_start(sc, &sc->tempInt, &temp_int);
						appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->temp);
						PfIf_else(sc);
						appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->temp);
						PfIf_end(sc);
					}
				}
				else {
					if (sc->readToRegisters) {
						appendGlobalToRegisters(sc, &sc->regIDs[k * sc->registers_per_thread + i], &sc->inputsStruct, &sc->inoutID);
					}
					else {
						appendGlobalToShared(sc, &sc->sdataID, &sc->inputsStruct, &sc->inoutID);
					}
				}
				if ((sc->zeropad[readWrite]) || ((sc->numAxisUploads > 1) && (sc->zeropadBluestein[readWrite]))) {
					PfIf_else(sc);
					if (sc->readToRegisters) {
						PfSetToZero(sc, &sc->regIDs[k * sc->registers_per_thread + i]);
					}
					else {
						PfSetToZeroShared(sc, &sc->sdataID);
					}
				}
			}
			else {
				if ((type == 6) || (type == 110) || (type == 120) || (type == 130) || (type == 144)) {
					if (sc->writeFromRegisters) {
						appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[k * sc->registers_per_thread + i]);
						if (sc->mergeSequencesR2C) {
							if ((sc->size[1].data.i % 2) != 0) {
								temp_int.data.i = sc->size[1].data.i - 1;
								PfIf_lt_start(sc, &sc->inoutID_y, &temp_int);
							}
							PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->outputStride[1]);
							appendRegistersToGlobal_y(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[k * sc->registers_per_thread + i]);
							if ((sc->size[1].data.i % 2) != 0) {
								PfIf_end(sc);
							}
						}
					}
					else {
						appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
						appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->temp);
						if (sc->mergeSequencesR2C) {
							if ((sc->size[1].data.i % 2) != 0) {
								temp_int.data.i = sc->size[1].data.i - 1;
								PfIf_lt_start(sc, &sc->inoutID_y, &temp_int);
							}
							PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->outputStride[1]);
							appendRegistersToGlobal_y(sc, &sc->outputsStruct, &sc->inoutID, &sc->temp);
							if ((sc->size[1].data.i % 2) != 0) {
								PfIf_end(sc);
							}
						}
					}
				}
				else  if (type == 142) {
					if (sc->writeFromRegisters) {
						appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[k * sc->registers_per_thread + i]);
						PfAdd(sc, &sc->inoutID, &sc->inoutID, &fftDim);
						appendRegistersToGlobal_y(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[k * sc->registers_per_thread + i]);
					}
					else {
					}
				}
				else {
					if (sc->writeFromRegisters) {
						appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[k * sc->registers_per_thread + i]);
					}
					else {
						appendSharedToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->sdataID);
					}
				}
			}
			if ((sc->zeropad[readWrite]) || ((sc->numAxisUploads > 1) && (sc->zeropadBluestein[readWrite]))) {
				PfIf_end(sc);
			}
			temp_int.data.i = (k * used_registers.data.i + i + 1) * sc->localSize[0].data.i * sc->localSize[1].data.i;

			temp_int1.data.i = fftDim.data.i * batching_localSize.data.i;

			if ((sc->mergeSequencesR2C) && (mult.data.i == 1))
				temp_int1.data.i *= 2;

			if (temp_int.data.i > temp_int1.data.i) {
				PfIf_end(sc);
			}

			if (sc->fftDim.data.i == sc->fft_dim_full.data.i) {
				PfCheckZeropadEnd(sc, 1);
				temp_int.data.i = batching_localSize.data.i;
				//we switched to reading 2x more data, but we still need to check out of bounds for odd size1
				if ((sc->mergeSequencesR2C) && (mult.data.i == 1))
					temp_int.data.i *= 2;
				if ((size1.data.i % temp_int.data.i) != 0) {
#if (VKFFT_BACKEND!=2) //AMD compiler fix
					if ((sc->mergeSequencesR2C) && (sc->size[1].data.i % 2) && (readWrite == 0)) {
					}
					else {
						PfIf_end(sc);
					}
#else
					if (readWrite != 0) {
						PfIf_end(sc);
					}
#endif
				}
			}
			else {
				if ((sc->reorderFourStep) && (readWrite == 1)) {
				}
				else {
					//PfIf_end(sc);
				}
			}
		}
	}


	if (sc->fftDim.data.i != sc->fft_dim_full.data.i) {
		PfIf_end(sc);
	}
	if (sc->useDisableThreads) {
		PfIf_end(sc);
	}
	return;
}
static inline void appendReadWriteDataPfSolve_strided(PfSolveSpecializationConstantsLayout* sc, int readWrite, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	PfContainer temp_int1 = {};
	temp_int1.type = 31;

	PfContainer used_registers = {};
	used_registers.type = 31;

	PfContainer* bufferStride = (readWrite) ? sc->outputStride : sc->inputStride;

	if ((!sc->writeFromRegisters) && (readWrite == 1))
		appendBarrierPfSolve(sc);
	//char shiftX[500] = "";
	//if (&sc->performWorkGroupShift[0])
	//	sprintf(shiftX, " + consts.workGroupShiftX * %s ", &sc->gl_WorkGroupSize_x);
	PfContainer fftDim = {};
	fftDim.type = 31;

	if (sc->zeropadBluestein[readWrite]) {
		if (sc->numAxisUploads == 1) {
			if (readWrite) {
				fftDim.data.i = sc->fft_zeropad_Bluestein_left_write[sc->axis_id].data.i;
			}
			else {
				if (sc->readToRegisters == 0) {
					appendSetSMToZero(sc);
					appendBarrierPfSolve(sc);
				}
				fftDim.data.i = sc->fft_zeropad_Bluestein_left_read[sc->axis_id].data.i;
			}
		}
		else {
			fftDim.data.i = sc->fftDim.data.i;
		}
	}
	else
		fftDim.data.i = sc->fftDim.data.i;

	if (type == 111) {
		fftDim.data.i = (fftDim.data.i + 2) / 2;
	}
	else if ((type == 143) && (readWrite == 0)) {
		fftDim.data.i = 2 * fftDim.data.i;
	}

	if (readWrite == 0) {
		if (sc->performWorkGroupShift[0]) {
			PfMul(sc, &sc->shiftX, &sc->workGroupShiftX, &sc->gl_WorkGroupSize_x, 0);
			PfAdd(sc, &sc->shiftX, &sc->gl_GlobalInvocationID_x, &sc->shiftX);
		}
		else {
			PfMov(sc, &sc->shiftX, &sc->gl_GlobalInvocationID_x);
		}
		if (sc->axis_id == 0) {
			if (sc->size[1].data.i > 1) {
				if (sc->performWorkGroupShift[1]) {
					PfAdd(sc, &sc->shiftY, &sc->gl_WorkGroupID_y, &sc->workGroupShiftY);
					PfCheckZeropad(sc, &sc->shiftY, 1);
					PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->inputStride[1], 0);
				}
				else
				{
					PfMov(sc, &sc->shiftY, &sc->gl_WorkGroupID_y);
					PfCheckZeropad(sc, &sc->shiftY, 1);
					PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->inputStride[1], 0);
				}
			}
			else
			{
				PfSetToZero(sc, &sc->shiftZ);
			}
		}
		else {
			PfSetToZero(sc, &sc->shiftZ);
		}
		appendOffset(sc, readWrite, type);
		if (sc->axis_id > 0) {
			PfMod(sc, &sc->inoutID_x, &sc->shiftX, &sc->fft_dim_x);
			PfCheckZeropad(sc, &sc->inoutID_x, 0);
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		disableThreads = (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") ? 1 : 0;\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x, &sc->stageStartSize, &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x * &sc->stageStartSize, &sc->fftDim * &sc->stageStartSize, &sc->size[&sc->axis_id]);

			PfDiv(sc, &sc->tempInt2, &sc->shiftX, &sc->fft_dim_x);

			PfMod(sc, &sc->tempInt2, &sc->tempInt2, &sc->stageStartSize);

			PfMul(sc, &temp_int, &sc->fft_dim_x, &sc->stageStartSize, 0);
			PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int); // disableThreads - tempInt3

			PfMul(sc, &temp_int, &fftDim, &sc->stageStartSize, 0);
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);

			PfAdd(sc, &sc->tempInt2, &sc->tempInt2, &sc->tempInt);

			if (sc->numAxisUploads > 1) {
				PfIf_lt_start(sc, &sc->tempInt2, &sc->fft_dim_full);
			}
			else {
				PfIf_lt_start(sc, &sc->tempInt2, &sc->sourceFFTSize);
			}
		}
		else {
			//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		disableThreads = (((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") ? 1 : 0;\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim, &sc->fft_dim_full);
			PfDiv(sc, &sc->tempInt2, &sc->shiftX, &sc->stageStartSize);
			PfMul(sc, &temp_int, &sc->fftDim, &sc->stageStartSize, 0);
			PfMul(sc, &sc->tempInt2, &sc->tempInt2, &temp_int, 0);

			PfIf_lt_start(sc, &sc->tempInt2, &sc->fft_dim_full);
		}
		//PfIf_gt_start(sc, &sc->disableThreads, &temp_int1);
	}
	else {
		if ((sc->inputStride[1].type>100) || (sc->outputStride[1].type>100) || (sc->inputStride[1].data.i != sc->outputStride[1].data.i) || (sc->performPostCompilationInputOffset) || (sc->performPostCompilationOutputOffset) || ((sc->inputOffset.data.i != sc->outputOffset.data.i) && (sc->inputOffset.type < 100) && (sc->outputOffset.type < 100)) || ((sc->convolutionStep) && (sc->matrixConvolution > 1)) || (sc->batchID.data.i > 0)) {
			if (sc->axis_id == 0) {
				if (sc->size[1].data.i > 1) {
					if (sc->performWorkGroupShift[1]) {
						PfAdd(sc, &sc->shiftY, &sc->gl_WorkGroupID_y, &sc->workGroupShiftY);
						PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->outputStride[1], 0);
					}
					else
					{
						PfMov(sc, &sc->shiftY, &sc->gl_WorkGroupID_y);
						PfMul(sc, &sc->shiftZ, &sc->shiftY, &sc->outputStride[1], 0);
					}
				}
				else {
					PfSetToZero(sc, &sc->shiftZ);
				}
			}
			else {
				PfSetToZero(sc, &sc->shiftZ);
			}

			appendOffset(sc, readWrite, type);
		}
		if ((sc->reorderFourStep) && (sc->stageStartSize.data.i == 1) && (sc->numAxisUploads > 1)) {
			PfDiv(sc, &sc->inoutID, &sc->shiftX, &sc->fft_dim_x);
			temp_int.data.i = sc->firstStageStartSize.data.i / fftDim.data.i;
			PfMod(sc, &sc->tempInt2, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->fft_dim_full.data.i / sc->firstStageStartSize.data.i;
			PfMul(sc, &sc->tempInt2, &sc->tempInt2, &temp_int, 0);

			temp_int.data.i = sc->fft_dim_x.data.i * (sc->firstStageStartSize.data.i / fftDim.data.i);
			PfDiv(sc, &sc->tempInt, &sc->shiftX, &temp_int);

			PfAdd(sc, &sc->tempInt2, &sc->tempInt2, &sc->tempInt);
			//sc->tempLen = sprintf(sc->tempStr, "		if (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize, sc->size[sc->axis_id]);
			PfMod(sc, &sc->tempInt, &sc->inoutID, &sc->stageStartSize);
			temp_int.data.i = sc->fft_dim_x.data.i * sc->stageStartSize.data.i;
			PfDiv(sc, &sc->inoutID, &sc->shiftX, &temp_int);
			temp_int.data.i = fftDim.data.i * sc->stageStartSize.data.i;
			PfMul(sc, &sc->inoutID, &sc->inoutID, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID);
			PfIf_lt_start(sc, &sc->tempInt, &sc->fft_dim_full);
		}
		else {
			if (sc->axis_id > 0) {
				if (sc->numAxisUploads > 1) {
					PfIf_lt_start(sc, &sc->tempInt2, &sc->fft_dim_full);
				}
				else {
					PfIf_lt_start(sc, &sc->tempInt2, &sc->sourceFFTSize);
				}
			}
			else {
				PfIf_lt_start(sc, &sc->tempInt2, &sc->fft_dim_full);
			}
		}
	}

	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->disableThreads, &temp_int);
	}

	PfDivCeil(sc, &used_registers, &fftDim, &sc->localSize[1]);

	if (sc->registerBoost > 1) {
		temp_int.data.i = sc->registerBoost;
		PfDiv(sc, &used_registers, &used_registers, &temp_int);
	}
	if (sc->axis_id > 0) {
		//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 "));\n", &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x, &sc->stageStartSize, &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x * &sc->stageStartSize, &sc->fftDim * &sc->stageStartSize);
		if ((readWrite == 1) && (sc->reorderFourStep) && (sc->stageStartSize.data.i == 1) && (sc->numAxisUploads > 1)) {
			temp_int1.data.i = sc->fft_dim_full.data.i / fftDim.data.i;
			PfMul(sc, &sc->inoutID_y, &sc->gl_LocalInvocationID_y, &temp_int1, 0);
			PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &sc->tempInt2);
		}
		else {
			PfMul(sc, &sc->inoutID_y, &sc->gl_LocalInvocationID_y, &sc->stageStartSize, 0);
			PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &sc->tempInt2);
		}

	}
	else {
		//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim);
		PfMod(sc, &sc->inoutID_x, &sc->shiftX, &sc->stageStartSize);

		PfMul(sc, &sc->tempInt, &sc->gl_LocalInvocationID_y, &sc->stageStartSize, 0);

		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt);

		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->tempInt2);
	}
	if (bufferStride[0].data.i != 1)
		PfMul(sc, &sc->inoutID, &sc->inoutID_x, &bufferStride[0], 0);
	else
		PfMov(sc, &sc->inoutID, &sc->inoutID_x);

	if (sc->axis_id > 0) {
		PfMul(sc, &sc->tempInt, &sc->inoutID_y, &bufferStride[1], 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
	}
	PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->shiftZ);

	//for (uint64_t k = 0; k < &sc->registerBoost; k++) {
	for (int k = 0; k < sc->registerBoost; k++) {
		//for (uint64_t i = 0; i < used_registers; i++) {
		for (int i = 0; i < used_registers.data.i; i++) {

			temp_int1.data.i = (k * used_registers.data.i + i + 1) * sc->localSize[1].data.i;

			if (temp_int1.data.i > fftDim.data.i) {
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		if(%s < %" PRIu64 "){\n", &sc->gl_LocalInvocationID_y, &sc->fftDim - (i + k * used_registers) * &sc->localSize[1]);
				temp_int1.data.i = sc->localSize[1].data.i - (temp_int1.data.i - fftDim.data.i);
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_y, &temp_int1);
			}

			/*if (bufferStride[0].data.i != 1)
				PfMul(sc, &sc->inoutID, &sc->inoutID_x, &bufferStride[0], 0);
			else
				PfMov(sc, &sc->inoutID, &sc->inoutID_x);

			if (sc->axis_id > 0) {
				PfMul(sc, &sc->tempInt, &sc->inoutID_y, &bufferStride[1], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
			}

			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->shiftZ);*/
			if ((i > 0) || (k > 0)) {
				if (sc->axis_id > 0) {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 "));\n", &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x, &sc->stageStartSize, &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x * &sc->stageStartSize, &sc->fftDim * &sc->stageStartSize);
					if ((readWrite == 1) && (sc->reorderFourStep) && (sc->stageStartSize.data.i == 1) && (sc->numAxisUploads > 1)) {
						temp_int1.data.i = sc->fft_dim_full.data.i / fftDim.data.i * bufferStride[1].data.i * sc->localSize[1].data.i;
						PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int1);
					}
					else {
						temp_int1.data.i = sc->stageStartSize.data.i * bufferStride[1].data.i * sc->localSize[1].data.i;
						PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int1);
					}

				}
				else {
					//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim);
					temp_int1.data.i = sc->stageStartSize.data.i * bufferStride[0].data.i * sc->localSize[1].data.i;
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int1);
				}
			}

			temp_int.data.i = (k * used_registers.data.i + i) * sc->localSize[1].data.i;

			if ((sc->readToRegisters && (readWrite == 0)) || (sc->writeFromRegisters && (readWrite == 1))) {
				//if (&sc->inputBufferBlockNum == 1)
				//	&sc->tempIntLen = sprintf(&sc->tempIntStr, "			%s=%s%s[%s]%s;\n", &sc->regIDs[i + k * &sc->registers_per_thread], convTypeLeft, &inputsStruct, &sc->inoutID, convTypeRight);
				//else
				//	&sc->tempIntLen = sprintf(&sc->tempIntStr, "			%s=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", &sc->regIDs[i + k * &sc->registers_per_thread], convTypeLeft, &sc->inoutID, &sc->inputBufferBlockSize, &inputsStruct, &sc->inoutID, &sc->inputBufferBlockSize, convTypeRight);
			}
			else {
				//if (&sc->inputBufferBlockNum == 1)
				//	&sc->tempIntLen = sprintf(&sc->tempIntStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%s%s[%s]%s;\n", &sc->sharedStride, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_LocalInvocationID_x, convTypeLeft, &inputsStruct, &sc->inoutID, convTypeRight);
				//else
				//	&sc->tempIntLen = sprintf(&sc->tempIntStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", &sc->sharedStride, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_LocalInvocationID_x, convTypeLeft, &sc->inoutID, &sc->inputBufferBlockSize, &inputsStruct, &sc->inoutID, &sc->inputBufferBlockSize, convTypeRight);
				PfAdd(sc, &sc->sdataID, &sc->gl_LocalInvocationID_y, &temp_int);
				if ((type == 143) && (!sc->readToRegisters) && (readWrite == 0)) {
					temp_int1.data.i = 2;
					PfDiv(sc, &sc->sdataID, &sc->sdataID, &temp_int1);
				}
				PfMul(sc, &sc->sdataID, &sc->sharedStride, &sc->sdataID, 0);
				PfAdd(sc, &sc->sdataID, &sc->sdataID, &sc->gl_LocalInvocationID_x);
			}
			if ((sc->zeropad[readWrite]) || ((sc->numAxisUploads > 1) && (sc->zeropadBluestein[readWrite]))) {
				//sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
				temp_int.data.i = 1;
				PfSetToZero(sc, &sc->tempInt);
				if ((i > 0) && (k == 0)) {
					if (sc->axis_id > 0) {
						//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 "));\n", &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x, &sc->stageStartSize, &sc->gl_GlobalInvocationID_x, shiftX, &sc->fft_dim_x * &sc->stageStartSize, &sc->fftDim * &sc->stageStartSize);
						if ((readWrite == 1) && (sc->reorderFourStep) && (sc->stageStartSize.data.i == 1) && (sc->numAxisUploads > 1)) {
							temp_int1.data.i = sc->fft_dim_full.data.i / fftDim.data.i * sc->localSize[1].data.i;
							PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &temp_int1);
						}
						else {
							temp_int1.data.i = sc->stageStartSize.data.i * sc->localSize[1].data.i;
							PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &temp_int1);
						}

					}
					else {
						//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize, &sc->gl_LocalInvocationID_y, (i + k * used_registers) * &sc->localSize[1], &sc->gl_GlobalInvocationID_x, shiftX, &sc->stageStartSize, &sc->stageStartSize * &sc->fftDim);
						temp_int1.data.i = sc->stageStartSize.data.i * sc->localSize[1].data.i;
						PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &temp_int1);
					}
				}
				if (sc->axis_id > 0)
					PfMod(sc, &sc->combinedID, &sc->inoutID_y, &sc->fft_dim_full);
				else
					PfMod(sc, &sc->combinedID, &sc->inoutID_x, &sc->fft_dim_full);

				if (sc->zeropad[readWrite]) {
					if (readWrite)
						PfIf_lt_start(sc, &sc->combinedID, &sc->fft_zeropad_left_write[sc->axis_id]);
					else
						PfIf_lt_start(sc, &sc->combinedID, &sc->fft_zeropad_left_read[sc->axis_id]);
					PfMov(sc, &sc->tempInt, &temp_int);
					PfIf_else(sc);

					if (readWrite)
						PfIf_ge_start(sc, &sc->combinedID, &sc->fft_zeropad_right_write[sc->axis_id]);
					else
						PfIf_ge_start(sc, &sc->combinedID, &sc->fft_zeropad_right_read[sc->axis_id]);
					PfMov(sc, &sc->tempInt, &temp_int);
					PfIf_end(sc);

					PfIf_end(sc);
				}
				if (sc->numAxisUploads > 1) {
					if (sc->zeropadBluestein[readWrite]) {
						if (readWrite)
							PfIf_lt_start(sc, &sc->combinedID, &sc->fft_zeropad_Bluestein_left_write[sc->axis_id]);
						else
							PfIf_lt_start(sc, &sc->combinedID, &sc->fft_zeropad_Bluestein_left_read[sc->axis_id]);
						PfMov(sc, &sc->tempInt, &temp_int);
						PfIf_end(sc);
					}
				}
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->tempInt, &temp_int);
			}

			temp_int1.data.i = k * used_registers.data.i + i;

			if (readWrite == 0) {
				if ((type == 111) || (type == 121) || (type == 131) || (type == 145)) {
					if (sc->readToRegisters) {
						appendGlobalToRegisters_x(sc, &sc->regIDs[temp_int1.data.i], &sc->inputsStruct, &sc->inoutID);
					}
					else {
						appendGlobalToRegisters_x(sc, &sc->temp, &sc->inputsStruct, &sc->inoutID);
						appendRegistersToShared(sc, &sc->sdataID, &sc->temp);
					}
				}
				else  if (type == 143) {
					if (sc->readToRegisters) {
						if (i < used_registers.data.i / 2) {
							appendGlobalToRegisters_x(sc, &sc->regIDs[k * sc->registers_per_thread + i], &sc->inputsStruct, &sc->inoutID);
						}
						else {
							appendGlobalToRegisters_y(sc, &sc->regIDs[k * sc->registers_per_thread + i - used_registers.data.i / 2], &sc->inputsStruct, &sc->inoutID);
						}
					}
					else {
						PfAdd(sc, &sc->combinedID, &sc->gl_LocalInvocationID_y, &temp_int);
						temp_int.data.i = 2;
						PfMod(sc, &sc->tempInt, &sc->combinedID, &temp_int);

						appendGlobalToRegisters_x(sc, &sc->temp, &sc->inputsStruct, &sc->inoutID);

						temp_int.data.i = 0;
						PfIf_eq_start(sc, &sc->tempInt, &temp_int);
						appendRegistersToShared_x_x(sc, &sc->sdataID, &sc->temp);
						PfIf_else(sc);
						appendRegistersToShared_y_x(sc, &sc->sdataID, &sc->temp);
						PfIf_end(sc);
					}
				}
				else {
					if (sc->readToRegisters) {
						appendGlobalToRegisters(sc, &sc->regIDs[temp_int1.data.i], &sc->inputsStruct, &sc->inoutID);
					}
					else {
						appendGlobalToShared(sc, &sc->sdataID, &sc->inputsStruct, &sc->inoutID);
					}
				}
				if ((sc->zeropad[readWrite]) || ((sc->numAxisUploads > 1) && (sc->zeropadBluestein[readWrite]))) {
					PfIf_else(sc);

					if (sc->readToRegisters) {
						PfSetToZero(sc, &sc->regIDs[temp_int1.data.i]);
					}
					else {
						PfSetToZeroShared(sc, &sc->sdataID);
					}
				}
			}
			else {
				if ((type == 111) || (type == 121) || (type == 131) || (type == 145)) {
					if (sc->writeFromRegisters) {
						appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[temp_int1.data.i]);
					}
					else {
						appendSharedToRegisters(sc, &sc->temp, &sc->sdataID);
						appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->temp);
					}
				}
				else  if (type == 143) {
					if (sc->writeFromRegisters) {
						appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[k * sc->registers_per_thread + i]);
						temp_int.data.i = fftDim.data.i * bufferStride[1].data.i;
						PfAdd(sc, &sc->tempInt, &sc->inoutID, &temp_int);
						appendRegistersToGlobal_y(sc, &sc->outputsStruct, &sc->tempInt, &sc->regIDs[k * sc->registers_per_thread + i]);
					}
					else {
					}
				}
				else {
					if (sc->writeFromRegisters) {
						appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->regIDs[temp_int1.data.i]);
					}
					else {
						appendSharedToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->sdataID);
					}
				}
			}
			if ((sc->zeropad[readWrite]) || ((sc->numAxisUploads > 1) && (sc->zeropadBluestein[readWrite]))) {
				PfIf_end(sc);
			}


			temp_int1.data.i = (k * used_registers.data.i + i + 1) * sc->localSize[1].data.i;
			if (temp_int1.data.i > fftDim.data.i) {
				//&sc->tempIntLen = sprintf(&sc->tempIntStr, "		}\n");
				PfIf_end(sc);
			}
		}
	}
	//&sc->tempIntLen = sprintf(&sc->tempIntStr, "	}\n");

	if (sc->useDisableThreads) {
		temp_int.data.i = 0;
		PfIf_end(sc);
	}

	PfIf_end(sc);
	return;
}

static inline void appendReadDataPfSolve(PfSolveSpecializationConstantsLayout* sc, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	switch (type) {
	case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144:
		appendReadWriteDataPfSolve_nonstrided(sc, 0, type);
		break;
	case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145://grouped_c2c + single_c2c_strided
		appendReadWriteDataPfSolve_strided(sc, 0, type);
		break;
	}
	return;
}
static inline void appendWriteDataPfSolve(PfSolveSpecializationConstantsLayout* sc, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	switch (type) {
	case 0: case 5: case 6: case 110: case 120: case 130: case 140: case 142: case 144:
		appendReadWriteDataPfSolve_nonstrided(sc, 1, type);
		break;
	case 1: case 2: case 111: case 121: case 131: case 141: case 143: case 145://grouped_c2c + single_c2c_strided
		appendReadWriteDataPfSolve_strided(sc, 1, type);
		break;
	}
	return;
}

static inline void appendGlobalToRegisters_mat_ParallelThomas(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	uint64_t used_registers = sc->registers_per_thread;//(sc->M_size.data.i + sc->warpSize - 1) / sc->warpSize;
	
	PfMov(sc, &sc->inoutID, &sc->warpInvocationID);
	for (uint64_t i = 0; i < used_registers; i++) {
		uint64_t activeThreads = (sc->M_size.data.i + used_registers - 1) / used_registers;

		if (((activeThreads - 1) * used_registers + i) >= sc->M_size.data.i) activeThreads--;

		temp_int.data.i = activeThreads;
		PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);

		if ((!sc->md_zero) && (((sc->inputBufferId == 0) && (!sc->ud_zero)) || ((sc->inputBufferId == (sc->numConsecutiveJWIterations-1)) && (!sc->ld_zero)) || (sc->numConsecutiveJWIterations == 1))) {
			PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->offset_md_global);
			if(sc->useMultipleInputBuffers)
				appendGlobalToRegistersMultipleBuffers(sc, &sc->md[i], &sc->inputsStruct, &sc->tempInt);
			else
				appendGlobalToRegisters(sc, &sc->md[i], &sc->inputsStruct, &sc->tempInt);

		}
		if (!sc->ld_zero) {
			PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->offset_ld_global);
			temp_int.data.i = 1;
            //PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			if(sc->useMultipleInputBuffers)
				appendGlobalToRegistersMultipleBuffers(sc, &sc->ld[i], &sc->inputsStruct, &sc->tempInt);
			else
				appendGlobalToRegisters(sc, &sc->ld[i], &sc->inputsStruct, &sc->tempInt);
		}

		if (!sc->ud_zero) {
			PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->offset_ud_global);
			temp_int.data.i = 1;
            //PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			if(sc->useMultipleInputBuffers)
				appendGlobalToRegistersMultipleBuffers(sc, &sc->ud[i], &sc->inputsStruct, &sc->tempInt);
			else
				appendGlobalToRegisters(sc, &sc->ud[i], &sc->inputsStruct, &sc->tempInt);
		}
		
		PfIf_else(sc);

		if (!sc->md_zero) {
			PfSetToZero(sc, &sc->md[i]);
		}
		if (!sc->ld_zero) {
			PfSetToZero(sc, &sc->ld[i]);
		}
		if (!sc->ud_zero) {
			PfSetToZero(sc, &sc->ud[i]);
		}

		PfIf_end(sc);
		temp_int.data.i = activeThreads;
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
	}
	return;
}

static inline void appendGlobalToRegisters_mat(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	if ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul)) {
		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul)) {
			if (i > 0) {
				temp_int.data.i = sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
		}else{
			temp_int.data.i = i * sc->num_threads;
			PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
		}
		
		if (((i + 1) * sc->num_threads > sc->M_size.data.i) || ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul))) {
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		}
		if (!sc->md_zero) {
			PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->offset_md_global);
			if(sc->useMultipleInputBuffers)
				appendGlobalToRegistersMultipleBuffers(sc, &sc->md[i], &sc->inputsStruct, &sc->tempInt);
			else
				appendGlobalToRegisters(sc, &sc->md[i], &sc->inputsStruct, &sc->tempInt);

		}
		else
		{
			//PfSetToZero(sc, &sc->md[i]);//fix
		}
		/*if((i==0)){
			sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f\\n\", inoutID, md_%" PRIu64 ");\n", i);
			
			
		}*/
		if (!sc->ld_zero) {
			PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->offset_ld_global);
			temp_int.data.i = 1;
            //PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			if(sc->useMultipleInputBuffers)
				appendGlobalToRegistersMultipleBuffers(sc, &sc->ld[i], &sc->inputsStruct, &sc->tempInt);
			else
				appendGlobalToRegisters(sc, &sc->ld[i], &sc->inputsStruct, &sc->tempInt);
		}
		else {
			//PfSetToZero(sc, &sc->ld[i]);//fix

		}
		/*if((i==0)){
					sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f\\n\", inoutID, ld_%" PRIu64 ");\n", i);
					
					
				}*/
		if (!sc->ud_zero) {
			PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->offset_ud_global);
			temp_int.data.i = 1;
            //PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
			if(sc->useMultipleInputBuffers)
				appendGlobalToRegistersMultipleBuffers(sc, &sc->ud[i], &sc->inputsStruct, &sc->tempInt);
			else
				appendGlobalToRegisters(sc, &sc->ud[i], &sc->inputsStruct, &sc->tempInt);
		}
		else {
			//PfSetToZero(sc, &sc->ud[i]);//fix
			
		}
		
		if (((i + 1) * sc->num_threads > sc->M_size.data.i) || ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul))) {
			PfIf_end(sc);
		}
	}
    /*for(uint64_t i = 0; i < sc->registers_per_thread; i++)
    {
		if (!sc->ud_zero) {
			if ((sc->registers_per_thread > 1) && (i < sc->registers_per_thread-1)) {
				temp_int.data.i = 1;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfMov(sc, &sc->temp, &sc->ud[i + 1]);
				PfIf_else(sc);
				PfMov(sc, &sc->temp, &sc->ud[i]);
				PfIf_end(sc);

				PfSubgroupShuffleDownCyclic(sc, &sc->ud[i], &sc->temp, 1);
			}
			else {
				PfSubgroupShuffleDown(sc, &sc->ud[i], &sc->ud[i], 1);
			}
		}
		if (!sc->ld_zero) {

			if ((sc->registers_per_thread > 1) && (i > 0)) {
				temp_int.data.i = sc->warpSize - 1;
				PfIf_ge_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfMov(sc, &sc->temp, &sc->ld[i - 1]);
				PfIf_else(sc);
				PfMov(sc, &sc->temp, &sc->ld[i]);
				PfIf_end(sc);

				PfSubgroupShuffleUpCyclic(sc, &sc->ld[i], &sc->temp, 1);
			}
			else {
				PfSubgroupShuffleUp(sc, &sc->ld[i], &sc->ld[i], 1);
			}
		}
	}*/
	return;
}
static inline void appendReadWrite_rd(PfSolveSpecializationConstantsLayout* sc, int readWrite) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	if ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul)) {
		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul)) {
			if (i > 0) {
				temp_int.data.i = sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
		}else{
			temp_int.data.i = i * sc->num_threads;
			PfAdd(sc, &sc->inoutID, &sc->gl_LocalInvocationID_x, &temp_int);
		}
		if (((i + 1) * sc->num_threads > sc->M_size.data.i) || ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul))) {
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		}
		int control_zp = 0;
		PfConfigureZeropad(sc, &sc->inoutID, &control_zp, readWrite);
		temp_int.data.i = 0;
		if (control_zp) PfIf_gt_start(sc, &sc->tempInt, &temp_int);

		//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = %s%s[inoutID + %s*%" PRIu64 "+%" PRIu64 "]%s;\n", i, sc->convTypeLeftInput, sc->inputsStructRes, sc->gl_WorkGroupID_x, sc->outputStride[2].x_num, sc->offset_res_global, sc->convTypeRightInput);
		PfMul(sc, &sc->tempInt, &sc->gl_WorkGroupID_x, &sc->outputStride[1], 0);
		PfAdd(sc, &sc->tempInt, &sc->inoutID, &sc->tempInt);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->offset_res_global);
		if (sc->readToRegisters || sc->writeFromRegisters) {
			if (readWrite)
				appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->rd[i]);
			else {
				appendGlobalToRegisters(sc, &sc->rd[i], &sc->outputsStruct, &sc->tempInt);
				if (control_zp) {
					PfIf_else(sc);
					PfSetToZero(sc, &sc->rd[i]);
				}
			}
		}
		else {
			temp_int.data.i = i * sc->num_threads + sc->offset_res.data.i;
			PfAdd(sc, &sc->inoutID_x, &sc->gl_LocalInvocationID_x, &temp_int);
			
			if (readWrite)
				appendSharedToGlobal(sc, &sc->outputsStruct, &sc->tempInt, &sc->inoutID_x);
			else {
				appendGlobalToShared(sc, &sc->inoutID_x, &sc->outputsStruct, &sc->tempInt);
				if (control_zp) {
					PfIf_else(sc);
					PfSetToZeroShared(sc, &sc->inoutID_x);
				}
			}
		}
		if (control_zp) PfIf_end(sc);
		if (((i + 1) * sc->num_threads > sc->M_size.data.i) || ((sc->warpSize != sc->num_threads) && (sc->performTriSolve || sc->performMatVecMul))) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendReadWrite_block(PfSolveSpecializationConstantsLayout* sc, int readWrite) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	PfMul(sc, &sc->combinedID, &sc->localSize[0], &sc->gl_LocalInvocationID_y, 0);
	PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->gl_LocalInvocationID_x);

	temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i * sc->registers_per_thread;
	PfMul(sc, &sc->tempInt, &sc->gl_WorkGroupID_x, &temp_int, 0);
	PfAdd(sc, &sc->combinedID, &sc->combinedID, &sc->tempInt);

	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		//sc->tempLen = sprintf(sc->tempStr, "	id_y = %" PRIu64 " + %" PRIu64 " * %s + %" PRIu64 " * %s;\n", i, sc->registers_per_thread, sc->gl_LocalInvocationID_y, sc->localSize[1] * sc->registers_per_thread, sc->gl_WorkGroupID_x);
		PfMod(sc, &sc->inoutID_x, &sc->combinedID, &sc->size[0]);
		PfDiv(sc, &sc->inoutID_y, &sc->combinedID, &sc->size[0]);

		PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
		int control_zp = 0;
		PfConfigureZeropad(sc, &sc->inoutID_x, &control_zp, readWrite);

		temp_int.data.i = 0;
		if (control_zp) PfIf_gt_start(sc, &sc->tempInt, &temp_int);

		if(readWrite)
        {
			PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->outputStride[1], 0);
            PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->offset_res_global);
		}
		else
		{
			PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->offset_md_global);
		}
		if ((((sc->block % 10) == 4)&&(readWrite)) || (((sc->block % 10) == 7) && (!readWrite))){
			temp_int.data.i = 2;
			PfMul(sc, &sc->inoutID, &sc->inoutID, &temp_int, 0);
		}
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);

		if (readWrite) {
			if (((sc->block / 10) % 10) == 2) {
				if ((sc->block % 10) == 2)
					appendGlobalToRegisters_x(sc, &sc->temp, &sc->outputsStruct, &sc->inoutID);
				else if ((sc->block % 10) == 3)
					appendGlobalToRegisters_y(sc, &sc->temp, &sc->outputsStruct, &sc->inoutID);
                else
					appendGlobalToRegisters(sc, &sc->temp, &sc->outputsStruct, &sc->inoutID);

				PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp);
			}

			if ((sc->block % 10) == 2)
				appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->rd[i]);
			else if ((sc->block % 10) == 3)
				appendRegistersToGlobal_y(sc, &sc->outputsStruct, &sc->inoutID, &sc->rd[i]);
			else if ((sc->block % 10) == 4){
				appendRegistersToGlobal_x(sc, &sc->outputsStruct, &sc->inoutID, &sc->rd[i]);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->outputStride[1]);
				appendRegistersToGlobal_y(sc, &sc->outputsStruct, &sc->inoutID, &sc->rd[i]);
			}
			else
				appendRegistersToGlobal(sc, &sc->outputsStruct, &sc->inoutID, &sc->rd[i]);
			
			//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = %s%s[id_x + id_y * %" PRIu64 " + %" PRIu64 "]%s;\n", i, sc->convTypeLeftInput, sc->inputsStruct, sc->inputStride[1], sc->offset_md_global, sc->convTypeRightInput);
		}
		else {
			if ((sc->block % 10) == 5)
				appendGlobalToRegisters_x(sc, &sc->rd[i], &sc->inputsStruct, &sc->inoutID);
			else if ((sc->block % 10) == 6)
				appendGlobalToRegisters_y(sc, &sc->rd[i], &sc->inputsStruct, &sc->inoutID);
			else if ((sc->block % 10) == 7){
				appendGlobalToRegisters_x(sc, &sc->rd[i], &sc->inputsStruct, &sc->inoutID);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);
				appendGlobalToRegisters_y(sc, &sc->rd[i], &sc->inputsStruct, &sc->inoutID);
			}
            else
				appendGlobalToRegisters(sc, &sc->rd[i], &sc->inputsStruct, &sc->inoutID);
			//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = %s%s[id_x + id_y * %" PRIu64 " + %" PRIu64 "]%s;\n", i, sc->convTypeLeftInput, sc->inputsStruct, sc->inputStride[1], sc->offset_md_global, sc->convTypeRightInput);
			if (control_zp) {
				PfIf_else(sc);
				PfSetToZero(sc, &sc->rd[i]);
			}
		}

		if (control_zp) PfIf_end(sc);

		PfIf_end(sc);
		temp_int.data.i = sc->localSize[0].data.i * sc->localSize[1].data.i;
		PfAdd(sc, &sc->combinedID, &sc->combinedID, &temp_int);
	}
	return;
}

#endif