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
#ifndef PFSOLVE_JONESWORLAND_SEQUENTIAL_H
#define PFSOLVE_JONESWORLAND_SEQUENTIAL_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_InputOutputLayout.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_SharedMemory.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_Registers.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"

#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_GlobalToRegisters.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_RegistersToGlobal.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_SharedToRegisters.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_RegistersToShared.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_GlobalToShared.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_SharedToGlobal.h"

#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_MatVecMul.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_TridiagonalSolve.h"

#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_KernelUtils.h"
#include "pfSolve_CodeGen/pfSolve_KernelBuildingBlocks/pfSolve_KernelStartEnd.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"
static inline PfSolveResult PfSolve_shaderGen_JonesWorlandMV_sequential(PfSolveSpecializationConstantsLayout* sc) {
	PfSolveResult res = PFSOLVE_SUCCESS;
	res = appendVersion(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	res = appendExtensions(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	if ((((sc->floatTypeCode/10)%10) == 3) || (((sc->floatTypeInputMemoryCode/10)%10) == 3) || (((sc->floatTypeOutputMemoryCode/10)%10) == 3)) {
		appendQuadDoubleDoubleStruct(sc);
	}
	if ((sc->floatTypeCode != sc->floatTypeInputMemoryCode) || (sc->floatTypeCode != sc->floatTypeOutputMemoryCode)) {
		appendConversion(sc);
	}	
	res = appendInputLayout(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	res = appendOutputLayout(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	res = appendPushConstants(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	res = appendKernelStart(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	res = appendSharedMemoryInitialization(sc);
	if (res != PFSOLVE_SUCCESS) return res;

	res = appendRegistersInitialization_transposed(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	for (uint64_t i = 0; i < (uint64_t)ceil(sc->M_size.x_num / (double)sc->size[0]); i++) {
		sc->start_shift_x = i * sc->size[0];
		sc->offset_md = 0;
		//sc->offset_res = 2 * sc->inputStride[1];
		sc->offset_res_global = sc->offsetSolution.x_num;
		if (!sc->upperBanded) {
			//sc->offset_ld = sc->inputStride[1];
			sc->offset_ld_global = sc->offsetM.x_num + sc->inputStride[1].x_num;
			sc->offset_md_global = sc->offsetM.x_num;
			sc->inverseGlobal = 1;
		}
		else {
			//sc->offset_ud = sc->inputStride[1];
			sc->offset_md_global = sc->offsetM.x_num + sc->inputStride[1].x_num;
			sc->offset_ud_global = sc->offsetM.x_num;
			sc->inverseGlobal = 0;
		}
		res = appendGlobalToShared_res_transposed(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		res = appendBarrier(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		res = appendSharedToRegisters_res_transposed(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		if (sc->performMatVecMul){
			if (!sc->upperBanded) {
				res = appendMatVecMul_transposed_lt(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			else {
				res = appendMatVecMul_transposed_ut(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}
		sc->offset_md = 0;
		//sc->offset_res = 2 * sc->inputStride[1];
		sc->offset_res_global = sc->offsetSolution.x_num;
		if (sc->performTriSolve){
			if (!sc->upperBanded) {
				//sc->offset_ud = sc->inputStride[1];
				sc->offset_md_global = sc->offsetV.x_num + sc->inputStride[1].x_num;
				sc->offset_ud_global = sc->offsetV.x_num;
				res = appendBidiagonalSolve_ut(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->inverseGlobal = 1;
			}
			else {
				//sc->offset_ld = sc->inputStride[1];
				sc->offset_ld_global = sc->offsetV.x_num + sc->inputStride[1].x_num;
				sc->offset_md_global = sc->offsetV.x_num;
				res = appendBidiagonalSolve_lt(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->inverseGlobal = 0;
			}
		}
		res = appendRegistersToShared_res_transposed(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		res = appendBarrier(sc);
		if (res != PFSOLVE_SUCCESS) return res;

		res = appendSharedToGlobal_res_transposed(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}
	res = appendKernelEnd(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	return res;
	}
#endif
