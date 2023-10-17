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
#ifndef PFSOLVE_MATVECMUL_H
#define PFSOLVE_MATVECMUL_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_Transfers.h"

static inline void appendMatVecMul(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	/*
		Kernel can have two input types - from registers or from shared memory. In second case, a copy of state in registers is created.
	*/
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;
	
	/*if (sc->read_SharedToRegisters) {
		appendBarrier(sc);
		
		appendSharedToRegisters_init(sc);
		
	}
	if (sc->read_RegistersToShared) {
		appendRegistersToShared_mat(sc);
		
	}*/
	//appendBarrier(sc);
	
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);
		
		*/
		if (!sc->ud_zero) {
			if ((sc->registers_per_thread > 1) && (i < sc->registers_per_thread-1)) {
				temp_int.data.i = 1;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfMov(sc, &sc->rd_copy[i], &sc->rd[i + 1]);
				PfIf_else(sc);
				PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
				PfIf_end(sc);

				PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->rd_copy[i], 1);
			}
			else {
				PfSubgroupShuffleDown(sc, &sc->temp, &sc->rd[i], 1);
			}
		}
		if (!sc->ld_zero) {

			if ((sc->registers_per_thread > 1) && (i > 0)) {
				temp_int.data.i = sc->warpSize - 1;
				PfIf_ge_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfMov(sc, &sc->rd_copy[i], &sc->rd[i - 1]);
				PfIf_else(sc);
				PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
				PfIf_end(sc);

				PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->rd_copy[i], 1);
			}
			else {
				PfSubgroupShuffleUp(sc, &sc->temp1, &sc->rd[i], 1);
			}
		}
		PfMul(sc, &sc->rd_copy[i], &sc->rd[i], &sc->md[i], 0);
		if ((sc->M_size.data.i - 1 - i * sc->num_threads) > 0){
			if(i==sc->registers_per_thread-1){
				temp_int.data.i = sc->M_size.data.i - 1 - i * sc->num_threads;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			}
			if (!sc->ud_zero){
				PfMul(sc, &sc->temp, &sc->temp, &sc->ud[i], 0);
				PfAdd(sc, &sc->rd_copy[i], &sc->rd_copy[i], &sc->temp);
			//	sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += ud_%" PRIu64 " * temp_0;\n", i, i);
				
				
			}
			
			if(i==sc->registers_per_thread-1){
				PfIf_end(sc);
				
			}
		}
		if(i==0){
			temp_int.data.i = 0;
			PfIf_gt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			
		}
		if (!sc->ld_zero){
			//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += ld_%" PRIu64 " * temp_1;\n", i, i);
			PfMul(sc, &sc->temp1, &sc->temp1, &sc->ld[i], 0);
			PfAdd(sc, &sc->rd_copy[i], &sc->rd_copy[i], &sc->temp1);
			
		}
		
		if(i==0){
			PfIf_end(sc);
		}
		/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);
		
		*/
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		PfMov(sc, &sc->rd[i], &sc->rd_copy[i]);
	}
	/*if (sc->write_RegistersToShared) {
		appendBarrier(sc);
		
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			appendRegistersToShared_res(sc,i);
			
		}
	}
	if (sc->write_SharedToRegisters) {
		appendSharedToRegisters(sc);
		
	}*/
	return;
}

static inline void appendMatVecMul_fromGlobal(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	for (uint64_t j = 0; j < sc->LDA; j++) {
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;*/
			if ((sc->M_size.data.i - i * sc->localSize[0].data.i) > 0){
				if ((i + 1) * sc->localSize[0].data.i > sc->M_size.data.i) {
					temp_int.data.i = sc->M_size.data.i - i * sc->localSize[0].data.i;
					PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				}
				temp_int.data.i = -sc->KU - i * sc->localSize[0].data.i + j;
				if(temp_int.data.i > 0)
					PfIf_ge_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);

				temp_int.data.i = sc->M_size.data.i - sc->KU - i * sc->localSize[0].data.i + j;
				if (temp_int.data.i > 0){
					PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
					//sc->tempLen = sprintf(sc->tempStr, "	if((%s + %" PRIi64 ">= 0)&&((%s + %" PRIi64 "< 0))){\n", sc->gl_LocalInvocationID_x, (int64_t)i * sc->localSize[0] - sc->KU + j, sc->gl_LocalInvocationID_x, (int64_t)i * sc->localSize[0] + sc->KU - j - sc->M_size);
					
					int64_t shift = sc->KU - j;
					//if (shift < 0) shift = 0;
					temp_int.data.i = i * sc->localSize[0].data.i + shift;
					PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &temp_int);
					appendSharedToRegisters(sc, &sc->temp, &sc->tempInt);

					if (sc->offset_md_global.type > 100) {
						temp_int.data.i = i * sc->localSize[0].data.i + shift + j * sc->M_size.data.i;
						PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &temp_int);
						PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->offset_md_global);
					}
					else {
						temp_int.data.i = i * sc->localSize[0].data.i + shift + j * sc->M_size.data.i + sc->offset_md_global.data.i;
						PfAdd(sc, &sc->tempInt, &sc->gl_LocalInvocationID_x, &temp_int);
					}
					appendGlobalToRegisters(sc, &sc->temp1, &sc->inputsStruct, &sc->tempInt);
					PfMul(sc, &sc->temp, &sc->temp, &sc->temp1, &sc->temp2);
					PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp);
					//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += sdata[%s + %" PRIu64 "] * %s%s[%s +%" PRIu64 "]%s;\n", i, sc->gl_LocalInvocationID_x, i * sc->localSize[0] + sc->KU - j, sc->convTypeLeftInput, sc->inputsStruct, sc->gl_LocalInvocationID_x, shift + j * sc->M_size + sc->offset_md_global, sc->convTypeRightInput);
					
					PfIf_end(sc);
				}
				temp_int.data.i = -sc->KU - i * sc->localSize[0].data.i + j;
				if(temp_int.data.i > 0)
					PfIf_end(sc);

				if ((i + 1) * sc->localSize[0].data.i > sc->M_size.data.i) {
					PfIf_end(sc);
				}
			}
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;*/
		}
	}

	return;
}

#endif
