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

static inline void appendMatVecMul_ParallelThomas(PfSolveSpecializationConstantsLayout* sc) {
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
		//PfPrintReg(sc, &sc->gl_LocalInvocationID_x, &sc->temp1);
	}
	uint64_t used_registers = sc->registers_per_thread;// (sc->M_size.data.i + sc->warpSize - 1) / sc->warpSize;
	if (!sc->ud_zero){
		PfSubgroupShuffleDown(sc, &sc->temp, &sc->rd[0], 1);
	}
	if (!sc->ld_zero){
		PfSubgroupShuffleUp(sc, &sc->temp, &sc->rd[used_registers-1], 1);
	}
	for (uint64_t i = 0; i < used_registers; i++) {
		/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);

		*/
		//uint64_t activeThreads = (sc->M_size.data.i + used_registers - 1) / used_registers;

		//temp_int.data.i = activeThreads;
		//PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);

		if (sc->ud_zero && sc->ld_zero) {
			PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->md[i], 0);
		}
		if (!sc->ud_zero) {
			if ((sc->inputBufferId == 0) || (sc->numConsecutiveJWIterations == 1))
				PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->md[i], 0);
			if (i == (used_registers-1)) {
				PfMul(sc, &sc->temp, &sc->temp, &sc->ud[i], 0);
				PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp);
				/*temp_int.data.i = sc->M_size.data.i - (used_registers - 1) * sc->warpSize;
				if (temp_int.data.i > 0) {
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
				}

				PfMul(sc, &sc->temp, &sc->temp, &sc->ud[i], 0);
				PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp);
				//	sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += ud_%" PRIu64 " * temp_0;\n", i, i);

				if (temp_int.data.i > 0) {
					PfIf_end(sc);
				}*/
			}
			else {
				PfMul(sc, &sc->temp1, &sc->rd[i+1], &sc->ud[i], 0);
				PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp1);
			}
		}
		if (!sc->ld_zero) {
			if ((sc->inputBufferId == (sc->numConsecutiveJWIterations-1)) || (sc->numConsecutiveJWIterations == 1))
				PfMul(sc, &sc->temp1, &sc->rd[i], &sc->md[i], 0);
			else 
				PfMov(sc, &sc->temp1, &sc->rd[i]);
			if (i == 0) {
				PfMul(sc, &sc->temp, &sc->temp, &sc->ld[i], 0);
				PfAdd(sc, &sc->md[i], &sc->temp1, &sc->temp);
				/*temp_int.data.i = 0;
				if (temp_int.data.i > 0) {
					PfIf_gt_start(sc, &sc->warpInvocationID, &temp_int);
				}
				PfMul(sc, &sc->temp, &sc->temp, &sc->ld[i], 0);
				PfAdd(sc, &sc->md[i], &sc->temp1, &sc->temp);
				
				if (temp_int.data.i > 0) {
					PfIf_end(sc);
				}*/
			}		
			else {
				PfMul(sc, &sc->temp, &sc->rd[i-1], &sc->ld[i], 0);
				PfMov(sc, &sc->rd[i - 1], &sc->md[i-1]);
				PfAdd(sc, &sc->md[i], &sc->temp1, &sc->temp);
			}
			if (i == used_registers-1) {
				PfMov(sc, &sc->rd[i], &sc->md[i]);
			}
		}

		//PfIf_end(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);
		//res = PfAppendLine(sc);
		
	}
	return;
}

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
		//PfPrintReg(sc, &sc->gl_LocalInvocationID_x, &sc->temp1);
	}
	if (sc->num_threads != sc->warpSize){
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);

			*/

			if (i * sc->num_threads < sc->M_size_pow2.data.i){
				if (!sc->ud_zero) {
					if (i < sc->registers_per_thread-1) {
						temp_int.data.i = 1;
						PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
						PfMov(sc, &sc->rd_copy[i], &sc->rd[i + 1]);
						PfIf_else(sc);
						PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
						PfIf_end(sc);
					}
					else {
						PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
					}
					PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->rd_copy[i], 1);
					if (i == sc->registers_per_thread-1) {
						temp_int.data.i = 0;
						PfIf_eq_start(sc, &sc->warpInvocationID, &temp_int);
						appendRegistersToShared(sc, &sc->warpID, &sc->rd[0]);
						PfIf_end(sc);

						appendBarrierPfSolve(sc);
						
						temp_int.data.i = sc->warpSize - 1;
						PfIf_eq_start(sc, &sc->warpInvocationID, &temp_int);

						temp_int.data.i = sc->num_threads/sc->warpSize - 1;
						PfIf_lt_start(sc, &sc->warpID, &temp_int);
						temp_int.data.i = 1;
						PfAdd(sc, &sc->tempInt, &sc->warpID, &temp_int);
						appendSharedToRegisters(sc, &sc->temp, &sc->tempInt);
						PfIf_end(sc);

						PfIf_end(sc);

						appendBarrierPfSolve(sc);
					}
				}
				if (!sc->ld_zero) {
					if (i > 0) {
						temp_int.data.i = sc->warpSize - 1;
						PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
						PfMov(sc, &sc->rd_copy[i], &sc->rd[i - 1]);
						PfIf_else(sc);
						PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
						PfIf_end(sc);
					}
					else {
						PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
					}
					PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->rd_copy[i], 1);
					if (i == 0) {
						temp_int.data.i = sc->warpSize - 1;
						PfIf_eq_start(sc, &sc->warpInvocationID, &temp_int);
						appendRegistersToShared(sc, &sc->warpID, &sc->rd[sc->registers_per_thread-1]);
						PfIf_end(sc);

						appendBarrierPfSolve(sc);
						
						temp_int.data.i = 0;
						PfIf_eq_start(sc, &sc->warpInvocationID, &temp_int);
						
						temp_int.data.i = 0;
						PfIf_gt_start(sc, &sc->warpID, &temp_int);
						temp_int.data.i = 1;
						PfSub(sc, &sc->tempInt, &sc->warpID, &temp_int);
						appendSharedToRegisters(sc, &sc->temp1, &sc->tempInt);
						PfIf_end(sc);
						
						PfIf_end(sc);

						appendBarrierPfSolve(sc);
					}
					
				}
				PfMul(sc, &sc->rd_copy[i], &sc->rd[i], &sc->md[i], 0);
				if (!sc->ud_zero) {
					temp_int.data.i = sc->warpSize * sc->registers_per_thread;
					PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
					PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
					temp_int.data.i = i * sc->warpSize;
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

					temp_int.data.i = sc->M_size.data.i - 1;
					PfIf_lt_start(sc, &sc->inoutID, &temp_int);
					PfMul(sc, &sc->temp, &sc->temp, &sc->ud[i], 0);
					PfAdd(sc, &sc->rd_copy[i], &sc->rd_copy[i], &sc->temp);
					//	sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += ud_%" PRIu64 " * temp_0;\n", i, i);
					PfIf_end(sc);
				}
				if (!sc->ld_zero) {
					temp_int.data.i = sc->warpSize * sc->registers_per_thread;
					PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
					PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
					temp_int.data.i = i * sc->warpSize;
					PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

					temp_int.data.i = 0;
					PfIf_gt_start(sc, &sc->inoutID, &temp_int);
					PfMul(sc, &sc->temp1, &sc->temp1, &sc->ld[i], 0);
					PfAdd(sc, &sc->rd_copy[i], &sc->rd_copy[i], &sc->temp1);
					PfIf_end(sc);

				}
			}
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);

			*/
		}

		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			PfMov(sc, &sc->rd[i], &sc->rd_copy[i]);
		}
	}
	else {
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);

			*/

			if (i * sc->num_threads < sc->M_size.data.i){
				if (!sc->ud_zero) {
					if ((sc->registers_per_thread > 1) && (i < sc->registers_per_thread - 1)) {
						temp_int.data.i = 1;
						PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
						PfMov(sc, &sc->temp, &sc->rd[i + 1]);
						PfIf_else(sc);
						PfMov(sc, &sc->temp, &sc->rd[i]);
						PfIf_end(sc);
						if (sc->performALT)
							PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->temp, 2);
						else
							PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->temp, 1);
					}
					else {
						if (sc->performALT)
							PfSubgroupShuffleDown(sc, &sc->temp, &sc->rd[i], 2);
						else
							PfSubgroupShuffleDown(sc, &sc->temp, &sc->rd[i], 1);
					}
				}
				if (!sc->ld_zero) {

					if ((sc->registers_per_thread > 1) && (i > 0)) {
						temp_int.data.i = sc->warpSize - 1;
						PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
						PfMov(sc, &sc->temp1, &sc->rd_copy[0]);
						PfIf_else(sc);
						PfMov(sc, &sc->temp1, &sc->rd[i]);
						PfIf_end(sc);
						if (sc->performALT)
							PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->temp1, 2);
						else
							PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->temp1, 1);
					}
					else {
						if (sc->performALT)
							PfSubgroupShuffleUp(sc, &sc->temp1, &sc->rd[i], 2);
						else
							PfSubgroupShuffleUp(sc, &sc->temp1, &sc->rd[i], 1);
					}
					if (sc->registers_per_thread > 1)
						PfMov(sc, &sc->rd_copy[0], &sc->rd[i]);
				}
				PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->md[i], 0);
				if (!sc->ud_zero) {
					temp_int.data.i = sc->M_size.data.i - 1 - i * sc->num_threads;
					if (temp_int.data.i > 0) {
						if (temp_int.data.i < sc->localSize[0].data.i) {
							PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
						}
						PfMul(sc, &sc->temp, &sc->temp, &sc->ud[i], 0);
						PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp);
						//	sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += ud_%" PRIu64 " * temp_0;\n", i, i);

						if (temp_int.data.i < sc->localSize[0].data.i) {
							PfIf_end(sc);

						}
					}
				}
				if (!sc->ld_zero) {
					if (i == 0) {
						temp_int.data.i = 0;
						PfIf_gt_start(sc, &sc->warpInvocationID, &temp_int);

					}
					//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " += ld_%" PRIu64 " * temp_1;\n", i, i);
					PfMul(sc, &sc->temp1, &sc->temp1, &sc->ld[i], 0);
					PfAdd(sc, &sc->rd[i], &sc->rd[i], &sc->temp1);

					if (i == 0) {
						PfIf_end(sc);
					}

				}
			}
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);

			*/
		}
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
		if ((sc->performALT) && ((((j % 2) == 1 - ((sc->LDA % 2))) && sc->KL) || (((j % 2) == (sc->LDA % 2)) && sc->KU))) continue;
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d  %%f  %%f  %%f\\n\", inoutID, res_%" PRIu64 ", md_%" PRIu64 ", ld_%" PRIu64 ");\n", i, i, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;*/
			temp_int.data.i = sc->M_size.data.i - i * sc->localSize[0].data.i;
			if (temp_int.data.i > 0){
				if ((i + 1) * sc->localSize[0].data.i > sc->M_size.data.i) {
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
