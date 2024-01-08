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
#ifndef PFSOLVE_TRIDIAGONAL_H
#define PFSOLVE_TRIDIAGONAL_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"

static inline void appendTridiagonalSolve_PCR(PfSolveSpecializationConstantsLayout* sc) {
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
		
		appendBarrier(sc);
		
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			appendRegistersToShared_mat_lu(sc, i);
			
		}
		appendBarrier(sc);
		
	}
	if (sc->read_RegistersToShared) {
		appendRegistersToShared_mat(sc);
		
		appendBarrier(sc);
		
	}*/
	//int64_t maxPCRiteration = (int64_t)ceil(log2(sc->M_size_pow2.data.i));
	//int64_t BSsystemSize = 0;// sc->M_size_pow2.data.i / sc->warpSize;
	int64_t maxSharedMemPCRIteration =  (int64_t)ceil(log2(sc->num_threads / sc->warpSize));
	int64_t maxPCRiteration = (int64_t)ceil(log2(sc->warpSize)); //(int64_t)ceil(log2(sc->M_size_pow2.data.i));
	int64_t maxBSiteration = sc->M_size_pow2.data.i / sc->num_threads;
	int64_t stride = 1;
	int64_t next_stride = 1;
    /* if(!sc->ud_zero)
        {
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if ((sc->registers_per_thread > 1) && (i < sc->registers_per_thread - 1)) {
				temp_int.data.i = 1;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfMov(sc, &sc->ud_copy[i], &sc->ud[i + 1]);
				PfIf_else(sc);
				PfMov(sc, &sc->ud_copy[i], &sc->ud[i]);
				PfIf_end(sc);

				PfSubgroupShuffleDownCyclic(sc, &sc->ud_copy[i], &sc->ud_copy[i], 1);
			}
			else {
				PfSubgroupShuffleDown(sc, &sc->ud_copy[i], &sc->ud[i], 1);
			}
		}
	}
	if (!sc->ld_zero) {
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if ((sc->registers_per_thread > 1) && (i > 0)) {
				temp_int.data.i = sc->warpSize - 1;
				PfIf_ge_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfMov(sc, &sc->ld_copy[i], &sc->ld[i - 1]);
				PfIf_else(sc);
				PfMov(sc, &sc->ld_copy[i], &sc->ld[i]);
				PfIf_end(sc);

				PfSubgroupShuffleUpCyclic(sc, &sc->ld_copy[i], &sc->ld_copy[i], 1);
			}
			else {
				PfSubgroupShuffleUp(sc, &sc->ld_copy[i], &sc->ld[i], 1);
			}
		}
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (!sc->ld_zero) {
			PfMov(sc, &sc->ld[i], &sc->ld_copy[i]);
		}
		if (!sc->ud_zero) {
			PfMov(sc, &sc->ud[i], &sc->ud_copy[i]);
		}
	}*/
	for (int64_t sharedMemPCRIteration = 0; sharedMemPCRIteration < maxSharedMemPCRIteration; sharedMemPCRIteration++) {
		next_stride = stride * 2;
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*if ((i == 0)) {
							sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f  %%f  %%f  %%f\\n\", inoutID, ld_%" PRIu64 ", md_%" PRIu64 ", ud_%" PRIu64 ", res_%" PRIu64 ");\n", i,i,i,i);
							PfAppendLine(sc);
							
						}*/
			//sc->tempLen = sprintf(sc->tempStr, "	{//if (acc_%" PRIi64 ">0){\n", i);
			//PfAppendLine(sc);
			//
			if (!sc->ld_zero) {
				PfMov(sc, &sc->temp, &sc->ld[i]);

				if (i > 0) {
					temp_int.data.i = sc->warpSize - stride;
					PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
					PfMov(sc, &sc->ld_copy[i], &sc->ld[i - 1]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i - 1]);
					PfIf_else(sc);
					PfMov(sc, &sc->ld_copy[i], &sc->ld[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
					PfIf_end(sc);
				}
				else {
					PfMov(sc, &sc->ld_copy[i], &sc->ld[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
				}
				PfSubgroupShuffleUpCyclic(sc, &sc->ld_copy[i], &sc->ld_copy[i], stride);

				PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->rd_copy[i], stride);
					
				if (i == 0) {
					temp_int.data.i = sc->warpSize - stride;
					PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);

					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
					
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
                                        temp_int.data.i = sc->warpSize - stride;
					PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);

					appendRegistersToShared(sc, &sc->tempInt, &sc->rd[sc->registers_per_thread - 1]);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendRegistersToShared(sc, &sc->tempInt, &sc->ld[sc->registers_per_thread - 1]);
					PfIf_end(sc);

					appendBarrierPfSolve(sc);

					temp_int.data.i = stride;
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
					
					temp_int.data.i = 0;
					PfIf_gt_start(sc, &sc->warpID, &temp_int);

					temp_int.data.i = 1;
					PfSub(sc, &sc->tempInt, &sc->warpID, &temp_int);
					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);

					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);

					appendSharedToRegisters(sc, &sc->temp1, &sc->tempInt);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendSharedToRegisters(sc, &sc->ld_copy[i], &sc->tempInt);
					PfIf_end(sc);

					PfIf_end(sc);

					appendBarrierPfSolve(sc);
				}
				temp_int.data.i = sc->warpSize * sc->registers_per_thread;
				PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
				temp_int.data.i = i * sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				temp_int.data.i = stride;
				PfIf_lt_start(sc, &sc->inoutID, &temp_int);
				{
				//temp_int.data.i = stride - i * sc->warpSize;
				//if (temp_int.data.i > 0) {
					PfSetToZero(sc, &sc->ld_copy[i]);
					PfSetToZero(sc, &sc->temp);
				}
				PfIf_else(sc);
				{
					temp_int.data.i = next_stride;
					PfIf_lt_start(sc, &sc->inoutID, &temp_int);
					PfSetToZero(sc, &sc->ld_copy[i]);
					PfIf_end(sc);
				}
				PfIf_end(sc);

				PfMul(sc, &sc->ld_copy[i], &sc->ld_copy[i], &sc->temp, 0);
				PfMovNeg(sc, &sc->ld_copy[i], &sc->ld_copy[i]);
				PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp, 0);
					
				if (!sc->ud_zero) {
					PfSub(sc, &sc->md[i], &sc->md[i], &sc->temp2);
					//sc->tempLen = sprintf(sc->tempStr, "\n\
		md_%" PRIi64 " = md_%" PRIi64 " - temp_ac;\n", i, i);
						
				}
				PfSub(sc, &sc->rd_copy[i], &sc->rd[i], &sc->temp1);
			}

			if (!sc->ud_zero) {
				PfMov(sc, &sc->temp, &sc->ud[i]);

				if (i < (sc->registers_per_thread - 1)) {
					temp_int.data.i = stride;
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
					PfMov(sc, &sc->ud_copy[i], &sc->ud[i + 1]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i + 1]);
					PfIf_else(sc);
					PfMov(sc, &sc->ud_copy[i], &sc->ud[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
					PfIf_end(sc);
				}
				else {
					PfMov(sc, &sc->ud_copy[i], &sc->ud[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
				}

				PfSubgroupShuffleDownCyclic(sc, &sc->ud_copy[i], &sc->ud_copy[i], stride);
				PfSubgroupShuffleDownCyclic(sc, &sc->temp1, &sc->rd_copy[i], stride);

				if (i == (sc->registers_per_thread - 1)) {
					temp_int.data.i = stride;
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);

					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);

					appendRegistersToShared(sc, &sc->tempInt, &sc->rd[0]);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendRegistersToShared(sc, &sc->tempInt, &sc->ud[0]);
					PfIf_end(sc);

					appendBarrierPfSolve(sc);

					temp_int.data.i = sc->warpSize - stride;
					PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
					
					temp_int.data.i = sc->num_threads/sc->warpSize - 1;
					PfIf_lt_start(sc, &sc->warpID, &temp_int);

					temp_int.data.i = 1;
					PfAdd(sc, &sc->tempInt, &sc->warpID, &temp_int);
					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);

					temp_int.data.i = sc->warpSize - stride;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
					PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);

					appendSharedToRegisters(sc, &sc->temp1, &sc->tempInt);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendSharedToRegisters(sc, &sc->ud_copy[i], &sc->tempInt);

					PfIf_end(sc);

					PfIf_end(sc);

					appendBarrierPfSolve(sc);
				}

				temp_int.data.i = sc->warpSize * sc->registers_per_thread;
				PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
				temp_int.data.i = i * sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				temp_int.data.i = sc->M_size.data.i - stride;
				PfIf_ge_start(sc, &sc->inoutID, &temp_int);
				{
				//temp_int.data.i = stride - i * sc->warpSize;
				//if (temp_int.data.i > 0) {
					PfSetToZero(sc, &sc->ud_copy[i]);
					PfSetToZero(sc, &sc->temp);
				}
				PfIf_else(sc);
				{
					temp_int.data.i = sc->M_size.data.i - next_stride;
					PfIf_ge_start(sc, &sc->inoutID, &temp_int);
					PfSetToZero(sc, &sc->ud_copy[i]);
					PfIf_end(sc);
				}
				PfIf_end(sc);	
					
				PfMul(sc, &sc->ud_copy[i], &sc->ud_copy[i], &sc->temp, 0);
				PfMovNeg(sc, &sc->ud_copy[i], &sc->ud_copy[i]);
				PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp, 0);

				if (!sc->ld_zero) {
					PfSub(sc, &sc->md[i], &sc->md[i], &sc->temp2);
					//sc->tempLen = sprintf(sc->tempStr, "\n\
		md_%" PRIi64 " = md_%" PRIi64 " - temp_ac;\n", i, i);

				}
				PfSub(sc, &sc->rd_copy[i], &sc->rd[i], &sc->temp1);
			}
		}
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (!sc->ld_zero) {
				PfMov(sc, &sc->ld[i], &sc->ld_copy[i]);
			}
			if (!sc->ud_zero) {
				PfMov(sc, &sc->ud[i], &sc->ud_copy[i]);
			}
			PfMov(sc, &sc->rd[i], &sc->rd_copy[i]);
		}
		//appendBarrier(sc);
		

		stride = next_stride;
		
	}
	/*for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		PfPrintReg(sc, &sc->gl_LocalInvocationID_x, &sc->rd[i]);
	}*/
	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (i > 0) {
				temp_int.data.i = sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->num_threads / sc->warpSize + 1;
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

			appendRegistersToShared(sc, &sc->tempInt, &sc->rd[i]);
			PfIf_end(sc);
		}
		appendBarrierPfSolve(sc);

		temp_int.data.i = sc->num_threads / sc->warpSize;
		PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (i > 0) {
				temp_int.data.i = sc->num_threads;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->num_threads / sc->warpSize + 1;
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

			appendSharedToRegisters(sc, &sc->rd[i], &sc->tempInt);
			PfIf_end(sc);
		}
		appendBarrierPfSolve(sc);

		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (i > 0) {
				temp_int.data.i = sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->num_threads / sc->warpSize + 1;
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

			if (!sc->ld_zero) {
				appendRegistersToShared(sc, &sc->tempInt, &sc->ld[i]);
			}
			if (!sc->ud_zero) {
				appendRegistersToShared(sc, &sc->tempInt, &sc->ud[i]);
			}
			PfIf_end(sc);
		}
		appendBarrierPfSolve(sc);

		temp_int.data.i = sc->num_threads / sc->warpSize;
		PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (i > 0) {
				temp_int.data.i = sc->num_threads;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->num_threads / sc->warpSize + 1;
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

			if (!sc->ld_zero) {
				appendSharedToRegisters(sc, &sc->ld[i], &sc->tempInt);
			}
			if (!sc->ud_zero) {
				appendSharedToRegisters(sc, &sc->ud[i], &sc->tempInt);
			}
			PfIf_end(sc);
		}
		appendBarrierPfSolve(sc);
	}
	stride = 1;
	next_stride = 1;
	for (int64_t PCRiteration = 0; PCRiteration < maxPCRiteration; PCRiteration++) {
		next_stride = stride * 2;
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			//if (PCRiteration == 1) PfPrintReg(sc, &sc->gl_LocalInvocationID_x, &sc->rd[i]);
			/*if ((i == 0)) {
							sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f  %%f  %%f  %%f\\n\", inoutID, ld_%" PRIu64 ", md_%" PRIu64 ", ud_%" PRIu64 ", res_%" PRIu64 ");\n", i,i,i,i);
							PfAppendLine(sc);
							
						}*/
			//sc->tempLen = sprintf(sc->tempStr, "	{//if (acc_%" PRIi64 ">0){\n", i);
			//PfAppendLine(sc);
			//
			if (!sc->ld_zero) {
				if (i - stride / sc->warpSize >= 0)
				{
					PfMov(sc, &sc->temp, &sc->ld[i]);
				
					if ((stride < sc->warpSize)) {
						if ((sc->registers_per_thread > 1) && (i > 0)) {
							temp_int.data.i = sc->warpSize - stride;
							if(temp_int.data.i<=0){
								PfMov(sc, &sc->ld_copy[i], &sc->ld[i - 1]);
								PfMov(sc, &sc->rd_copy[i], &sc->rd[i - 1]);
							}else{
								if (temp_int.data.i < 0) 
									temp_int.data.i = 0;
								PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
								PfMov(sc, &sc->ld_copy[i], &sc->ld[i - 1]);
								PfMov(sc, &sc->rd_copy[i], &sc->rd[i - 1]);
								PfIf_else(sc);
								PfMov(sc, &sc->ld_copy[i], &sc->ld[i]);
								PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
								PfIf_end(sc);
							}
							PfSubgroupShuffleUpCyclic(sc, &sc->ld_copy[i], &sc->ld_copy[i], stride);

							PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->rd_copy[i], stride);

						}
						else {
							PfSubgroupShuffleUp(sc, &sc->ld_copy[i], &sc->ld[i], stride);

							PfSubgroupShuffleUp(sc, &sc->temp1, &sc->rd[i], stride);
						}
					}
					else {
						if ((sc->registers_per_thread > 1) && (i >= stride / sc->warpSize)) {
							PfMov(sc, &sc->ld_copy[i], &sc->ld[i - stride / sc->warpSize]);
							PfMov(sc, &sc->temp1, &sc->rd[i - stride / sc->warpSize]);
							
						}
					}
					//sc->tempLen = sprintf(sc->tempStr, "\n\
			if (id_x < %" PRIi64 ") {\n\
				ld_copy_%" PRIi64 " = 0;\n\
				temp_k = 0;\n\
				//temp_ac = 0;\n\
				//temp_d = 0;\n\
			}\n", stride - i * sc->num_threads, i);
				

					temp_int.data.i = stride - i * sc->num_threads;
					if (temp_int.data.i > 0) {
						if (temp_int.data.i >= sc->warpSize){
							PfSetToZero(sc, &sc->ld_copy[i]);
							PfSetToZero(sc, &sc->temp);
						}
						else {
							PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
							PfSetToZero(sc, &sc->ld_copy[i]);
							PfSetToZero(sc, &sc->temp);
							PfIf_else(sc);
							temp_int.data.i = next_stride - i * sc->num_threads;
							if (temp_int.data.i >= sc->warpSize){
								PfSetToZero(sc, &sc->ld_copy[i]);
							}else{
								if (temp_int.data.i > 0) {
									PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
									PfSetToZero(sc, &sc->ld_copy[i]);
									if (!sc->ud_zero) {
										//sc->tempLen = sprintf(sc->tempStr, "\n\
								temp_ac = sdata[id_x + %" PRIi64 "] * temp_k;\n", sc->offset_ud - stride + i * sc->num_threads);

									}
									PfIf_end(sc);
								}
							}
							PfIf_end(sc);
						}
					}
					else {
						temp_int.data.i = next_stride - i * sc->num_threads;
						if (temp_int.data.i > 0) {
							if (temp_int.data.i >= sc->warpSize) {
								PfSetToZero(sc, &sc->ld_copy[i]);
							}
							else{
								if (temp_int.data.i > 0) {
									PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
									PfSetToZero(sc, &sc->ld_copy[i]);
									if (!sc->ud_zero) {
										//sc->tempLen = sprintf(sc->tempStr, "\n\
								temp_ac = sdata[id_x + %" PRIi64 "] * temp_k;\n", sc->offset_ud - stride + i * sc->num_threads);

									}
									PfIf_end(sc);
								}
							}
						}
					}
					//sc->tempLen = sprintf(sc->tempStr, "\n\
			else if (id_x < %" PRIi64 ") {\n\
				//temp_k = ld_%" PRIi64 ";// / sdata[id_x + %" PRIi64 "];\n\
				ld_copy_%" PRIi64 " = 0;\n\
				//temp_d = sdata[id_x + %" PRIi64 "] * temp_k;\n", next_stride - i * sc->num_threads, i, sc->offset_md - stride + i * sc->num_threads, i, sc->offset_res - stride + i * sc->num_threads);
					

					/*sc->tempLen = sprintf(sc->tempStr, "\n\
			else {\n\
				temp_k = ld_%" PRIi64 ";// / sdata[id_x + %" PRIi64 "];\n\
				ld_%" PRIi64 " = -sdata[id_x + %" PRIi64 "] * temp_k;\n\
				temp_d = sdata[id_x + %" PRIi64 "] * temp_k;\n", i, sc->offset_md - stride + i * sc->num_threads, i, sc->offset_ld - stride + i * sc->num_threads, sc->offset_res - stride + i * sc->num_threads);
					PfAppendLine(sc);
					

					if (!sc->ud_zero) {
						sc->tempLen = sprintf(sc->tempStr, "\n\
				temp_ac = sdata[id_x + %" PRIi64 "] * temp_k;\n", sc->offset_ud - stride + i * sc->num_threads);
						PfAppendLine(sc);
						
					}

					sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
					PfAppendLine(sc);
					*/
					PfMul(sc, &sc->ld_copy[i], &sc->ld_copy[i], &sc->temp, 0);
					PfMovNeg(sc, &sc->ld_copy[i], &sc->ld_copy[i]);
					PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp, 0);
					
					if (!sc->ud_zero) {
						PfSub(sc, &sc->md[i], &sc->md[i], &sc->temp2);
						//sc->tempLen = sprintf(sc->tempStr, "\n\
			md_%" PRIi64 " = md_%" PRIi64 " - temp_ac;\n", i, i);
						
					}
					PfSub(sc, &sc->rd_copy[i], &sc->rd[i], &sc->temp1);
					//sc->tempLen = sprintf(sc->tempStr, "\n\
			res_copy_%" PRIi64 " = res_%" PRIi64 " - temp_d;\n", i, i);
					
				}
				//sc->tempLen = sprintf(sc->tempStr, "	}\n");
				//PfAppendLine(sc);
				//
			}

			if (!sc->ud_zero) {
				if (i + stride / sc->warpSize < sc->registers_per_thread)
				{
					PfMov(sc, &sc->temp, &sc->ud[i]);

					if ((stride < sc->warpSize)) {
						if ((sc->registers_per_thread > 1) && (i < sc->registers_per_thread - 1)) {
							temp_int.data.i = stride;
							PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
							PfMov(sc, &sc->ud_copy[i], &sc->ud[i + 1]);
							PfMov(sc, &sc->rd_copy[i], &sc->rd[i + 1]);
							PfIf_else(sc);
							PfMov(sc, &sc->ud_copy[i], &sc->ud[i]);
							PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
							PfIf_end(sc);

							PfSubgroupShuffleDownCyclic(sc, &sc->ud_copy[i], &sc->ud_copy[i], stride);

							PfSubgroupShuffleDownCyclic(sc, &sc->temp1, &sc->rd_copy[i], stride);
						}
						else {
							PfSubgroupShuffleDown(sc, &sc->ud_copy[i], &sc->ud[i], stride);

							PfSubgroupShuffleDown(sc, &sc->temp1, &sc->rd[i], stride);

						}
					}
					else {
						if ((sc->registers_per_thread > 1) && (i < sc->registers_per_thread - stride / sc->warpSize)) {
							PfMov(sc, &sc->ud_copy[i], &sc->ud[i + stride / sc->warpSize]);
							PfMov(sc, &sc->temp1, &sc->rd[i + stride / sc->warpSize]);
						}
					}


					//sc->tempLen = sprintf(sc->tempStr, "\n\
			if (id_x >= %" PRIi64 ") {\n\
				ud_copy_%" PRIi64 " = 0;\n\
				temp_k = 0;\n\
				//temp_ac = 0;\n\
				//temp_d = 0;\n\
			}\n", sc->M_size.x_num - stride - i * sc->num_threads, i);
					
					temp_int.data.i = sc->M_size.data.i - stride - i * sc->num_threads;
					if (sc->warpSize > temp_int.data.i) {
						if (temp_int.data.i > 0) {
							PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
							PfSetToZero(sc, &sc->ud_copy[i]);
							PfSetToZero(sc, &sc->temp);
							temp_int.data.i = sc->M_size.data.i - next_stride - i * sc->num_threads;
							if (sc->warpSize > temp_int.data.i) {
								PfIf_else(sc);
								if (temp_int.data.i > 0) {
									PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
									PfSetToZero(sc, &sc->ud_copy[i]);
									PfIf_end(sc);
								}else{
									PfSetToZero(sc, &sc->ud_copy[i]);
								}
							}
							PfIf_end(sc);
						}
						else {
							PfSetToZero(sc, &sc->ud_copy[i]);
							PfSetToZero(sc, &sc->temp);
						}
					}
					/*if (sc->M_size.mode) {
						sc->tempLen = sprintf(sc->tempStr, "\n\
			else if (id_x >= %s - %" PRIi64 ") {\n\
				//temp_k = ud_%" PRIi64 ";// / sdata[id_x + %" PRIi64 "];\n\
				ud_copy_%" PRIi64 " = 0;\n\
				//temp_d = sdata[id_x + %" PRIi64 "] * temp_k;\n", sc->M_size.x_str, next_stride + i * sc->num_threads, i, sc->offset_md + stride + i * sc->num_threads, i, sc->offset_res + stride + i * sc->num_threads);
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "\n\
			else if (id_x >= %" PRIi64 ") {\n\
				//temp_k = ud_%" PRIi64 ";// / sdata[id_x + %" PRIi64 "];\n\
				ud_copy_%" PRIi64 " = 0;\n\
				//temp_d = sdata[id_x + %" PRIi64 "] * temp_k;\n", sc->M_size.x_num - next_stride - i * sc->num_threads, i, sc->offset_md + stride + i * sc->num_threads, i, sc->offset_res + stride + i * sc->num_threads);
					}
					PfAppendLine(sc);
					

					if (!sc->ld_zero) {
						sc->tempLen = sprintf(sc->tempStr, "\n\
				temp_ac = sdata[id_x + %" PRIi64 "] * temp_k;\n", sc->offset_ld + stride + i * sc->num_threads);
						PfAppendLine(sc);
						
					}
					*/
					
					
					PfMul(sc, &sc->ud_copy[i], &sc->ud_copy[i], &sc->temp, 0);
					PfMovNeg(sc, &sc->ud_copy[i], &sc->ud_copy[i]);
					PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp, 0);

					if (!sc->ld_zero) {
						PfSub(sc, &sc->md[i], &sc->md[i], &sc->temp2);
						//sc->tempLen = sprintf(sc->tempStr, "\n\
			md_%" PRIi64 " = md_%" PRIi64 " - temp_ac;\n", i, i);

					}
					PfSub(sc, &sc->rd_copy[i], &sc->rd[i], &sc->temp1);

				}
				//sc->tempLen = sprintf(sc->tempStr, "	}\n");
				//PfAppendLine(sc);
				//
			}
		}
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (!sc->ld_zero) {
				PfMov(sc, &sc->ld[i], &sc->ld_copy[i]);
			}
			if (!sc->ud_zero) {
				PfMov(sc, &sc->ud[i], &sc->ud_copy[i]);
			}
			PfMov(sc, &sc->rd[i], &sc->rd_copy[i]);
		}
		//appendBarrier(sc);
		

		stride = next_stride;
		/*sc->tempLen = sprintf(sc->tempStr, "		printf(\" %%f %%f %%f %%f\\n\\n\", sdata[id_x], sdata[id_x+4], sdata[id_x+8], sdata[id_x+12]);\n");
		PfAppendLine(sc);
		*/
		//if (PCRiteration==maxPCRiteration-1)appendBarrier(sc);
		
	}

	for (int64_t BSiteration = 1; BSiteration < maxBSiteration; BSiteration++) {
		if (!sc->ld_zero) {
			PfMul(sc, &sc->temp, &sc->ld[BSiteration], &sc->rd[BSiteration-1], 0);
			PfSub(sc, &sc->rd[BSiteration], &sc->rd[BSiteration], &sc->temp);
		}
		if (!sc->ud_zero) {
			PfMul(sc, &sc->temp, &sc->ud[maxBSiteration-BSiteration- 1], &sc->rd[maxBSiteration-BSiteration], 0);
			PfSub(sc, &sc->rd[maxBSiteration-BSiteration -1], &sc->rd[maxBSiteration-BSiteration -1], &sc->temp);
		}
	}

	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->num_threads / sc->warpSize;
		PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (i > 0) {
				temp_int.data.i = sc->num_threads;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->num_threads / sc->warpSize + 1;
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

			appendRegistersToShared(sc, &sc->tempInt, &sc->rd[i]);
			PfIf_end(sc);
		}
		appendBarrierPfSolve(sc);

		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (i > 0) {
				temp_int.data.i = sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
			}
			PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
			PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
			temp_int.data.i = sc->num_threads / sc->warpSize + 1;
			PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
			PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

			appendSharedToRegisters(sc, &sc->rd[i], &sc->tempInt);
			PfIf_end(sc);
		}
		appendBarrierPfSolve(sc);
	}

	if ((sc->scaleC.data.d != 1.0) || (sc->scaleC.type > 100)) {
		temp_int.data.i = 0;
		PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
		//sc->tempLen = sprintf(sc->tempStr, "	if (%s == 0 ) res_%" PRIu64 " *= %s%s;\n", sc->gl_LocalInvocationID_x, 0, sc->scaleC.x_str, sc->LFending);
		PfMul(sc, &sc->rd[0], &sc->rd[0], &sc->scaleC, 0);
		PfIf_end(sc);
	}
	/*for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		//sc->tempLen = sprintf(sc->tempStr, "	if (acc_%" PRIi64 ">0){\n", i);

		//PfAppendLine(sc);
		//
		sc->tempLen = sprintf(sc->tempStr, "		acc_%" PRIi64 " = 0;\n", i);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "		if (id_x>=%" PRIi64 ") acc_%" PRIi64 "++;\n", 2 * stride - i*sc->num_threads, i);
		//PfAppendLine(sc);
		//
		sc->tempLen = sprintf(sc->tempStr, "		if (id_x>=%" PRIi64 ") acc_%" PRIi64 "++;\n", stride - i * sc->num_threads, i);
		PfAppendLine(sc);
		
		sc->tempLen = sprintf(sc->tempStr, "		if (id_x < %" PRIi64 ") acc_%" PRIi64 "++;\n", sc->M_size - stride - i * sc->num_threads, i);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "		if (id_x < %" PRIi64 ") acc_%" PRIi64 "++;\n", sc->M_size-2 * stride - i*sc->num_threads, i);
		//PfAppendLine(sc);
		//

		//if (PCRiteration == maxPCRiteration - 1)appendRegistersToShared_mat_lu(sc, i);
		
		//if (PCRiteration == maxPCRiteration - 1)appendRegistersToShared_res(sc, i);
		

		//sc->tempLen = sprintf(sc->tempStr, "	}\n");
		//PfAppendLine(sc);
		//
	}*/
	for (uint64_t i = 0; i < 0*sc->registers_per_thread; i++) {

			/*if ((i == 0)) {
										sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f  %%f  %%f  %%f\\n\", inoutID, ld_%" PRIu64 ", md_%" PRIu64 ", ud_%" PRIu64 ", res_%" PRIu64 ");\n", i,i,i,i);
										PfAppendLine(sc);
										
									}*/
		if (!sc->ud_zero) {
			if (i + (stride / sc->warpSize) < sc->registers_per_thread) {
				PfSubgroupShuffleDown(sc, &sc->temp, &sc->rd[i + (stride / sc->warpSize)], stride - (stride / sc->warpSize) * sc->warpSize);
			}
			if (i + (stride / 2 / sc->warpSize) < sc->registers_per_thread) {
				PfSubgroupShuffleDown(sc, &sc->temp1, &sc->rd[i + (stride / 2 / sc->warpSize)], stride / 2 - (stride / 2 / sc->warpSize) * sc->warpSize);
			}
		}
		if (!sc->ld_zero) {
			if (i >= (stride / sc->warpSize)) {
				PfSubgroupShuffleUp(sc, &sc->temp, &sc->rd[i - (stride / sc->warpSize)], stride - (stride / sc->warpSize) * sc->warpSize);
			}
			if (i >= (stride / 2 / sc->warpSize)) {
				//PfSubgroupShuffleUp(sc, &sc->temp1, &sc->rd[i + (stride / 2 / sc->warpSize)], stride / 2 - (stride / 2 / sc->warpSize) * sc->warpSize);
			}
		}
		//sc->tempLen = sprintf(sc->tempStr, "	if (acc_%" PRIi64 "==0){\n",i);
		////sc->tempLen = sprintf(sc->tempStr, "	if ((id_x<%" PRIi64 ") && (id_x>=%" PRIi64 ")){\n", stride - i * sc->num_threads, sc->M_size.x_num - stride - i * sc->num_threads);
		temp_int.data.i = stride - i * sc->num_threads;
		if (temp_int.data.i < 0) 
			temp_int.data.i = 0;
		PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);

		temp_int.data.i = sc->M_size.data.i - stride - i * sc->num_threads;
		if (temp_int.data.i < 0) 
			temp_int.data.i = 0;
		PfIf_ge_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);

		PfMov(sc, &sc->rd[i], &sc->rd_copy[i]);
		//sc->tempLen = sprintf(sc->tempStr, "		res_%" PRIi64 " = res_%" PRIi64 ";// / md_%" PRIi64 ";\n", i, i, i);
		//PfAppendLine(sc);
		
		PfIf_else(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "	else if (id_x < %" PRIi64 ") {\n", stride - i*sc->num_threads);
		//PfAppendLine(sc);
		
		if (!sc->ud_zero) {
			PfMul(sc, &sc->temp1, &sc->temp1, &sc->ud[i], 0);
			PfSub(sc, &sc->rd[i], &sc->rd[i], &sc->temp1);
		}
		else {
		}
		PfIf_end(sc);
		PfIf_else(sc);
		if (!sc->ld_zero){
			//sc->tempLen = sprintf(sc->tempStr, "	ld_%" PRIi64 " = sdata[id_x + %" PRIi64 "];\n", i , sc->offset_ld + i*sc->num_threads);
			//PfAppendLine(sc);
			
		}
		if (!sc->ld_zero){
			if (!sc->ud_zero){
				//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIi64 " = (res_%" PRIi64 " - sdata[id_x+%" PRIi64 "] * ld_%" PRIi64 ") / (1.0%s - sdata[id_x+%" PRIi64 "] * ld_%" PRIi64 ");\n", i, i, sc->offset_res - stride + i*sc->num_threads, i, sc->LFending, sc->offset_ud - stride + i*sc->num_threads, i);
				//PfAppendLine(sc);
				
			}else{
				//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIi64 " = (res_%" PRIi64 " - temp_k * ld_%" PRIi64 ") ;\n", i, i,  i);
				PfMul(sc, &sc->temp, &sc->temp, &sc->ld[i], 0);
				PfSub(sc, &sc->rd[i], &sc->rd[i], &sc->temp);
			}
		}else{
			//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIi64 " = (res_%" PRIi64 ") ;\n", i, i);
		}
		PfIf_end(sc);
		//if ((sc->scaleC)&&(i==0)){
		if ((sc->scaleC.data.d != 1.0) || (sc->scaleC.type>100)) {
			temp_int.data.i = 0;
			PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			//sc->tempLen = sprintf(sc->tempStr, "	if (%s == 0 ) res_%" PRIu64 " *= %s%s;\n", sc->gl_LocalInvocationID_x, 0, sc->scaleC.x_str, sc->LFending);
			PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->scaleC, 0);
			PfIf_end(sc);
		}
		//}
	}
	
	/*if (sc->write_RegistersToShared) {
		appendBarrier(sc);
		
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			appendRegistersToShared_res(sc, i);
			
		}
	}	
	if (sc->write_SharedToRegisters) {
		appendSharedToRegisters(sc);
		
	}*/
	return;
}

static inline void appendTridiagonalSolve_ParallelThomas(PfSolveSpecializationConstantsLayout* sc) {
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
		
		appendBarrier(sc);
		
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			appendRegistersToShared_mat_lu(sc, i);
			
		}
		appendBarrier(sc);
		
	}
	if (sc->read_RegistersToShared) {
		appendRegistersToShared_mat(sc);
		
		appendBarrier(sc);
		
	}*/
	//int64_t maxPCRiteration = (int64_t)ceil(log2(sc->M_size_pow2.data.i));
	//int64_t BSsystemSize = 0;// sc->M_size_pow2.data.i / sc->warpSize;
	int64_t maxSharedMemPCRIteration =  (int64_t)ceil(log2(sc->num_threads / sc->warpSize));
	int64_t stride = 1;
	int64_t next_stride = 1;
	int64_t shared_stride = (sc->num_threads == sc->warpSize) ? 2*(sc->registers_per_thread / 2) + 1 : sc->num_threads / sc->warpSize * sc->registers_per_thread + 1;

	for (int64_t sharedMemPCRIteration = 0; sharedMemPCRIteration < maxSharedMemPCRIteration; sharedMemPCRIteration++) {
		next_stride = stride * 2;
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*if ((i == 0)) {
							sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f  %%f  %%f  %%f\\n\", inoutID, ld_%" PRIu64 ", md_%" PRIu64 ", ud_%" PRIu64 ", res_%" PRIu64 ");\n", i,i,i,i);
							PfAppendLine(sc);
							
						}*/
			//sc->tempLen = sprintf(sc->tempStr, "	{//if (acc_%" PRIi64 ">0){\n", i);
			//PfAppendLine(sc);
			//
			if (!sc->ld_zero) {
				PfMov(sc, &sc->temp, &sc->ld[i]);

				if (i > 0) {
					temp_int.data.i = sc->warpSize - stride;
					PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
					PfMov(sc, &sc->ld_copy[i], &sc->ld[i - 1]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i - 1]);
					PfIf_else(sc);
					PfMov(sc, &sc->ld_copy[i], &sc->ld[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
					PfIf_end(sc);
				}
				else {
					PfMov(sc, &sc->ld_copy[i], &sc->ld[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
				}
				PfSubgroupShuffleUpCyclic(sc, &sc->ld_copy[i], &sc->ld_copy[i], stride);

				PfSubgroupShuffleUpCyclic(sc, &sc->temp1, &sc->rd_copy[i], stride);
					
				if (i == 0) {
					temp_int.data.i = sc->warpSize - stride;
					PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);

					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
					
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
                                        temp_int.data.i = sc->warpSize - stride;
					PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);

					appendRegistersToShared(sc, &sc->tempInt, &sc->rd[sc->registers_per_thread - 1]);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendRegistersToShared(sc, &sc->tempInt, &sc->ld[sc->registers_per_thread - 1]);
					PfIf_end(sc);

					appendBarrierPfSolve(sc);

					temp_int.data.i = stride;
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
					
					temp_int.data.i = 0;
					PfIf_gt_start(sc, &sc->warpID, &temp_int);

					temp_int.data.i = 1;
					PfSub(sc, &sc->tempInt, &sc->warpID, &temp_int);
					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);

					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);

					appendSharedToRegisters(sc, &sc->temp1, &sc->tempInt);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendSharedToRegisters(sc, &sc->ld_copy[i], &sc->tempInt);
					PfIf_end(sc);

					PfIf_end(sc);

					appendBarrierPfSolve(sc);
				}
				temp_int.data.i = sc->warpSize * sc->registers_per_thread;
				PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
				temp_int.data.i = i * sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				temp_int.data.i = stride;
				PfIf_lt_start(sc, &sc->inoutID, &temp_int);
				{
				//temp_int.data.i = stride - i * sc->warpSize;
				//if (temp_int.data.i > 0) {
					PfSetToZero(sc, &sc->ld_copy[i]);
					PfSetToZero(sc, &sc->temp);
				}
				PfIf_else(sc);
				{
					temp_int.data.i = next_stride;
					PfIf_lt_start(sc, &sc->inoutID, &temp_int);
					PfSetToZero(sc, &sc->ld_copy[i]);
					PfIf_end(sc);
				}
				PfIf_end(sc);

				PfMul(sc, &sc->ld_copy[i], &sc->ld_copy[i], &sc->temp, 0);
				PfMovNeg(sc, &sc->ld_copy[i], &sc->ld_copy[i]);
				PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp, 0);
					
				if (!sc->ud_zero) {
					PfSub(sc, &sc->md[i], &sc->md[i], &sc->temp2);
					//sc->tempLen = sprintf(sc->tempStr, "\n\
		md_%" PRIi64 " = md_%" PRIi64 " - temp_ac;\n", i, i);
						
				}
				PfSub(sc, &sc->rd_copy[i], &sc->rd[i], &sc->temp1);
			}

			if (!sc->ud_zero) {
				PfMov(sc, &sc->temp, &sc->ud[i]);

				if (i < (sc->registers_per_thread - 1)) {
					temp_int.data.i = stride;
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
					PfMov(sc, &sc->ud_copy[i], &sc->ud[i + 1]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i + 1]);
					PfIf_else(sc);
					PfMov(sc, &sc->ud_copy[i], &sc->ud[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
					PfIf_end(sc);
				}
				else {
					PfMov(sc, &sc->ud_copy[i], &sc->ud[i]);
					PfMov(sc, &sc->rd_copy[i], &sc->rd[i]);
				}

				PfSubgroupShuffleDownCyclic(sc, &sc->ud_copy[i], &sc->ud_copy[i], stride);
				PfSubgroupShuffleDownCyclic(sc, &sc->temp1, &sc->rd_copy[i], stride);

				if (i == (sc->registers_per_thread - 1)) {
					temp_int.data.i = stride;
					PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);

					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);

					appendRegistersToShared(sc, &sc->tempInt, &sc->rd[0]);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendRegistersToShared(sc, &sc->tempInt, &sc->ud[0]);
					PfIf_end(sc);

					appendBarrierPfSolve(sc);

					temp_int.data.i = sc->warpSize - stride;
					PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
					
					temp_int.data.i = sc->num_threads/sc->warpSize - 1;
					PfIf_lt_start(sc, &sc->warpID, &temp_int);

					temp_int.data.i = 1;
					PfAdd(sc, &sc->tempInt, &sc->warpID, &temp_int);
					temp_int.data.i = stride;
					PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);

					temp_int.data.i = sc->warpSize - stride;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
					PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);

					appendSharedToRegisters(sc, &sc->temp1, &sc->tempInt);
					temp_int.data.i = stride * sc->num_threads / sc->warpSize;
					PfAdd(sc, &sc->tempInt, &sc->tempInt, &temp_int);
					appendSharedToRegisters(sc, &sc->ud_copy[i], &sc->tempInt);

					PfIf_end(sc);

					PfIf_end(sc);

					appendBarrierPfSolve(sc);
				}

				temp_int.data.i = sc->warpSize * sc->registers_per_thread;
				PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
				PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
				temp_int.data.i = i * sc->warpSize;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				temp_int.data.i = sc->M_size.data.i - stride;
				PfIf_ge_start(sc, &sc->inoutID, &temp_int);
				{
				//temp_int.data.i = stride - i * sc->warpSize;
				//if (temp_int.data.i > 0) {
					PfSetToZero(sc, &sc->ud_copy[i]);
					PfSetToZero(sc, &sc->temp);
				}
				PfIf_else(sc);
				{
					temp_int.data.i = sc->M_size.data.i - next_stride;
					PfIf_ge_start(sc, &sc->inoutID, &temp_int);
					PfSetToZero(sc, &sc->ud_copy[i]);
					PfIf_end(sc);
				}
				PfIf_end(sc);	
					
				PfMul(sc, &sc->ud_copy[i], &sc->ud_copy[i], &sc->temp, 0);
				PfMovNeg(sc, &sc->ud_copy[i], &sc->ud_copy[i]);
				PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp, 0);

				if (!sc->ld_zero) {
					PfSub(sc, &sc->md[i], &sc->md[i], &sc->temp2);
					//sc->tempLen = sprintf(sc->tempStr, "\n\
		md_%" PRIi64 " = md_%" PRIi64 " - temp_ac;\n", i, i);

				}
				PfSub(sc, &sc->rd_copy[i], &sc->rd[i], &sc->temp1);
			}
		}
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			if (!sc->ld_zero) {
				PfMov(sc, &sc->ld[i], &sc->ld_copy[i]);
			}
			if (!sc->ud_zero) {
				PfMov(sc, &sc->ud[i], &sc->ud_copy[i]);
			}
			PfMov(sc, &sc->rd[i], &sc->rd_copy[i]);
		}
		//appendBarrier(sc);
		

		stride = next_stride;
		
	}
	
	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
	}
	else
		PfMov(sc, &sc->inoutID, &sc->warpInvocationID);
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (i > 0) {
			temp_int.data.i = sc->warpSize;
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		}
		PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
		temp_int.data.i = shared_stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

		appendRegistersToShared(sc, &sc->tempInt, &sc->rd[i]);
		PfIf_end(sc);
	}
	appendBarrierPfSolve(sc);

	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);
	}
	else {
		temp_int.data.i = sc->registers_per_thread;
		PfMul(sc, &sc->inoutID, &sc->warpInvocationID, &temp_int, 0);
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (i > 0) {
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		}
		PfSetToZero(sc, &sc->rd[i]);
		PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
		temp_int.data.i = shared_stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

		appendSharedToRegisters(sc, &sc->rd[i], &sc->tempInt);
		PfIf_end(sc);
	}
	appendBarrierPfSolve(sc);

	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
	}
	else
		PfMov(sc, &sc->inoutID, &sc->warpInvocationID);
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (i > 0) {
			temp_int.data.i = sc->warpSize;
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		}
		PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
		temp_int.data.i = shared_stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

		if (!sc->ld_zero) {
			appendRegistersToShared(sc, &sc->tempInt, &sc->ld[i]);
		}
		if (!sc->ud_zero) {
			appendRegistersToShared(sc, &sc->tempInt, &sc->ud[i]);
		}
		PfIf_end(sc);
	}
	appendBarrierPfSolve(sc);

	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);
	}
	else {
		temp_int.data.i = sc->registers_per_thread;
		PfMul(sc, &sc->inoutID, &sc->warpInvocationID, &temp_int, 0);
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (i > 0) {
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		}
		PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
		temp_int.data.i = shared_stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

		if (!sc->ld_zero) {
			appendSharedToRegisters(sc, &sc->ld[i], &sc->tempInt);
		}
		if (!sc->ud_zero) {
			appendSharedToRegisters(sc, &sc->ud[i], &sc->tempInt);
		}
		PfIf_end(sc);
	}
	appendBarrierPfSolve(sc);
	for (uint64_t i = 1; i < sc->registers_per_thread; i++) {
		if (!sc->ld_zero) {
			PfMul(sc, &sc->temp, &sc->ld[i], &sc->rd[i-1], 0);
			PfSub(sc, &sc->rd[i], &sc->rd[i], &sc->temp);

			PfMul(sc, &sc->temp, &sc->ld[i], &sc->ld[i-1], 0);
			PfMovNeg(sc, &sc->ld[i], &sc->temp);
		}
		if (!sc->ud_zero) {
			PfMul(sc, &sc->temp, &sc->ud[sc->registers_per_thread - i - 1], &sc->rd[sc->registers_per_thread - i], 0);
			PfSub(sc, &sc->rd[sc->registers_per_thread - i - 1], &sc->rd[sc->registers_per_thread - i - 1], &sc->temp);

			PfMul(sc, &sc->ud[sc->registers_per_thread - i - 1], &sc->ud[sc->registers_per_thread - i - 1], &sc->ud[sc->registers_per_thread - i], 0);
			PfMovNeg(sc, &sc->ud[sc->registers_per_thread - i - 1], &sc->ud[sc->registers_per_thread - i - 1]);
		}
	}
	if (!sc->ld_zero) {
		PfSwapContainers(sc, &sc->rd[0], &sc->rd[sc->registers_per_thread - 1]);
		PfSwapContainers(sc, &sc->ld[0], &sc->ld[sc->registers_per_thread - 1]);
	}

	int64_t temp_registers_per_thread = sc->registers_per_thread;
	int64_t temp_M_size = sc->M_size.data.i;
	int64_t temp_M_size_pow2 = sc->M_size_pow2.data.i;
	int64_t temp_scaleC_type = sc->scaleC.type;
	int64_t temp_num_threads = sc->num_threads;
	pfLD temp_scaleC_d = sc->scaleC.data.d;
	if (sc->scaleC.type < 100) sc->scaleC.data.d = pfFPinit("1.0");
	sc->scaleC.type = 32;
	sc->registers_per_thread = 1;
	sc->M_size.data.i = sc->warpSize;
	sc->M_size_pow2.data.i = sc->warpSize;
	sc->num_threads = sc->warpSize;

	appendTridiagonalSolve_PCR(sc);

	sc->registers_per_thread = temp_registers_per_thread;
	sc->M_size.data.i = temp_M_size;
	sc->M_size_pow2.data.i = temp_M_size_pow2;
	sc->num_threads = temp_num_threads;
	if (temp_scaleC_type < 100) sc->scaleC.data.d = temp_scaleC_d;
	sc->scaleC.type = temp_scaleC_type;

	if (!sc->ld_zero) {
		PfSwapContainers(sc, &sc->rd[0], &sc->rd[sc->registers_per_thread - 1]);
		PfSwapContainers(sc, &sc->ld[0], &sc->ld[sc->registers_per_thread - 1]);
	}

	if (!sc->ld_zero) {
		PfSubgroupShuffleUp(sc, &sc->temp, &sc->rd[sc->registers_per_thread - 1], 1);
	}

	if (!sc->ud_zero) {
		PfSubgroupShuffleDown(sc, &sc->temp, &sc->rd[0], 1);
	}

	if (!sc->ld_zero) {
		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->warpInvocationID, &temp_int);
	}
	if (!sc->ud_zero) {
		temp_int.data.i = sc->warpSize - 1;
		PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
	}
	for (uint64_t i = 1; i < sc->registers_per_thread; i++) {
		if (!sc->ld_zero) {
			PfMul(sc, &sc->temp2, &sc->ld[i-1], &sc->temp, 0);
			PfSub(sc, &sc->rd[i-1], &sc->rd[i-1], &sc->temp2);
		}
		if (!sc->ud_zero) {
			PfMul(sc, &sc->temp2, &sc->ud[sc->registers_per_thread - i], &sc->temp, 0);
			PfSub(sc, &sc->rd[sc->registers_per_thread - i], &sc->rd[sc->registers_per_thread - i], &sc->temp2);
		}
	}
	if (!sc->ld_zero) {
		PfIf_end(sc);
	}
	if (!sc->ud_zero) {
		PfIf_end(sc);
	}
	/*temp_int.data.i = sc->warpSize * sc->registers_per_thread;
	PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
	PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);*/
	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);
	}
	else {
		temp_int.data.i = sc->registers_per_thread;
		PfMul(sc, &sc->inoutID, &sc->warpInvocationID, &temp_int, 0);
	}
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (i > 0) {
			temp_int.data.i = sc->num_threads / sc->warpSize;
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		}
		PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
		temp_int.data.i = shared_stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

		appendRegistersToShared(sc, &sc->tempInt, &sc->rd[i]);
		PfIf_end(sc);
	}
	appendBarrierPfSolve(sc);

	/*temp_int.data.i = sc->num_threads / sc->warpSize;
	PfMul(sc, &sc->tempInt, &sc->warpInvocationID, &temp_int, 0);
	PfAdd(sc, &sc->inoutID, &sc->warpID, &sc->tempInt);*/
	if (maxSharedMemPCRIteration > 0) {
		temp_int.data.i = sc->warpSize * sc->registers_per_thread;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->inoutID, &sc->warpInvocationID, &sc->tempInt);
	}
	else
		PfMov(sc, &sc->inoutID, &sc->warpInvocationID);

	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		if (i > 0) {
			temp_int.data.i = sc->warpSize;
			PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
		}
		PfIf_lt_start(sc, &sc->inoutID, &sc->M_size);
		temp_int.data.i = sc->num_threads / sc->warpSize * sc->registers_per_thread;
		PfDiv(sc, &sc->tempInt, &sc->inoutID, &temp_int);
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &temp_int);
		temp_int.data.i = shared_stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_x);

		appendSharedToRegisters(sc, &sc->rd[i], &sc->tempInt);
		PfIf_end(sc);
	}
	appendBarrierPfSolve(sc);

	if ((sc->scaleC.data.d != 1.0) || (sc->scaleC.type > 100)) {
		temp_int.data.i = 0;
		PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
		//sc->tempLen = sprintf(sc->tempStr, "	if (%s == 0 ) res_%" PRIu64 " *= %s%s;\n", sc->gl_LocalInvocationID_x, 0, sc->scaleC.x_str, sc->LFending);
		PfMul(sc, &sc->rd[0], &sc->rd[0], &sc->scaleC, 0);
		PfIf_end(sc);
	}
	return;
}

/*static inline PfSolveResult appendBidiagonalSolve_lt(PfSolveSpecializationConstantsLayout* sc) {
	PfSolveResult PFSOLVE_SUCCESS;
	
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {

		if (sc->start_shift_x + i < sc->M_size.x_num){
			if (i>0){
				sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = (res_%" PRIu64 " - %s%s[%" PRIu64 "]%s * res_%" PRIu64 ") / %s%s[%" PRIu64 "]%s;\n", i, i, sc->convTypeLeftInput, sc->inputsStruct, sc->start_shift_x + i-1+sc->offset_ld_global, sc->convTypeRightInput, i-1, sc->convTypeLeftInput, sc->inputsStruct, sc->start_shift_x + i+sc->offset_md_global, sc->convTypeRightInput);
				PfAppendLine(sc);
				
				if (i == sc->registers_per_thread-1) {
					sc->tempLen = sprintf(sc->tempStr, "	temp0 = res_%" PRIu64 ";\n", i);
					PfAppendLine(sc);
					
				}
			}
			else{
				if (sc->start_shift_x > 0) {
					sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = (res_%" PRIu64 " - %s%s[%" PRIu64 "]%s * temp0) / %s%s[%" PRIu64 "]%s;\n", i, i, sc->convTypeLeftInput, sc->inputsStruct, sc->start_shift_x + i-1 + sc->offset_ld_global, sc->convTypeRightInput, sc->convTypeLeftInput, sc->inputsStruct, sc->start_shift_x + i + sc->offset_md_global, sc->convTypeRightInput);
					PfAppendLine(sc);
					
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = res_%" PRIu64 " / %s%s[%" PRIu64 "]%s;\n", i, i, sc->convTypeLeftInput, sc->inputsStruct, sc->start_shift_x + i + sc->offset_md_global, sc->convTypeRightInput);
					PfAppendLine(sc);
					
					if (sc->scaleC.mode==0){
						sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " *= %.17e%s;\n", 0, sc->scaleC.x_num, sc->LFending);
						PfAppendLine(sc);
						
					}
				}
			}
		}
	}
	
	return res;
}
static inline PfSolveResult appendBidiagonalSolve_ut(PfSolveSpecializationConstantsLayout* sc) {
	PfSolveResult PFSOLVE_SUCCESS;
	
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {

		uint64_t lower_band_pad = (sc->inputStride[1].x_num == sc->M_size.x_num -1) ? 1 : 0;
		if (sc->start_shift_x + i < sc->M_size.x_num){
			if (i>0){
				sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = (res_%" PRIu64 " - %s%s[%" PRIu64 "]%s * res_%" PRIu64 ") / %s%s[%" PRIu64 "]%s;\n", i, i, sc->convTypeLeftInput, sc->inputsStruct, sc->M_size.x_num -(sc->start_shift_x + i)+sc->offset_ud_global, sc->convTypeRightInput, i-1, sc->convTypeLeftInput, sc->inputsStruct,  sc->M_size.x_num -1-(sc->start_shift_x + i)+sc->offset_md_global, sc->convTypeRightInput);
				PfAppendLine(sc);
				
				if (i == sc->registers_per_thread - 1) {
					sc->tempLen = sprintf(sc->tempStr, "	temp0 = res_%" PRIu64 ";\n", i);
					PfAppendLine(sc);
					
				}
			}
			else{
				if (sc->start_shift_x > 0) {
					sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = (res_%" PRIu64 " - %s%s[%" PRIu64 "]%s * temp0) / %s%s[%" PRIu64 "]%s;\n", i, i, sc->convTypeLeftInput, sc->inputsStruct, sc->M_size.x_num - (sc->start_shift_x + i) + sc->offset_ud_global, sc->convTypeRightInput, sc->convTypeLeftInput, sc->inputsStruct, sc->M_size.x_num - 1 - (sc->start_shift_x + i) + sc->offset_md_global, sc->convTypeRightInput);
					PfAppendLine(sc);
					
				}
				else {
					if (lower_band_pad)
						sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = 0;\n", i);
					else
						sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = res_%" PRIu64 " / %s%s[%" PRIu64 "]%s;\n", i, i, sc->convTypeLeftInput, sc->inputsStruct, sc->M_size.x_num -1- (sc->start_shift_x + i) + sc->offset_md_global, sc->convTypeRightInput);
					PfAppendLine(sc);
					
				}
			}
			if ((sc->scaleC.x_num)&&((sc->start_shift_x + i + 1) == sc->M_size.x_num)){
					sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " *= %.17e%s;\n", i, sc->scaleC, sc->LFending);
					PfAppendLine(sc);
					
			}
		}

	}

	return res;
}*/
#endif
