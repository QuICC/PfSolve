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
#ifndef PFSOLVE_JONESWORLAND_H
#define PFSOLVE_JONESWORLAND_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_InputOutputLayout.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_SharedMemory.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_Registers.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_ReadWrite.h"

#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_MatVecMul.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_TridiagonalSolve.h"

#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelUtils.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelStartEnd.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"
static inline PfSolveResult PfSolve_shaderGen_JonesWorlandMV(PfSolveSpecializationConstantsLayout* sc) {
	appendVersion(sc);
	
	appendExtensions(sc);
	if ((((sc->floatTypeCode/10)%10) == 3) || (((sc->floatTypeInputMemoryCode/10)%10) == 3) || (((sc->floatTypeOutputMemoryCode/10)%10) == 3)) {
		appendQuadDoubleDoubleStruct(sc);
	}
	if ((sc->floatTypeCode != sc->floatTypeInputMemoryCode) || (sc->floatTypeCode != sc->floatTypeOutputMemoryCode)) {
		appendConversion(sc);
	}	
	int id = 0;
	appendInputLayoutPfSolve(sc, id, 0);
	id++;
	appendOutputLayoutPfSolve(sc, id, 0);

	appendPushConstants_jw(sc);
	
#if(VKFFT_BACKEND==0)
	//appendSharedMemoryInitialization(sc);
	
#endif
	appendKernelStart_jw(sc, 0);
	
#if(VKFFT_BACKEND!=0)
	//appendSharedMemoryInitialization(sc);
	
#endif
	appendRegistersInitialization_compute_JW(sc, 0);

	sc->offset_md.type = 31;
	sc->offset_md.data.i = 0;
	sc->offset_res.type = 31;
	sc->offset_res.data.i = 2 * sc->M_size.data.i;
	
	sc->offset_ld.type = 31;
	sc->offset_ud.type = 31;

	sc->offset_md_global.type = 31;
	sc->offset_ud_global.type = 31;
	sc->offset_ld_global.type = 31;

	if (sc->offsetSolution.type > 100) {
		sc->offset_res_global.type = sc->offsetSolution.type;
		PfAllocateContainerFlexible(sc, &sc->offset_res_global, 50);
		PfCopyContainer(sc, &sc->offset_res_global, &sc->offsetSolution);
	}
	else {
		sc->offset_res_global.type = 31;
		sc->offset_res_global.data.i = sc->offsetSolution.data.i;
	}

	sc->readToRegisters = 1;
	if ((sc->axis_upload_id != 0) && (sc->axis_upload_id != (sc->numAxisUploads-1))) {
		appendReadWrite_mat_rd_reduced(sc, 0);
		//PfPrintReg(sc, &sc->gl_GlobalInvocationID_x, &sc->ud[0]);
		appendTridiagonalSolve_ParallelThomas_sharedShuffleRead(sc, sc->ld);
		appendTridiagonalSolve_ParallelThomas_sharedShuffleRead(sc, sc->ud);
		appendTridiagonalSolve_ParallelThomas_sharedShuffleRead(sc, sc->rd);
		//PfPrintReg(sc, &sc->gl_GlobalInvocationID_x, &sc->ud[0]);
	}
	else {
		if (sc->useUncoalescedJWTnoSharedMemory)
			appendReadWrite_rd_uncoalesced(sc, 0);
		else {
			appendReadWrite_rd(sc, 0);
			if (sc->useParallelThomas)
				appendTridiagonalSolve_ParallelThomas_sharedShuffleRead(sc, sc->rd);
		}
		
	}
	for (int i = 0; i < sc->numConsecutiveJWIterations; i++) {
		if (i > 0) sc->M_size.data.i = (sc->upperBanded) ? sc->M_size.data.i - 1 : sc->M_size.data.i + 1;
		sc->M_size_pow2.data.i = (int64_t)pow(2, (int)ceil(log2((double)sc->M_size.data.i)));
		sc->inputBufferId = i;
		if (sc->upperBanded == 2) {
			sc->md_zero = 0;
			sc->ld_zero = 0;
			sc->ud_zero = 0;
			if (sc->numAxisUploads > 1) {
				sc->offset_ld_global.data.i = 0;// sc->offsetM.data.i;
				sc->offset_md_global.data.i = sc->size[0].data.i;;//  sc->offsetM.data.i + sc->M_size.data.i;
				sc->offset_ud_global.data.i = 2 * sc->size[0].data.i;// sc->offsetM.data.i;
			}
			else {
				sc->offset_ld_global.data.i = 0;// sc->offsetM.data.i;
				sc->offset_md_global.data.i = sc->M_size.data.i;;//  sc->offsetM.data.i + sc->M_size.data.i;
				sc->offset_ud_global.data.i = 2 * sc->M_size.data.i;// sc->offsetM.data.i;
			}
		}
		else {
			if (sc->performTriSolve == 2)
			{
				sc->upperBanded = !sc->upperBanded;
			}

			sc->md_zero = 0;
			if (sc->upperBanded) {
				sc->ud_zero = 0;
				sc->ld_zero = 1;

				sc->offset_ud.data.i = sc->M_size.data.i;

				if (sc->offsetM.type > 100) {
					sc->offset_md_global.type = 100 + sc->uintTypeCode;
					PfAllocateContainerFlexible(sc, &sc->offset_md_global, 50);
					sprintf(sc->offset_md_global.name, "offset_md_global");
					PfDefine(sc, &sc->offset_md_global, sc->offset_md_global.name);

					sc->offset_ud_global.type = 100 + sc->uintTypeCode;
					PfAllocateContainerFlexible(sc, &sc->offset_ud_global, 50);
					sprintf(sc->offset_ud_global.name, "offset_lud_global");
					PfDefine(sc, &sc->offset_ud_global, sc->offset_ud_global.name);

					PfAdd(sc, &sc->offset_md_global, &sc->offsetM, &sc->M_size);
					PfMov(sc, &sc->offset_ud_global, &sc->offsetM);
				}
				else {
					if (i == 0) {
						sc->offset_md_global.data.i = 0;// sc->offsetM.data.i;
						sc->offset_ud_global.data.i = sc->M_size.data.i;//  sc->offsetM.data.i + sc->M_size.data.i;
					}
					else {
						sc->offset_ud_global.data.i = 0;//  sc->offsetM.data.i + sc->M_size.data.i;
						sc->md_zero = 1;
					}
				}
			}
			else {
				sc->ud_zero = 1;
				sc->ld_zero = 0;
				sc->offset_ld.data.i = sc->M_size.data.i;

				if (sc->offsetM.type > 100) {
					sc->offset_md_global.type = 100 + sc->uintTypeCode;
					PfAllocateContainerFlexible(sc, &sc->offset_md_global, 50);
					sprintf(sc->offset_md_global.name, "offset_md_global");
					PfDefine(sc, &sc->offset_md_global, sc->offset_md_global.name);

					sc->offset_ld_global.type = 100 + sc->uintTypeCode;
					PfAllocateContainerFlexible(sc, &sc->offset_ld_global, 50);
					sprintf(sc->offset_ld_global.name, "offset_lud_global");
					PfDefine(sc, &sc->offset_ld_global, sc->offset_ld_global.name);

					PfAdd(sc, &sc->offset_ld_global, &sc->offsetM, &sc->M_size);
					PfMov(sc, &sc->offset_md_global, &sc->offsetM);
				}
				else {
					if (i == (sc->numConsecutiveJWIterations - 1)) {
						sc->offset_ld_global.data.i = 0;// sc->offsetM.data.i;
						sc->offset_md_global.data.i = sc->M_size.data.i;;//  sc->offsetM.data.i + sc->M_size.data.i;
					}
					else {
						sc->offset_ld_global.data.i = 0;// sc->offsetM.data.i;
						sc->md_zero = 1;
					}

				}

			}

			if (sc->performTriSolve == 2)
			{
				sc->upperBanded = !sc->upperBanded;
			}
		}
		if((sc->num_warps_data_parallel > 1) && (sc->sharedMatricesUpload)){
			appendGlobalToShared_mat_ParallelThomas(sc);
			appendBarrierPfSolve(sc); //this shouldn't be needed, but it is?
		}
		
		if (sc->performMatVecMul) {
			if (sc->useParallelThomas) {
				if ((sc->num_warps_data_parallel > 1)&& (sc->sharedMatricesUpload)) {
					appendSharedToRegisters_mat_ParallelThomas(sc);
				}
				else
					appendGlobalToRegisters_mat_ParallelThomas(sc);
			}
			else
				appendGlobalToRegisters_mat(sc);

			//sc->read_SharedToRegisters = 0;
			//sc->write_RegistersToShared=0;
			if (sc->performTriSolve == 2)
			{
				sc->ud_zero = 1;
				sc->ld_zero = 1;
			}
			if(sc->useParallelThomas)
				appendMatVecMul_ParallelThomas(sc);
			else
				appendMatVecMul(sc);

			//sc->read_SharedToRegisters = 0;
			//sc->write_RegistersToShared=0;
		}
		//appendBarrier(sc);
		if (sc->upperBanded) {
			if (sc->offsetM.type > 100) {
				PfDeallocateContainer(sc, &sc->offset_md_global);
				PfDeallocateContainer(sc, &sc->offset_ud_global);
			}
		}
		else {
			if (sc->offsetM.type > 100) {
				PfDeallocateContainer(sc, &sc->offset_md_global);
				PfDeallocateContainer(sc, &sc->offset_ld_global);
			}
		}

		if (sc->upperBanded != 2) {
			sc->offset_md_global.type = 31;
			sc->offset_ud_global.type = 31;
			sc->offset_ld_global.type = 31;
			sc->offset_md.data.i = 0;
			sc->offset_res.data.i = 2 * sc->M_size.data.i;
			//sc->offset_res_global.data.i = sc->offsetSolution.data.i;
			if (sc->upperBanded) {
				sc->ud_zero = 1;
				sc->ld_zero = 0;
				sc->offset_ld.data.i = sc->M_size.data.i;

				if (sc->offsetV.type > 100) {
					sc->offset_ld_global.type = 100 + sc->uintTypeCode;
					PfAllocateContainerFlexible(sc, &sc->offset_ld_global, 50);
					sprintf(sc->offset_ld_global.name, "offset_lud_global");
					if (sc->offsetM.type < 100)
						PfDefine(sc, &sc->offset_ld_global, sc->offset_ld_global.name);

					PfMov(sc, &sc->offset_ld_global, &sc->offsetV);
				}
				else {
					//sc->offset_ld_global.data.i = sc->offsetV.data.i + sc->M_size.data.i;
					if (i == 0)
						sc->offset_ld_global.data.i = 2 * sc->M_size.data.i;// sc->offsetV.data.i;
					else
						sc->offset_ld_global.data.i = sc->M_size.data.i;// sc->offsetV.data.i;
				}
			}
			else {
				sc->ud_zero = 0;
				sc->ld_zero = 1;
				sc->offset_ud.data.i = sc->M_size.data.i;

				if (sc->offsetV.type > 100) {
					/*PfAllocateContainerFlexible(sc, &sc->offset_md_global, 50);
					sc->offset_md_global.type = 100 + sc->uintTypeCode;
					sprintf(sc->offset_md_global.name, "offset_md_global");
					if (sc->offsetM.type < 100)
						PfDefine(sc, &sc->offset_md_global);*/

					sc->offset_ud_global.type = 100 + sc->uintTypeCode;
					PfAllocateContainerFlexible(sc, &sc->offset_ud_global, 50);
					sprintf(sc->offset_ud_global.name, "offset_lud_global");
					if (sc->offsetM.type < 100)
						PfDefine(sc, &sc->offset_ud_global, sc->offset_ud_global.name);

					//PfAdd(sc, &sc->offset_md_global, &sc->offsetV, &sc->M_size);
					PfMov(sc, &sc->offset_ud_global, &sc->offsetV);
				}
				else {
					//sc->offset_md_global.data.i = sc->offsetV.data.i + sc->M_size.data.i;
					if (i == (sc->numConsecutiveJWIterations - 1))
						sc->offset_ud_global.data.i = 2 * sc->M_size.data.i;// sc->offsetV.data.i;
					else
						sc->offset_ud_global.data.i = sc->M_size.data.i;// sc->offsetV.data.i;
				}
			}

			sc->md_zero = 1;
			if (sc->performTriSolve != 2)
			{
				if ((sc->num_warps_data_parallel > 1) && (sc->sharedMatricesUpload)) {
					appendSharedToRegisters_mat_ParallelThomas(sc);
				}
				else
					appendGlobalToRegisters_mat_ParallelThomas(sc);
			}

			if ((sc->num_warps_data_parallel > 1) && (sc->sharedMatricesUpload))
				appendBarrierPfSolve(sc);
		}
		else {
			sc->md_zero = 1;
			sc->ld_zero = 0;
			sc->ud_zero = 0;
		}
		if (sc->performTriSolve) {
			if(sc->useParallelThomas)
				appendTridiagonalSolve_ParallelThomas(sc);
			else
				appendTridiagonalSolve_PCR_Thomas(sc);
		}
		if (sc->upperBanded) {
			if (sc->offsetV.type > 100) {
				//PfDeallocateContainer(sc, &sc->offset_md_global);
				PfDeallocateContainer(sc, &sc->offset_ld_global);
			}
		}
		else {
			if (sc->offsetV.type > 100) {
				//PfDeallocateContainer(sc, &sc->offset_md_global);
				PfDeallocateContainer(sc, &sc->offset_ud_global);
			}
		}
	}

	//int i =0;
	//if((i==0)){
	//							sc->tempLen = sprintf(sc->tempStr, "	printf(\"%%d %%f  %%f  %%f  %%f\\n\", inoutID, ld_%" PRIu64 ", md_%" PRIu64 ", ud_%" PRIu64 ", res_%" PRIu64 ");\n", i,i,i,i);
	//							PfAppendLine(sc);
	//							
	//						}
	sc->writeFromRegisters = 1;
	if ((sc->numAxisUploads > 1) && (sc->axis_upload_id < (sc->numAxisUploads/2))) {
		appendWrite_mat_rd_for_reduced_kernel(sc, 1);
	}
	else if ((sc->numAxisUploads > 1) && (sc->axis_upload_id != (sc->numAxisUploads-1))) {
		appendTridiagonalSolve_ParallelThomas_sharedShuffleWrite(sc, sc->rd);
		
		sc->ld_zero = 1;
		sc->ud_zero = 1;
		appendReadWrite_mat_rd_reduced(sc, 1);

	}else {
		if (sc->useUncoalescedJWTnoSharedMemory)
			appendReadWrite_rd_uncoalesced(sc, 1);
		else {
			if (sc->useParallelThomas)
				appendTridiagonalSolve_ParallelThomas_sharedShuffleWrite(sc, sc->rd);
			appendReadWrite_rd(sc, 1);
		}
	}
	PfDeallocateContainer(sc, &sc->offset_res_global);

	
	appendKernelEnd(sc);
	freeRegistersInitialization_compute_JW(sc, 0);
	return sc->res;
	}
#endif
