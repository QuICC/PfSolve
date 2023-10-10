// This file is part of PfSolve
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This file is provided for informational purposes only. Redistribution without permission is not allowed.
#ifndef PFSOLVE_COMPUTEPF_H
#define PFSOLVE_COMPUTEPF_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryTransfers/pfSolve_Transfers.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"

static inline void appendGlobalToRegisters_compute_Pf(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		temp_int.data.i = (i % (sc->logicBlock[0].data.i / sc->warpSize)) * sc->localSize[0].data.i;
		PfMul(sc, &sc->inoutID_x, &sc->gl_WorkGroupID_x, &sc->logicBlock[0], 0);
		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->gl_LocalInvocationID_x);
		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &temp_int);

		temp_int.data.i = ((i / (sc->logicBlock[0].data.i / sc->warpSize)) % sc->logicBlock[1].data.i);
		PfMul(sc, &sc->inoutID_y, &sc->gl_WorkGroupID_y, &sc->logicBlock[1], 0);
		PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &temp_int);

		temp_int.data.i = ((i / (sc->logicBlock[0].data.i / sc->warpSize)) / sc->logicBlock[1].data.i);
		PfMul(sc, &sc->inoutID_z, &sc->gl_WorkGroupID_z, &sc->logicBlock[2], 0);
		PfAdd(sc, &sc->inoutID_z, &sc->inoutID_z, &temp_int);
		
		
		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
		}

		PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
		PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

		appendGlobalToRegisters(sc, &sc->regIDs_x[i], &sc->qDxStruct, &sc->inoutID);

		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}

		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			temp_int.data.i = sc->size[1].data.i;
			PfIf_lt_start(sc, &sc->inoutID_y, &temp_int);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
		}

		PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
		PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

		appendGlobalToRegisters(sc, &sc->regIDs_y[i], &sc->qDyStruct, &sc->inoutID);


		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
		}


		PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
		PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

		appendGlobalToRegisters(sc, &sc->regIDs_z[i], &sc->qDzStruct, &sc->inoutID);

		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendComputePf_2(PfSolveSpecializationConstantsLayout* sc) {
	
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	int64_t stride = 1;
	int64_t next_stride = 1;
	for (int64_t i = 0; i < sc->registers_per_thread; i++) {
		temp_int.data.i = (i % (sc->logicBlock[0].data.i / sc->warpSize)) * sc->localSize[0].data.i;
		PfMul(sc, &sc->inoutID_x, &sc->gl_WorkGroupID_x, &sc->logicBlock[0], 0);
		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->gl_LocalInvocationID_x);
		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &temp_int);

		temp_int.data.i = ((i / (sc->logicBlock[0].data.i / sc->warpSize)) % sc->logicBlock[1].data.i);
		PfMul(sc, &sc->inoutID_y, &sc->gl_WorkGroupID_y, &sc->logicBlock[1], 0);
		PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &temp_int);

		temp_int.data.i = ((i / (sc->logicBlock[0].data.i / sc->warpSize)) / sc->logicBlock[1].data.i);
		PfMul(sc, &sc->inoutID_z, &sc->gl_WorkGroupID_z, &sc->logicBlock[2], 0);
		PfAdd(sc, &sc->inoutID_z, &sc->inoutID_z, &temp_int);

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);
		PfSetToZero(sc, &sc->temp2);

		if (i % (sc->logicBlock[0].data.i / sc->localSize[0].data.i) != (sc->logicBlock[0].data.i / sc->localSize[0].data.i - 1)) {

			temp_int.data.i = 1;
			PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i+1]);
			PfIf_else(sc);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i]);
			PfIf_end(sc);
			
			PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->temp, 1);


			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}
		}
		else {
			PfSubgroupShuffleDown(sc, &sc->regIDs_x[i], &sc->temp, 1);
			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}


			if (sc->size[0].data.i > sc->logicBlock[0].data.i) {
				temp_int.data.i = sc->warpSize - 1;
				PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfDivCeil(sc, &temp_int, &sc->size[0], &sc->logicBlock[0]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_x, &temp_int);

				//sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/ (double)sc->logicBlock[0])-1);
				
				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfInc(sc, &sc->inoutID);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDxStruct, &sc->inoutID);

				PfIf_end(sc);
				PfIf_end(sc);
			}
		}

		PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
		
		PfSub(sc, &sc->temp, &sc->temp, &sc->regIDs_x[i]);
		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * %.17e%s;\n", i, sc->s_dx, sc->LFending);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * s_dx;\n", i);
		
		
		PfIf_end(sc);
		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}

		PfSetToZero(sc, &sc->temp);

		if (i % (sc->logicBlock[0].data.i / sc->localSize[0].data.i) != 0) {

			temp_int.data.i = sc->warpSize - 1;
			PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i - 1]);
			PfIf_else(sc);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i]);
			PfIf_end(sc);

			PfSubgroupShuffleUpCyclic(sc, &sc->temp, &sc->temp, 1);


			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}
		}
		else {
			PfSubgroupShuffleUp(sc, &sc->regIDs_x[i], &sc->temp, 1);
			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}


			if (sc->size[0].data.i > sc->logicBlock[0].data.i) {
				temp_int.data.i = 0;
				PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_x, &temp_int);

				//sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/ (double)sc->logicBlock[0])-1);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 1;
				PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDxStruct, &sc->inoutID);

				PfIf_end(sc);
				PfIf_end(sc);
			}
		}

		PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);

		PfSub(sc, &sc->temp, &sc->temp, &sc->regIDs_x[i]);
		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * %.17e%s;\n", i, sc->s_dx, sc->LFending);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * s_dx;\n", i);


		PfIf_end(sc);

		temp_double.data.d = sc->s_dx.data.d*2;
		PfMul(sc, &sc->temp2, &sc->temp2, &temp_double, 0);

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);

		PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
		
		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) % sc->logicBlock[1].data.i) != (sc->logicBlock[1].data.i - 1)) {

			PfSub(sc, &sc->temp, &sc->regIDs_y[i + sc->logicBlock[0].data.i / sc->localSize[0].data.i], &sc->regIDs_y[i]);
			PfMul(sc, &sc->temp, &sc->temp, &sc->s_dy, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[1].data.i > sc->logicBlock[1].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[1], &sc->logicBlock[1]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_y, &temp_int);
				
				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);
				
				appendGlobalToRegisters(sc, &sc->temp, &sc->qDyStruct, &sc->inoutID);
				
				PfSub(sc, &sc->temp, &sc->temp, &sc->regIDs_y[i]);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) % sc->logicBlock[1].data.i) != 0) {

			PfSub(sc, &sc->temp, &sc->regIDs_y[i - sc->logicBlock[0].data.i / sc->localSize[0].data.i], &sc->regIDs_y[i]);
			PfMul(sc, &sc->temp, &sc->temp, &sc->s_dy, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[1].data.i > sc->logicBlock[1].data.i) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_y, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDyStruct, &sc->inoutID);

				PfSub(sc, &sc->temp, &sc->temp, &sc->regIDs_y[i]);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);
				PfIf_end(sc);
			}
		}

		temp_double.data.d = sc->s_dy.data.d * 2;
		PfMul(sc, &sc->temp1, &sc->temp1, &temp_double, 0);
		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);

		PfIf_end(sc);

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);
		PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
		
		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) / sc->logicBlock[1].data.i) != (sc->logicBlock[2].data.i - 1)) {

			PfSub(sc, &sc->temp, &sc->regIDs_z[i + (sc->logicBlock[0].data.i / sc->localSize[0].data.i) * sc->logicBlock[1].data.i], &sc->regIDs_z[i]);
			PfMul(sc, &sc->temp, &sc->temp, &sc->s_dz, 0);
			PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp);

		}
		else {
			if (sc->size[2].data.i > sc->logicBlock[2].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[2], &sc->logicBlock[2]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_z, &temp_int);
			
				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[2]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDzStruct, &sc->inoutID);
				
				PfSub(sc, &sc->temp, &sc->temp, &sc->regIDs_z[i]);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) / sc->logicBlock[1].data.i) != 0) {

			PfSub(sc, &sc->temp, &sc->regIDs_z[i - (sc->logicBlock[0].data.i / sc->localSize[0].data.i) * sc->logicBlock[1].data.i], &sc->regIDs_z[i]);
			PfMul(sc, &sc->temp, &sc->temp, &sc->s_dz, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[2].data.i > sc->logicBlock[2].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[2], &sc->logicBlock[2]);
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_z, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[2]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDzStruct, &sc->inoutID);

				PfSub(sc, &sc->temp, &sc->temp, &sc->regIDs_z[i]);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}
		temp_double.data.d = sc->s_dz.data.d * 2;
		PfMul(sc, &sc->temp1, &sc->temp1, &temp_double, 0);
		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);

		PfIf_end(sc);
		
		PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
		PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
		
		//appendGlobalToRegisters(sc, &sc->temp, &sc->PfStruct, &sc->inoutID);
		
		//PfMul(sc, &sc->temp2, &sc->temp2, &sc->s_dt_D, 0);
		//PfSub(sc, &sc->temp, &sc->temp, &sc->temp2);
		
		appendRegistersToGlobal(sc, &sc->PfStruct, &sc->inoutID, &sc->temp2);

		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}
	}
	return;
}

static inline void appendComputePf_4(PfSolveSpecializationConstantsLayout* sc) {

	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	int64_t stride = 1;
	int64_t next_stride = 1;
	for (int64_t i = 0; i < sc->registers_per_thread; i++) {
		temp_int.data.i = (i % (sc->logicBlock[0].data.i / sc->warpSize)) * sc->localSize[0].data.i;
		PfMul(sc, &sc->inoutID_x, &sc->gl_WorkGroupID_x, &sc->logicBlock[0], 0);
		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &sc->gl_LocalInvocationID_x);
		PfAdd(sc, &sc->inoutID_x, &sc->inoutID_x, &temp_int);

		temp_int.data.i = ((i / (sc->logicBlock[0].data.i / sc->warpSize)) % sc->logicBlock[1].data.i);
		PfMul(sc, &sc->inoutID_y, &sc->gl_WorkGroupID_y, &sc->logicBlock[1], 0);
		PfAdd(sc, &sc->inoutID_y, &sc->inoutID_y, &temp_int);

		temp_int.data.i = ((i / (sc->logicBlock[0].data.i / sc->warpSize)) / sc->logicBlock[1].data.i);
		PfMul(sc, &sc->inoutID_z, &sc->gl_WorkGroupID_z, &sc->logicBlock[2], 0);
		PfAdd(sc, &sc->inoutID_z, &sc->inoutID_z, &temp_int);

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);
		temp_double.data.d = -sc->s_dx.data.d * 5.0l / 2.0l;
		PfMul(sc, &sc->temp2, &sc->regIDs_x[i], &temp_double, 0);


		if (i % (sc->logicBlock[0].data.i / sc->localSize[0].data.i) != (sc->logicBlock[0].data.i / sc->localSize[0].data.i - 1)) {

			temp_int.data.i = 1;
			PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i + 1]);
			PfIf_else(sc);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i]);
			PfIf_end(sc);

			PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->temp, 1);


			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}
		}
		else {
			PfSubgroupShuffleDown(sc, &sc->regIDs_x[i], &sc->temp, 1);
			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}


			if (sc->size[0].data.i > sc->logicBlock[0].data.i) {
				temp_int.data.i = sc->warpSize - 1;
				PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfDivCeil(sc, &temp_int, &sc->size[0], &sc->logicBlock[0]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_x, &temp_int);

				//sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/ (double)sc->logicBlock[0])-1);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfInc(sc, &sc->inoutID);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDxStruct, &sc->inoutID);

				PfIf_end(sc);
				PfIf_end(sc);
			}
		}

		PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);

		temp_double.data.d = sc->s_dx.data.d * 4.0l/3.0l;
		PfMul(sc, &sc->temp1, &sc->temp, &temp_double, 0);

		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * %.17e%s;\n", i, sc->s_dx, sc->LFending);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * s_dx;\n", i);

		PfIf_end(sc);


		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);

		if (i % (sc->logicBlock[0].data.i / sc->localSize[0].data.i) != (sc->logicBlock[0].data.i / sc->localSize[0].data.i - 1)) {

			temp_int.data.i = 2;
			PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i + 1]);
			PfIf_else(sc);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i]);
			PfIf_end(sc);

			PfSubgroupShuffleDownCyclic(sc, &sc->temp, &sc->temp, 2);


			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}
		}
		else {
			PfSubgroupShuffleDown(sc, &sc->regIDs_x[i], &sc->temp, 2);
			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}


			if (sc->size[0].data.i > sc->logicBlock[0].data.i) {
				temp_int.data.i = sc->warpSize - 2;
				PfIf_gt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfDivCeil(sc, &temp_int, &sc->size[0], &sc->logicBlock[0]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_x, &temp_int);

				//sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/ (double)sc->logicBlock[0])-1);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 2;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDxStruct, &sc->inoutID);

				PfIf_end(sc);
				PfIf_end(sc);
			}
		}
		PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);

		temp_double.data.d = - sc->s_dx.data.d / 12.0l;
		PfMul(sc, &sc->temp1, &sc->temp, &temp_double, 0);

		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);

		PfIf_end(sc);


		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}


		if (i % (sc->logicBlock[0].data.i / sc->localSize[0].data.i) != 0) {

			temp_int.data.i = sc->warpSize - 1;
			PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i - 1]);
			PfIf_else(sc);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i]);
			PfIf_end(sc);

			PfSubgroupShuffleUpCyclic(sc, &sc->temp, &sc->temp, 1);


			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}
		}
		else {
			PfSubgroupShuffleUp(sc, &sc->regIDs_x[i], &sc->temp, 1);
			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}


			if (sc->size[0].data.i > sc->logicBlock[0].data.i) {
				temp_int.data.i = 0;
				PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_x, &temp_int);

				//sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/ (double)sc->logicBlock[0])-1);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 1;
				PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDxStruct, &sc->inoutID);

				PfIf_end(sc);
				PfIf_end(sc);
			}
		}

		PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);

		temp_double.data.d = sc->s_dx.data.d * 4.0l / 3.0l;
		PfMul(sc, &sc->temp1, &sc->temp, &temp_double, 0);

		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * %.17e%s;\n", i, sc->s_dx, sc->LFending);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * s_dx;\n", i);

		PfIf_end(sc);


		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}


		if (i % (sc->logicBlock[0].data.i / sc->localSize[0].data.i) != 0) {

			temp_int.data.i = sc->warpSize - 2;
			PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i - 1]);
			PfIf_else(sc);
			PfMov(sc, &sc->temp, &sc->regIDs_x[i]);
			PfIf_end(sc);

			PfSubgroupShuffleUpCyclic(sc, &sc->temp, &sc->temp, 2);


			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}
		}
		else {
			PfSubgroupShuffleUp(sc, &sc->regIDs_x[i], &sc->temp, 2);
			if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);
			}
			if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);
			}
			if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
				PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);
			}


			if (sc->size[0].data.i > sc->logicBlock[0].data.i) {
				temp_int.data.i = 2;
				PfIf_lt_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_x, &temp_int);

				//sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/ (double)sc->logicBlock[0])-1);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 2;
				PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDxStruct, &sc->inoutID);

				PfIf_end(sc);
				PfIf_end(sc);
			}
		}

		PfIf_lt_start(sc, &sc->inoutID_x, &sc->size[0]);

		temp_double.data.d = -sc->s_dx.data.d / 12.0l;
		PfMul(sc, &sc->temp1, &sc->temp, &temp_double, 0);

		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * %.17e%s;\n", i, sc->s_dx, sc->LFending);
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 += (temp_0 - reg_x_%" PRIu64 ") * s_dx;\n", i);

		PfIf_end(sc);
		

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);

		temp_double.data.d = -sc->s_dy.data.d * 5.0l / 2.0l;
		PfMul(sc, &sc->temp1, &sc->regIDs_y[i], &temp_double, 0);

		PfIf_lt_start(sc, &sc->inoutID_y, &sc->size[1]);

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) % sc->logicBlock[1].data.i) != (sc->logicBlock[1].data.i - 1)) {

			temp_double.data.d = sc->s_dy.data.d * 4.0l / 3.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_y[i + sc->logicBlock[0].data.i / sc->localSize[0].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[1].data.i > sc->logicBlock[1].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[1], &sc->logicBlock[1]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_y, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDyStruct, &sc->inoutID);

				temp_double.data.d = sc->s_dy.data.d * 4.0l / 3.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) % sc->logicBlock[1].data.i) < (sc->logicBlock[1].data.i - 2)) {

			temp_double.data.d = -sc->s_dy.data.d / 12.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_y[i + 2 * sc->logicBlock[0].data.i / sc->localSize[0].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[1].data.i > sc->logicBlock[1].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[1], &sc->logicBlock[1]);
				if (sc->logicBlock[1].data.i>1)
					temp_int.data.i = temp_int.data.i - 1;
				else
					temp_int.data.i = temp_int.data.i - 2;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_y, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 2 * sc->inputStride[1].data.i;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDyStruct, &sc->inoutID);

				temp_double.data.d = -sc->s_dy.data.d / 12.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) % sc->logicBlock[1].data.i) != 0) {

			temp_double.data.d = sc->s_dy.data.d * 4.0l / 3.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_y[i - sc->logicBlock[0].data.i / sc->localSize[0].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[1].data.i > sc->logicBlock[1].data.i) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_y, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[1]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDyStruct, &sc->inoutID);

				temp_double.data.d = sc->s_dy.data.d * 4.0l / 3.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) % sc->logicBlock[1].data.i) > 1) {

			temp_double.data.d = -sc->s_dy.data.d / 12.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_y[i - 2 * sc->logicBlock[0].data.i / sc->localSize[0].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[1].data.i > sc->logicBlock[1].data.i) {
				if (sc->logicBlock[1].data.i > 1)
					temp_int.data.i = 1;
				else
					temp_int.data.i = 2;
				PfIf_ge_start(sc, &sc->gl_WorkGroupID_y, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 2* sc->inputStride[1].data.i;
				PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDyStruct, &sc->inoutID);

				temp_double.data.d = -sc->s_dy.data.d / 12.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);

		PfIf_end(sc);

		PfSetToZero(sc, &sc->temp);
		PfSetToZero(sc, &sc->temp1);
		PfIf_lt_start(sc, &sc->inoutID_z, &sc->size[2]);

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) / sc->logicBlock[1].data.i) != (sc->logicBlock[2].data.i - 1)) {

			temp_double.data.d = sc->s_dz.data.d * 4.0l / 3.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_z[i + (sc->logicBlock[0].data.i / sc->localSize[0].data.i) * sc->logicBlock[1].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[2].data.i > sc->logicBlock[2].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[2], &sc->logicBlock[2]);
				temp_int.data.i = temp_int.data.i - 1;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_z, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[2]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDzStruct, &sc->inoutID);

				temp_double.data.d = sc->s_dz.data.d * 4.0l / 3.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) / sc->logicBlock[1].data.i) < (sc->logicBlock[2].data.i - 2)) {

			temp_double.data.d = -sc->s_dz.data.d / 12.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_z[i + 2 * (sc->logicBlock[0].data.i / sc->localSize[0].data.i) * sc->logicBlock[1].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[2].data.i > sc->logicBlock[2].data.i) {
				PfDivCeil(sc, &temp_int, &sc->size[2], &sc->logicBlock[2]);
				if (sc->logicBlock[2].data.i > 1)
					temp_int.data.i = temp_int.data.i - 1;
				else
					temp_int.data.i = temp_int.data.i - 2;
				PfIf_lt_start(sc, &sc->gl_WorkGroupID_z, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 2 * sc->inputStride[2].data.i;
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDzStruct, &sc->inoutID);

				temp_double.data.d = -sc->s_dz.data.d / 12.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) / sc->logicBlock[1].data.i) != 0) {

			temp_double.data.d = sc->s_dz.data.d * 4.0l / 3.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_z[i - (sc->logicBlock[0].data.i / sc->localSize[0].data.i) * sc->logicBlock[1].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[2].data.i > sc->logicBlock[2].data.i) {
				temp_int.data.i = 0;
				PfIf_gt_start(sc, &sc->gl_WorkGroupID_z, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				PfSub(sc, &sc->inoutID, &sc->inoutID, &sc->inputStride[2]);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDzStruct, &sc->inoutID);

				temp_double.data.d = sc->s_dz.data.d * 4.0l / 3.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		if ((i / (sc->logicBlock[0].data.i / sc->localSize[0].data.i) / sc->logicBlock[1].data.i) > 1) {

			temp_double.data.d = -sc->s_dz.data.d / 12.0l;
			PfMul(sc, &sc->temp, &sc->regIDs_z[i - 2 * (sc->logicBlock[0].data.i / sc->localSize[0].data.i) * sc->logicBlock[1].data.i], &temp_double, 0);
			PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

		}
		else {
			if (sc->size[2].data.i > sc->logicBlock[2].data.i) {
				if (sc->logicBlock[2].data.i > 1)
					temp_int.data.i = 1;
				else
					temp_int.data.i = 2;
				PfIf_ge_start(sc, &sc->gl_WorkGroupID_z, &temp_int);

				PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
				PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
				PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);
				temp_int.data.i = 2 * sc->inputStride[2].data.i;
				PfSub(sc, &sc->inoutID, &sc->inoutID, &temp_int);

				appendGlobalToRegisters(sc, &sc->temp, &sc->qDzStruct, &sc->inoutID);

				temp_double.data.d = -sc->s_dz.data.d / 12.0l;
				PfMul(sc, &sc->temp, &sc->temp, &temp_double, 0);
				PfAdd(sc, &sc->temp1, &sc->temp1, &sc->temp);

				PfIf_end(sc);
			}
		}

		PfAdd(sc, &sc->temp2, &sc->temp2, &sc->temp1);

		PfIf_end(sc);

		PfMul(sc, &sc->inoutID, &sc->inoutID_y, &sc->inputStride[1], 0);
		PfMul(sc, &sc->tempInt, &sc->inoutID_z, &sc->inputStride[2], 0);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->inoutID_x);
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->tempInt);

		//appendGlobalToRegisters(sc, &sc->temp, &sc->PfStruct, &sc->inoutID);

		//PfMul(sc, &sc->temp2, &sc->temp2, &sc->s_dt_D, 0);
		//PfSub(sc, &sc->temp, &sc->temp, &sc->temp2);

		appendRegistersToGlobal(sc, &sc->PfStruct, &sc->inoutID, &sc->temp2);

		if (sc->size[0].data.i % sc->logicBlock[0].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[1].data.i % sc->logicBlock[1].data.i) {
			PfIf_end(sc);
		}
		if (sc->size[2].data.i % sc->logicBlock[2].data.i) {
			PfIf_end(sc);
		}
	}
	return;
}

#endif
