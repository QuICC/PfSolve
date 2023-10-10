// This file is part of PfSolve
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This file is provided for informational purposes only. Redistribution without permission is not allowed.
#ifndef PFSOLVE_COMPUTEFLUXD_H
#define PFSOLVE_COMPUTEFLUXD_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
/*
static inline PfSolveResult appendComputeFluxD_x(PfSolveSpecializationConstantsLayout* sc) {
	
	char tempStr0[100];
	char tempStr1[100];
	PfSolveResult res = PFSOLVE_SUCCESS;
	int64_t stride = 1;
	int64_t next_stride = 1;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_x = %s + %" PRIu64 " + %s * %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i % (sc->logicBlock[0] / sc->warpSize)) * sc->localSize[0], sc->gl_WorkGroupID_x, sc->logicBlock[0]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_y = %" PRIu64 " + %s * %" PRIu64 ";\n", ((i / (sc->logicBlock[0] / sc->warpSize)) % sc->logicBlock[1]), sc->gl_WorkGroupID_y, sc->logicBlock[1]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_z = %" PRIu64 " + %s * %" PRIu64 ";\n", ((i / (sc->logicBlock[0] / sc->warpSize)) / sc->logicBlock[1]), sc->gl_WorkGroupID_z, sc->logicBlock[2]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		if (i % (sc->logicBlock[0] / sc->localSize[0]) != (sc->logicBlock[0] / sc->localSize[0] - 1)) {

			sc->tempLen = sprintf(sc->tempStr, "\n\
			if (id_x < %" PRIi64 ") {\n\
				temp_0 = reg_%" PRIi64 ";\n\
			}else{\n\
				temp_0 = reg_%" PRIi64 ";\n\
			}\n", 1, i + 1, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sprintf(tempStr0, "temp_0");
			sprintf(tempStr1, "temp_0");
			subgroupShuffleDownCyclic(sc, tempStr0, tempStr1, 1);
			if (sc->size[0] % sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_x < %" PRIu64 "){\n", sc->size[0]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] % sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_y < %" PRIu64 "){\n", sc->size[1]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] % sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_z < %" PRIu64 "){\n", sc->size[2]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}
		else {
			sprintf(tempStr0, "reg_%" PRIi64 "", i);
			sprintf(tempStr1, "temp_0");
			subgroupShuffleDown(sc, tempStr0, tempStr1, 1);
			if (sc->size[0] % sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_x < %" PRIu64 "){\n", sc->size[0]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] % sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_y < %" PRIu64 "){\n", sc->size[1]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] % sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_z < %" PRIu64 "){\n", sc->size[2]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[0] > sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "\n\
			if ((id_x == %" PRIi64 ") && (%s != %" PRIi64 ")) {\n", sc->warpSize - 1, sc->gl_WorkGroupID_x, (uint64_t)ceil(sc->size[0]/(double)sc->logicBlock[0])-1);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;

				sc->tempLen = sprintf(sc->tempStr, "	temp_0 = %sPf[inoutID_x+1 + inoutID_y*%" PRIu64 "+inoutID_z*%" PRIu64 "]%s;\n", sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput);

				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}

		sc->tempLen = sprintf(sc->tempStr, "\n\
			if (inoutID_x < %" PRIi64 ") {\n", sc->size[0] - 1);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	temp_1 = %sqDx[inoutID_x+1 + inoutID_y*%" PRIu64 "+inoutID_z*%" PRIu64 "]%s;\n", sc->convTypeLeftInput, sc->inputStride[1].x_num+1, (sc->inputStride[1].x_num + 1)*sc->size[1], sc->convTypeRightInput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 = temp_1 - (temp_1 + %.17e%s * (temp_0 - reg_%" PRIu64 ")) * %.17e%s;\n", sc->k_nf*sc->s_dx, sc->LFending, i, sc->s_dt_D, sc->LFending);
		sc->tempLen = sprintf(sc->tempStr, "	temp_1 = temp_1 - (temp_1 + k_nf * (temp_0 - reg_%" PRIu64 ") * s_dx) * s_dt_D;\n", i);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	qDx[inoutID_x+1 + inoutID_y*%" PRIu64 "+inoutID_z*%" PRIu64 "]= %stemp_1%s;\n", sc->inputStride[1].x_num + 1, (sc->inputStride[1].x_num + 1) * sc->size[1], sc->convTypeLeftOutput, sc->convTypeRightOutput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;

		if (sc->size[0] % sc->logicBlock[0]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		if (sc->size[1] % sc->logicBlock[1]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		if (sc->size[2] % sc->logicBlock[2]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
	}
	return res;
}

static inline PfSolveResult appendComputeFluxD_y(PfSolveSpecializationConstantsLayout* sc) {

	char tempStr0[100];
	char tempStr1[100];
	PfSolveResult res = PFSOLVE_SUCCESS;
	int64_t stride = 1;
	int64_t next_stride = 1;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_x = %s + %" PRIu64 " + %s * %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i % (sc->logicBlock[0] / sc->warpSize)) * sc->localSize[0], sc->gl_WorkGroupID_x, sc->logicBlock[0]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_y = %" PRIu64 " + %s * %" PRIu64 ";\n", ((i / (sc->logicBlock[0] / sc->warpSize)) % sc->logicBlock[1]), sc->gl_WorkGroupID_y, sc->logicBlock[1]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_z = %" PRIu64 " + %s * %" PRIu64 ";\n", ((i / (sc->logicBlock[0] / sc->warpSize)) / sc->logicBlock[1]), sc->gl_WorkGroupID_z, sc->logicBlock[2]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		if ((i / (sc->logicBlock[0] / sc->localSize[0]) % sc->logicBlock[1]) != (sc->logicBlock[1]-1)) {

			sc->tempLen = sprintf(sc->tempStr, "	temp_0 = reg_%" PRIu64 ";\n", i + sc->logicBlock[0] / sc->localSize[0]);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			if (sc->size[0] % sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_x < %" PRIu64 "){\n", sc->size[0]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] % sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_y < %" PRIu64 "){\n", sc->size[1]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] % sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_z < %" PRIu64 "){\n", sc->size[2]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}
		else {
			if (sc->size[0] % sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_x < %" PRIu64 "){\n", sc->size[0]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] % sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_y < %" PRIu64 "){\n", sc->size[1]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] % sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_z < %" PRIu64 "){\n", sc->size[2]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] > sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "\n\
			if (%s != %" PRIi64 ") {\n", sc->gl_WorkGroupID_y, (uint64_t)ceil(sc->size[1] / (double)sc->logicBlock[1]) - 1);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;

				sc->tempLen = sprintf(sc->tempStr, "	temp_0 = %sPf[inoutID_x + (inoutID_y+1)*%" PRIu64 "+inoutID_z*%" PRIu64 "]%s;\n", sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput);

				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}

		sc->tempLen = sprintf(sc->tempStr, "\n\
			if (inoutID_y < %" PRIi64 ") {\n", sc->size[1] - 1);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	temp_1 = %sqDy[inoutID_x + (inoutID_y+1)*%" PRIu64 "+inoutID_z*%" PRIu64 "]%s;\n", sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[1].x_num* (sc->size[1]+1), sc->convTypeRightInput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 = temp_1 - (temp_1 + %.17e%s * (temp_0 - reg_%" PRIu64 ")) * %.17e%s;\n", sc->k_nf*sc->s_dy, sc->LFending, i, sc->s_dt_D, sc->LFending);
		sc->tempLen = sprintf(sc->tempStr, "	temp_1 = temp_1 - (temp_1 + k_nf * (temp_0 - reg_%" PRIu64 ") * s_dy) * s_dt_D;\n", i);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	qDy[inoutID_x + (inoutID_y+1)*%" PRIu64 "+inoutID_z*%" PRIu64 "]= %stemp_1%s;\n", sc->inputStride[1].x_num, sc->inputStride[1].x_num * (sc->size[1] + 1), sc->convTypeLeftOutput, sc->convTypeRightOutput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;

		if (sc->size[0] % sc->logicBlock[0]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		if (sc->size[1] % sc->logicBlock[1]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		if (sc->size[2] % sc->logicBlock[2]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
	}
	return res;
}

static inline PfSolveResult appendComputeFluxD_z(PfSolveSpecializationConstantsLayout* sc) {

	char tempStr0[100];
	char tempStr1[100];
	PfSolveResult res = PFSOLVE_SUCCESS;
	int64_t stride = 1;
	int64_t next_stride = 1;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_x = %s + %" PRIu64 " + %s * %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i % (sc->logicBlock[0] / sc->warpSize)) * sc->localSize[0], sc->gl_WorkGroupID_x, sc->logicBlock[0]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_y = %" PRIu64 " + %s * %" PRIu64 ";\n", ((i / (sc->logicBlock[0] / sc->warpSize)) % sc->logicBlock[1]), sc->gl_WorkGroupID_y, sc->logicBlock[1]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	inoutID_z = %" PRIu64 " + %s * %" PRIu64 ";\n", ((i / (sc->logicBlock[0] / sc->warpSize)) / sc->logicBlock[1]), sc->gl_WorkGroupID_z, sc->logicBlock[2]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		if ((i / (sc->logicBlock[0] / sc->localSize[0]) / sc->logicBlock[1]) != (sc->logicBlock[2]-1)) {

			sc->tempLen = sprintf(sc->tempStr, "	temp_0 = reg_%" PRIu64 ";\n", i + (sc->logicBlock[0] / sc->localSize[0])*sc->logicBlock[1]);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			if (sc->size[0] % sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_x < %" PRIu64 "){\n", sc->size[0]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] % sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_y < %" PRIu64 "){\n", sc->size[1]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] % sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_z < %" PRIu64 "){\n", sc->size[2]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}
		else {
			if (sc->size[0] % sc->logicBlock[0]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_x < %" PRIu64 "){\n", sc->size[0]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[1] % sc->logicBlock[1]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_y < %" PRIu64 "){\n", sc->size[1]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] % sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "	if(inoutID_z < %" PRIu64 "){\n", sc->size[2]);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			if (sc->size[2] > sc->logicBlock[2]) {
				sc->tempLen = sprintf(sc->tempStr, "\n\
			if (%s != %" PRIi64 ") {\n", sc->gl_WorkGroupID_z, (uint64_t)ceil(sc->size[2] / (double)sc->logicBlock[2]) - 1);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;

				sc->tempLen = sprintf(sc->tempStr, "	temp_0 = %sPf[inoutID_x + (inoutID_y)*%" PRIu64 "+(inoutID_z+1)*%" PRIu64 "]%s;\n", sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput);

				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}

		sc->tempLen = sprintf(sc->tempStr, "\n\
			if (inoutID_z < %" PRIi64 ") {\n", sc->size[2] - 1);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	temp_1 = %sqDz[inoutID_x + (inoutID_y)*%" PRIu64 "+ (inoutID_z+1)*%" PRIu64 "]%s;\n", sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		//sc->tempLen = sprintf(sc->tempStr, "	temp_1 = temp_1 - (temp_1 + %.17e%s * ((temp_0 - reg_%" PRIu64 ")* %.17e%s -  %.17e%s*(%sT[inoutID_x + (inoutID_y)*%" PRIu64 "+ (inoutID_z)*%" PRIu64 "]%s + %sT[inoutID_x + (inoutID_y)*%" PRIu64 "+ (inoutID_z+1)*%" PRIu64 "]%s))) * %.17e%s;\n", sc->k_nf, sc->LFending, i, sc->s_dz, sc->LFending, sc->arg * 0.5, sc->LFending, sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput, sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput,sc->s_dt_D, sc->LFending);
		sc->tempLen = sprintf(sc->tempStr, "	temp_1 = temp_1 - (temp_1 + k_nf * ((temp_0 - reg_%" PRIu64 ")* s_dz -  arg *(%sT[inoutID_x + (inoutID_y)*%" PRIu64 "+ (inoutID_z)*%" PRIu64 "]%s + %sT[inoutID_x + (inoutID_y)*%" PRIu64 "+ (inoutID_z+1)*%" PRIu64 "]%s))) * s_dt_D;\n", i, sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput, sc->convTypeLeftInput, sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeRightInput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	qDz[inoutID_x + (inoutID_y)*%" PRIu64 "+(inoutID_z+1)*%" PRIu64 "]= %stemp_1%s;\n", sc->inputStride[1].x_num, sc->inputStride[2].x_num, sc->convTypeLeftOutput, sc->convTypeRightOutput);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "\n\
			}\n");
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;

		if (sc->size[0] % sc->logicBlock[0]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		if (sc->size[1] % sc->logicBlock[1]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		if (sc->size[2] % sc->logicBlock[2]) {
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
	}
	return res;
}
*/
#endif
