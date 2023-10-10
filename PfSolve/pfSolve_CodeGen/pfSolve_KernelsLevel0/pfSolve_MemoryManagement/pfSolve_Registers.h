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
#ifndef PFSOLVE_KERNELSTART_H
#define PFSOLVE_KERNELSTART_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"

static inline PfSolveResult appendInitialization(PfSolveSpecializationConstantsLayout* sc) {
	PfSolveResult res = PFSOLVE_SUCCESS;
	
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s md_%" PRIu64 ";\n", sc->dataType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
	}
	sc->offset_md = 0;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s ud_%" PRIu64 ";\n", sc->dataType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
	}
	sc->offset_ud = sc->M_size;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s ld_%" PRIu64 ";\n", sc->dataType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
	}
	sc->offset_ld = 2*sc->M_size;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s x_%" PRIu64 ";\n", sc->dataType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
	}
	sc->offset_x = 3*sc->M_size;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s res_%" PRIu64 ";\n", sc->dataType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
	}
	sc->offset_res = 4*sc->M_size;
	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.x;\n");
	uint64_t logicalStoragePerThread = sc->registers_per_thread * sc->registerBoost;
	uint64_t logicalRegistersPerThread = sc->registers_per_thread;
	if (sc->convolutionStep) {
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".x=0;\n", i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".y=0;\n", i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
		for (uint64_t j = 1; j < sc->matrixConvolution; j++) {
			for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
				sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 "_%" PRIu64 ";\n", vecType, i, j);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 "_%" PRIu64 ".x=0;\n", i, j);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 "_%" PRIu64 ".y=0;\n", i, j);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
		}
	}
	else {
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".x=0;\n", i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".y=0;\n", i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.y;//gl_LocalInvocationID.x/gl_WorkGroupSize.x;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dum=dum/gl_LocalInvocationID.x-1;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dummy=dummy/gl_LocalInvocationID.x-1;\n");
	sc->regIDs = (char**)malloc(sizeof(char*) * logicalStoragePerThread);
	if (!sc->regIDs) return PFSOLVE_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < logicalStoragePerThread; i++) {
		sc->regIDs[i] = (char*)malloc(sizeof(char) * 50);
		if (!sc->regIDs[i]) {
			for (uint64_t j = 0; j < i; j++) {
				free(sc->regIDs[j]);
				sc->regIDs[j] = 0;
			}
			free(sc->regIDs);
			sc->regIDs = 0;
			return PFSOLVE_ERROR_MALLOC_FAILED;
		}
		if (i < logicalRegistersPerThread)
			sprintf(sc->regIDs[i], "temp_%" PRIu64 "", i);
		else
			sprintf(sc->regIDs[i], "temp_%" PRIu64 "", i);
		//sprintf(sc->regIDs[i], "%" PRIu64 "[%" PRIu64 "]", i / logicalRegistersPerThread, i % logicalRegistersPerThread);
		//sprintf(sc->regIDs[i], "s[%" PRIu64 "]", i - logicalRegistersPerThread);

	}
	if (sc->registerBoost > 1) {
		//sc->tempLen = sprintf(sc->tempStr, "	%s sort0;\n", vecType);
		//sc->tempLen = sprintf(sc->tempStr, "	%s temps[%" PRIu64 "];\n", vecType, (sc->registerBoost -1)* logicalRegistersPerThread);
		for (uint64_t i = 1; i < sc->registerBoost; i++) {
			//sc->tempLen = sprintf(sc->tempStr, "	%s temp%" PRIu64 "[%" PRIu64 "];\n", vecType, i, logicalRegistersPerThread);
			for (uint64_t j = 0; j < sc->registers_per_thread; j++) {
				sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, j + i * sc->registers_per_thread);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".x=0;\n", j + i * sc->registers_per_thread);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	temp_%" PRIu64 ".y=0;\n", j + i * sc->registers_per_thread);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			/*sc->tempLen = sprintf(sc->tempStr, "\
for(uint i=0; i<%" PRIu64 "; i++)\n\
temp%" PRIu64 "[i]=%s(dum, dum);\n", logicalRegistersPerThread, i, vecType);*/
		}
	}
	sc->tempLen = sprintf(sc->tempStr, "	%s w;\n", vecType);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	w.x=0;\n");
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	w.y=0;\n");
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sprintf(sc->w, "w");
	uint64_t maxNonPow2Radix = 1;
	if (sc->fftDim % 3 == 0) maxNonPow2Radix = 3;
	if (sc->fftDim % 5 == 0) maxNonPow2Radix = 5;
	if (sc->fftDim % 7 == 0) maxNonPow2Radix = 7;
	if (sc->fftDim % 11 == 0) maxNonPow2Radix = 11;
	if (sc->fftDim % 13 == 0) maxNonPow2Radix = 13;
	for (uint64_t i = 0; i < maxNonPow2Radix; i++) {
		sprintf(sc->locID[i], "loc_%" PRIu64 "", i);
		sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->locID[i]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	%s.x=0;\n", sc->locID[i]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	%s.y=0;\n", sc->locID[i]);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}
	sprintf(sc->temp, "%s", sc->locID[0]);
	uint64_t useRadix8 = 0;
	for (uint64_t i = 0; i < sc->numStages; i++)
		if (sc->stageRadix[i] == 8) useRadix8 = 1;
	if (useRadix8 == 1) {
		if (maxNonPow2Radix > 1) sprintf(sc->iw, "%s", sc->locID[1]);
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s iw;\n", vecType);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	iw.x=0;\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	iw.y=0;\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sprintf(sc->iw, "iw");
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->tempReg);
	sc->tempLen = sprintf(sc->tempStr, "	%s %s=0;\n", uintType, sc->stageInvocationID);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s=0;\n", uintType, sc->blockInvocationID);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s=0;\n", uintType, sc->sdataID);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s=0;\n", uintType, sc->combinedID);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s=0;\n", uintType, sc->inoutID);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, "	%s LUTId=0;\n", uintType);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "	%s angle=0;\n", floatType);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}
	if (((sc->stageStartSize > 1) && (!((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse))) || (sc->performDCT)) {
		sc->tempLen = sprintf(sc->tempStr, "	%s mult;\n", vecType);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	mult.x = 0;\n");
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	mult.y = 0;\n");
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}
	if (sc->cacheShuffle) {
		sc->tempLen = sprintf(sc->tempStr, "\
	%s tshuffle= ((%s>>1))%%(%" PRIu64 ");\n\
	%s shuffle[%" PRIu64 "];\n", uintType, sc->gl_LocalInvocationID_x, sc->registers_per_thread, vecType, sc->registers_per_thread);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
			/*sc->tempLen = sprintf(sc->tempStr, "\
shuffle[%" PRIu64 "];\n", i, vecType);*/
			sc->tempLen = sprintf(sc->tempStr, "	shuffle[%" PRIu64 "].x = 0;\n", i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	shuffle[%" PRIu64 "].y = 0;\n", i);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
		}
	}
	return res;
}
#endif
