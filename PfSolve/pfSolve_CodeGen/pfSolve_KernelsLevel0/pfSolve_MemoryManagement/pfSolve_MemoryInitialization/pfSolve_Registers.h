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
#ifndef PFSOLVE_REGISTERS_H
#define PFSOLVE_REGISTERS_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"

static inline void appendRegistersInitialization_compute_Pf(PfSolveSpecializationConstantsLayout* sc, int type) {

	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
    char name[50];

	sc->regIDs_x = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->regIDs_x == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->regIDs_y = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->regIDs_y == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->regIDs_z = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->regIDs_z == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->regIDs_x[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->regIDs_x[i], 50);
		sprintf(name, "reg_x_%d", i);
		PfDefine(sc, &sc->regIDs_x[i], name);
		PfSetToZero(sc, &sc->regIDs_x[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->regIDs_z[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->regIDs_y[i], 50);
		sc->regIDs_y[i].type = 100 + sc->floatTypeCode;
		sprintf(name, "reg_y_%d", i);
		PfDefine(sc, &sc->regIDs_y[i], name);
		PfSetToZero(sc, &sc->regIDs_y[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfAllocateContainerFlexible(sc, &sc->regIDs_z[i], 50);
		sprintf(name, "reg_z_%d", i);
		PfDefine(sc, &sc->regIDs_z[i], name);
		PfSetToZero(sc, &sc->regIDs_z[i]);
	}

	sc->temp.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp, 50);
	sprintf(name, "temp_0");
	PfDefine(sc, &sc->temp, name);
	PfSetToZero(sc, &sc->temp);

	sc->temp1.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp1, 50);
	sprintf(name, "temp_1");
	PfDefine(sc, &sc->temp1, name);
	PfSetToZero(sc, &sc->temp1);

	sc->temp2.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp2, 50);
	sprintf(name, "temp_2");
	PfDefine(sc, &sc->temp2, name);
	PfSetToZero(sc, &sc->temp2);

	sc->tempInt.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sprintf(name, "tempInt");
	PfDefine(sc, &sc->tempInt, name);
	PfSetToZero(sc, &sc->tempInt);  
    if (((sc->floatTypeCode % 100) / 10) == 3) {
		sc->tempQuad.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad, 50);
		sprintf(name, "tempQuad");
		PfDefine(sc, &sc->tempQuad, name);
		PfSetToZero(sc, &sc->tempQuad);

		sc->tempQuad2.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad2, 50);
		sprintf(name, "tempQuad2");
		PfDefine(sc, &sc->tempQuad2, name);
		PfSetToZero(sc, &sc->tempQuad2);

		sc->tempQuad3.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad3, 50);
		sprintf(name, "tempQuad3");
		PfDefine(sc, &sc->tempQuad3, name);
		PfSetToZero(sc, &sc->tempQuad3);

		sc->tempIntQuad.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempIntQuad, 50);
		sprintf(name, "tempIntQuad");
		PfDefine(sc, &sc->tempIntQuad, name);
		PfSetToZero(sc, &sc->tempIntQuad);
	}

	sc->inoutID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sprintf(name, "inoutID");
	PfDefine(sc, &sc->inoutID, name);
	PfSetToZero(sc, &sc->inoutID);

	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sprintf(name, "inoutID_x");
	PfDefine(sc, &sc->inoutID_x, name);
	PfSetToZero(sc, &sc->inoutID_x);

	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sprintf(name, "inoutID_y");
	PfDefine(sc, &sc->inoutID_y, name);
	PfSetToZero(sc, &sc->inoutID_y);

	sc->inoutID_z.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_z, 50);
	sprintf(name, "inoutID_z");
	PfDefine(sc, &sc->inoutID_z, name);
	PfSetToZero(sc, &sc->inoutID_z);

	
	return;
}

static inline void appendRegistersInitialization_compute_JW(PfSolveSpecializationConstantsLayout* sc, int type) {

	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
    char name[50];
	sc->rd = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->rd == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->rd_copy = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->rd == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->ud = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->ud == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->ud_copy = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->ud == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->ld = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->ld == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->ld_copy = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->ld == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	sc->md = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->md == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->rd[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->rd[i], 50);
		sprintf(name, "rd_%d", i);
		PfDefine(sc, &sc->rd[i], name);
		PfSetToZero(sc, &sc->rd[i]);
	}
	int64_t num_copy_registers = sc->registers_per_thread;
	if (sc->useParallelThomas) num_copy_registers = 1;

	for (int i = 0; i < num_copy_registers; i++) {
		sc->rd_copy[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->rd_copy[i], 50);
		sprintf(name, "rd_copy_%d", i);
		PfDefine(sc, &sc->rd_copy[i], name);
		PfSetToZero(sc, &sc->rd_copy[i]);
	}
	//we focus on bidiagonal systems for now 
	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->ud[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->ud[i], 50);
		sprintf(name, "uld_%d", i);
		PfDefine(sc, &sc->ud[i], name);
		PfSetToZero(sc, &sc->ud[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->ld[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->ld[i], 50);
		sprintf(name, "uld_%d", i);
		PfSetContainerName(sc, &sc->ld[i], name);
		//PfDefine(sc, &sc->ld[i], name);
		//PfSetToZero(sc, &sc->ld[i]);
	}
	if (!sc->upperBanded) {
		for (int i = 0; i < sc->registers_per_thread; i++) {
			sc->ud_copy[i].type = 100 + sc->floatTypeCode;
			PfAllocateContainerFlexible(sc, &sc->ud_copy[i], 50);
			sprintf(name, "copy_%d", i);
			PfSetContainerName(sc, &sc->ud_copy[i], name);
			//PfDefine(sc, &sc->ud_copy[i], name);
			//PfSetToZero(sc, &sc->ud_copy[i]);
		}
	}
	else {
		for (int i = 0; i < sc->registers_per_thread; i++) {
			sc->ld_copy[i].type = 100 + sc->floatTypeCode;
			PfAllocateContainerFlexible(sc, &sc->ld_copy[i], 50);
			sprintf(name, "copy_%d", i);
			PfSetContainerName(sc, &sc->ld_copy[i], name);
			//PfDefine(sc, &sc->ld_copy[i], name);
			//PfSetToZero(sc, &sc->ld_copy[i]);
		}
	}
	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->md[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->md[i], 50);
		sprintf(name, "copy_%d", i);
		PfDefine(sc, &sc->md[i], name);
		PfSetToZero(sc, &sc->md[i]);
	}
	sc->temp.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp, 50);
	sprintf(name, "temp_0");
	PfDefine(sc, &sc->temp, name);
	PfSetToZero(sc, &sc->temp);

	sc->temp1.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp1, 50);
	sprintf(name, "temp_1");
	PfDefine(sc, &sc->temp1, name);
	PfSetToZero(sc, &sc->temp1);

	sc->temp2.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp2, 50);
	sprintf(name, "temp_2");
	PfDefine(sc, &sc->temp2, name);
	PfSetToZero(sc, &sc->temp2);

	sc->tempInt.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sprintf(name, "tempInt");
	PfDefine(sc, &sc->tempInt, name);
	PfSetToZero(sc, &sc->tempInt);

	sc->warpInvocationID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->warpInvocationID, 50);
	sprintf(name, "warpInvocationID");
	PfDefine(sc, &sc->warpInvocationID, name);
	temp_int.data.i = sc->warpSize;
	PfMod(sc, &sc->warpInvocationID, &sc->gl_LocalInvocationID_x, &temp_int);

	sc->warpID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->warpID, 50);
	sprintf(name, "warpID");
	PfDefine(sc, &sc->warpID, name);
	temp_int.data.i = sc->warpSize;
	PfDiv(sc, &sc->warpID, &sc->gl_LocalInvocationID_x, &temp_int);

    if (((sc->floatTypeCode % 100) / 10) == 3) {
		sc->tempQuad.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad, 50);
		sprintf(name, "tempQuad");
		PfDefine(sc, &sc->tempQuad, name);
		PfSetToZero(sc, &sc->tempQuad);

		sc->tempQuad2.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad2, 50);
		sprintf(name, "tempQuad2");
		PfDefine(sc, &sc->tempQuad2, name);
		PfSetToZero(sc, &sc->tempQuad2);

		sc->tempQuad3.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad3, 50);
		sprintf(name, "tempQuad3");
		PfDefine(sc, &sc->tempQuad3, name);
		PfSetToZero(sc, &sc->tempQuad3);

		sc->tempIntQuad.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempIntQuad, 50);
		sprintf(name, "tempIntQuad");
		PfDefine(sc, &sc->tempIntQuad, name);
		PfSetToZero(sc, &sc->tempIntQuad);
	}

	sc->inoutID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sprintf(name, "inoutID");
	PfDefine(sc, &sc->inoutID, name);
	PfSetToZero(sc, &sc->inoutID);

	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sprintf(name, "inoutID_x");
	PfDefine(sc, &sc->inoutID_x, name);
	PfSetToZero(sc, &sc->inoutID_x);

	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sprintf(name, "inoutID_y");
	PfDefine(sc, &sc->inoutID_y, name);
	PfSetToZero(sc, &sc->inoutID_y);

	sc->inoutID_z.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_z, 50);
	sprintf(name, "inoutID_z");
	PfDefine(sc, &sc->inoutID_z, name);
	PfSetToZero(sc, &sc->inoutID_z);


	return;
}

static inline void appendRegistersInitialization_block(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
    char name[50];
	int typeCode = sc->floatTypeCode;
	if ((sc->block%10 == 2) || (sc->block%10 == 3))
		typeCode = sc->vecTypeCode;
    else 
		typeCode = sc->floatTypeCode;

	sc->rd = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->rd == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->rd[i].type = 100 + typeCode;
		PfAllocateContainerFlexible(sc, &sc->rd[i], 50);
		sprintf(name, "rd_%d", i);
		PfDefine(sc, &sc->rd[i], name);
		PfSetToZero(sc, &sc->rd[i]);
	}

	sc->temp.type = 100 + typeCode;
	PfAllocateContainerFlexible(sc, &sc->temp, 50);
	sprintf(name, "temp_0");
	PfDefine(sc, &sc->temp, name);
	PfSetToZero(sc, &sc->temp);

	sc->temp1.type = 100 + typeCode;
	PfAllocateContainerFlexible(sc, &sc->temp1, 50);
	sprintf(name, "temp_1");
	PfDefine(sc, &sc->temp1, name);
	PfSetToZero(sc, &sc->temp1);

	sc->temp2.type = 100 + typeCode;
	PfAllocateContainerFlexible(sc, &sc->temp2, 50);
	sprintf(name, "temp_2");
	PfDefine(sc, &sc->temp2, name);
	PfSetToZero(sc, &sc->temp2);

	sc->tempInt.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sprintf(name, "tempInt");
	PfDefine(sc, &sc->tempInt, name);
	PfSetToZero(sc, &sc->tempInt);

    if (((sc->floatTypeCode % 100) / 10) == 3) {
		sc->tempQuad.type = 100 + typeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad, 50);
		sprintf(name, "tempQuad");
		PfDefine(sc, &sc->tempQuad, name);
		PfSetToZero(sc, &sc->tempQuad);

		sc->tempQuad2.type = 100 + typeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad2, 50);
		sprintf(name, "tempQuad2");
		PfDefine(sc, &sc->tempQuad2, name);
		PfSetToZero(sc, &sc->tempQuad2);

		sc->tempQuad3.type = 100 + typeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad3, 50);
		sprintf(name, "tempQuad3");
		PfDefine(sc, &sc->tempQuad3, name);
		PfSetToZero(sc, &sc->tempQuad3);

		sc->tempIntQuad.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempIntQuad, 50);
		sprintf(name, "tempIntQuad");
		PfDefine(sc, &sc->tempIntQuad, name);
		PfSetToZero(sc, &sc->tempIntQuad);
	}

	sc->combinedID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->combinedID, 50);
	sprintf(name, "combinedID");
	PfDefine(sc, &sc->combinedID, name);
	PfSetToZero(sc, &sc->combinedID);

	sc->inoutID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sprintf(name, "inoutID");
	PfDefine(sc, &sc->inoutID, name);
	PfSetToZero(sc, &sc->inoutID);

	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sprintf(name, "inoutID_x");
	PfDefine(sc, &sc->inoutID_x, name);
	PfSetToZero(sc, &sc->inoutID_x);

	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sprintf(name, "inoutID_y");
	PfDefine(sc, &sc->inoutID_y, name);
	PfSetToZero(sc, &sc->inoutID_y);
	
	return;
}
static inline void appendRegistersInitialization_dgbmv(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
    char name[50];
	sc->rd = (PfContainer*)calloc(sc->registers_per_thread, sizeof(PfContainer));
	if (sc->rd == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		sc->rd[i].type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->rd[i], 50);
		sprintf(name, "rd_%d", i);
		PfDefine(sc, &sc->rd[i], name);
		PfSetToZero(sc, &sc->rd[i]);
	}

	sc->temp.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp, 50);
	sprintf(name, "temp_0");
	PfDefine(sc, &sc->temp, name);
	PfSetToZero(sc, &sc->temp);

	sc->temp1.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp1, 50);
	sprintf(name, "temp_1");
	PfDefine(sc, &sc->temp1, name);
	PfSetToZero(sc, &sc->temp1);

	sc->temp2.type = 100 + sc->floatTypeCode;
	PfAllocateContainerFlexible(sc, &sc->temp2, 50);
	sprintf(name, "temp_2");
	PfDefine(sc, &sc->temp2, name);
	PfSetToZero(sc, &sc->temp2);

	sc->tempInt.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->tempInt, 50);
	sprintf(name, "tempInt");
	PfDefine(sc, &sc->tempInt, name);
	PfSetToZero(sc, &sc->tempInt);

    if (((sc->floatTypeCode % 100) / 10) == 3) {
		sc->tempQuad.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad, 50);
		sprintf(name, "tempQuad");
		PfDefine(sc, &sc->tempQuad, name);
		PfSetToZero(sc, &sc->tempQuad);

		sc->tempQuad2.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad2, 50);
		sprintf(name, "tempQuad2");
		PfDefine(sc, &sc->tempQuad2, name);
		PfSetToZero(sc, &sc->tempQuad2);

		sc->tempQuad3.type = 100 + sc->floatTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempQuad3, 50);
		sprintf(name, "tempQuad3");
		PfDefine(sc, &sc->tempQuad3, name);
		PfSetToZero(sc, &sc->tempQuad3);

		sc->tempIntQuad.type = 100 + sc->uintTypeCode;
		PfAllocateContainerFlexible(sc, &sc->tempIntQuad, 50);
		sprintf(name, "tempIntQuad");
		PfDefine(sc, &sc->tempIntQuad, name);
		PfSetToZero(sc, &sc->tempIntQuad);
	}
    
	sc->combinedID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->combinedID, 50);
	sprintf(name, "combinedID");
	PfDefine(sc, &sc->combinedID, name);
	PfSetToZero(sc, &sc->combinedID);

	sc->inoutID.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID, 50);
	sprintf(name, "inoutID");
	PfDefine(sc, &sc->inoutID, name);
	PfSetToZero(sc, &sc->inoutID);

	sc->inoutID_x.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_x, 50);
	sprintf(name, "inoutID_x");
	PfDefine(sc, &sc->inoutID_x, name);
	PfSetToZero(sc, &sc->inoutID_x);

	sc->inoutID_y.type = 100 + sc->uintTypeCode;
	PfAllocateContainerFlexible(sc, &sc->inoutID_y, 50);
	sprintf(name, "inoutID_y");
	PfDefine(sc, &sc->inoutID_y, name);
	PfSetToZero(sc, &sc->inoutID_y);

	return;
}
/*static inline PfSolveResult appendRegistersInitialization_dgbmv(PfSolveSpecializationConstantsLayout* sc) {
	PfSolveResult res = PFSOLVE_SUCCESS;


	res = appendRegistersInitialization_res(sc);
	if (res != PFSOLVE_SUCCESS) return res;

	sc->tempLen = sprintf(sc->tempStr, "	%s inoutID;\n", sc->uintType);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s id_x = %s;\n", sc->intType, sc->gl_LocalInvocationID_x);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	return res;
}
*/
static inline void freeRegisterInitialization_Pf(PfSolveSpecializationConstantsLayout* sc, int type) {

	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->regIDs_x[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->regIDs_y[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->regIDs_z[i]);
	}

	free(sc->regIDs_x);
	free(sc->regIDs_y);
	free(sc->regIDs_z);

    if (((sc->floatTypeCode % 100) / 10) == 3) {
		PfDeallocateContainer(sc, &sc->tempQuad);
		PfDeallocateContainer(sc, &sc->tempQuad2);
		PfDeallocateContainer(sc, &sc->tempQuad3);
		PfDeallocateContainer(sc, &sc->tempIntQuad);
	}
	PfDeallocateContainer(sc, &sc->temp);
	PfDeallocateContainer(sc, &sc->temp1);
	PfDeallocateContainer(sc, &sc->temp2);

	PfDeallocateContainer(sc, &sc->tempInt);

	PfDeallocateContainer(sc, &sc->tempInt2);
	
	PfDeallocateContainer(sc, &sc->inoutID);
	PfDeallocateContainer(sc, &sc->inoutID_x);
	PfDeallocateContainer(sc, &sc->inoutID_y);
	PfDeallocateContainer(sc, &sc->inoutID_z);
	
	return;
}

static inline void freeRegistersInitialization_compute_JW(PfSolveSpecializationConstantsLayout* sc, int type) {

	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->rd[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->rd_copy[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->ud[i]);
	}

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->ld[i]);
	}
	if (!sc->upperBanded) {
		for (int i = 0; i < sc->registers_per_thread; i++) {
			PfDeallocateContainer(sc, &sc->ud_copy[i]);
		}
	}
	else {
		for (int i = 0; i < sc->registers_per_thread; i++) {
			PfDeallocateContainer(sc, &sc->ld_copy[i]);
		}
	}
	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->md[i]);
	}

	free(sc->rd);
	free(sc->rd_copy); 

	free(sc->ud); 
	free(sc->ud_copy);
	free(sc->ld);
	free(sc->ld_copy);
	free(sc->md);

    if (((sc->floatTypeCode % 100) / 10) == 3) {
		PfDeallocateContainer(sc, &sc->tempQuad);
		PfDeallocateContainer(sc, &sc->tempQuad2);
		PfDeallocateContainer(sc, &sc->tempQuad3);
		PfDeallocateContainer(sc, &sc->tempIntQuad);
	}
	PfDeallocateContainer(sc, &sc->temp);
	PfDeallocateContainer(sc, &sc->temp1);
	PfDeallocateContainer(sc, &sc->temp2);

	PfDeallocateContainer(sc, &sc->tempInt);

	PfDeallocateContainer(sc, &sc->tempInt2);

	PfDeallocateContainer(sc, &sc->inoutID);
	PfDeallocateContainer(sc, &sc->inoutID_x);
	PfDeallocateContainer(sc, &sc->inoutID_y);
	PfDeallocateContainer(sc, &sc->inoutID_z);

	return;
}

static inline void freeRegistersInitialization_block(PfSolveSpecializationConstantsLayout* sc, int type) {

	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->rd[i]);
	}

	
	free(sc->rd);
	
    if (((sc->floatTypeCode % 100) / 10) == 3) {
		PfDeallocateContainer(sc, &sc->tempQuad);
		PfDeallocateContainer(sc, &sc->tempQuad2);
		PfDeallocateContainer(sc, &sc->tempQuad3);
		PfDeallocateContainer(sc, &sc->tempIntQuad);
	}
	PfDeallocateContainer(sc, &sc->temp);
	PfDeallocateContainer(sc, &sc->temp1);
	PfDeallocateContainer(sc, &sc->temp2);

	PfDeallocateContainer(sc, &sc->tempInt);

	PfDeallocateContainer(sc, &sc->combinedID);

	PfDeallocateContainer(sc, &sc->inoutID);
	PfDeallocateContainer(sc, &sc->inoutID_x);
	PfDeallocateContainer(sc, &sc->inoutID_y);
	
	return;
}
static inline void freeRegistersInitialization_dgbmv(PfSolveSpecializationConstantsLayout* sc, int type) {

	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;

	for (int i = 0; i < sc->registers_per_thread; i++) {
		PfDeallocateContainer(sc, &sc->rd[i]);
	}


	free(sc->rd);

    if (((sc->floatTypeCode % 100) / 10) == 3) {
		PfDeallocateContainer(sc, &sc->tempQuad);
		PfDeallocateContainer(sc, &sc->tempQuad2);
		PfDeallocateContainer(sc, &sc->tempQuad3);
		PfDeallocateContainer(sc, &sc->tempIntQuad);
	}
	PfDeallocateContainer(sc, &sc->temp);
	PfDeallocateContainer(sc, &sc->temp1);
	PfDeallocateContainer(sc, &sc->temp2);

	PfDeallocateContainer(sc, &sc->tempInt);

	PfDeallocateContainer(sc, &sc->combinedID);

	PfDeallocateContainer(sc, &sc->inoutID);
	PfDeallocateContainer(sc, &sc->inoutID_x);
	PfDeallocateContainer(sc, &sc->inoutID_y);

	return;
}


#endif
