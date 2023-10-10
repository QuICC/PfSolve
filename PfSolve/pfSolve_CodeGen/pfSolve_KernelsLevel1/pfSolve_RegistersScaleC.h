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
#ifndef PFSOLVE_REGISTERSSCALEC_H
#define PFSOLVE_REGISTERSSCALEC_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"

static inline void appendRegistersScaleC(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->scaleC, &sc->temp);
	}
	return;
}
static inline void appendRegistersScaleD(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	temp_int.data.i = sc->registers_per_thread * sc->num_threads;
	PfMul(sc, &sc->inoutID, &sc->gl_WorkGroupID_x, &temp_int, 0);
	PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);

	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {

		//sc->tempLen = sprintf(sc->tempStr, "	inoutID = %s + %" PRIu64 " + %s * %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, i * sc->num_threads, sc->gl_WorkGroupID_x, sc->registers_per_thread * sc->num_threads);
		
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &sc->size[0]);
		
		temp_double.data.d = 1;
		PfAdd(sc, &sc->temp, &sc->inoutID_x, &temp_double);
		if (sc->scaleC.type > 100) {
			PfAdd(sc, &sc->temp1, &sc->inoutID_x, &sc->scaleC);
			temp_double.data.d = sc->lshift + 1.5;
			PfAdd(sc, &sc->temp1, &sc->temp1, &temp_double);
		}
		else {
			temp_double.data.d = sc->lshift + 1.5 + sc->scaleC.data.d;
			PfAdd(sc, &sc->temp1, &sc->temp1, &temp_double);
		}

		PfDiv(sc, &sc->temp, &sc->temp, &sc->temp1);
		PfSqrt(sc, &sc->temp, &sc->temp);
		temp_double.data.d = sc->lshift + 1.0;
		PfAdd(sc, &sc->temp1, &sc->inoutID_x, &temp_double);
		temp_double.data.d = 2.0;
		PfMul(sc, &sc->temp1, &sc->temp1, &temp_double, 0);
		PfMul(sc, &sc->temp, &sc->temp, &sc->temp1, 0);

		PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->temp, 0);
		//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = res_%" PRIu64 " * 2.0%s*(%.17e%s + id_x)*sqrt((%s)((id_x+1.0%s)/(id_x + %s%s + %.17e%s)));\n", i, i, sc->LFending, (double)sc->lshift + 1.0, sc->LFending, sc->dataType, sc->LFending, sc->scaleC.x_str, sc->LFending, (double)sc->lshift + 1.5, sc->LFending);

		
		
		temp_int.data.i = sc->num_threads;
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
	}
	return;
}
static inline void appendRegistersScaleSphLaplA(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	temp_int.data.i = sc->registers_per_thread * sc->num_threads;
	PfMul(sc, &sc->inoutID, &sc->gl_WorkGroupID_x, &temp_int, 0);
	PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x);

	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		//sc->tempLen = sprintf(sc->tempStr, "	inoutID = %s + %" PRIu64 " + %s * %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, i * sc->num_threads, sc->gl_WorkGroupID_x, sc->registers_per_thread * sc->num_threads);
		

		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &sc->size[0]);

		temp_double.data.d = 1;
		PfAdd(sc, &sc->temp, &sc->inoutID_x, &temp_double);
		temp_double.data.d = 2;
		PfAdd(sc, &sc->temp1, &sc->inoutID_x, &temp_double);
		PfMul(sc, &sc->temp, &sc->temp, &sc->temp1, 0);

		if (sc->scaleC.type > 100) {
			PfAdd(sc, &sc->temp1, &sc->inoutID_x, &sc->scaleC);
			temp_double.data.d = sc->lshift + 2.5;
			PfAdd(sc, &sc->temp1, &sc->temp1, &temp_double);

			PfAdd(sc, &sc->temp2, &sc->inoutID_x, &sc->scaleC);
			temp_double.data.d = sc->lshift + 3.5;
			PfAdd(sc, &sc->temp2, &sc->temp2, &temp_double);
		}
		else {
			temp_double.data.d = sc->lshift + 2.5 + sc->scaleC.data.d;
			PfAdd(sc, &sc->temp1, &sc->temp1, &temp_double);

			temp_double.data.d = sc->lshift + 3.5 + sc->scaleC.data.d;
			PfAdd(sc, &sc->temp2, &sc->temp2, &temp_double);
		}
		PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp2, 0);

		PfDiv(sc, &sc->temp, &sc->temp, &sc->temp1);
		PfSqrt(sc, &sc->temp, &sc->temp);

		temp_double.data.d = sc->lshift + 2.0;
		PfAdd(sc, &sc->temp1, &sc->inoutID_x, &temp_double);

		temp_double.data.d = sc->lshift + 3.0;
		PfAdd(sc, &sc->temp2, &sc->inoutID_x, &temp_double);
		PfMul(sc, &sc->temp1, &sc->temp1, &sc->temp2, 0);
		temp_double.data.d = 4.0;
		PfMul(sc, &sc->temp1, &sc->temp1, &temp_double, 0);
		PfMul(sc, &sc->temp, &sc->temp, &sc->temp1, 0);

		PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->temp, 0);
		
		//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = res_%" PRIu64 " * 4.0%s*(%.17e%s + id_x)*(%.17e%s + id_x)*sqrt((%s)((id_x+1.0%s)*(id_x+2.0%s)/((id_x + %s%s + %.17e%s)*(id_x + %s%s + %.17e%s))));\n", i, i, sc->LFending, (double)sc->lshift + 2.0, sc->LFending, (double)sc->lshift + 3.0, sc->LFending, sc->dataType, sc->LFending, sc->LFending, sc->scaleC.x_str, sc->LFending, (double)sc->lshift + 2.5, sc->LFending, sc->scaleC.x_str, sc->LFending, (double)sc->lshift + 3.5, sc->LFending);

		temp_int.data.i = sc->num_threads;
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
	}
	return;
}
static inline void appendRegistersScaleSphLaplB(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfContainer temp_int1 = {};
	temp_int1.type = 31;
	PfContainer temp_double = {};
	temp_double.type = 32;

	temp_int.data.i = sc->registers_per_thread * sc->num_threads;
	PfMul(sc, &sc->inoutID, &sc->gl_WorkGroupID_x, &temp_int, 0);
	PfAdd(sc, &sc->inoutID, &sc->inoutID, &sc->gl_LocalInvocationID_x); 
	
	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		//sc->tempLen = sprintf(sc->tempStr, "	inoutID = %s + %" PRIu64 " + %s * %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, i * sc->num_threads, sc->gl_WorkGroupID_x, sc->registers_per_thread * sc->num_threads);
		
		PfMod(sc, &sc->inoutID_x, &sc->inoutID, &sc->size[0]);

		temp_double.data.d = 1;
		PfAdd(sc, &sc->temp, &sc->inoutID_x, &temp_double);
		if (sc->scaleC.type > 100) {
			PfAdd(sc, &sc->temp1, &sc->inoutID_x, &sc->scaleC);
			temp_double.data.d = sc->lshift + 1.5;
			PfAdd(sc, &sc->temp1, &sc->temp1, &temp_double);
		}
		else {
			temp_double.data.d = sc->lshift + 1.5 + sc->scaleC.data.d;
			PfAdd(sc, &sc->temp1, &sc->inoutID_x, &temp_double);
		}

		PfDiv(sc, &sc->temp, &sc->temp, &sc->temp1);
		PfSqrt(sc, &sc->temp, &sc->temp);
		temp_double.data.d = sc->lshift + 1.0;
		PfAdd(sc, &sc->temp1, &sc->inoutID_x, &temp_double);
		temp_double.data.d = 4.0 * (double)sc->lshift + 6.0;
		PfMul(sc, &sc->temp1, &sc->temp1, &temp_double, 0);
		PfMul(sc, &sc->temp, &sc->temp, &sc->temp1, 0);

		PfMul(sc, &sc->rd[i], &sc->rd[i], &sc->temp, 0);
		
		//sc->tempLen = sprintf(sc->tempStr, "	res_%" PRIu64 " = res_%" PRIu64 " * 2.0%s*(%.17e%s + id_x)*%.17e%s*sqrt((%s)((id_x+1.0%s)/(id_x + %s%s + %.17e%s)));\n", i, i, sc->LFending, (double)sc->lshift + 1.0, sc->LFending, 2.0 * (double)sc->lshift + 3.0, sc->LFending, sc->dataType, sc->LFending, sc->scaleC.x_str, sc->LFending, (double)sc->lshift + 1.5, sc->LFending);

		
		temp_int.data.i = sc->num_threads;
		PfAdd(sc, &sc->inoutID, &sc->inoutID, &temp_int);
	}
	return;
}
#endif
