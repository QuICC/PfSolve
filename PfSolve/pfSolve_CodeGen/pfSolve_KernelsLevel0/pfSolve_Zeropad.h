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
#ifndef PFSOLVE_ZEROPAD_H
#define PFSOLVE_ZEROPAD_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"

static inline void PfCheckZeropadStart(PfSolveSpecializationConstantsLayout* sc, PfContainer* location, int axisCheck) {
	//return if sequence is full of zeros from the start
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {
			if (axisCheck == 0) {
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0].data.i < sc->fft_zeropad_right_full[0].data.i) {
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[0]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[0]);
					}
				}
			}
			break;
		}
		case 2: {
			if (axisCheck == 0) {
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0].data.i < sc->fft_zeropad_right_full[0].data.i) {
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[0]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[0]);
					}
				}
			}
			if (axisCheck == 1) {
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1].data.i < sc->fft_zeropad_right_full[1].data.i) {
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[1]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[1]);
					}
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			if (axisCheck == 1) {
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1].data.i < sc->fft_zeropad_right_full[1].data.i) {
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[1]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[1]);
					}
				}
			}
			if (axisCheck == 2) {
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2].data.i < sc->fft_zeropad_right_full[2].data.i) {
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[2]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[2]);
					}
				}
			}
			break;
		}
		case 1: {
			if (axisCheck == 2) {
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2].data.i < sc->fft_zeropad_right_full[2].data.i) {
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[2]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[2]);
					}
				}
			}
			break;
		}
		case 2: {

			break;
		}
		}
	}
	return;
}
static inline void PfCheckZeropadEnd(PfSolveSpecializationConstantsLayout* sc, int axisCheck) {
	//return if sequence is full of zeros from the start
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {
			if (axisCheck == 0) {
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0].data.i < sc->fft_zeropad_right_full[0].data.i) {
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		case 2: {
			if (axisCheck == 0) {
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0].data.i < sc->fft_zeropad_right_full[0].data.i) {
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			if (axisCheck == 1) {
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1].data.i < sc->fft_zeropad_right_full[1].data.i) {
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			if (axisCheck == 1) {
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1].data.i < sc->fft_zeropad_right_full[1].data.i) {
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			if (axisCheck == 2) {
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2].data.i < sc->fft_zeropad_right_full[2].data.i) {
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		case 1: {
			if (axisCheck == 2) {
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2].data.i < sc->fft_zeropad_right_full[2].data.i) {
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		case 2: {

			break;
		}
		}
	}
	return;
}

static inline void PfCheckZeropad(PfSolveSpecializationConstantsLayout* sc, PfContainer* location, int axisCheck) {
	//return if sequence is full of zeros from the start
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {
			if (axisCheck == 0) {
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0].data.i < sc->fft_zeropad_right_full[0].data.i) {
						sc->useDisableThreads = 1;
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[0]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[0]);
						temp_int.data.i = 0;
						PfMov(sc, &sc->disableThreads, &temp_int);
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		case 2: {
			if (axisCheck == 0) {
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0].data.i < sc->fft_zeropad_right_full[0].data.i) {
						sc->useDisableThreads = 1; 
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[0]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[0]);
						temp_int.data.i = 0;
						PfMov(sc, &sc->disableThreads, &temp_int);
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			if (axisCheck == 1) {
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1].data.i < sc->fft_zeropad_right_full[1].data.i) {
						sc->useDisableThreads = 1; 
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[1]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[1]);
						temp_int.data.i = 0;
						PfMov(sc, &sc->disableThreads, &temp_int);
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			if (axisCheck == 1) {
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1].data.i < sc->fft_zeropad_right_full[1].data.i) {
						sc->useDisableThreads = 1; 
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[1]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[1]);
						temp_int.data.i = 0;
						PfMov(sc, &sc->disableThreads, &temp_int);
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			if (axisCheck == 2) {
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2].data.i < sc->fft_zeropad_right_full[2].data.i) {
						sc->useDisableThreads = 1; 
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[2]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[2]);
						temp_int.data.i = 0;
						PfMov(sc, &sc->disableThreads, &temp_int);
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		case 1: {
			if (axisCheck == 2) {
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2].data.i < sc->fft_zeropad_right_full[2].data.i) {
						sc->useDisableThreads = 1; 
						PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[2]);
						PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[2]);
						temp_int.data.i = 0;
						PfMov(sc, &sc->disableThreads, &temp_int);
						PfIf_end(sc);
						PfIf_end(sc);
					}
				}
			}
			break;
		}
		case 2: {

			break;
		}
		}
	}
	return;
}

static inline void PfConfigureZeropad(PfSolveSpecializationConstantsLayout* sc, PfContainer* location, int* control, int readWrite) {
	//return if sequence is full of zeros from the start
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer temp_int = {};
	temp_int.type = 31;
	PfSetToZero(sc, &sc->tempInt);
	temp_int.data.i = 1;
	if (readWrite) {
		if (sc->outputZeropad[0].type > 100) {
			PfIf_ge_start(sc, location, &sc->outputZeropad[0]);
			PfIf_lt_start(sc, location, &sc->outputZeropad[1]);
			PfMov(sc, &sc->tempInt, &temp_int);
			PfIf_end(sc);
			PfIf_end(sc);
			control[0] = 1;
		}
		else {
			if (sc->outputZeropad[0].data.i) {
				PfIf_ge_start(sc, location, &sc->outputZeropad[0]);
				if (sc->outputZeropad[1].data.i) {
					PfIf_lt_start(sc, location, &sc->outputZeropad[1]);
				}
				PfMov(sc, &sc->tempInt, &temp_int);
				if (sc->outputZeropad[1].data.i) {
					PfIf_end(sc);
				}
				PfIf_end(sc);
				control[0] = 1;
			}
			else {
				if (sc->outputZeropad[1].data.i) {
					PfIf_lt_start(sc, location, &sc->outputZeropad[1]);
					PfMov(sc, &sc->tempInt, &temp_int);
					PfIf_end(sc);
					control[0] = 1;
				}
			}
		}
	}
	else {
		if (sc->inputZeropad[0].type > 100) {
			PfIf_ge_start(sc, location, &sc->inputZeropad[0]);
			PfIf_lt_start(sc, location, &sc->inputZeropad[1]);
			PfMov(sc, &sc->tempInt, &temp_int);
			PfIf_end(sc);
			PfIf_end(sc);
			control[0] = 1;
		}
		else {
			if (sc->inputZeropad[0].data.i) {
				PfIf_ge_start(sc, location, &sc->inputZeropad[0]);
				if (sc->inputZeropad[1].data.i) {
					PfIf_lt_start(sc, location, &sc->inputZeropad[1]);
				}
				PfMov(sc, &sc->tempInt, &temp_int);
				if (sc->inputZeropad[1].data.i) {
					PfIf_end(sc);
				}
				PfIf_end(sc);
				control[0] = 1;
			}
			else {
				if (sc->inputZeropad[1].data.i) {
					PfIf_lt_start(sc, location, &sc->inputZeropad[1]);
					PfMov(sc, &sc->tempInt, &temp_int);
					PfIf_end(sc);
					control[0] = 1;
				}
			}
		}
	}
	return;
}

#endif
