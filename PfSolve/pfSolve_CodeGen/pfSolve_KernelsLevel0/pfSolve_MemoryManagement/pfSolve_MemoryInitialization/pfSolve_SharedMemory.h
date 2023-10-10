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
#ifndef PFSOLVE_SHAREDMEMORY_H
#define PFSOLVE_SHAREDMEMORY_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"

static inline void appendSharedMemoryPfSolve(PfSolveSpecializationConstantsLayout* sc, int type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType); 
#if(VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "shared %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", floatType->name, sc->usedSharedMemory.data.i / sc->complexSize);
		PfAppendLine(sc);
		
#elif(VKFFT_BACKEND==1)
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", floatType->name, floatType->name);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType);
#elif(VKFFT_BACKEND==2)
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType, sc->localSize[1] * sc->maxSharedStride);
		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", floatType->name, floatType->name);
		PfAppendLine(sc);
		
		//sc->tempLen = sprintf(sc->tempStr, "%s %s sdata[];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", sharedDefinitions, vecType);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
		sc->tempLen = sprintf(sc->tempStr, "__local %s sdata[%" PRIu64 "];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts\n\n", vecType->name, sc->usedSharedMemory.data.i / sc->complexSize);
		PfAppendLine(sc);
		
#endif
	return;
}
#endif
