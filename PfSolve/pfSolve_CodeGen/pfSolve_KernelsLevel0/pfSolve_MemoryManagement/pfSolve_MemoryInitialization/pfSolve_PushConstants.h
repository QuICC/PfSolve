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
#ifndef PFSOLVE_PUSHCONSTANTS_H
#define PFSOLVE_PUSHCONSTANTS_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"
static inline void appendPushConstant(PfSolveSpecializationConstantsLayout* sc, PfContainer* var) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (var->type > 100) {
		PfContainer* varType;
		PfGetTypeFromCode(sc, var->type, &varType);
		sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", varType->name, var->name);
		PfAppendLine(sc);
	}
	else {
		sc->res = PFSOLVE_ERROR_MATH_FAILED;
	}
	return;
}
static inline void appendPushConstants(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (sc->pushConstantsStructSize == 0)
		return;
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "layout(push_constant) uniform PushConsts\n{\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);

#endif
	char tempCopyStr[60];
	if (sc->performWorkGroupShift[0]) {
		appendPushConstant(sc, &sc->workGroupShiftX);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftX.name);
		sprintf(sc->workGroupShiftX.name, "%s", tempCopyStr);
	}
	if (sc->performWorkGroupShift[1]) {
		appendPushConstant(sc, &sc->workGroupShiftY);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftY.name);
		sprintf(sc->workGroupShiftY.name, "%s", tempCopyStr);
	}
	if (sc->performWorkGroupShift[2]) {
		appendPushConstant(sc, &sc->workGroupShiftZ);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftZ.name);
		sprintf(sc->workGroupShiftZ.name, "%s", tempCopyStr);
	}
	if (sc->performPostCompilationInputOffset) {
		appendPushConstant(sc, &sc->inputOffset);
		sprintf(tempCopyStr, "consts.%s", sc->inputOffset.name);
		sprintf(sc->inputOffset.name, "%s", tempCopyStr);
	}
	if (sc->performPostCompilationOutputOffset) {
		appendPushConstant(sc, &sc->outputOffset);
		sprintf(tempCopyStr, "consts.%s", sc->outputOffset.name);
		sprintf(sc->outputOffset.name, "%s", tempCopyStr);
	}
	if (sc->performPostCompilationKernelOffset) {
		appendPushConstant(sc, &sc->kernelOffset);
		sprintf(tempCopyStr, "consts.%s", sc->kernelOffset.name);
		sprintf(sc->kernelOffset.name, "%s", tempCopyStr);
	}
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "} consts;\n\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);
	//sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	//PfAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);

	//sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	//PfAppendLine(sc);

#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);

#endif
	return;
}
static inline void getPushConstantsSize_jw(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer* uintType;
	PfGetTypeFromCode(sc, sc->uintTypeCode, &uintType);
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
	sc->pushConstantsStructSize = 0;
	if (sc->performWorkGroupShift[0]) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->performWorkGroupShift[1]) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->performWorkGroupShift[2]) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_MSIZE)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OFFSETM)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OFFSETV)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OFFSETSOLUTION)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_INPUTZEROPAD)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OUTPUTZEROPAD)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_INPUTBUFFERSTRIDE)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OUTPUTBUFFERSTRIDE)) {
		sc->pushConstantsStructSize += ((sc->uintTypeCode%100) / 10 == 0) ? sizeof(uint32_t) : sizeof(uint64_t);
	}
	if (sc->jw_control_bitmask & (RUNTIME_SCALEC)) {
		switch ((sc->floatTypeCode % 100) / 10) {
		case 0:
			sc->pushConstantsStructSize += 2;
			return;
		case 1:
			sc->pushConstantsStructSize += 4;
			return;
		case 2:
			sc->pushConstantsStructSize += 8;
			return;
		break;
		}
	}
	
	return;
}
static inline void appendPushConstants_jw(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return; 
	getPushConstantsSize_jw(sc);
	if (sc->pushConstantsStructSize == 0)
		return;
	PfContainer* intType;
	PfGetTypeFromCode(sc, sc->intTypeCode, &intType);
	PfContainer* floatType;
	PfGetTypeFromCode(sc, sc->floatTypeCode, &floatType);
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "layout(push_constant) uniform PushConsts\n{\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	PfAppendLine(sc);

#endif
	char tempCopyStr[60];
	if (sc->performWorkGroupShift[0]) {
		appendPushConstant(sc, &sc->workGroupShiftX);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftX.name);
		sprintf(sc->workGroupShiftX.name, "%s", tempCopyStr);
	}
	if (sc->performWorkGroupShift[1]) {
		appendPushConstant(sc, &sc->workGroupShiftY);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftY.name);
		sprintf(sc->workGroupShiftY.name, "%s", tempCopyStr);
	}
	if (sc->performWorkGroupShift[2]) {
		appendPushConstant(sc, &sc->workGroupShiftZ);
		sprintf(tempCopyStr, "consts.%s", sc->workGroupShiftZ.name);
		sprintf(sc->workGroupShiftZ.name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_MSIZE)) {
		appendPushConstant(sc, &sc->M_size);
		sprintf(tempCopyStr, "consts.%s", sc->M_size.name);
		sprintf(sc->M_size.name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OFFSETM)) {
		appendPushConstant(sc, &sc->offsetM);
		sprintf(tempCopyStr, "consts.%s", sc->offsetM.name);
		sprintf(sc->offsetM.name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OFFSETV)) {
		appendPushConstant(sc, &sc->offsetV);
		sprintf(tempCopyStr, "consts.%s", sc->offsetV.name);
		sprintf(sc->offsetV.name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OFFSETSOLUTION)) {
		appendPushConstant(sc, &sc->offsetSolution);
		sprintf(tempCopyStr, "consts.%s", sc->offsetSolution.name);
		sprintf(sc->offsetSolution.name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_INPUTZEROPAD)) {
		appendPushConstant(sc, &sc->inputZeropad[0]);
		appendPushConstant(sc, &sc->inputZeropad[1]);
		sprintf(tempCopyStr, "consts.%s", sc->inputZeropad[0].name);
		sprintf(sc->inputZeropad[0].name, "%s", tempCopyStr);
		sprintf(tempCopyStr, "consts.%s", sc->inputZeropad[1].name);
		sprintf(sc->inputZeropad[1].name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OUTPUTZEROPAD)) {
		appendPushConstant(sc, &sc->outputZeropad[0]);
		appendPushConstant(sc, &sc->outputZeropad[1]);
		sprintf(tempCopyStr, "consts.%s", sc->outputZeropad[0].name);
		sprintf(sc->outputZeropad[0].name, "%s", tempCopyStr);
		sprintf(tempCopyStr, "consts.%s", sc->outputZeropad[1].name);
		sprintf(sc->outputZeropad[1].name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_INPUTBUFFERSTRIDE)) {
		appendPushConstant(sc, &sc->inputStride[1]);
		sprintf(tempCopyStr, "consts.%s", sc->inputStride[1].name);
		sprintf(sc->inputStride[1].name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_OUTPUTBUFFERSTRIDE)) {
		appendPushConstant(sc, &sc->outputStride[1]);
		sprintf(tempCopyStr, "consts.%s", sc->outputStride[1].name);
		sprintf(sc->outputStride[1].name, "%s", tempCopyStr);
	}
	if (sc->jw_control_bitmask & (RUNTIME_SCALEC)) {
		appendPushConstant(sc, &sc->scaleC);
		sprintf(tempCopyStr, "consts.%s", sc->scaleC.name);
		sprintf(sc->scaleC.name, "%s", tempCopyStr);
	}


#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "} consts;\n\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);
	sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);

	sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	PfAppendLine(sc);

#elif(VKFFT_BACKEND==3)
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	PfAppendLine(sc);

#endif
	return;
}
#endif
