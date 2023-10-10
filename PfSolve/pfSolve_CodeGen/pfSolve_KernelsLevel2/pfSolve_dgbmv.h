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
#ifndef PFSOLVE_DGBMV_H
#define PFSOLVE_DGBMV_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_InputOutputLayout.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_SharedMemory.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_Registers.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_ReadWrite.h"

//#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_RegistersScaleC.h"

#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelUtils.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelStartEnd.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"

static inline PfSolveResult PfSolve_shaderGen_dgbmv(PfSolveSpecializationConstantsLayout* sc) {
	
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

	appendKernelStart_jw(sc, 0);
	
	appendRegistersInitialization_dgbmv(sc);

	sc->offset_md.type = 31;
	sc->offset_md.data.i = 0;
	sc->offset_res.type = 31;
	sc->offset_res.data.i = 0;//2 * sc->M_size.data.i;
	
	sc->offset_ld.type = 31;
	sc->offset_ud.type = 31;

	sc->offset_md_global.type = 31;
	sc->offset_ud_global.type = 31;
	sc->offset_ld_global.type = 31;

	if (sc->offsetSolution.type > 100) {
		PfAllocateContainerFlexible(sc, &sc->offset_res_global, 50);
		sc->offset_res_global.type = sc->offsetSolution.type;
		PfCopyContainer(sc, &sc->offset_res_global, &sc->offsetSolution);
	}
	else {
		sc->offset_res_global.type = 31;
		sc->offset_res_global.data.i = sc->offsetSolution.data.i;
	}

	if (sc->offsetM.type > 100) {
		PfAllocateContainerFlexible(sc, &sc->offset_md_global, 50);
		sc->offset_md_global.type = sc->offsetM.type;
		PfCopyContainer(sc, &sc->offset_md_global, &sc->offsetM);
	}
	else {
		sc->offset_md_global.type = 31;
		sc->offset_md_global.data.i = sc->offsetM.data.i;
	}

	sc->readToRegisters = 0;
	appendReadWrite_rd(sc, 0);
	
	appendBarrierPfSolve(sc);
	
	appendMatVecMul_fromGlobal(sc);
	
	sc->writeFromRegisters = 1;
	appendReadWrite_rd(sc, 1);
	
	appendKernelEnd(sc);
	freeRegistersInitialization_dgbmv(sc, 0);
	if (sc->offsetM.type > 100) {
		PfDeallocateContainer(sc, &sc->offset_md_global);
	}
	if (sc->offsetSolution.type > 100) {
		PfDeallocateContainer(sc, &sc->offset_res_global);
	}

	return sc->res;
}
#endif
