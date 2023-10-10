// This file is part of PfSolve
//
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// This file is provided for informational purposes only. Redistribution without permission is not allowed.
#ifndef PFSOLVE_FINITEDIFFERENCES_H
#define PFSOLVE_FINITEDIFFERENCES_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_InputOutputLayout.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_SharedMemory.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_Registers.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_MemoryManagement/pfSolve_MemoryInitialization/pfSolve_PushConstants.h"

#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_compute_fluxD.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel1/pfSolve_compute_Pf.h"

#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelUtils.h"
#include "pfSolve_CodeGen/pfSolve_KernelsLevel0/pfSolve_KernelStartEnd.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"
/*static inline PfSolveResult PfSolve_shaderGen_compute_fluxD(PfSolveSpecializationConstantsLayout* sc) {
	appendVersion(sc);
	
	appendExtensions(sc);
	
	appendInputLayout(sc);
	
	appendOutputLayout(sc);
	
	appendPushConstants(sc);
	
#if(VKFFT_BACKEND==0)
	//appendSharedMemoryInitialization(sc);
	
#endif
	appendKernelStart_compute_fluxD(sc);
	
#if(VKFFT_BACKEND!=0)
	//appendSharedMemoryInitialization(sc);
	
#endif


	appendRegistersInitialization_compute_fluxD(sc);
	
	
	appendGlobalToRegisters_compute_fluxD(sc);
	
	appendComputeFluxD_x(sc);
	
	appendComputeFluxD_y(sc);
	
	appendComputeFluxD_z(sc);
	

	appendKernelEnd(sc);
	
	return sc->res;
}*/

static inline PfSolveResult PfSolve_shaderGen_compute_Pf(PfSolveSpecializationConstantsLayout* sc) {
	appendVersion(sc);
	
	appendExtensions(sc);
	if ((((sc->floatTypeCode/10)%10) == 3) || (((sc->floatTypeInputMemoryCode/10)%10) == 3) || (((sc->floatTypeOutputMemoryCode/10)%10) == 3)) {
		appendQuadDoubleDoubleStruct(sc);
	}
	if ((sc->floatTypeCode != sc->floatTypeInputMemoryCode) || (sc->floatTypeCode != sc->floatTypeOutputMemoryCode)) {
		appendConversion(sc);
	}	
	appendPushConstants(sc);

	int id = 0;
	appendInputLayoutPfSolve(sc, id, 0);
	id++;
	appendOutputLayoutPfSolve(sc, id, 0);
	
#if(VKFFT_BACKEND==0)
	//appendSharedMemoryInitialization(sc);
	
#endif
	appendKernelStart_compute_Pf(sc);
	
#if(VKFFT_BACKEND!=0)
	//appendSharedMemoryInitialization(sc);
	
#endif


	appendRegistersInitialization_compute_Pf(sc, 0);
	
	appendGlobalToRegisters_compute_Pf(sc);
	
	appendComputePf_2(sc);
	

	appendKernelEnd(sc);
	
	freeRegisterInitialization_Pf(sc, 0);
	return sc->res;
}

#endif
