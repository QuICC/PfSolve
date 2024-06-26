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
#ifndef PFSOLVE_MATHUTILS_H
#define PFSOLVE_MATHUTILS_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"

static inline void PfPrintReg(PfSolveSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* in);

static inline void appendBarrierPfSolve(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
#if(VKFFT_BACKEND==0)
	sc->tempLen = sprintf(sc->tempStr, "barrier();\n\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==1)
	sc->tempLen = sprintf(sc->tempStr, "__syncthreads();\n\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==2)
	sc->tempLen = sprintf(sc->tempStr, "__syncthreads();\n\n");
	PfAppendLine(sc);
#elif((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
	sc->tempLen = sprintf(sc->tempStr, "barrier(CLK_LOCAL_MEM_FENCE);\n\n");
	PfAppendLine(sc);
#elif(VKFFT_BACKEND==5)
	sc->tempLen = sprintf(sc->tempStr, "threadgroup_barrier(mem_flags::mem_none);\n\n");
	PfAppendLine(sc);
#endif
	return;
}

//register manipulation functions: mov, add, sub, etc.
static inline void PfCopyContainer(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfCopyContainer(sc, &out->data.dd[0], &in->data.dd[0]);
		PfCopyContainer(sc, &out->data.dd[1], &in->data.dd[1]);
	}
	if (out->type > 100) {
		if (in->type > 100) {
			if (out->type == in->type) {
				int len = 0;
				len = sprintf(out->name, "%s", in->name);
				if (len > out->size) sc->res = PFSOLVE_ERROR_MATH_FAILED;

				switch (out->type % 10) {
				case 3:
					PfCopyContainer(sc, &out->data.c[0], &in->data.c[0]);
					PfCopyContainer(sc, &out->data.c[1], &in->data.c[1]);
					return;
				}
				return;
			}
		}
		else {
		}
	}
	else {
		if (in->type > 100) {
		}
		else {
			if (out->type == in->type) {
				switch (out->type % 10) {
				case 1:
					out->data.i = in->data.i;
					return;
				case 2:
					out->data.d = in->data.d;
					return;
				case 3:
					out->data.c[0].data.d = in->data.c[0].data.d;
					out->data.c[1].data.d = in->data.c[1].data.d;
					return;
				}
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfSwapContainers(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfSwapContainers(sc, &out->data.dd[0], &in->data.dd[0]);
		PfSwapContainers(sc, &out->data.dd[1], &in->data.dd[1]);
	}
	if (out->type > 100) {
		if (in->type > 100) {
			if (out->type == in->type) {
				int len = in->size;
				in->size = out->size;
				out->size = len;

				char* temp = in->name;
				in->name = out->name;
				out->name = temp;

				switch (out->type % 10) {
				case 3:
					PfSwapContainers(sc, &out->data.c[0], &in->data.c[0]);
					PfSwapContainers(sc, &out->data.c[1], &in->data.c[1]);
					return;
				}
				return;
			}
		}
		else {
		}
	}
	else {
		if (in->type > 100) {
		}
		else {
			if (out->type == in->type) {
				switch (out->type % 10) {
				case 1:
				{
					pfINT temp;
					temp = in->data.i;
					in->data.i = out->data.i;
					out->data.i = temp;
					return;
				}
				case 2:
				{
					pfLD temp;
					temp = in->data.d;
					in->data.d = out->data.d;
					out->data.d = temp;
					return;
				}
				case 3:
				{
					pfLD temp;
					temp = in->data.c[0].data.d;
					in->data.c[0].data.d = out->data.c[0].data.d;
					out->data.c[0].data.d = temp;

					temp = in->data.c[1].data.d;
					in->data.c[1].data.d = out->data.c[1].data.d;
					out->data.c[1].data.d = temp;
					return;
				}
				}
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfAllocateContainerFlexible(PfSolveSpecializationConstantsLayout* sc, PfContainer* container, int size) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (container->size != 0) return;

	if (container->type > 100){
		container->name = (char*)calloc(size, sizeof(char));
		container->size = size;

		if (container->name == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;
	}
	if(container->type < 200){
		if ((((container->type % 100) / 10) == 3) && ((container->type % 10) == 2)) {
			if (container->data.dd == 0) container->data.dd = (PfContainer*) calloc(2, sizeof(PfContainer));
			if (container->data.dd == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;
			container->data.dd[0].type = container->type-10;
			container->data.dd[1].type = container->type-10;
			PfAllocateContainerFlexible(sc, &container->data.dd[0], 50);
			PfAllocateContainerFlexible(sc, &container->data.dd[1], 50);
		}
		else if ((container->type % 10) == 3){
			if (container->data.c == 0) container->data.c = (PfContainer*) calloc(2, sizeof(PfContainer));
			if (container->data.c == 0) sc->res = PFSOLVE_ERROR_MALLOC_FAILED;
			container->data.c[0].type = container->type-1;
			container->data.c[1].type = container->type-1;
			PfAllocateContainerFlexible(sc, &container->data.c[0], 50);
			PfAllocateContainerFlexible(sc, &container->data.c[1], 50);
		}
	}
	return;
}

static inline void PfDeallocateContainer(PfSolveSpecializationConstantsLayout* sc, PfContainer* container) {
	if (container->type > 0) {
		if (container->type > 100) {
			if (container->name)
				free(container->name);
			container->name = 0;
		}
		container->size = 0;
		container->type = 0;
		if(container->type < 200){
			if ((((container->type % 100) / 10) == 3) && ((container->type % 10) == 2)) {
				PfDeallocateContainer(sc, &container->data.dd[0]);
				PfDeallocateContainer(sc, &container->data.dd[1]);
				if (container->data.dd)
					free(container->data.dd);
				container->data.dd = 0;
			}
			else if ((container->type % 10) == 3){
				PfDeallocateContainer(sc, &container->data.c[0]);
				PfDeallocateContainer(sc, &container->data.c[1]);
				if (container->data.c)
					free(container->data.c);
				container->data.c = 0;
			}
		}
	}
	return;
}

static inline void PfConvToDoubleDouble(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((in->type > 100) || (((in->type % 100) / 10) == 3)) {
		if(out->type==0){
			out->type = in->type;
			PfAllocateContainerFlexible(sc, out, 50);
		}
		PfCopyContainer(sc, out, in);
		return;
	}else{
		if(out->type==0){
			out->type = in->type + 10;
			PfAllocateContainerFlexible(sc, out, 50);
		}
		double high, low;
		if ((in->type % 10)== 2) {
			high = (double) in->data.d;
		    if (isnan (high) || isinf (high)){
		    	low = 0.0;
		    }else{
		    	low = (double) (in->data.d - (pfLD)high);
		    	double temp = high + low;
		    	low = (high - temp) + low;
		    	high = temp;
		    }
		    out->data.dd[0].data.d = high;
			out->data.dd[1].data.d = low;
		}
		return;
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfGetTypeFromCode(PfSolveSpecializationConstantsLayout* sc, int code, PfContainer** type) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	switch (code % 10) {
	case 1:
		switch ((code % 100) / 10) {
		case 0:
			type[0] = &sc->uintDef;
			return;
		case 1:
			type[0] = &sc->intDef;
			return;
		case 2:
			type[0] = &sc->uint64Def;
			return;
		case 3:
			type[0] = &sc->int64Def;
			return;
		}
		break;
	case 2:
		switch ((code % 100) / 10) {
		case 0:
			type[0] = &sc->halfDef;
			return;
		case 1:
			type[0] = &sc->floatDef;
			return;
		case 2:
			type[0] = &sc->doubleDef;
			return;
		case 3:
			type[0] = &sc->quadDef;
			return;
		}
		break;
	case 3:
		switch ((code % 100) / 10) {
		case 0:
			type[0] = &sc->half2Def;
			return;
		case 1:
			type[0] = &sc->float2Def;
			return;
		case 2:
			type[0] = &sc->double2Def;
			return;
		case 3:
			type[0] = &sc->quad2Def;
			return;
		}
		break;
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfAppendNumberLiteral(PfSolveSpecializationConstantsLayout* sc, PfContainer* number) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (((number->type % 10) == 2) || ((number->type % 10) == 3)) {
		switch ((number->type % 100) / 10) {
		case 0:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->halfLiteral.name);
			PfAppendLine(sc);
			return;
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->floatLiteral.name);
			PfAppendLine(sc);
			return;
		case 2:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->doubleLiteral.name);
			PfAppendLine(sc);
			return;
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "%s", sc->doubleLiteral.name);
			PfAppendLine(sc);
			return;
		}
	}
	return;
}
static inline void PfAppendConversionStart(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (((out->type % 100) / 10) == ((in->type % 100) / 10))
		return;
	if ((out->type < 100) || (in->type < 100))
		return;
	switch (in->type % 10) {
	case 1:
		return;
	case 2:
		switch ((out->type % 100) / 10) {
		case 0:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "float16_t(");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
			sc->tempLen = sprintf(sc->tempStr, "(half)");
			PfAppendLine(sc);
#elif(VKFFT_BACKEND==5)
			sc->tempLen = sprintf(sc->tempStr, "half(");
			PfAppendLine(sc);
#endif
			return;
		case 1:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "float(");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
			sc->tempLen = sprintf(sc->tempStr, "(float)");
			PfAppendLine(sc);
#endif
			return;
		case 2:
			switch ((in->type % 100) / 10) {
			case 0: case 1: case 2:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
				sc->tempLen = sprintf(sc->tempStr, "double(");
				PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
				sc->tempLen = sprintf(sc->tempStr, "(double)");
				PfAppendLine(sc);
#endif
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "conv_pf_quad_to_double(");
				PfAppendLine(sc);
				return;
			}
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "conv_double_to_pf_quad(");
			PfAppendLine(sc);
			return;
		}
	case 3:
		switch ((out->type % 100) / 10) {
		case 0:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "f16vec2(");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "conv_half2(");
			PfAppendLine(sc);
#endif
			return;
		case 1:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "vec2(");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "conv_float2(");
			PfAppendLine(sc);
#endif
			return;
		case 2:
			switch ((in->type % 100) / 10) {
			case 0: case 1: case 2:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "dvec2(");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, "conv_double2(");
			PfAppendLine(sc);
#endif
			return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "conv_pf_quad2_to_double2(");
				PfAppendLine(sc);
				return;
			}
		case 3:
			sc->tempLen = sprintf(sc->tempStr, "conv_double2_to_pf_quad2(");
			PfAppendLine(sc);
			return;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfAppendConversionEnd(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (((out->type % 100) / 10) == ((in->type % 100) / 10))
		return;
	if ((out->type < 100) || (in->type < 100))
		return;
	switch (in->type % 10) {
	case 1:
		return;
	case 2:
		switch ((out->type % 100) / 10) {
		case 0:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
			return;
		case 1:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
			return;
		case 2:
			switch ((in->type % 100) / 10) {
			case 0: case 1: case 2:
#if((VKFFT_BACKEND==0)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4))
#endif
			return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, ")");
				PfAppendLine(sc);
				return;
			}
		case 3:
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
			return;
		}
	case 3:
		switch ((out->type % 100) / 10) {
		case 0:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#endif
			return;
		case 1:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#endif
			return;
		case 2:
			switch ((in->type % 100) / 10) {
			case 0: case 1: case 2:
#if(VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#elif((VKFFT_BACKEND==1)||(VKFFT_BACKEND==2)||(VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
#endif
			return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, ")");
				PfAppendLine(sc);
				return;
			}
		case 3:
			sc->tempLen = sprintf(sc->tempStr, ")");
			PfAppendLine(sc);
			return;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfSetContainerName(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, const char* name) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		sprintf(out->name, "%s", name);
		if(out->type < 200){
			if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)){
				sprintf(out->data.dd[0].name, "%s.x", name);
				sprintf(out->data.dd[1].name, "%s.y", name);
			}else{
				if (((out->type % 10) == 3) && (out->type > 100)) {
					sprintf(out->data.c[0].name, "%s.x", name);
					sprintf(out->data.c[1].name, "%s.y", name);
					if (((out->type % 100) / 10) == 3){
						sprintf(out->data.c[0].data.dd[0].name, "%s.x.x", name);
						sprintf(out->data.c[0].data.dd[1].name, "%s.x.y", name);
						sprintf(out->data.c[1].data.dd[0].name, "%s.y.x", name);
						sprintf(out->data.c[1].data.dd[1].name, "%s.y.y", name);
					}
				}
			}
		}
		return;
	}
	return;
}
static inline void PfDefine(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, const char* name) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		PfSetContainerName(sc, out, name);
		switch (out->type % 10) {
		case 1:
			switch ((out->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->uintDef.name, name);
				PfAppendLine(sc);
				return;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->intDef.name, name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->uint64Def.name, name);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->int64Def.name, name);
				PfAppendLine(sc);
				return;
			}
			break;
		case 2:
			switch ((out->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->halfDef.name, name);
				PfAppendLine(sc);
				return;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->floatDef.name, name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->doubleDef.name, name);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->quadDef.name, name);
				PfAppendLine(sc);
				return;
			}
			break;
		case 3:
			switch ((out->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->half2Def.name, name);
				PfAppendLine(sc);
				return;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->float2Def.name, name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->double2Def.name, name);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s %s;\n", sc->quad2Def.name, name);
				PfAppendLine(sc);
				return;
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfDefineConstant(PfSolveSpecializationConstantsLayout* sc, PfContainer* name, PfContainer* value) {
	//needs to be fixed for double-double
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (name->type > 100) {
		switch (name->type % 10) {
		case 1:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->uintDef.name, name->name);
				PfAppendLine(sc);
				break;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->intDef.name, name->name);
				PfAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->uint64Def.name, name->name);
				PfAppendLine(sc);
				break;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->int64Def.name, name->name);
				PfAppendLine(sc);
				break;
			}
			break;
		case 2:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->halfDef.name, name->name);
				PfAppendLine(sc);
				break;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->floatDef.name, name->name);
				PfAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->doubleDef.name, name->name);
				PfAppendLine(sc);
				break;
			}
			break;
		case 3:
			switch ((name->type % 100) / 10) {
			case 0:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->half2Def.name, name->name);
				PfAppendLine(sc);
				break;
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->float2Def.name, name->name);
				PfAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%s %s %s", sc->constDef.name, sc->double2Def.name, name->name);
				PfAppendLine(sc);
				break;
			}
			break;
		}
		if (value->type < 100) {
			sc->tempLen = sprintf(sc->tempStr, " = ");
			PfAppendLine(sc);
			switch (value->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", value->data.i);
				PfAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)value->data.d);
				PfAppendLine(sc);
				break;
			case 3:
				//fix
				sc->res = PFSOLVE_ERROR_MATH_FAILED;
				break;
			}
			PfAppendNumberLiteral(sc, name);
			sc->tempLen = sprintf(sc->tempStr, ";");
			PfAppendLine(sc);
			return;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfSetToZero(PfSolveSpecializationConstantsLayout* sc, PfContainer* out) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	//out
	if ((out->type % 10) == 3){
		PfSetToZero(sc, &out->data.c[0]);
		PfSetToZero(sc, &out->data.c[1]);
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfSetToZero(sc, &out->data.dd[0]);
		PfSetToZero(sc, &out->data.dd[1]);
		return;
	}
	else{
		if (out->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
			PfAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			PfAppendLine(sc);
			switch (out->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "0");
				PfAppendLine(sc);
				break;
			case 2: case 3:
				sc->tempLen = sprintf(sc->tempStr, "0.0");
				PfAppendLine(sc);
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
			return;
		}
		else {
			switch (out->type % 10) {
			case 1:
				out->data.i = 0;
				return;
			case 2:
				out->data.d = 0;
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfSetToZeroShared(PfSolveSpecializationConstantsLayout* sc, PfContainer* sdataID) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if(sc->storeSharedComplexComponentsSeparately){
		if ((((sc->sdataStruct.type % 100) / 10) == 3) && ((sc->sdataStruct.type % 10) > 1)) {
			if (sdataID->type > 100) {
				switch (sc->sdataStruct.type % 10){
				case 2: 
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = 0;\n\
sdata[%s].y = 0;\n", sdataID->name, sdataID->name);
					PfAppendLine(sc);
					return;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = 0;\n\
sdata[%s].y = 0;\n", sdataID->name, sdataID->name);
					PfAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s + %" PRIi64 "].x = 0;\n\
sdata[%s + %" PRIi64 "].y = 0;\n", sdataID->name, sc->offsetImaginaryShared.data.i, sdataID->name, sc->offsetImaginaryShared.data.i);
					PfAppendLine(sc);
					return;
				}
			}
			else {
				switch (sc->sdataStruct.type % 10){
				case 2: 
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i, sdataID->data.i);
					PfAppendLine(sc);
					return;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i, sdataID->data.i);
					PfAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i + sc->offsetImaginaryShared.data.i, sdataID->data.i + sc->offsetImaginaryShared.data.i);
					PfAppendLine(sc);
					return;
				}
			}
		}
		if (sdataID->type > 100) {
			switch (sc->sdataStruct.type % 10){
			case 2: 
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s] = 0;\n", sdataID->name);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = 0;\n\
sdata[%s + %" PRIi64 "].y = 0;\n", sdataID->name, sdataID->name, sc->offsetImaginaryShared.data.i);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (sc->sdataStruct.type % 10){
			case 2: 
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "] = 0;\n", sdataID->data.i);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i, sdataID->data.i + sc->offsetImaginaryShared.data.i);
				PfAppendLine(sc);
				return;
			}
		}
	}else{
		if ((((sc->sdataStruct.type % 100) / 10) == 3) && ((sc->sdataStruct.type % 10) > 1)) {
			if (sdataID->type > 100) {
				switch (sc->sdataStruct.type % 10){
				case 2: 
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = 0;\n\
sdata[%s].y = 0;\n", sdataID->name, sdataID->name);
					PfAppendLine(sc);
					return;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x.x = 0;\n\
sdata[%s].x.y = 0;\n", sdataID->name, sdataID->name);
					PfAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].y.x = 0;\n\
sdata[%s].y.y = 0;\n", sdataID->name, sdataID->name);
					PfAppendLine(sc);
					return;
				}
			}
			else {
				switch (sc->sdataStruct.type % 10){
				case 2: 
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i, sdataID->data.i);
					PfAppendLine(sc);
					return;
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x.x = 0;\n\
sdata[%" PRIi64 "].x.y = 0;\n", sdataID->data.i, sdataID->data.i);
					PfAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].y.x = 0;\n\
sdata[%" PRIi64 "].y.y = 0;\n", sdataID->data.i, sdataID->data.i);
					PfAppendLine(sc);
					return;
				}
			}
		}
		if (sdataID->type > 100) {
			switch (sc->sdataStruct.type % 10){
			case 2: 
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s] = 0;\n", sdataID->name);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s].x = 0;\n\
sdata[%s].y = 0;\n", sdataID->name, sdataID->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (sc->sdataStruct.type % 10){
			case 2: 
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "] = 0;\n", sdataID->data.i);
				PfAppendLine(sc);
				return;
			case 3:
				sc->tempLen = sprintf(sc->tempStr, "\
sdata[%" PRIi64 "].x = 0;\n\
sdata[%" PRIi64 "].y = 0;\n", sdataID->data.i, sdataID->data.i);
				PfAppendLine(sc);
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfMov(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
		if ((out->type > 100) && (in->type > 100) && ((out->type % 10) == (in->type % 10))) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
			PfAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, "%s", in->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
			return;
		}
		PfMov(sc, &out->data.c[0], &in->data.c[0]);
		PfMov(sc, &out->data.c[1], &in->data.c[1]);
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfContainer temp = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp, in);
		PfMov(sc, &out->data.dd[0], &temp.data.dd[0]);
		PfMov(sc, &out->data.dd[1], &temp.data.dd[1]);
		PfDeallocateContainer(sc, &temp);
		return;
	}
	if (out->type > 100) {
		if ((out->type > 100) && (in->type > 100) && ((out->type % 10) == (in->type % 10))) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
			PfAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, "%s", in->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
			return;
		}
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		PfAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "%s", in->name);
			PfAppendLine(sc);
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in->data.i);
				PfAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in->data.d);
				PfAppendLine(sc);
				break;
			}
			PfAppendNumberLiteral(sc, out);
		}
		PfAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		PfAppendLine(sc);
		return;
	}
	else {
		if (in->type > 100) {
		}
		else {
			switch (out->type % 10) {
			case 1:
				switch (in->type % 10) {
				case 1:
					out->data.i = in->data.i;
					return;
				case 2:
					out->data.i = (pfINT)in->data.d;
					return;
				}
				return;
			case 2:
				switch (in->type % 10) {
				case 1:
					out->data.d = (double)in->data.i;
					return;
				case 2:
					out->data.d = in->data.d;
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfMovNeg(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
		PfMovNeg(sc, &out->data.c[0], &in->data.c[0]);
		PfMovNeg(sc, &out->data.c[1], &in->data.c[1]);
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfContainer temp = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp, in);
		PfMovNeg(sc, &out->data.dd[0], &temp.data.dd[0]);
		PfMovNeg(sc, &out->data.dd[1], &temp.data.dd[1]);
		PfDeallocateContainer(sc, &temp);
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		PfAppendConversionStart(sc, out, in);
		if (in->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "-%s", in->name);
			PfAppendLine(sc);
		}
		else {
			switch (in->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in->data.i);
				PfAppendLine(sc);
				break;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (-in->data.d));
				PfAppendLine(sc);
				break;
			}
			PfAppendNumberLiteral(sc, out);
		}
		PfAppendConversionEnd(sc, out, in);
		sc->tempLen = sprintf(sc->tempStr, ";\n");
		PfAppendLine(sc);
		return;
	}
	else {
		if (in->type > 100) {
		}
		else {
			switch (out->type % 10) {
			case 1:
				switch (in->type % 10) {
				case 1:
					out->data.i = -in->data.i;
					return;
				case 2:
					out->data.i = (pfINT)-in->data.d;
					return;
				}
				return;
			case 2:
				switch (in->type % 10) {
				case 1:
					out->data.d = (double)-in->data.i;
					return;
				case 2:
					out->data.d = -in->data.d;
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfAdd(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2);

static inline void PfSub(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2);

static inline void PfQuadQuickSum(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {// double-double, double, double
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			if ((in_2->type % 10) == 3){
				PfQuadQuickSum(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0]);
				PfQuadQuickSum(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[1]);
			}else{
				PfQuadQuickSum(sc, &out->data.c[0], &in_1->data.c[0], in_2);
				PfQuadQuickSum(sc, &out->data.c[1], &in_1->data.c[1], in_2);
			}
		}else{
			if ((in_2->type % 10) == 3){
				PfQuadQuickSum(sc, &out->data.c[0], in_1, &in_2->data.c[0]);
				PfQuadQuickSum(sc, &out->data.c[1], in_1, &in_2->data.c[1]);
			}else{
				PfQuadQuickSum(sc, &out->data.c[0], in_1, in_2);
				PfMov(sc, &out->data.c[1], &out->data.c[0]);
			}
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfAdd(sc, &out->data.dd[0], in_1, in_2);
		PfSub(sc, &out->data.dd[1], &out->data.dd[0], in_1);
		PfSub(sc, &out->data.dd[1], in_2, &out->data.dd[1]);
	}
	return;
}

static inline void PfQuadSum(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {// double-double, double, double
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			if ((in_2->type % 10) == 3){
				PfQuadSum(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0], temp);
				PfQuadSum(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[1], temp);
			}else{
				PfQuadSum(sc, &out->data.c[0], &in_1->data.c[0], in_2, temp);
				PfQuadSum(sc, &out->data.c[1], &in_1->data.c[1], in_2, temp);
			}
		}else{
			if ((in_2->type % 10) == 3){
				PfQuadSum(sc, &out->data.c[0], in_1, &in_2->data.c[0], temp);
				PfQuadSum(sc, &out->data.c[1], in_1, &in_2->data.c[1], temp);
			}else{
				PfQuadSum(sc, &out->data.c[0], in_1, in_2, temp);
				PfMov(sc, &out->data.c[1], &out->data.c[0]);
			}
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfAdd(sc, &out->data.dd[0], in_1, in_2);
		PfSub(sc, &out->data.dd[1], &out->data.dd[0], in_1);
		PfSub(sc, temp, &out->data.dd[0], &out->data.dd[1]);
		PfSub(sc, temp, in_1, temp);
		PfSub(sc, &out->data.dd[1], in_2, &out->data.dd[1]);
		PfAdd(sc, &out->data.dd[1], &out->data.dd[1], temp);
	}
	return;
}

static inline void PfAdd(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100) && (((out->type % 100) / 10) != 3)) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
			PfAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
			return;
		}
#endif
		if ((in_2->type % 10) == 3){
			PfAdd(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0]);
			PfAdd(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[1]);
		}else{
			PfAdd(sc, &out->data.c[0], &in_1->data.c[0], in_2);
			PfAdd(sc, &out->data.c[1], &in_1->data.c[1], in_2);
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfContainer temp1 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp1, in_1);
		PfContainer temp2 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp2, in_2);
		PfQuadSum(sc, &sc->tempQuad.data.c[0], &temp1.data.dd[0], &temp2.data.dd[0], &sc->tempQuad3.data.c[0].data.dd[0]);
		PfAdd(sc, &out->data.dd[0], &temp1.data.dd[1], &temp2.data.dd[1]);
		PfAdd(sc, &sc->tempQuad.data.c[0].data.dd[1], &sc->tempQuad.data.c[0].data.dd[1], &out->data.dd[0]);
		PfQuadQuickSum(sc, out, &sc->tempQuad.data.c[0].data.dd[0], &sc->tempQuad.data.c[0].data.dd[1]);
		PfDeallocateContainer(sc, &temp1);
		PfDeallocateContainer(sc, &temp2);
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1: 
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i + in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i + in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d + (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d + in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i + in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i + (pfINT)in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (pfINT)in_1->data.d + in_2->data.i;
							return;
						case 2:
							out->data.i = (pfINT)(in_1->data.d + in_2->data.d);
							return;
						}
					}
					break;
				}
			}
		break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (pfLD)(in_1->data.i + in_2->data.i);
							return;
						case 2:
							out->data.d = (pfLD)in_1->data.i + in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d + (pfLD)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d + in_2->data.d;
							return;
						}
					}
					break;
				}
			}
		break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfInc(PfSolveSpecializationConstantsLayout* sc, PfContainer* out) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
		case 2:
			sc->tempLen = sprintf(sc->tempStr, "\
%s = %s + 1;\n", out->name, out->name);
			PfAppendLine(sc);
			return;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			out->data.i = out->data.i + 1;
			return;
		case 2:
			out->data.d = out->data.d + 1;
			return;
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfQuadQuickDiff(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {// double-double, double, double
	if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfSub(sc, &out->data.dd[0], in_1, in_2);
		PfSub(sc, &out->data.dd[1], in_1, &out->data.dd[0]);
		PfSub(sc, &out->data.dd[1], &out->data.dd[1], in_2);
	}
	return;
}

static inline void PfQuadDiff(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {// double-double, double, double
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			if ((in_2->type % 10) == 3){
				PfQuadDiff(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0], temp);
				PfQuadDiff(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[1], temp);
			}else{
				PfQuadDiff(sc, &out->data.c[0], &in_1->data.c[0], in_2, temp);
				PfQuadDiff(sc, &out->data.c[1], &in_1->data.c[1], in_2, temp);
			}
		}else{
			if ((in_2->type % 10) == 3){
				PfQuadDiff(sc, &out->data.c[0], in_1, &in_2->data.c[0], temp);
				PfQuadDiff(sc, &out->data.c[1], in_1, &in_2->data.c[1], temp);
			}else{
				PfQuadDiff(sc, &out->data.c[0], in_1, in_2, temp);
				PfMov(sc, &out->data.c[1], &out->data.c[0]);
			}
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfSub(sc, &out->data.dd[0], in_1, in_2);
		PfSub(sc, &out->data.dd[1], &out->data.dd[0], in_1);
		PfSub(sc, temp, &out->data.dd[0], &out->data.dd[1]);
		PfSub(sc, temp, in_1, temp);
		PfAdd(sc, &out->data.dd[1], in_2, &out->data.dd[1]);
		PfSub(sc, &out->data.dd[1], temp, &out->data.dd[1]);
	}
	return;
}

static inline void PfSub(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100) && (((out->type % 100) / 10) != 3)) {
			//packed instructions workaround if all values are in registers
			sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
			PfAppendLine(sc);
			sc->tempLen = sprintf(sc->tempStr, " = ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
			return;
		}
#endif
		if ((in_2->type % 10) == 3){
			PfSub(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0]);
			PfSub(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[1]);
		}else{
			PfSub(sc, &out->data.c[0], &in_1->data.c[0], in_2);
			PfSub(sc, &out->data.c[1], &in_1->data.c[1], in_2);
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfContainer temp1 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp1, in_1);
		PfContainer temp2 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp2, in_2);
		PfQuadDiff(sc, &sc->tempQuad.data.c[0], &temp1.data.dd[0], &temp2.data.dd[0], &sc->tempQuad3.data.c[0].data.dd[0]);
		PfSub(sc, &out->data.dd[0], &temp1.data.dd[1], &temp2.data.dd[1]);
		PfAdd(sc, &sc->tempQuad.data.c[0].data.dd[1], &sc->tempQuad.data.c[0].data.dd[1], &out->data.dd[0]);
		PfQuadQuickSum(sc, out, &sc->tempQuad.data.c[0].data.dd[0], &sc->tempQuad.data.c[0].data.dd[1]);
		PfDeallocateContainer(sc, &temp1);
		PfDeallocateContainer(sc, &temp2);
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i - in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i - in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d - (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d - in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " - ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i - in_2->data.i;
							return;
						case 2:
							out->data.i = in_1->data.i - (pfINT)in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (pfINT)in_1->data.d - in_2->data.i;
							return;
						case 2:
							out->data.i = (pfINT)(in_1->data.d - in_2->data.d);
							return;

						}
					}
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (pfLD)(in_1->data.i - in_2->data.i);
							return;
						case 2:
							out->data.d = (pfLD)in_1->data.i - in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d - (pfLD)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d - in_2->data.d;
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfIf_eq_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right);
static inline void PfIf_gt_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right);
static inline void PfIf_lt_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right);
static inline void PfIf_else(PfSolveSpecializationConstantsLayout* sc);
static inline void PfIf_end(PfSolveSpecializationConstantsLayout* sc);

static inline void PfMul(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp);
static inline void PfMulNeg(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp);
static inline void PfDiv(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2);
static inline void PfDivCeil(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2);

static inline void PfQuadSplit(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* temp) {// double-double, double, double
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			PfQuadSplit(sc, &out->data.c[0], &in_1->data.c[0], temp);
			PfQuadSplit(sc, &out->data.c[1], &in_1->data.c[1], temp);
		}else{
			PfQuadSplit(sc, &out->data.c[0], in_1, temp);
			PfQuadSplit(sc, &out->data.c[1], in_1, temp);
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		if (in_1->type > 100){
			PfContainer temp_double = PFSOLVE_ZERO_INIT;
			temp_double.type = 22;
			temp_double.data.d = pfFPinit("3.7252902984619140625e-09"); // 2^-28
			PfContainer temp_int = PFSOLVE_ZERO_INIT;
			temp_int.type = 31;
			/*PfSetToZero(sc, &sc->tempIntQuad);
			temp_int.data.i = 134217729; // 2^27+1
			PfIf_gt_start(sc, in_1, &temp_int);
			temp_int.data.i = 1;
			PfMov(sc, &sc->tempIntQuad, &temp_int);
			PfIf_else(sc);
			temp_int.data.i = -134217729; // 2^27+1
			PfIf_lt_start(sc, in_1, &temp_int);
			temp_int.data.i = 1;
			PfMov(sc, &sc->tempIntQuad, &temp_int);
			PfIf_end(sc);
			PfIf_end(sc);
			PfIf_eq_start(sc, &sc->tempIntQuad, &temp_int);

			temp_double.data.d = pfFPinit("3.7252902984619140625e-09"); // 2^-28
			PfMul(sc, &out->data.dd[1], in_1, &temp_double, 0);
			temp_double.data.d = pfFPinit("134217729.0"); // 2^27+1
			PfMul(sc, temp, &out->data.dd[1], &temp_double, 0);
			PfSub(sc, &out->data.dd[0], temp, &out->data.dd[1]);
			PfSub(sc, &out->data.dd[0], temp, &out->data.dd[0]);
			PfSub(sc, &out->data.dd[1], &out->data.dd[1], &out->data.dd[0]);
			temp_int.data.i = 268435456; // 2^27+1
			PfMul(sc, &out->data.dd[0], &out->data.dd[0], &temp_int, 0);
			PfMul(sc, &out->data.dd[1], &out->data.dd[1], &temp_int, 0);

			PfIf_else(sc);*/

			temp_double.data.d =  pfFPinit("134217729.0"); // 2^27+1
			PfMul(sc, temp, in_1, &temp_double, 0);
			PfSub(sc, &out->data.dd[0], temp, in_1);
			PfSub(sc, &out->data.dd[0], temp, &out->data.dd[0]);
			PfSub(sc, &out->data.dd[1], in_1, &out->data.dd[0]);
			
			//PfIf_end(sc);
		}else{
			PfContainer temp_double = PFSOLVE_ZERO_INIT;
			temp_double.type = 22;
			temp_double.data.d = pfFPinit("3.7252902984619140625e-09"); // 2^-28
			PfContainer temp_int = PFSOLVE_ZERO_INIT;
			temp_int.type = 31;
			double temp_double2;
			double temp_double3;
			double temp_double4;

			if ((in_1->data.d > 134217729) || (in_1->data.d < -134217729)){

			temp_double.data.d = pfFPinit("3.7252902984619140625e-09"); // 2^-28
			temp_double2 = ((double)in_1->data.d) * (double)temp_double.data.d;
			temp_double3 = temp_double2 * 134217729;
			temp_double4 = temp_double3 - temp_double2;
			temp_double4 = temp_double3 - temp_double4;
			temp_double2 = temp_double2 - temp_double4;
			temp_double.data.d = temp_double4 * 268435456;
			PfMov(sc, &out->data.dd[0], &temp_double);
			temp_double.data.d = temp_double3 * 268435456;
			PfMov(sc, &out->data.dd[1], &temp_double);

			}else{

			temp_double3 = ((double)in_1->data.d) * 134217729;
			temp_double4 = temp_double3 - ((double)in_1->data.d);
			temp_double3 = temp_double3 - temp_double4;
			temp_double.data.d = temp_double3;
			PfMov(sc, &out->data.dd[0], &temp_double);
			temp_double3 = ((double)in_1->data.d) - temp_double3;
			temp_double.data.d = temp_double3;
			PfMov(sc, &out->data.dd[1], &temp_double);

			}
		}
	}
	return;
}

static inline void PfFMA(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* in_3);

static inline void PfQuadProd(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1,  PfContainer* in_2, PfContainer* temp) {// double-double, double, double
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			if ((in_2->type % 10) == 3){
				PfQuadProd(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0], &temp->data.c[0]);
				PfQuadProd(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[1], &temp->data.c[1]);
			}else{
				PfQuadProd(sc, &out->data.c[0], &in_1->data.c[0], in_2, &temp->data.c[0]);
				PfQuadProd(sc, &out->data.c[1], &in_1->data.c[1], in_2, &temp->data.c[1]);
			}
		}else{
			if ((in_2->type % 10) == 3){
				PfQuadProd(sc, &out->data.c[0], in_1, &in_2->data.c[0], &temp->data.c[0]);
				PfQuadProd(sc, &out->data.c[1], in_1, &in_2->data.c[1], &temp->data.c[1]);
			}else{
				PfQuadProd(sc, &out->data.c[0], in_1, in_2, &temp->data.c[0]);
				PfMov(sc, &out->data.c[1], &out->data.c[0]);
			}
		}
		return;
	}
	if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		//v1
		/*PfMul(sc, &out->data.dd[0], in_1, in_2, 0);
		PfQuadSplit(sc, &temp->data.c[0], in_1, &out->data.dd[1]);
		PfQuadSplit(sc, &sc->tempQuad2.data.c[0], in_2, &out->data.dd[1]);

		//PfMovNeg(sc, &sc->tempQuad2.data.c[1].data.dd[1], &out->data.dd[0]);
		//PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[0], &sc->tempQuad2.data.c[0].data.dd[0],  &sc->tempQuad2.data.c[1].data.dd[1]);
		PfMul(sc, &out->data.dd[1], &temp->data.c[0].data.dd[0], &sc->tempQuad2.data.c[0].data.dd[0], 0);
		PfSub(sc, &out->data.dd[1], &out->data.dd[1], &out->data.dd[0]);

		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[0], &sc->tempQuad2.data.c[0].data.dd[1], &out->data.dd[1]);
		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[1], &sc->tempQuad2.data.c[0].data.dd[0], &out->data.dd[1]);
		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[1], &sc->tempQuad2.data.c[0].data.dd[1], &out->data.dd[1]);*/

		//v2
		/*PfMul(sc, &out->data.dd[0], in_1, in_2, 0);
		PfQuadSplit(sc, &temp->data.c[0], in_1, &out->data.dd[1]);
		PfQuadSplit(sc, &sc->tempQuad2.data.c[0], in_2, &out->data.dd[1]);
		//important
		PfMovNeg(sc, &sc->tempQuad2.data.c[1].data.dd[1], in_2);
		PfFMA(sc, &sc->tempQuad2.data.c[1].data.dd[0], &sc->tempQuad2.data.c[1].data.dd[1], in_1, &out->data.dd[0]);

		PfMovNeg(sc, &sc->tempQuad2.data.c[1].data.dd[1], &out->data.dd[0]);
		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[0], &sc->tempQuad2.data.c[0].data.dd[0], &sc->tempQuad2.data.c[1].data.dd[1]);
		//PfPrintReg(sc, &sc->inoutID, &sc->tempQuad2.data.c[1].data.dd[0]);
		PfAdd(sc, &out->data.dd[1], &out->data.dd[1], &sc->tempQuad2.data.c[1].data.dd[0]);
		//PfSub(sc, &out->data.dd[1], &out->data.dd[1], &sc->tempQuad2.data.c[1].data.dd[0]);
		//PfPrintReg(sc, &sc->inoutID, &out->data.dd[1]);

		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[0], &sc->tempQuad2.data.c[0].data.dd[1], &out->data.dd[1]);
		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[1], &sc->tempQuad2.data.c[0].data.dd[0], &out->data.dd[1]);
		PfFMA(sc, &out->data.dd[1], &temp->data.c[0].data.dd[1], &sc->tempQuad2.data.c[0].data.dd[1], &out->data.dd[1]);*/

		//v3
		PfMul(sc, &out->data.dd[0], in_1, in_2, 0);
		PfMovNeg(sc, &out->data.dd[1], &out->data.dd[0]);
		PfFMA(sc, &out->data.dd[1], in_1, in_2, &out->data.dd[1]);
	}
	return;
}


static inline void PfFMA(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* in_3) {
	//fma inlining is not correct if all three numbers are complex for now
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100) && (in_3->type > 100) && (((out->type % 100) / 10) != 3)) {
			
			//packed instructions workaround if all values are in registers
			if (((in_1->type % 10) != 3) || ((in_2->type % 10) != 3)) {
				sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
				PfAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, " = ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, " + ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_3->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				PfAppendLine(sc);
				return;
			}
		}
#endif
		if ((in_3->type % 10) == 3){
			if ((in_2->type % 10) == 3){
				if ((in_1->type % 10) == 3){
					sc->res = PFSOLVE_ERROR_MATH_FAILED;
				}else{
					PfFMA(sc, &out->data.c[0], in_1, &in_2->data.c[0], &in_3->data.c[0]);
					PfFMA(sc, &out->data.c[1], in_1, &in_2->data.c[1], &in_3->data.c[1]);
				}
			}else{
				if ((in_1->type % 10) == 3){
					PfFMA(sc, &out->data.c[0], &in_1->data.c[0], in_2, &in_3->data.c[0]);
					PfFMA(sc, &out->data.c[1], &in_1->data.c[1], in_2, &in_3->data.c[1]);
				}else{
					if ((((out->type % 100) / 10) == 3)){
						PfMul(sc, &out->data.c[0], in_1, in_2, &out->data.c[1]);
						PfAdd(sc, &out->data.c[1], &out->data.c[0], &in_3->data.c[1]);
						PfAdd(sc, &out->data.c[0], &out->data.c[0], &in_3->data.c[0]);
					}else {
						PfFMA(sc, &out->data.c[0], in_1, in_2, &in_3->data.c[0]);
						PfFMA(sc, &out->data.c[1], in_1, in_2, &in_3->data.c[1]);
					}
				}
			}
		}else{
			if ((in_2->type % 10) == 3){
				if ((in_1->type % 10) == 3){
					sc->res = PFSOLVE_ERROR_MATH_FAILED;
				}else{
					PfFMA(sc, &out->data.c[0], in_1, &in_2->data.c[0], in_3);
					PfFMA(sc, &out->data.c[1], in_1, &in_2->data.c[1], in_3);
				}
			}else{
				if ((in_1->type % 10) == 3){
					PfFMA(sc, &out->data.c[0], &in_1->data.c[0], in_2, in_3);
					PfFMA(sc, &out->data.c[1], &in_1->data.c[1], in_2, in_3);
				}else{
					PfFMA(sc, &out->data.c[0], in_1, in_2, in_3);
					PfMov(sc, &out->data.c[1], &out->data.c[0]);
				}
			}
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfMul(sc, &sc->tempQuad.data.c[1], in_1, in_2, 0);
		PfMov(sc, &sc->tempQuad2.data.c[1], &sc->tempQuad.data.c[1]);
		PfAdd(sc, out, &sc->tempQuad2.data.c[1], in_3);
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100) && (in_3->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i + in_3->data.i);
						PfAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)(in_1->data.i * in_2->data.i) + in_3->data.d));
						PfAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i * in_2->data.d + (pfLD)in_3->data.i));
						PfAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i * in_2->data.d + in_3->data.d));
						PfAppendLine(sc);
						break;
					}
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * (pfLD)in_2->data.i + (pfLD)in_3->data.i));
						PfAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * in_2->data.i + in_3->data.d));
						PfAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * in_2->data.d + (pfLD)in_3->data.i));
						PfAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * in_2->data.d + in_3->data.d));
						PfAppendLine(sc);
						break;
					}
					break;
				}
				break;
			case 3:
				switch (in_2->type % 10) {
				case 1:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.c[0].data.d * (pfLD)in_2->data.i + (pfLD)in_3->data.i));
						PfAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.c[0].data.d * in_2->data.i + in_3->data.d));
						PfAppendLine(sc);
						break;
					}
					break;
				case 2:
					switch (in_3->type % 10) {
					case 1:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.c[0].data.d * in_2->data.d + (pfLD)in_3->data.i));
						PfAppendLine(sc);
						break;
					case 2:
						sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.c[0].data.d * in_2->data.d + in_3->data.d));
						PfAppendLine(sc);
						break;
					}
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else if ((in_1->type < 100) && (in_2->type < 100) && (in_3->type > 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i * in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, " + ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_3);
			sc->tempLen = sprintf(sc->tempStr, "%s", in_3->name);
			PfAppendLine(sc);
			PfAppendConversionEnd(sc, out, in_3);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "fma(");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, ", ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ", ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_3);
			if (in_3->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_3->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_3->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_3->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_3->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_3);
			sc->tempLen = sprintf(sc->tempStr, ");\n");
			PfAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = in_1->data.i * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.i = in_1->data.i * in_2->data.i + (pfINT)in_3->data.d;
									return;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = (pfINT)(in_1->data.i * in_2->data.d + in_3->data.i);
									return;
								case 2:
									out->data.i = (pfINT)(in_1->data.i * in_2->data.d + in_3->data.d);
									return;
								}
							}
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = (pfINT)(in_1->data.d * in_2->data.i + in_3->data.i);
									return;
								case 2:
									out->data.i = (pfINT)(in_1->data.d * in_2->data.i + in_3->data.d);
									return;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.i = (pfINT)(in_1->data.d * in_2->data.d + in_3->data.i);
									return;
								case 2:
									out->data.i = (pfINT)(in_1->data.d * in_2->data.d + in_3->data.d);
									return;
								}
							}
							break;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = (pfLD)(in_1->data.i * in_2->data.i + in_3->data.i);
									return;
								case 2:
									out->data.d = (pfLD)(in_1->data.i * in_2->data.i + in_3->data.d);
									return;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.i * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.i * in_2->data.d + in_3->data.d;
									return;
								}
							}
							break;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.d * in_2->data.i + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.d * in_2->data.i + in_3->data.d;
									return;
								}
							}
							break;
						case 2:
							if (in_3->type > 100) {
							}
							else {
								switch (in_3->type % 10) {
								case 1:
									out->data.d = in_1->data.d * in_2->data.d + in_3->data.i;
									return;
								case 2:
									out->data.d = in_1->data.d * in_2->data.d + in_3->data.d;
									return;
								}
							}
							break;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfMul(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100) && (((out->type % 100) / 10) != 3)) {
			//packed instructions workaround if all values are in registers
			if (((in_1->type % 10) != 3) || ((in_2->type % 10) != 3)) {
				sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
				PfAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, " = ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				PfAppendLine(sc);
				return;
			}
			else {
				if ((((out->type % 100) / 10) < 2) && (out->type == in_1->type) && (out->type == in_2->type)) {
					if ((strcmp(out->name, in_1->name)) && (strcmp(out->name, in_2->name))) {
						PfMovNeg(sc, &out->data.c[0], &in_1->data.c[1]);
						PfMov(sc, &out->data.c[1], &in_1->data.c[0]);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[1].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);
						
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[0].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " + ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);
					}
					else {
						PfMovNeg(sc, &temp->data.c[0], &in_1->data.c[1]);
						PfMov(sc, &temp->data.c[1], &in_1->data.c[0]);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[1].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);

						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[0].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " + ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);
					}
					return;
				}
			}
		}
#endif
		if ((in_2->type % 10) == 3){
			if ((in_1->type % 10) == 3){
				if ((in_1->type < 100) || (in_2->type < 100) || ((strcmp(out->name, in_1->name)) && (strcmp(out->name, in_2->name)))) {
					PfMul(sc, &out->data.c[0], &in_1->data.c[1], &in_2->data.c[1], 0);
					PfMovNeg(sc, &out->data.c[0], &out->data.c[0]);
					PfFMA(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0], &out->data.c[0]);

					PfMul(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[0], 0);
					PfFMA(sc, &out->data.c[1], &in_1->data.c[0], &in_2->data.c[1], &out->data.c[1]);
				}else{
					PfMul(sc, &temp->data.c[0], &in_1->data.c[1], &in_2->data.c[1], 0);
					PfMovNeg(sc, &temp->data.c[0], &temp->data.c[0]);
					PfFMA(sc, &temp->data.c[0], &in_1->data.c[0], &in_2->data.c[0], &temp->data.c[0]);

					PfMul(sc, &temp->data.c[1], &in_1->data.c[1], &in_2->data.c[0], 0);
					PfFMA(sc, &out->data.c[1], &in_1->data.c[0], &in_2->data.c[1], &temp->data.c[1]);
					PfMov(sc, &out->data.c[0], &temp->data.c[0]);
				}
			}else{
				PfMul(sc, &out->data.c[0], in_1, &in_2->data.c[0], 0);
				PfMul(sc, &out->data.c[1], in_1, &in_2->data.c[1], 0);
			}
		}else{
			if ((in_1->type % 10) == 3){
				PfMul(sc, &out->data.c[0], &in_1->data.c[0], in_2, 0);
				PfMul(sc, &out->data.c[1], &in_1->data.c[1], in_2, 0);
			}else{
				PfMul(sc, &out->data.c[0], in_1, in_2, 0);
				PfMov(sc, &out->data.c[1], &out->data.c[0]);
			}
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfContainer temp1 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp1, in_1);
		PfContainer temp2 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp2, in_2);
		
		PfQuadProd(sc, &sc->tempQuad.data.c[0], &temp1.data.dd[0], &temp2.data.dd[0], &sc->tempQuad3);
		PfFMA(sc, &sc->tempQuad.data.c[0].data.dd[1], &temp1.data.dd[0], &temp2.data.dd[1], &sc->tempQuad.data.c[0].data.dd[1]);
		PfFMA(sc, &sc->tempQuad.data.c[0].data.dd[1], &temp1.data.dd[1], &temp2.data.dd[0], &sc->tempQuad.data.c[0].data.dd[1]);
		PfQuadQuickSum(sc, out, &sc->tempQuad.data.c[0].data.dd[0], &sc->tempQuad.data.c[0].data.dd[1]);

		PfDeallocateContainer(sc, &temp1);
		PfDeallocateContainer(sc, &temp2);
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i * in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i * in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d * in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " * ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}

		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i * in_2->data.i;
							return;
						case 2:
							out->data.i = (pfINT)(in_1->data.i * in_2->data.d);
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (pfINT)(in_1->data.d * in_2->data.i);
							return;
						case 2:
							out->data.i = (pfINT)(in_1->data.d * in_2->data.d);
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (pfLD)(in_1->data.i * in_2->data.i);
							return;
						case 2:
							out->data.d = (pfLD)in_1->data.i * in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d * (pfLD)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d * in_2->data.d;
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfMulNeg(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
#if(VKFFT_BACKEND == 2)
		if ((in_1->type > 100) && (in_2->type > 100) && (((out->type % 100) / 10) != 3)) {
			//packed instructions workaround if all values are in registers
			if (((in_1->type % 10) != 3) || ((in_2->type % 10) != 3)) {
				sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
				PfAppendLine(sc);
				sc->tempLen = sprintf(sc->tempStr, " = ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, "(-%s)", in_1->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_1);
				sc->tempLen = sprintf(sc->tempStr, " * ");
				PfAppendLine(sc);
				PfAppendConversionStart(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
				PfAppendConversionEnd(sc, out, in_2);
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				PfAppendLine(sc);
				return;
			}
			else {
				if ((((out->type % 100) / 10) < 2) && (out->type == in_1->type) && (out->type == in_2->type)) {
					if ((strcmp(out->name, in_1->name)) && (strcmp(out->name, in_2->name))) {
						PfMovNeg(sc, &out->data.c[0], &in_1->data.c[1]);
						PfMov(sc, &out->data.c[1], &in_1->data.c[0]);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[1].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);
						
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "(-%s)", in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[0].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " - ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);
					}
					else {
						PfMovNeg(sc, &temp->data.c[0], &in_1->data.c[1]);
						PfMov(sc, &temp->data.c[1], &in_1->data.c[0]);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[1].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);

						sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " = ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "(-%s)", in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " * ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", in_2->data.c[0].name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, " - ");
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "%s", temp->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						PfAppendLine(sc);
					}
					return;
				}
			}
		}
#endif
		if ((in_2->type % 10) == 3){
			if ((in_1->type % 10) == 3){
				if ((in_1->type < 100) || (in_2->type < 100) || ((strcmp(out->name, in_1->name)) && (strcmp(out->name, in_2->name)))) {
					PfMul(sc, &out->data.c[0], &in_1->data.c[1], &in_2->data.c[1], 0);
					PfMovNeg(sc, &out->data.c[0], &out->data.c[0]);
					PfFMA(sc, &out->data.c[0], &in_1->data.c[0], &in_2->data.c[0], &out->data.c[0]);

					PfMul(sc, &out->data.c[1], &in_1->data.c[1], &in_2->data.c[0], 0);
					PfFMA(sc, &out->data.c[1], &in_1->data.c[0], &in_2->data.c[1], &out->data.c[1]);
					PfMovNeg(sc, out, out);
				}else{
					PfMul(sc, &temp->data.c[0], &in_1->data.c[1], &in_2->data.c[1], 0);
					PfMovNeg(sc, &temp->data.c[0], &temp->data.c[0]);
					PfFMA(sc, &temp->data.c[0], &in_1->data.c[0], &in_2->data.c[0], &temp->data.c[0]);

					PfMul(sc, &temp->data.c[1], &in_1->data.c[1], &in_2->data.c[0], 0);
					PfFMA(sc, &out->data.c[1], &in_1->data.c[0], &in_2->data.c[1], &temp->data.c[1]);
					PfMov(sc, &out->data.c[0], &temp->data.c[0]);
					PfMovNeg(sc, out, out);
				}
			}else{
				PfMulNeg(sc, &out->data.c[0], in_1, &in_2->data.c[0], 0);
				PfMulNeg(sc, &out->data.c[1], in_1, &in_2->data.c[1], 0);
			}
		}else{
			if ((in_1->type % 10) == 3){
				PfMulNeg(sc, &out->data.c[0], &in_1->data.c[0], in_2, 0);
				PfMulNeg(sc, &out->data.c[1], &in_1->data.c[1], in_2, 0);
			}else{
				PfMulNeg(sc, &out->data.c[0], in_1, in_2, 0);
				PfMov(sc, &out->data.c[1], &out->data.c[0]);
			}
		}
		return;
	}
	else if ((((out->type % 100) / 10) == 3) && ((out->type % 10) == 2)) {
		PfContainer temp1 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp1, in_1);
		PfContainer temp2 = PFSOLVE_ZERO_INIT;
		PfConvToDoubleDouble(sc, &temp2, in_2);
		
		PfQuadProd(sc, &sc->tempQuad.data.c[0], &temp1.data.dd[0], &temp2.data.dd[0], &sc->tempQuad3);
		PfFMA(sc, &sc->tempQuad.data.c[0].data.dd[1], &temp1.data.dd[0], &temp2.data.dd[1], &sc->tempQuad.data.c[0].data.dd[1]);
		PfFMA(sc, &sc->tempQuad.data.c[0].data.dd[1], &temp1.data.dd[1], &temp2.data.dd[0], &sc->tempQuad.data.c[0].data.dd[1]);
		PfQuadQuickSum(sc, out, &sc->tempQuad.data.c[0].data.dd[0], &sc->tempQuad.data.c[0].data.dd[1]);
		PfMovNeg(sc, out, out);

		PfDeallocateContainer(sc, &temp1);
		PfDeallocateContainer(sc, &temp2);
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in_1->data.i * in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)-in_1->data.i * in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (-in_1->data.d * (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (-in_1->data.d * in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "(%s)", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", -in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) -in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " * ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "(-%s)", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}

		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = -in_1->data.i * in_2->data.i;
							return;
						case 2:
							out->data.i = (pfINT)(-in_1->data.i * in_2->data.d);
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (pfINT)(-in_1->data.d * in_2->data.i);
							return;
						case 2:
							out->data.i = (pfINT)(-in_1->data.d * in_2->data.d);
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (pfLD)(-in_1->data.i * in_2->data.i);
							return;
						case 2:
							out->data.d = (pfLD)-in_1->data.i * in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = -in_1->data.d * (pfLD)in_2->data.i;
							return;
						case 2:
							out->data.d = -in_1->data.d * in_2->data.d;
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfFMA3(PfSolveSpecializationConstantsLayout* sc, PfContainer* out_1, PfContainer* out_2, PfContainer* in_1, PfContainer* in_num, PfContainer* in_conj) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfFMA(sc, &out_1->data.c[0], &in_1->data.c[0], &in_num->data.c[0], &out_1->data.c[0]);
	PfFMA(sc, &out_1->data.c[1], &in_conj->data.c[1], &in_num->data.c[0], &out_1->data.c[1]);
	PfFMA(sc, &out_2->data.c[0], &in_1->data.c[1], &in_num->data.c[1], &out_2->data.c[0]);
	PfFMA(sc, &out_2->data.c[1], &in_conj->data.c[0], &in_num->data.c[1], &out_2->data.c[1]);
	/*out_1->data.c[0].data.d = in_1->data.c[0].data.d * in_num->data.c[0].data.d + out_1->data.c[0].data.d;
				out_1->data.c[1].data.d = in_conj->data.c[1].data.d * in_num->data.c[0].data.d + out_1->data.c[1].data.d;
				out_2->data.c[0].data.d = in_1->data.c[1].data.d * in_num->data.c[1].data.d + out_2->data.c[0].data.d;
				out_2->data.c[1].data.d = in_conj->data.c[0].data.d * in_num->data.c[1].data.d + out_2->data.c[1].data.d;
				*/
	return;
}
static inline void PfFMA3_const_w(PfSolveSpecializationConstantsLayout* sc, PfContainer* out_1, PfContainer* out_2, PfContainer* in_1, PfContainer* in_num_x, PfContainer* in_num_y, PfContainer* in_conj, PfContainer* temp, PfContainer* tempx) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out_1->type > 100) {
#if(VKFFT_BACKEND==2)
		if (((out_1->type%100)/10) < 2) {
			PfMov(sc, &temp->data.c[0], &in_1->data.c[0]);
			PfMov(sc, &temp->data.c[1], &in_conj->data.c[1]);
			PfFMA(sc, out_1, temp, in_num_x, out_1);

			PfMov(sc, &temp->data.c[0], &in_1->data.c[1]);
			PfMov(sc, &temp->data.c[1], &in_conj->data.c[0]);
			PfFMA(sc, out_2, temp, in_num_y, out_2);
			return;
		}
#endif
		//in_1 has to be same type as out
	}
	PfFMA(sc, &out_1->data.c[0], &in_1->data.c[0], in_num_x, &out_1->data.c[0]);
	PfFMA(sc, &out_1->data.c[1], &in_conj->data.c[1], in_num_x, &out_1->data.c[1]);
	PfFMA(sc, &out_2->data.c[0], &in_1->data.c[1], in_num_y, &out_2->data.c[0]);
	PfFMA(sc, &out_2->data.c[1], &in_conj->data.c[0], in_num_y, &out_2->data.c[1]);
	/*out_1->data.c[0].data.d = in_1->data.c[0].data.d * in_num_x->data.d + out_1->data.c[0].data.d;
				out_1->data.c[1].data.d = in_conj->data.c[1].data.d * in_num_x->data.d + out_1->data.c[1].data.d;
				out_2->data.c[0].data.d = in_1->data.c[1].data.d * in_num_y->data.d + out_2->data.c[0].data.d;
				out_2->data.c[1].data.d = in_conj->data.c[0].data.d * in_num_y->data.d + out_2->data.c[1].data.d;*/
	return;
}

//no quad implementation needed so far, will add later
static inline void PfDiv(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			PfDiv(sc, &out->data.c[0], &in_1->data.c[0], in_2);
			PfDiv(sc, &out->data.c[1], &in_1->data.c[1], in_2);
		}else{
			PfDiv(sc, &out->data.c[0], in_1, in_2);
			PfMov(sc, &out->data.c[1], &out->data.c[0]);
		}
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i / in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)((pfLD)in_1->data.i / in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d / (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) (in_1->data.d / in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " / ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
				sc->res = PFSOLVE_ERROR_MATH_FAILED;
			}
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i / in_2->data.i;
							return;
						case 2:
							out->data.i = (pfINT)(in_1->data.i / in_2->data.d);
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (pfINT)(in_1->data.d / in_2->data.i);
							return;
						case 2:
							out->data.i = (pfINT)(in_1->data.d / in_2->data.d);
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (pfLD)(in_1->data.i / in_2->data.i);
							return;
						case 2:
							out->data.d = (pfLD)in_1->data.i / in_2->data.d;
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = in_1->data.d / (pfLD)in_2->data.i;
							return;
						case 2:
							out->data.d = in_1->data.d / in_2->data.d;
							return;
						}
					}
					break;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfDivCeil(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if ((out->type % 10) == 3){
		if ((in_1->type % 10) == 3){
			PfDivCeil(sc, &out->data.c[0], &in_1->data.c[0], in_2);
			PfDivCeil(sc, &out->data.c[1], &in_1->data.c[1], in_2);
		}else{
			PfDivCeil(sc, &out->data.c[0], in_1, in_2);
			PfMov(sc, &out->data.c[1], &out->data.c[0]);
		}
		return;
	}
	if (out->type > 100) {
		sc->tempLen = sprintf(sc->tempStr, "%s", out->name);
		PfAppendLine(sc);
		sc->tempLen = sprintf(sc->tempStr, " = ");
		PfAppendLine(sc);
		if ((in_1->type < 100) && (in_2->type < 100)) {
			switch (in_1->type % 10) {
			case 1:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", (pfINT)pfceil(in_1->data.i / (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)pfceil((pfLD)in_1->data.i / in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			case 2:
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)pfceil(in_1->data.d / (pfLD)in_2->data.i));
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double)pfceil(in_1->data.d / in_2->data.d));
					PfAppendLine(sc);
					break;
				}
				break;
			}
			PfAppendNumberLiteral(sc, out);
			sc->tempLen = sprintf(sc->tempStr, ";\n");
			PfAppendLine(sc);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "ceil(");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_1);
			if (in_1->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_1->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_1->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_1->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_1);
			sc->tempLen = sprintf(sc->tempStr, " / ");
			PfAppendLine(sc);
			PfAppendConversionStart(sc, out, in_2);
			if (in_2->type > 100) {
				sc->tempLen = sprintf(sc->tempStr, "%s", in_2->name);
				PfAppendLine(sc);
			}
			else {
				switch (in_2->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "%" PRIi64 "", in_2->data.i);
					PfAppendLine(sc);
					break;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "%.17Le", (long double) in_2->data.d);
					PfAppendLine(sc);
					break;
				}
				PfAppendNumberLiteral(sc, out);
			}
			PfAppendConversionEnd(sc, out, in_2);
			if (((in_1->type % 10) == 3) && ((in_2->type % 10) == 3)) {
				sc->res = PFSOLVE_ERROR_MATH_FAILED;
			}
			sc->tempLen = sprintf(sc->tempStr, ");\n");
			PfAppendLine(sc);
		}
		return;
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i / in_2->data.i + (in_1->data.i % in_2->data.i != 0);
							return;
						case 2:
							out->data.i = (pfINT)pfceil(in_1->data.i / in_2->data.d);
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = (pfINT)pfceil(in_1->data.d / in_2->data.i);
							return;
						case 2:
							out->data.i = (pfINT)pfceil(in_1->data.d / in_2->data.d);
							return;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = (pfLD)(in_1->data.i / in_2->data.i + (in_1->data.i % in_2->data.i != 0));
							return;
						case 2:
							out->data.d = (pfLD)pfceil(in_1->data.i / in_2->data.d);
							return;
						}
					}
					break;
				case 2:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.d = pfceil(in_1->data.d / in_2->data.i);
							return;
						case 2:
							out->data.d = pfceil(in_1->data.d / in_2->data.d);
							return;
						}
					}
					break;
				case 3:
					break;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfMod(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s %% %s;\n", out->name, in_1->name, in_2->name);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s %% %" PRIi64 ";\n", out->name, in_1->name, in_2->data.i);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 " %% %s;\n", out->name, in_1->data.i, in_2->name);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 ";\n", out->name, in_1->data.i % in_2->data.i);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
		break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i % in_2->data.i;
							return;
						}
					}
				break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfAnd(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s && %s;\n", out->name, in_1->name, in_2->name);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s && %" PRIi64 ";\n", out->name, in_1->name, in_2->data.i);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 " && %s;\n", out->name, in_1->data.i, in_2->name);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %d;\n", out->name, in_1->data.i && in_2->data.i);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i && in_2->data.i;
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfOr(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s || %s;\n", out->name, in_1->name, in_2->name);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %s || %" PRIi64 ";\n", out->name, in_1->name, in_2->data.i);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %" PRIi64 " || %s;\n", out->name, in_1->data.i, in_2->name);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							sc->tempLen = sprintf(sc->tempStr, "\
%s = %d;\n", out->name, in_1->data.i || in_2->data.i);
							PfAppendLine(sc);
							return;
						case 2:
							break;
						case 3:
							break;
						}
					}
					break;
				case 2:
					break;
				case 3:
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	else {
		switch (out->type % 10) {
		case 1:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 1:
					if (in_2->type > 100) {
					}
					else {
						switch (in_2->type % 10) {
						case 1:
							out->data.i = in_1->data.i || in_2->data.i;
							return;
						}
					}
					break;
				}
			}
			break;
		case 2:
			break;
		case 3:
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}


static inline void PfSinCos(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 2:
					switch ((out->type / 10) % 10) {
					case 0: case 1:
#if(VKFFT_BACKEND==0)
						sc->tempLen = sprintf(sc->tempStr, "\
%s.x = cos(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sin(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
#elif ((VKFFT_BACKEND == 1) || (VKFFT_BACKEND == 2))
						sc->tempLen = sprintf(sc->tempStr, "\
__sincosf(%s, &%s.y, &%s.x);\n", in_1->name, out->name, out->name);
						PfAppendLine(sc);
#elif ((VKFFT_BACKEND == 3) || (VKFFT_BACKEND == 4))
						sc->tempLen = sprintf(sc->tempStr, "\
%s.x = native_cos(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = native_sin(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
#elif (VKFFT_BACKEND == 5)
						sc->tempLen = sprintf(sc->tempStr, "\
%s.x = cos(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sin(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
#endif
						return;
					case 2:
#if(VKFFT_BACKEND==0)
						sc->tempLen = sprintf(sc->tempStr, "\
%s = sincos20(%s);\n", out->name, in_1->name);
						PfAppendLine(sc);
#elif ((VKFFT_BACKEND == 1) || (VKFFT_BACKEND == 2))
						sc->tempLen = sprintf(sc->tempStr, "\
sincos(%s, &%s.y, &%s.x);\n", in_1->name, out->name, out->name);
						PfAppendLine(sc);
#elif ((VKFFT_BACKEND == 3) || (VKFFT_BACKEND == 4) || (VKFFT_BACKEND == 5))
						sc->tempLen = sprintf(sc->tempStr, "\
%s.y = sincos(%s, &%s.x);\n", out->name, in_1->name, out->name);
						PfAppendLine(sc);
#endif
						return;
					}
				}
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
%s.x = %.17Le;\n", out->name, (long double)pfcos(in_1->data.d));
					PfAppendLine(sc);
					sc->tempLen = sprintf(sc->tempStr, "\
%s.y = %.17Le;\n", out->name, (long double)pfsin(in_1->data.d));
					PfAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 3:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					out->data.c[0].data.d = pfcos(in_1->data.d);
					out->data.c[1].data.d = pfsin(in_1->data.d);
					return;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfNorm(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %s.x*%s.x + %s.y * %s.y;\n", out->name, in_1->name, in_1->name, in_1->name, in_1->name);
					PfAppendLine(sc);
				}
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %.17Le;\n", out->name, (long double)(in_1->data.c[0].data.d * in_1->data.c[0].data.d + in_1->data.c[1].data.d * in_1->data.c[1].data.d));
					PfAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 3:
					out->data.d = in_1->data.c[0].data.d * in_1->data.c[0].data.d + in_1->data.c[1].data.d * in_1->data.c[1].data.d;
					return;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfSqrt(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 2:
#if(VKFFT_BACKEND==0)
					sc->tempLen = sprintf(sc->tempStr, "\
%s = sqrt(%s);\n", out->name, in_1->name);
					PfAppendLine(sc);
#else
					sc->tempLen = sprintf(sc->tempStr, "\
%s = sqrt(%s);\n", out->name, in_1->name);
					PfAppendLine(sc);
#endif
					return;
				}
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %.17Le;\n", out->name, pfsqrt(in_1->data.d));
					PfAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					out->data.d = pfsqrt(in_1->data.d);
					return;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfRsqrt(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (out->type > 100) {
		//in_1 has to be same type as out
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
				switch (in_1->type % 10) {
				case 2:
#if(VKFFT_BACKEND==0)
					sc->tempLen = sprintf(sc->tempStr, "\
%s = inversesqrt(%s);\n", out->name, in_1->name);
					PfAppendLine(sc);
#else
					sc->tempLen = sprintf(sc->tempStr, "\
%s = rsqrt(%s);\n", out->name, in_1->name);
					PfAppendLine(sc);
#endif
				}
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
%s = %.17Le;\n", out->name, (long double)(pfFPinit("1.0") / pfsqrt(in_1->data.d)));
					PfAppendLine(sc);
					return;
				}
			}
		}
	}
	else {
		switch (out->type % 10) {
		case 2:
			if (in_1->type > 100) {
			}
			else {
				switch (in_1->type % 10) {
				case 2:
					out->data.d = pfFPinit("1.0") / pfsqrt(in_1->data.d);
					return;
				}
			}
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfConjugate(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (strcmp(out->name, in_1->name)) 
		PfMov(sc, &out->data.c[0], &in_1->data.c[0]);
	PfMovNeg(sc, &out->data.c[1], &in_1->data.c[1]);
	return;
}

static inline void PfShuffleComplex(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfMovNeg(sc, &temp->data.c[0], &in_2->data.c[1]);
	PfMov(sc, &temp->data.c[1], &in_2->data.c[0]);
	PfAdd(sc, out, in_1, temp);
	return;
}
static inline void PfShuffleComplexInv(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in_1, PfContainer* in_2, PfContainer* temp) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfMov(sc, &temp->data.c[0], &in_2->data.c[1]);
	PfMovNeg(sc, &temp->data.c[1], &in_2->data.c[0]);
	PfAdd(sc, out, in_1, temp);

	return;
}

//logic functions: if, ge, gt, le, lt, etc.
static inline void PfIf_eq_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s == %s) {\n", left->name, right->name);
			PfAppendLine(sc);
			return;
		}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s == %" PRIi64 ") {\n", left->name, right->data.i);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s == %.17Le) {\n", left->name, (long double)right->data.d);
				PfAppendLine(sc);
				return;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " == %s) {\n", left->data.i, right->name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le == %s) {\n", (long double)left->data.d, right->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i == right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i == right->data.d));
					PfAppendLine(sc);
					return;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d == right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d == right->data.d));
					PfAppendLine(sc);
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfIf_neq_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s != %s) {\n", left->name, right->name);
			PfAppendLine(sc);
			return;
		}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s != %" PRIi64 ") {\n", left->name, right->data.i);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s != %.17Le) {\n", left->name, (long double)right->data.d);
				PfAppendLine(sc);
				return;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " != %s) {\n", left->data.i, right->name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le != %s) {\n", (long double)left->data.d, right->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i != right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i != right->data.d));
					PfAppendLine(sc);
					return;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d != right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d != right->data.d));
					PfAppendLine(sc);
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfIf_lt_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s < %s) {\n", left->name, right->name);
			PfAppendLine(sc);
			return;
		}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s < %" PRIi64 ") {\n", left->name, right->data.i);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s < %.17Le) {\n", left->name, (long double)right->data.d);
				PfAppendLine(sc);
				return;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " < %s) {\n", left->data.i, right->name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le < %s) {\n", (long double)left->data.d, right->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i < right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i < right->data.d));
					PfAppendLine(sc);
					return;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d < right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d < right->data.d));
					PfAppendLine(sc);
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfIf_le_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s <= %s) {\n", left->name, right->name);
			PfAppendLine(sc);
			return;
}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s <= %" PRIi64 ") {\n", left->name, right->data.i);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s <= %.17Le) {\n", left->name, (long double)right->data.d);
				PfAppendLine(sc);
				return;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " <= %s) {\n", left->data.i, right->name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le <= %s) {\n", (long double)left->data.d, right->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i <= right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i <= right->data.d));
					PfAppendLine(sc);
					return;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d <= right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d <= right->data.d));
					PfAppendLine(sc);
					return;
				}
				return;
			case 3:
				break;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfIf_gt_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s > %s) {\n", left->name, right->name);
			PfAppendLine(sc);
			return;
}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s > %" PRIi64 ") {\n", left->name, right->data.i);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s > %.17Le) {\n", left->name, (long double)right->data.d);
				PfAppendLine(sc);
				return;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " > %s) {\n", left->data.i, right->name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le > %s) {\n", (long double)left->data.d, right->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i > right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i > right->data.d));
					PfAppendLine(sc);
					return;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d > right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d > right->data.d));
					PfAppendLine(sc);
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfIf_ge_start(PfSolveSpecializationConstantsLayout* sc, PfContainer* left, PfContainer* right) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (left->type > 100) {
		if (right->type > 100) {
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s >= %s) {\n", left->name, right->name);
			PfAppendLine(sc);
			return;
}
		else {
			switch (right->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s >= %" PRIi64 ") {\n", left->name, right->data.i);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%s >= %.17Le) {\n", left->name, (long double)right->data.d);
				PfAppendLine(sc);
				return;
			}
		}
	}
	else {
		if (right->type > 100) {
			switch (left->type % 10) {
			case 1:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 " >= %s) {\n", left->data.i, right->name);
				PfAppendLine(sc);
				return;
			case 2:
				sc->tempLen = sprintf(sc->tempStr, "\
if (%.17Le >= %s) {\n", (long double)left->data.d, right->name);
				PfAppendLine(sc);
				return;
			}
		}
		else {
			switch (left->type % 10) {
			case 1:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i >= right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.i >= right->data.d));
					PfAppendLine(sc);
					return;
				}
				break;
			case 2:
				switch (right->type % 10) {
				case 1:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d >= right->data.i));
					PfAppendLine(sc);
					return;
				case 2:
					sc->tempLen = sprintf(sc->tempStr, "\
if (%d) {\n", (left->data.d >= right->data.d));
					PfAppendLine(sc);
					return;
				}
				return;
			}
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfIf_start(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
{\n");
	PfAppendLine(sc);
	return;
}
static inline void PfIfTrue(PfSolveSpecializationConstantsLayout* sc, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (in->type > 100) {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (%s) {\n", in->name);
			PfAppendLine(sc);
			return;
		case 2:
			break;
		}
	}
	else {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (%" PRIi64 ") {\n", in->data.i);
			PfAppendLine(sc);
			return;
		case 2:
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfIfFalse(PfSolveSpecializationConstantsLayout* sc, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (in->type > 100) {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (!%s) {\n", in->name);
			PfAppendLine(sc);
			return;
		case 2:
			break;
		}
	}
	else {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "\
if (!%" PRIi64 ") {\n", in->data.i);
			PfAppendLine(sc);
			return;
		case 2:
			break;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}
static inline void PfIf_else(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
}else{\n");
	PfAppendLine(sc);
	return;
}
static inline void PfIf_end(PfSolveSpecializationConstantsLayout* sc) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	sc->tempLen = sprintf(sc->tempStr, "\
}\n");
	PfAppendLine(sc);
	return;
}

static inline void PfPrintReg(PfSolveSpecializationConstantsLayout* sc, PfContainer* inoutID, PfContainer* in) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (in->type > 100) {
		switch (in->type % 10) {
		case 1:
			sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%d\\n\", %s, %s);", inoutID->name, in->name);
			PfAppendLine(sc);
			return;
		case 2:
			sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%.17e\\n\", %s, %s);", inoutID->name, in->name);
			PfAppendLine(sc); return;
		case 3:
			if (((in->type/10) % 10) == 3)
				sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%.17e %%.17e %%.17e %%.17e\\n\", %s, %s.x.x, %s.x.y, %s.y.x, %s.y.y);", inoutID->name, in->name, in->name, in->name, in->name);
			else
				sc->tempLen = sprintf(sc->tempStr, "printf(\"%%d %%f %%f\\n\", %s, %s.x, %s.y);", inoutID->name, in->name, in->name);
			PfAppendLine(sc);
			return;
		}
	}
	sc->res = PFSOLVE_ERROR_MATH_FAILED;
	return;
}

static inline void PfPermute(PfSolveSpecializationConstantsLayout* sc, pfUINT* permute, pfUINT num_elem, pfUINT type, PfContainer* regIDs, PfContainer* temp) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	PfContainer tempID[33] = PFSOLVE_ZERO_INIT;
	for (int i = 0; i < num_elem; i++) {
		tempID[i].type = 100 + sc->vecTypeCode;
		PfAllocateContainerFlexible(sc, &tempID[i], 50);
	}
	if (type == 0) {
		if (sc->locID[0].type > 100) {
			for (pfUINT i = 0; i < num_elem; i++)
				PfCopyContainer(sc, &tempID[i], &sc->locID[i]);
			for (pfUINT i = 0; i < num_elem; i++)
				PfCopyContainer(sc, &sc->locID[i], &tempID[permute[i]]);
		}
	}
	if (type == 1) {
		if (regIDs[0].type > 100) {
			for (pfUINT i = 0; i < num_elem; i++)
				PfCopyContainer(sc, &tempID[i], &regIDs[i]);
			for (pfUINT i = 0; i < num_elem; i++)
				PfCopyContainer(sc, &regIDs[i], &tempID[permute[i]]);
		}
	}
	for (int i = 0; i < num_elem; i++) {
		PfDeallocateContainer(sc, &tempID[i]);
	}
	return;
}
static inline void PfSubgroupAdd(PfSolveSpecializationConstantsLayout* sc, PfContainer* in, PfContainer* out, pfUINT subWarpSplit) {
	if (sc->res != PFSOLVE_SUCCESS) return;

#if (VKFFT_BACKEND==0)
	/*sc->tempLen = sprintf(sc->tempStr, "	%s.x = subgroupAdd(%s.x);\n", out, in);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s.y = subgroupAdd(%s.y);\n", out, in);
	res = PfAppendLine(sc);
	if (res != PFSOLVE_SUCCESS) return res;*/
#elif (VKFFT_BACKEND==1)
	//v1
	/*for (int i = 1; i < sc->warpSize / subWarpSplit; i *= 2) {
		sc->tempLen = sprintf(sc->tempStr, "	%s.x += __shfl_xor_sync(0xffffffff, %s.x, %d);\n", out, in, i);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	%s.y += __shfl_xor_sync(0xffffffff, %s.y, %d);\n", out, in, i);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}
	//v2
	for (int i = (int)sc->warpSize / 2 / subWarpSplit; i > 0; i /= 2) {
		sc->tempLen = sprintf(sc->tempStr, "	%s.x += __shfl_down_sync(0xffffffff, %s.x, %d);\n", out, in, i);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	%s.y += __shfl_down_sync(0xffffffff, %s.y, %d);\n", out, in, i);
		res = PfAppendLine(sc);
		if (res != PFSOLVE_SUCCESS) return res;
	}*/
#endif
	return;
}

static inline void PfSubgroupBroadcast(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in, int id) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (sc->logicalWarpSize > sc->warpSize) {
		PfContainer temp_int = PFSOLVE_ZERO_INIT;
		temp_int.type = 31;
		temp_int.data.i = id;
		PfIf_eq_start(sc, &sc->gl_LocalInvocationID_x, &temp_int);
		sc->tempLen = sprintf(sc->tempStr, "\
sdata[0] = %s;\n", in->name);
		PfAppendLine(sc);
		PfIf_end(sc);
		appendBarrierPfSolve(sc);
		sc->tempLen = sprintf(sc->tempStr, "\
%s = sdata[0];\n", out->name);
		PfAppendLine(sc);
		appendBarrierPfSolve(sc);
	}
	else {
		if (((out->type / 10) % 10) == 3) {
			PfSubgroupBroadcast(sc, &out->data.dd[0], &in->data.dd[0], id);
			PfSubgroupBroadcast(sc, &out->data.dd[1], &in->data.dd[1], id);
		}
		else {
#if (VKFFT_BACKEND==0)
			sc->tempLen = sprintf(sc->tempStr, "%s = subgroupAdd(%s);\n", out, in);
			PfAppendLine(sc);

#elif (VKFFT_BACKEND==1)
			sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_sync(0xffffffff, %s, %d);\n", out->name, in->name, id);
			PfAppendLine(sc);

#elif (VKFFT_BACKEND==2)
			sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_sync(%s, %d);\n", out->name, in->name, id);
			PfAppendLine(sc);

#endif
		}
	}
	return;
};

static inline void PfSubgroupShuffleDown(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in, int stride) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (sc->logicalWarpSize > sc->warpSize) {
		PfContainer temp_int = PFSOLVE_ZERO_INIT;
		temp_int.type = 31;
		temp_int.data.i = stride;
		PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);
		temp_int.data.i = stride;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
		if (sc->performALT) PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_z);
		sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s] = %s;\n", sc->tempInt.name, in->name);
		PfAppendLine(sc);
		PfIf_end(sc);
	}
	if (((out->type/10) % 10) == 3) {
		PfSubgroupShuffleDown(sc, &out->data.dd[0], &in->data.dd[0], stride);
		PfSubgroupShuffleDown(sc, &out->data.dd[1], &in->data.dd[1], stride);
	}
	else {
#if (VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "%s = subgroupAdd(%s);\n", out, in);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==1)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_down_sync(0xffffffff, %s, %d);\n", out->name, in->name, stride);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==2)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_down(%s, %d);\n", out->name, in->name, stride);
		PfAppendLine(sc);

#endif
	}
	if (sc->logicalWarpSize > sc->warpSize) {
		appendBarrierPfSolve(sc);
		PfContainer temp_int = PFSOLVE_ZERO_INIT;
		temp_int.type = 31;
		temp_int.data.i = sc->warpSize - stride;
		PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);

		temp_int.data.i = sc->logicalWarpSize / sc->warpSize - 1;
		PfIf_lt_start(sc, &sc->warpID, &temp_int);
		temp_int.data.i = 1;
		PfAdd(sc, &sc->tempInt, &sc->warpID, &temp_int);

		temp_int.data.i = stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
		temp_int.data.i = sc->warpSize - stride;
		PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		if (sc->performALT) PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_z);
		sc->tempLen = sprintf(sc->tempStr, "\
%s = sdata[%s];\n", out->name, sc->tempInt.name);
		PfAppendLine(sc);

		PfIf_end(sc);
		PfIf_end(sc);
		appendBarrierPfSolve(sc);
	}
	
	return;
};
static inline void PfSubgroupShuffleUp(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in, int stride) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (sc->logicalWarpSize > sc->warpSize) {
		PfContainer temp_int = PFSOLVE_ZERO_INIT;
		temp_int.type = 31;
		temp_int.data.i = sc->warpSize - stride;
		PfIf_ge_start(sc, &sc->warpInvocationID, &temp_int);
		temp_int.data.i = stride;
		PfMul(sc, &sc->tempInt, &sc->warpID, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
		temp_int.data.i = sc->warpSize - stride;
		PfSub(sc, &sc->tempInt, &sc->tempInt, &temp_int);
		if (sc->performALT) PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_z);
		sc->tempLen = sprintf(sc->tempStr, "\
sdata[%s] = %s;\n", sc->tempInt.name, in->name);
		PfAppendLine(sc);
		PfIf_end(sc);
	}
	if (((out->type/10) % 10) == 3) {
		PfSubgroupShuffleUp(sc, &out->data.dd[0], &in->data.dd[0], stride);
		PfSubgroupShuffleUp(sc, &out->data.dd[1], &in->data.dd[1], stride);
	}
	else {
#if (VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "%s = subgroupAdd(%s);\n", out, in);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==1)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_up_sync(0xffffffff, %s, %d);\n", out->name, in->name, stride);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==2)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_up(%s, %d);\n", out->name, in->name, stride);
		PfAppendLine(sc);

#endif
	}
	if (sc->logicalWarpSize > sc->warpSize) {
		appendBarrierPfSolve(sc);
		PfContainer temp_int = PFSOLVE_ZERO_INIT;
		temp_int.type = 31;
		temp_int.data.i = stride;
		PfIf_lt_start(sc, &sc->warpInvocationID, &temp_int);

		temp_int.data.i = 0;
		PfIf_gt_start(sc, &sc->warpID, &temp_int);
		temp_int.data.i = 1;
		PfSub(sc, &sc->tempInt, &sc->warpID, &temp_int);

		temp_int.data.i = stride;
		PfMul(sc, &sc->tempInt, &sc->tempInt, &temp_int, 0);
		PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->warpInvocationID);
		if (sc->performALT) PfAdd(sc, &sc->tempInt, &sc->tempInt, &sc->inoutID_z);
		sc->tempLen = sprintf(sc->tempStr, "\
%s = sdata[%s];\n", out->name, sc->tempInt.name);
		PfAppendLine(sc);

		PfIf_end(sc);
		PfIf_end(sc);
		appendBarrierPfSolve(sc);
	}
	return;
};

static inline void PfSubgroupShuffleUpCyclic(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in, int stride) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (((out->type/10) % 10) == 3) {
		PfSubgroupShuffleUpCyclic(sc, &out->data.dd[0], &in->data.dd[0], stride);
		PfSubgroupShuffleUpCyclic(sc, &out->data.dd[1], &in->data.dd[1], stride);
	}
	else {
#if (VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "%s = subgroupAdd(%s);\n", out, in);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==1)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_sync(0xffffffff, %s, (%s+%d) %% %d);\n", out->name, in->name, sc->gl_LocalInvocationID_x.name, sc->warpSize - stride, sc->warpSize);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==2)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl(%s, (%s+%d) %% %d);\n", out->name, in->name, sc->gl_LocalInvocationID_x.name, sc->warpSize - stride, sc->warpSize);
		PfAppendLine(sc);

#endif
	}
	return;
};
static inline void PfSubgroupShuffleDownCyclic(PfSolveSpecializationConstantsLayout* sc, PfContainer* out, PfContainer* in, int stride) {
	if (sc->res != PFSOLVE_SUCCESS) return;
	if (((out->type/10) % 10) == 3) {
		PfSubgroupShuffleDownCyclic(sc, &out->data.dd[0], &in->data.dd[0], stride);
		PfSubgroupShuffleDownCyclic(sc, &out->data.dd[1], &in->data.dd[1], stride);
	}
	else {
#if (VKFFT_BACKEND==0)
		sc->tempLen = sprintf(sc->tempStr, "%s = subgroupAdd(%s);\n", out->name, in->name);
		PfAppendLine(sc);

#elif (VKFFT_BACKEND==1)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl_sync(0xffffffff, %s, (%s+%d) %% %d);\n", out->name, in->name, sc->gl_LocalInvocationID_x.name, stride, sc->warpSize);
		PfAppendLine(sc);
#elif (VKFFT_BACKEND==2)
		sc->tempLen = sprintf(sc->tempStr, "%s = __shfl(%s, (%s+%d) %% %d);\n", out->name, in->name, sc->gl_LocalInvocationID_x.name, stride, sc->warpSize);
		PfAppendLine(sc);

#endif
	}
	return;
};
#endif
