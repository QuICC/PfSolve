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
#ifndef PFSOLVE_WARPREDUCE_H
#define PFSOLVE_WARPREDUCE_H
#include "pfSolve_Structs/pfSolve_Structs.h"
#include "pfSolve_CodeGen/pfSolve_StringManagement/pfSolve_StringManager.h"
#include "pfSolve_CodeGen/pfSolve_MathUtils/pfSolve_MathUtils.h"
static inline PfSolveResult appendWarpReduce(PfSolveSpecializationConstantsLayout* sc) {
	/*
		Kernel can have two input types - from registers or from shared memory. In second case, a copy of state in registers is created.
	*/
	PfSolveResult res = PFSOLVE_SUCCESS;
	for (int64_t w = ((uint64_t)ceil(log2(sc->warpSize)) + 1); w >= 0; w--) {
		uint64_t subWarpStep = (w >= ((uint64_t)ceil(log2(sc->warpSize)))) ? 1 : (uint64_t)pow(2, (uint64_t)(log2(sc->warpSize) - w));
		for (uint64_t i = sc->log2WarpGridPointsStart[w + 1]; i < sc->log2WarpGridPointsStart[w]; i += sc->gl_NumSubgroups * subWarpStep) {
			sc->tempLen = sprintf(sc->tempStr, "	temp_0 = 0;\n");
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			if (w >= ((uint64_t)ceil(log2(sc->warpSize)))) {
				uint64_t max_blocks_per_subgroup = 1;
				sc->tempLen = sprintf(sc->tempStr, "switch(%s){\n", sc->gl_SubgroupID);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				for (uint64_t j = 0; j < sc->gl_NumSubgroups; j++) {
					if (i + j < sc->log2WarpGridPointsStart[w]) {
						sc->tempLen = sprintf(sc->tempStr, "	case  %" PRIi64 ":\n", j);
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
						//all subgroups past numUsedReductionSubgroups will have 0 active threads
						max_blocks_per_subgroup = ((uint64_t)ceil(sc->activeThreadsReductionForEquispacedGrid[i + j] / (double)sc->warpSize) > max_blocks_per_subgroup) ? (uint64_t)ceil(sc->activeThreadsReductionForEquispacedGrid[i + j] / (double)sc->warpSize) : max_blocks_per_subgroup;
						for (int64_t k = 0; k < (uint64_t)ceil(sc->activeThreadsReductionForEquispacedGrid[i + j] / (double)sc->warpSize); k++) {
							if (k == ((uint64_t)ceil(sc->activeThreadsReductionForEquispacedGrid[i + j] / (double)sc->warpSize - 1))) {
								sc->tempLen = sprintf(sc->tempStr, "	if (%s < %" PRIi64 "){\n", sc->gl_SubgroupInvocationID, sc->activeThreadsReductionForEquispacedGrid[i + j] - ((uint64_t)ceil(sc->activeThreadsReductionForEquispacedGrid[i + j] / (double)sc->warpSize) - 1) * sc->warpSize);
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
							}
							sc->tempLen = sprintf(sc->tempStr, "		temp_start_shared_%" PRIi64 " = %" PRIi64 "+%s;\n", k, sc->activeSharedOffsetForEquispacedGrid[i + j] + sc->warpSize * k, sc->gl_SubgroupInvocationID);
							res = PfAppendLine(sc);
							if (res != PFSOLVE_SUCCESS) return res;
							sc->tempLen = sprintf(sc->tempStr, "		temp_start_LUT_%" PRIi64 " = %" PRIi64 "+%s;\n", k, sc->activeGlobalKernelOffsetForSubgroup[i + j] + sc->warpSize * k, sc->gl_SubgroupInvocationID);
							res = PfAppendLine(sc);
							if (res != PFSOLVE_SUCCESS) return res;
							if (k == ((uint64_t)ceil(sc->activeThreadsReductionForEquispacedGrid[i + j] / (double)sc->warpSize - 1))) {
								sc->tempLen = sprintf(sc->tempStr, "	}\n");
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
							}
						}
						sc->tempLen = sprintf(sc->tempStr, "	if (%s == %" PRIi64 "){\n", sc->gl_SubgroupInvocationID, (i / sc->gl_NumSubgroups) % sc->warpSize);
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
						if (sc->M_size.mode)
							sc->tempLen = sprintf(sc->tempStr, "		start_%" PRIu64 " = %s + %" PRIu64 ";\n", i / sc->gl_NumSubgroups / sc->warpSize, sc->M_size.x_str, sc->equispacedGridPoint[i + j]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		start_%" PRIu64 " = %" PRIu64 ";\n", i / sc->gl_NumSubgroups / sc->warpSize, sc->M_size.x_num + sc->equispacedGridPoint[i + j]);
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "	};\n");
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;

						sc->tempLen = sprintf(sc->tempStr, "	break;\n");
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
					}
				}
				sc->tempLen = sprintf(sc->tempStr, "	}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				for (int64_t k = 0; k < max_blocks_per_subgroup; k++) {
					sc->tempLen = sprintf(sc->tempStr, "	if (temp_start_shared_%" PRIi64 " < 1000000){\n", k);
					res = PfAppendLine(sc);
					if (res != PFSOLVE_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "		temp_0 += sdata[temp_start_shared_%" PRIi64 "] * %sLUT[temp_start_LUT_%" PRIi64 "]%s;\n", k, sc->convTypeLeftInput, k, sc->convTypeRightInput);
					res = PfAppendLine(sc);
					if (res != PFSOLVE_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "		temp_start_shared_%" PRIi64 " = 1000000;\n", k);
					res = PfAppendLine(sc);
					if (res != PFSOLVE_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = PfAppendLine(sc);
					if (res != PFSOLVE_SUCCESS) return res;
				}
				res = subgroupAdd(sc, "temp_0", "temp_0", subWarpStep);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	if (%s == %" PRIi64 "){\n", sc->gl_SubgroupInvocationID, (i / sc->gl_NumSubgroups) % sc->warpSize);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		res_%" PRIu64 " = temp_0;\n", i / sc->gl_NumSubgroups / sc->warpSize);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	};\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "switch(%s){\n", sc->gl_SubgroupID);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				for (uint64_t j = 0; j < sc->gl_NumSubgroups; j++) {
					if (i + j * subWarpStep < sc->log2WarpGridPointsStart[w]) {
						sc->tempLen = sprintf(sc->tempStr, "	case  %" PRIi64 ":\n", j);
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
						//all subgroups past numUsedReductionSubgroups will have 0 active threads
						for (int64_t k = 0; k < subWarpStep; k++) {
							if (i + j * subWarpStep + k < sc->log2WarpGridPointsStart[w]) {
								sc->tempLen = sprintf(sc->tempStr, "	if ((%s >= %" PRIi64 ")&& (%s < %" PRIi64 ")) {\n", sc->gl_SubgroupInvocationID, k * sc->warpSize / subWarpStep, sc->gl_SubgroupInvocationID, k * sc->warpSize / subWarpStep + sc->activeThreadsReductionForEquispacedGrid[i + j * subWarpStep + k]);
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "		temp_start_shared_0 = %" PRIi64 "+%s;\n", (int64_t)sc->activeSharedOffsetForEquispacedGrid[i + j * subWarpStep + k] - k * sc->warpSize / subWarpStep, sc->gl_SubgroupInvocationID);
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "		temp_start_LUT_0 = %" PRIi64 "+%s;\n", (int64_t)sc->activeGlobalKernelOffsetForSubgroup[i + j * subWarpStep + k] - k * sc->warpSize / subWarpStep, sc->gl_SubgroupInvocationID);
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "	}\n");
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "	if (%s == %" PRIi64 "){\n", sc->gl_SubgroupInvocationID, k * sc->warpSize / subWarpStep + ((i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups) % (sc->warpSize / subWarpStep));
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								if(sc->M_size.mode)
									sc->tempLen = sprintf(sc->tempStr, "		start_%" PRIu64 " = %s + %" PRIu64 ";\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize), sc->M_size.x_str, sc->equispacedGridPoint[i + j * subWarpStep + k]);
								else
									sc->tempLen = sprintf(sc->tempStr, "		start_%" PRIu64 " = %" PRIu64 ";\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize), sc->M_size.x_num + sc->equispacedGridPoint[i + j * subWarpStep + k]);
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "	};\n");
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
							}
						}

						sc->tempLen = sprintf(sc->tempStr, "	break;\n");
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
					}
				}
				sc->tempLen = sprintf(sc->tempStr, "	}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	if (temp_start_0  < 1000000) {\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		temp_0 += sdata[temp_start_shared_0] * %sLUT[temp_start_LUT_0]%s;\n", sc->convTypeLeftInput, sc->convTypeRightInput);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		temp_start_0 = 1000000;\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				res = subgroupAdd(sc, "temp_0", "temp_0", subWarpStep);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	if (start_%" PRIu64 " < %" PRIi64 "){\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize), (int64_t)100000);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		sdata[start_%" PRIu64 "] = temp_0;\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize));
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		start_%" PRIu64 " = 100000;\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize));
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	};\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				/*sc->tempLen = sprintf(sc->tempStr, "switch(%s){\n", sc->gl_SubgroupID);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				for (uint64_t j = 0; j < sc->gl_NumSubgroups; j++) {
					if (i + j* subWarpStep < sc->log2WarpGridPointsStart[w]) {
						sc->tempLen = sprintf(sc->tempStr, "	case  %" PRIi64 ":\n", j);
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
						//all subgroups past numUsedReductionSubgroups will have 0 active threads
						for (int k = 0; k < subWarpStep; k++) {
							if (i + j * subWarpStep + k < sc->log2WarpGridPointsStart[w]) {
								//sc->tempLen = sprintf(sc->tempStr, "	if (%s == %" PRIi64 "){\n", sc->gl_SubgroupInvocationID, k * sc->warpSize / subWarpStep + ((i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups) % (sc->warpSize / subWarpStep));
								sc->tempLen = sprintf(sc->tempStr, "	if (start_%" PRIu64 " < %" PRIi64 "){\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize), 100000);
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "		sdata[start_%" PRIu64 "] = temp_0;\n", (i - sc->log2WarpGridPointsStart[w + 1]) / sc->gl_NumSubgroups / (sc->warpSize));
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "	};\n");
								res = PfAppendLine(sc);
								if (res != PFSOLVE_SUCCESS) return res;
							}
						}
						sc->tempLen = sprintf(sc->tempStr, "	break;\n");
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
					}
				}
				sc->tempLen = sprintf(sc->tempStr, "	}\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;*/
			}
			/*sc->tempLen = sprintf(sc->tempStr, "	if (%s == %" PRIi64 ")\n", sc->gl_SubgroupInvocationID, i % sc->warpSize);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "		res_%" PRIu64 " = temp_0;\n", i/sc->warpSize);
			res = PfAppendLine(sc);
			if (res != PFSOLVE_SUCCESS) return res;*/
		}
		if (w == ((uint64_t)ceil(log2(sc->warpSize)))) {
			for (uint64_t i = 0; i < (uint64_t)ceil((sc->log2WarpGridPointsStart[w]) / (double)sc->localSize[0]); i++) {
				sc->tempLen = sprintf(sc->tempStr, "	if (start_%" PRIu64 " < %" PRIi64 "){\n", i, (int64_t)100000);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		sdata[start_%" PRIu64 "] = res_%" PRIu64 ";\n", i, i);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "		start_%" PRIu64 " = 100000;\n", i);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	};\n");
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				/*if (sc->log2WarpGridPointsStart[w] < sc->localSize[0]) {
					sc->tempLen = sprintf(sc->tempStr, "	if(%s*%" PRIu64 " < %" PRIu64 "){\n", sc->gl_SubgroupInvocationID, sc->gl_NumSubgroups, sc->log2WarpGridPointsStart[w]);
					res = PfAppendLine(sc);
					if (res != PFSOLVE_SUCCESS) return res;
				}
				else {
					if ((i == ((uint64_t)ceil(sc->log2WarpGridPointsStart[w] / (double)sc->localSize[0]) - 1)) && ((sc->log2WarpGridPointsStart[w] % sc->localSize[0]) != 0)) {
						sc->tempLen = sprintf(sc->tempStr, "	if(%s<%" PRIu64 "){\n", sc->gl_SubgroupID, sc->log2WarpGridPointsStart[w] % sc->localSize[0]);
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
					}
				}
				sc->tempLen = sprintf(sc->tempStr, "		sdata[start_%" PRIu64 "] = res_%" PRIu64 ";\n", i, i);
				res = PfAppendLine(sc);
				if (res != PFSOLVE_SUCCESS) return res;
				if (sc->size[0] < sc->localSize[0]) {
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = PfAppendLine(sc);
					if (res != PFSOLVE_SUCCESS) return res;
				}
				else {
					if ((i == ((uint64_t)ceil(sc->log2WarpGridPointsStart[w] / (double)sc->localSize[0]) - 1)) && ((sc->log2WarpGridPointsStart[w] % sc->localSize[0]) != 0)) {
						sc->tempLen = sprintf(sc->tempStr, "	}\n");
						res = PfAppendLine(sc);
						if (res != PFSOLVE_SUCCESS) return res;
					}
				}*/
			}
		}
	}
	return res;
}
#endif
