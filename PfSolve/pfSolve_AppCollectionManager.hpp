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
#ifndef PFSOLVE_APPCOLLECTIONMANAGER_H
#define PFSOLVE_APPCOLLECTIONMANAGER_H
//#include <iostream>
#include <map> // C++ for now

#include "QuICC/Math/PfSolve/pfSolve.h"

typedef struct PfSolve_MapKey_JonesWorland {
	int64_t size[2];
	int64_t upperBound;
	int64_t outputBufferStride;
	int64_t type;
} ;
typedef struct PfSolve_MapKey_dgbmv {
	int64_t size[2];
	int64_t outputBufferStride;
	int64_t LDA;
	int64_t KU;
	int64_t KL;
};
typedef struct PfSolve_MapKey_block {
	int64_t size[2];
	int64_t inputBufferStride;
	int64_t outputBufferStride;
	int64_t type;
	int64_t lshift;
};

typedef struct PfSolve_AppLibrary{
	std::map<PfSolve_MapKey_JonesWorland, PfSolveApplication> mapJonesWorland;
	//std::map<PfSolve_MapKey_scaleC, PfSolveApplication> mapScaleC;
	//std::map<PfSolve_MapKey_copy, PfSolveApplication> mapCopy;
	std::map<PfSolve_MapKey_block, PfSolveApplication> mapBlock;
	std::map<PfSolve_MapKey_dgbmv, PfSolveApplication> mapDGBMV;
};


bool inline operator<(const PfSolve_MapKey_JonesWorland& l, const PfSolve_MapKey_JonesWorland& r) {
        return (l.size[0] < r.size[0]
			|| (l.size[0] == r.size[0] && l.size[1] < r.size[1])
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.upperBound < r.upperBound)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.upperBound == r.upperBound && l.outputBufferStride < r.outputBufferStride)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.upperBound == r.upperBound && l.outputBufferStride == r.outputBufferStride && l.type < r.type)
			);
};

bool inline operator<(const PfSolve_MapKey_dgbmv& l, const PfSolve_MapKey_dgbmv& r) {
        return (l.size[0] < r.size[0]
			|| (l.size[0] == r.size[0] && l.size[1] < r.size[1])
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.LDA < r.LDA)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.LDA == r.LDA && l.outputBufferStride < r.outputBufferStride)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.LDA == r.LDA && l.outputBufferStride == r.outputBufferStride && l.KU < r.KU)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.LDA == r.LDA && l.outputBufferStride == r.outputBufferStride && l.KU == r.KU && l.KL < r.KL)
			);
};

bool inline operator<(const PfSolve_MapKey_block& l, const PfSolve_MapKey_block& r) {
        return (l.size[0] < r.size[0]
			|| (l.size[0] == r.size[0] && l.size[1] < r.size[1])
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.inputBufferStride < r.inputBufferStride)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.inputBufferStride == r.inputBufferStride && l.outputBufferStride < r.outputBufferStride)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.inputBufferStride == r.inputBufferStride && l.outputBufferStride == r.outputBufferStride && l.type < r.type)
			|| (l.size[0] == r.size[0] && l.size[1] == r.size[1] && l.inputBufferStride == r.inputBufferStride && l.outputBufferStride == r.outputBufferStride && l.type == r.type && l.lshift < r.lshift)
			);
};

static inline PfSolveResult checkLibrary_JonesWorland(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_JonesWorland mapKey_JonesWorland, PfSolveApplication** app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	auto search =  appLibrary->mapJonesWorland.find(mapKey_JonesWorland);
	if (search != appLibrary->mapJonesWorland.end()){
        app[0] = (PfSolveApplication*) &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult addToLibrary_JonesWorland(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_JonesWorland mapKey_JonesWorland, PfSolveApplication* app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	appLibrary->mapJonesWorland[mapKey_JonesWorland] = app[0];
	
	return resSolve;
}
static inline PfSolveResult checkLibrary_dgbmv(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_dgbmv mapKey_dgbmv, PfSolveApplication** app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	auto search =  appLibrary->mapDGBMV.find(mapKey_dgbmv);
	if (search != appLibrary->mapDGBMV.end()){
        app[0] = (PfSolveApplication*) &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult addToLibrary_dgbmv(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_dgbmv mapKey_dgbmv, PfSolveApplication* app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	appLibrary->mapDGBMV[mapKey_dgbmv] = app[0];
	
	return resSolve;
}

static inline PfSolveResult checkLibrary_block(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_block mapKey_block, PfSolveApplication** app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	auto search =  appLibrary->mapBlock.find(mapKey_block);
	if (search != appLibrary->mapBlock.end()){
        app[0] = (PfSolveApplication*) &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult addToLibrary_block(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_block mapKey_block, PfSolveApplication* app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	appLibrary->mapBlock[mapKey_block] = app[0];
	
	return resSolve;
}
/*
static inline PfSolveResult checkLibrary_scaleC(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_scaleC mapKey_ScaleC, PfSolveApplication** app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	auto search =  appLibrary->mapScaleC.find(mapKey_ScaleC);
	if (search != appLibrary->mapScaleC.end()){
		app[0] = &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult checkLibrary_copy(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_copy mapKey_Copy, PfSolveApplication** app) {

	PfSolveResult resSolve = PFSOLVE_SUCCESS;

	auto search =  appLibrary->mapCopy.find(mapKey_Copy);
	if (search != appLibrary->mapCopy.end()){
		app[0] = &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult checkLibrary_block(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_block mapKey_Block, PfSolveApplication** app) {

	PfSolveResult resSolve = PFSOLVE_SUCCESS;

	auto search =  appLibrary->mapBlock.find(mapKey_Block);
	if (search != appLibrary->mapBlock.end()){
		app[0] = &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult checkLibrary_dgbmv(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_dgbmv mapKey_dgbmv, PfSolveApplication** app) {

	PfSolveResult resSolve = PFSOLVE_SUCCESS;

	auto search =  appLibrary->mapDGBMV.find(mapKey_dgbmv);
	if (search != appLibrary->mapDGBMV.end()){
		app[0] = &search->second;
	}else{
		app[0]=0;
	}
	return resSolve;
}
static inline PfSolveResult addToLibrary_JonesWorland(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_JonesWorland mapKey_JonesWorland, PfSolveApplication* app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	appLibrary->mapJonesWorland[mapKey_JonesWorland] = app[0];
	
	return resSolve;
}

static inline PfSolveResult addToLibrary_JonesWorland_sequential(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_JonesWorland_sequential mapKey_JonesWorland_sequential, PfSolveApplication* app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	appLibrary->mapJonesWorland_sequential[mapKey_JonesWorland_sequential] = app[0];
	
	return resSolve;
}

static inline PfSolveResult addToLibrary_scaleC(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_scaleC mapKey_ScaleC, PfSolveApplication* app) {
	
	PfSolveResult resSolve = PFSOLVE_SUCCESS;
	
	appLibrary->mapScaleC[mapKey_ScaleC] = app[0];
	
	return resSolve;
}
static inline PfSolveResult addToLibrary_copy(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_copy mapKey_Copy, PfSolveApplication* app) {

	PfSolveResult resSolve = PFSOLVE_SUCCESS;

	appLibrary->mapCopy[mapKey_Copy] = app[0];

	return resSolve;
}
static inline PfSolveResult addToLibrary_block(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_block mapKey_Block, PfSolveApplication* app) {

	PfSolveResult resSolve = PFSOLVE_SUCCESS;

	appLibrary->mapBlock[mapKey_Block] = app[0];

	return resSolve;
}
static inline PfSolveResult addToLibrary_dgbmv(PfSolve_AppLibrary* appLibrary, PfSolve_MapKey_dgbmv mapKey_dgbmv, PfSolveApplication* app) {

	PfSolveResult resSolve = PFSOLVE_SUCCESS;

	appLibrary->mapDGBMV[mapKey_dgbmv] = app[0];

	return resSolve;
}*/
#endif
