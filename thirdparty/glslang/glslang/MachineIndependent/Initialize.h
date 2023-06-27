//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013-2016 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _INITIALIZE_INCLUDED_
#define _INITIALIZE_INCLUDED_

#include "../Include/ResourceLimits.h"
#include "../Include/Common.h"
#include "../Include/ShHandle.h"
#include "SymbolTable.h"
#include "Versions.h"

namespace glslang {

//
// This is made to hold parseable strings for almost all the built-in
// functions and variables for one specific combination of version
// and profile.  (Some still need to be added programmatically.)
// This is a base class for language-specific derivations, which
// can be used for language independent builtins.
//
// The strings are organized by
//    commonBuiltins:  intersection of all stages' built-ins, processed just once
//    stageBuiltins[]: anything a stage needs that's not in commonBuiltins
//
class TBuiltInParseables {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())
    TBuiltInParseables();
    virtual ~TBuiltInParseables();
    virtual void initialize(int version, EProfile, const SpvVersion& spvVersion) = 0;
    virtual void initialize(const TBuiltInResource& resources, int version, EProfile, const SpvVersion& spvVersion, EShLanguage) = 0;
    virtual const TString& getCommonString() const { return commonBuiltins; }
    virtual const TString& getStageString(EShLanguage language) const { return stageBuiltins[language]; }

    virtual void identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable) = 0;
    virtual void identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable, const TBuiltInResource &resources) = 0;

protected:
    TString commonBuiltins;
    TString stageBuiltins[EShLangCount];
};

//
// This is a GLSL specific derivation of TBuiltInParseables.  To present a stable
// interface and match other similar code, it is called TBuiltIns, rather
// than TBuiltInParseablesGlsl.
//
class TBuiltIns : public TBuiltInParseables {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())
    TBuiltIns();
    virtual ~TBuiltIns();
    void initialize(int version, EProfile, const SpvVersion& spvVersion);
    void initialize(const TBuiltInResource& resources, int version, EProfile, const SpvVersion& spvVersion, EShLanguage);

    void identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable);
    void identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable, const TBuiltInResource &resources);

protected:
    void addTabledBuiltins(int version, EProfile profile, const SpvVersion& spvVersion);
    void relateTabledBuiltins(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage, TSymbolTable&);
    void add2ndGenerationSamplingImaging(int version, EProfile profile, const SpvVersion& spvVersion);
    void addSubpassSampling(TSampler, const TString& typeName, int version, EProfile profile);
    void addQueryFunctions(TSampler, const TString& typeName, int version, EProfile profile);
    void addImageFunctions(TSampler, const TString& typeName, int version, EProfile profile);
    void addSamplingFunctions(TSampler, const TString& typeName, int version, EProfile profile);
    void addGatherFunctions(TSampler, const TString& typeName, int version, EProfile profile);

    // Helpers for making textual representations of the permutations
    // of texturing/imaging functions.
    const char* postfixes[5];
    const char* prefixes[EbtNumTypes];
    int dimMap[EsdNumDims];
};

} // end namespace glslang

#endif // _INITIALIZE_INCLUDED_
