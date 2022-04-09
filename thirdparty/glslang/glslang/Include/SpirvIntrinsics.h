//
// Copyright(C) 2021 Advanced Micro Devices, Inc.
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

#pragma once

#ifndef GLSLANG_WEB

//
// GL_EXT_spirv_intrinsics
//
#include "Common.h"

namespace glslang {

class TIntermTyped;
class TIntermConstantUnion;
class TType;

// SPIR-V requirements
struct TSpirvRequirement {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    // capability = [..]
    TSet<TString> extensions;
    // extension = [..]
    TSet<int> capabilities;
};

// SPIR-V execution modes
struct TSpirvExecutionMode {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    // spirv_execution_mode
    TMap<int, TVector<const TIntermConstantUnion*>> modes;
    // spirv_execution_mode_id
    TMap<int, TVector<const TIntermTyped*> > modeIds;
};

// SPIR-V decorations
struct TSpirvDecorate {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    // spirv_decorate
    TMap<int, TVector<const TIntermConstantUnion*> > decorates;
    // spirv_decorate_id
    TMap<int, TVector<const TIntermTyped*>> decorateIds;
    // spirv_decorate_string
    TMap<int, TVector<const TIntermConstantUnion*> > decorateStrings;
};

// SPIR-V instruction
struct TSpirvInstruction {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    TSpirvInstruction() { set = ""; id = -1; }

    bool operator==(const TSpirvInstruction& rhs) const { return set == rhs.set && id == rhs.id; }
    bool operator!=(const TSpirvInstruction& rhs) const { return !operator==(rhs); }

    // spirv_instruction
    TString set;
    int     id;
};

// SPIR-V type parameter
struct TSpirvTypeParameter {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    TSpirvTypeParameter(const TIntermConstantUnion* arg) { constant = arg; }

    bool operator==(const TSpirvTypeParameter& rhs) const { return constant == rhs.constant; }
    bool operator!=(const TSpirvTypeParameter& rhs) const { return !operator==(rhs); }

    const TIntermConstantUnion* constant;
};

typedef TVector<TSpirvTypeParameter> TSpirvTypeParameters;

// SPIR-V type
struct TSpirvType {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    bool operator==(const TSpirvType& rhs) const
    {
        return spirvInst == rhs.spirvInst && typeParams == rhs.typeParams;
    }
    bool operator!=(const TSpirvType& rhs) const { return !operator==(rhs); }

    // spirv_type
    TSpirvInstruction spirvInst;
    TSpirvTypeParameters typeParams;
};

} // end namespace glslang

#endif // GLSLANG_WEB
