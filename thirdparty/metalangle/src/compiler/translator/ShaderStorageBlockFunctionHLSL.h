//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderStorageBlockOutputHLSL: A traverser to translate a ssbo_access_chain to an offset of
// RWByteAddressBuffer.
//     //EOpIndexDirectInterfaceBlock
//     ssbo_variable :=
//       | the name of the SSBO
//       | the name of a variable in an SSBO backed interface block

//     // EOpIndexInDirect
//     // EOpIndexDirect
//     ssbo_array_indexing := ssbo_access_chain[expr_no_ssbo]

//     // EOpIndexDirectStruct
//     ssbo_structure_access := ssbo_access_chain.identifier

//     ssbo_access_chain :=
//       | ssbo_variable
//       | ssbo_array_indexing
//       | ssbo_structure_access
//

#ifndef COMPILER_TRANSLATOR_SHADERSTORAGEBLOCKFUNCTIONHLSL_H_
#define COMPILER_TRANSLATOR_SHADERSTORAGEBLOCKFUNCTIONHLSL_H_

#include <set>

#include "compiler/translator/InfoSink.h"
#include "compiler/translator/Types.h"

namespace sh
{

class TIntermSwizzle;
enum class SSBOMethod
{
    LOAD,
    STORE,
    LENGTH,
    ATOMIC_ADD,
    ATOMIC_MIN,
    ATOMIC_MAX,
    ATOMIC_AND,
    ATOMIC_OR,
    ATOMIC_XOR,
    ATOMIC_EXCHANGE,
    ATOMIC_COMPSWAP
};

class ShaderStorageBlockFunctionHLSL final : angle::NonCopyable
{
  public:
    TString registerShaderStorageBlockFunction(const TType &type,
                                               SSBOMethod method,
                                               TLayoutBlockStorage storage,
                                               bool rowMajor,
                                               int matrixStride,
                                               int unsizedArrayStride,
                                               TIntermSwizzle *node);

    void shaderStorageBlockFunctionHeader(TInfoSinkBase &out);

  private:
    struct ShaderStorageBlockFunction
    {
        bool operator<(const ShaderStorageBlockFunction &rhs) const;
        TString functionName;
        TString typeString;
        SSBOMethod method;
        TType type;
        bool rowMajor;
        int matrixStride;
        int unsizedArrayStride;
        TVector<int> swizzleOffsets;
        bool isDefaultSwizzle;
    };

    static void OutputSSBOLoadFunctionBody(TInfoSinkBase &out,
                                           const ShaderStorageBlockFunction &ssboFunction);
    static void OutputSSBOStoreFunctionBody(TInfoSinkBase &out,
                                            const ShaderStorageBlockFunction &ssboFunction);
    static void OutputSSBOLengthFunctionBody(TInfoSinkBase &out, int unsizedArrayStride);
    static void OutputSSBOAtomicMemoryFunctionBody(TInfoSinkBase &out,
                                                   const ShaderStorageBlockFunction &ssboFunction);
    using ShaderStorageBlockFunctionSet = std::set<ShaderStorageBlockFunction>;
    ShaderStorageBlockFunctionSet mRegisteredShaderStorageBlockFunctions;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_SHADERSTORAGEBLOCKFUNCTIONHLSL_H_
