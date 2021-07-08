//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderStorageBlockOutputHLSL: A traverser to translate a buffer variable of shader storage block
// to an offset of RWByteAddressBuffer.
//

#ifndef COMPILER_TRANSLATOR_SHADERSTORAGEBLOCKOUTPUTHLSL_H_
#define COMPILER_TRANSLATOR_SHADERSTORAGEBLOCKOUTPUTHLSL_H_

#include "compiler/translator/ShaderStorageBlockFunctionHLSL.h"
#include "compiler/translator/blocklayout.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{
class ResourcesHLSL;
class OutputHLSL;
class TSymbolTable;

struct TReferencedBlock : angle::NonCopyable
{
    POOL_ALLOCATOR_NEW_DELETE
    TReferencedBlock(const TInterfaceBlock *block, const TVariable *instanceVariable);
    const TInterfaceBlock *block;
    const TVariable *instanceVariable;  // May be nullptr if the block is not instanced.
};

// Maps from uniqueId to a variable.
using ReferencedInterfaceBlocks = std::map<int, const TReferencedBlock *>;

// Used to save shader storage block field member information.
using BlockMemberInfoMap = std::map<const TField *, BlockMemberInfo>;

using ShaderVarToFieldMap = std::map<std::string, const TField *>;

class ShaderStorageBlockOutputHLSL : public TIntermTraverser
{
  public:
    ShaderStorageBlockOutputHLSL(OutputHLSL *outputHLSL,
                                 TSymbolTable *symbolTable,
                                 ResourcesHLSL *resourcesHLSL,
                                 const std::vector<InterfaceBlock> &shaderStorageBlocks);

    ~ShaderStorageBlockOutputHLSL();

    // This writes part of the function call to store a value to a SSBO to the output stream. After
    // calling this, ", <stored value>)" should be written to the output stream to complete the
    // function call.
    void outputStoreFunctionCallPrefix(TIntermTyped *node);
    // This writes the function call to load a SSBO value to the output stream.
    void outputLoadFunctionCall(TIntermTyped *node);
    // This writes the function call to get the lengh of unsized array member of SSBO.
    void outputLengthFunctionCall(TIntermTyped *node);
    // Writes the atomic memory function calls for SSBO.
    void outputAtomicMemoryFunctionCallPrefix(TIntermTyped *node, TOperator op);

    void writeShaderStorageBlocksHeader(TInfoSinkBase &out) const;

  protected:
    void visitSymbol(TIntermSymbol *) override;
    void visitConstantUnion(TIntermConstantUnion *) override;
    bool visitSwizzle(Visit visit, TIntermSwizzle *node) override;
    bool visitBinary(Visit visit, TIntermBinary *) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitTernary(Visit visit, TIntermTernary *) override;
    bool visitUnary(Visit visit, TIntermUnary *) override;

  private:
    void traverseSSBOAccess(TIntermTyped *node, SSBOMethod method);
    void setMatrixStride(TIntermTyped *node, TLayoutBlockStorage storage, bool rowMajor);
    bool isEndOfSSBOAccessChain();
    void writeEOpIndexDirectOrIndirectOutput(TInfoSinkBase &out, Visit visit, TIntermBinary *node);
    // Common part in dot operations.
    void writeDotOperatorOutput(TInfoSinkBase &out, const TField *field);

    int mMatrixStride;
    bool mRowMajor;
    bool mLocationAsTheLastArgument;
    OutputHLSL *mOutputHLSL;
    ShaderStorageBlockFunctionHLSL *mSSBOFunctionHLSL;
    ResourcesHLSL *mResourcesHLSL;
    ReferencedInterfaceBlocks mReferencedShaderStorageBlocks;

    BlockMemberInfoMap mBlockMemberInfoMap;
    const std::vector<InterfaceBlock> &mShaderStorageBlocks;
};
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_SHADERSTORAGEBLOCKOUTPUTHLSL_H_
