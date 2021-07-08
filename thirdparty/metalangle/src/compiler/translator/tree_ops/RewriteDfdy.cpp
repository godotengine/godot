//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Implementation of dFdy viewport transformation.
// See header for more info.

#include "compiler/translator/tree_ops/RewriteDfdy.h"

#include "common/angleutils.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class Traverser : public TIntermTraverser
{
  public:
    ANGLE_NO_DISCARD static bool Apply(TCompiler *compiler,
                                       TIntermNode *root,
                                       const TSymbolTable &symbolTable,
                                       TIntermBinary *viewportYScale);

  private:
    Traverser(TIntermBinary *viewportYScale, TSymbolTable *symbolTable);
    bool visitUnary(Visit visit, TIntermUnary *node) override;

    TIntermBinary *mViewportYScale = nullptr;
};

Traverser::Traverser(TIntermBinary *viewportYScale, TSymbolTable *symbolTable)
    : TIntermTraverser(true, false, false, symbolTable), mViewportYScale(viewportYScale)
{}

// static
bool Traverser::Apply(TCompiler *compiler,
                      TIntermNode *root,
                      const TSymbolTable &symbolTable,
                      TIntermBinary *viewportYScale)
{
    TSymbolTable *pSymbolTable = const_cast<TSymbolTable *>(&symbolTable);
    Traverser traverser(viewportYScale, pSymbolTable);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

bool Traverser::visitUnary(Visit visit, TIntermUnary *node)
{
    // Decide if the node represents a call to dFdy()
    if (node->getOp() != EOpDFdy)
    {
        return true;
    }

    // Copy the dFdy node so we can replace it with the corrected value
    TIntermUnary *newDfdy = node->deepCopy()->getAsUnaryNode();

    size_t objectSize    = node->getType().getObjectSize();
    TOperator multiplyOp = (objectSize == 1) ? EOpMul : EOpVectorTimesScalar;

    // Correct dFdy()'s value:
    // (dFdy() * ANGLEUniforms.viewportYScale)
    TIntermBinary *correctedDfdy =
        new TIntermBinary(multiplyOp, newDfdy, mViewportYScale->deepCopy());

    // Replace the old dFdy node with the new node that contains the corrected value
    queueReplacement(correctedDfdy, OriginalNode::IS_DROPPED);

    return true;
}

}  // anonymous namespace

bool RewriteDfdy(TCompiler *compiler,
                 TIntermNode *root,
                 const TSymbolTable &symbolTable,
                 int shaderVersion,
                 TIntermBinary *viewportYScale)
{
    // dFdy is only valid in GLSL 3.0 and later.
    if (shaderVersion < 300)
        return true;

    return Traverser::Apply(compiler, root, symbolTable, viewportYScale);
}

}  // namespace sh
