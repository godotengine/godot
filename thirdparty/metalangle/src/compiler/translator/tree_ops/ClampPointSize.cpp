//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClampPointSize.cpp: Limit the value that is written to gl_PointSize.
//

#include "compiler/translator/tree_ops/ClampPointSize.h"

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/FindSymbolNode.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"

namespace sh
{

bool ClampPointSize(TCompiler *compiler,
                    TIntermBlock *root,
                    float maxPointSize,
                    TSymbolTable *symbolTable)
{
    // Only clamp gl_PointSize if it's used in the shader.
    if (!FindSymbolNode(root, ImmutableString("gl_PointSize")))
    {
        return true;
    }

    TIntermSymbol *pointSizeNode = new TIntermSymbol(BuiltInVariable::gl_PointSize());

    TConstantUnion *maxPointSizeConstant = new TConstantUnion();
    maxPointSizeConstant->setFConst(maxPointSize);
    TIntermConstantUnion *maxPointSizeNode =
        new TIntermConstantUnion(maxPointSizeConstant, TType(EbtFloat, EbpHigh, EvqConst));

    // min(gl_PointSize, maxPointSize)
    TIntermSequence *minArguments = new TIntermSequence();
    minArguments->push_back(pointSizeNode->deepCopy());
    minArguments->push_back(maxPointSizeNode);
    TIntermTyped *clampedPointSize =
        CreateBuiltInFunctionCallNode("min", minArguments, *symbolTable, 100);

    // gl_PointSize = min(gl_PointSize, maxPointSize)
    TIntermBinary *assignPointSize = new TIntermBinary(EOpAssign, pointSizeNode, clampedPointSize);

    return RunAtTheEndOfShader(compiler, root, assignPointSize, symbolTable);
}

}  // namespace sh
