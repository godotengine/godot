//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClampFragDepth.cpp: Limit the value that is written to gl_FragDepth to the range [0.0, 1.0].
// The clamping is run at the very end of shader execution, and is only performed if the shader
// statically accesses gl_FragDepth.
//

#include "compiler/translator/tree_ops/ClampFragDepth.h"

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/FindSymbolNode.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"

namespace sh
{

bool ClampFragDepth(TCompiler *compiler, TIntermBlock *root, TSymbolTable *symbolTable)
{
    // Only clamp gl_FragDepth if it's used in the shader.
    if (!FindSymbolNode(root, ImmutableString("gl_FragDepth")))
    {
        return true;
    }

    TIntermSymbol *fragDepthNode = new TIntermSymbol(BuiltInVariable::gl_FragDepth());

    TIntermTyped *minFragDepthNode = CreateZeroNode(TType(EbtFloat, EbpHigh, EvqConst));

    TConstantUnion *maxFragDepthConstant = new TConstantUnion();
    maxFragDepthConstant->setFConst(1.0);
    TIntermConstantUnion *maxFragDepthNode =
        new TIntermConstantUnion(maxFragDepthConstant, TType(EbtFloat, EbpHigh, EvqConst));

    // clamp(gl_FragDepth, 0.0, 1.0)
    TIntermSequence *clampArguments = new TIntermSequence();
    clampArguments->push_back(fragDepthNode->deepCopy());
    clampArguments->push_back(minFragDepthNode);
    clampArguments->push_back(maxFragDepthNode);
    TIntermTyped *clampedFragDepth =
        CreateBuiltInFunctionCallNode("clamp", clampArguments, *symbolTable, 100);

    // gl_FragDepth = clamp(gl_FragDepth, 0.0, 1.0)
    TIntermBinary *assignFragDepth = new TIntermBinary(EOpAssign, fragDepthNode, clampedFragDepth);

    return RunAtTheEndOfShader(compiler, root, assignFragDepth, symbolTable);
}

}  // namespace sh
