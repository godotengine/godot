//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// gl_FragColor needs to broadcast to all color buffers in ES2 if
// GL_EXT_draw_buffers is explicitly enabled in a fragment shader.
//
// We emulate this by replacing all gl_FragColor with gl_FragData[0], and in the end
// of main() function, assigning gl_FragData[1], ..., gl_FragData[maxDrawBuffers-1]
// with gl_FragData[0].
//

#include "compiler/translator/tree_ops/EmulateGLFragColorBroadcast.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"

namespace sh
{

namespace
{

constexpr const ImmutableString kGlFragDataString("gl_FragData");

class GLFragColorBroadcastTraverser : public TIntermTraverser
{
  public:
    GLFragColorBroadcastTraverser(int maxDrawBuffers, TSymbolTable *symbolTable, int shaderVersion)
        : TIntermTraverser(true, false, false, symbolTable),
          mGLFragColorUsed(false),
          mMaxDrawBuffers(maxDrawBuffers),
          mShaderVersion(shaderVersion)
    {}

    ANGLE_NO_DISCARD bool broadcastGLFragColor(TCompiler *compiler, TIntermBlock *root);

    bool isGLFragColorUsed() const { return mGLFragColorUsed; }

  protected:
    void visitSymbol(TIntermSymbol *node) override;

    TIntermBinary *constructGLFragDataNode(int index) const;
    TIntermBinary *constructGLFragDataAssignNode(int index) const;

  private:
    bool mGLFragColorUsed;
    int mMaxDrawBuffers;
    const int mShaderVersion;
};

TIntermBinary *GLFragColorBroadcastTraverser::constructGLFragDataNode(int index) const
{
    TIntermSymbol *symbol =
        ReferenceBuiltInVariable(kGlFragDataString, *mSymbolTable, mShaderVersion);
    TIntermTyped *indexNode = CreateIndexNode(index);

    TIntermBinary *binary = new TIntermBinary(EOpIndexDirect, symbol, indexNode);
    return binary;
}

TIntermBinary *GLFragColorBroadcastTraverser::constructGLFragDataAssignNode(int index) const
{
    TIntermTyped *fragDataIndex = constructGLFragDataNode(index);
    TIntermTyped *fragDataZero  = constructGLFragDataNode(0);

    return new TIntermBinary(EOpAssign, fragDataIndex, fragDataZero);
}

void GLFragColorBroadcastTraverser::visitSymbol(TIntermSymbol *node)
{
    if (node->variable().symbolType() == SymbolType::BuiltIn && node->getName() == "gl_FragColor")
    {
        queueReplacement(constructGLFragDataNode(0), OriginalNode::IS_DROPPED);
        mGLFragColorUsed = true;
    }
}

bool GLFragColorBroadcastTraverser::broadcastGLFragColor(TCompiler *compiler, TIntermBlock *root)
{
    ASSERT(mMaxDrawBuffers > 1);
    if (!mGLFragColorUsed)
    {
        return true;
    }

    TIntermBlock *broadcastBlock = new TIntermBlock();
    // Now insert statements
    //   gl_FragData[1] = gl_FragData[0];
    //   ...
    //   gl_FragData[maxDrawBuffers - 1] = gl_FragData[0];
    for (int colorIndex = 1; colorIndex < mMaxDrawBuffers; ++colorIndex)
    {
        broadcastBlock->appendStatement(constructGLFragDataAssignNode(colorIndex));
    }
    return RunAtTheEndOfShader(compiler, root, broadcastBlock, mSymbolTable);
}

}  // namespace

bool EmulateGLFragColorBroadcast(TCompiler *compiler,
                                 TIntermBlock *root,
                                 int maxDrawBuffers,
                                 std::vector<sh::ShaderVariable> *outputVariables,
                                 TSymbolTable *symbolTable,
                                 int shaderVersion)
{
    ASSERT(maxDrawBuffers > 1);
    GLFragColorBroadcastTraverser traverser(maxDrawBuffers, symbolTable, shaderVersion);
    root->traverse(&traverser);
    if (traverser.isGLFragColorUsed())
    {
        if (!traverser.updateTree(compiler, root))
        {
            return false;
        }
        if (!traverser.broadcastGLFragColor(compiler, root))
        {
            return false;
        }

        for (auto &var : *outputVariables)
        {
            if (var.name == "gl_FragColor")
            {
                // TODO(zmo): Find a way to keep the original variable information.
                var.name       = "gl_FragData";
                var.mappedName = "gl_FragData";
                var.arraySizes.push_back(maxDrawBuffers);
                ASSERT(var.arraySizes.size() == 1u);
            }
        }
    }

    return true;
}

}  // namespace sh
