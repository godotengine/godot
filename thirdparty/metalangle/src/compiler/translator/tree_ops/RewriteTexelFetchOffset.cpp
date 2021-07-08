//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Implementation of texelFetchOffset translation issue workaround.
// See header for more info.

#include "compiler/translator/tree_ops/RewriteTexelFetchOffset.h"

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
                                       int shaderVersion);

  private:
    Traverser(const TSymbolTable &symbolTable, int shaderVersion);
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    void nextIteration();

    const TSymbolTable *symbolTable;
    const int shaderVersion;
    bool mFound = false;
};

Traverser::Traverser(const TSymbolTable &symbolTable, int shaderVersion)
    : TIntermTraverser(true, false, false), symbolTable(&symbolTable), shaderVersion(shaderVersion)
{}

// static
bool Traverser::Apply(TCompiler *compiler,
                      TIntermNode *root,
                      const TSymbolTable &symbolTable,
                      int shaderVersion)
{
    Traverser traverser(symbolTable, shaderVersion);
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (traverser.mFound)
        {
            if (!traverser.updateTree(compiler, root))
            {
                return false;
            }
        }
    } while (traverser.mFound);

    return true;
}

void Traverser::nextIteration()
{
    mFound = false;
}

bool Traverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (mFound)
    {
        return false;
    }

    // Decide if the node represents the call of texelFetchOffset.
    if (node->getOp() != EOpCallBuiltInFunction)
    {
        return true;
    }

    ASSERT(node->getFunction()->symbolType() == SymbolType::BuiltIn);
    if (node->getFunction()->name() != "texelFetchOffset")
    {
        return true;
    }

    // Potential problem case detected, apply workaround.
    const TIntermSequence *sequence = node->getSequence();
    ASSERT(sequence->size() == 4u);

    // Decide if the sampler is a 2DArray sampler. In that case position is ivec3 and offset is
    // ivec2.
    bool is2DArray = sequence->at(1)->getAsTyped()->getNominalSize() == 3 &&
                     sequence->at(3)->getAsTyped()->getNominalSize() == 2;

    // Create new node that represents the call of function texelFetch.
    // Its argument list will be: texelFetch(sampler, Position+offset, lod).

    TIntermSequence *texelFetchArguments = new TIntermSequence();

    // sampler
    texelFetchArguments->push_back(sequence->at(0));

    // Position
    TIntermTyped *texCoordNode = sequence->at(1)->getAsTyped();
    ASSERT(texCoordNode);

    // offset
    TIntermTyped *offsetNode = nullptr;
    ASSERT(sequence->at(3)->getAsTyped());
    if (is2DArray)
    {
        // For 2DArray samplers, Position is ivec3 and offset is ivec2;
        // So offset must be converted into an ivec3 before being added to Position.
        TIntermSequence *constructOffsetIvecArguments = new TIntermSequence();
        constructOffsetIvecArguments->push_back(sequence->at(3)->getAsTyped());

        TIntermTyped *zeroNode = CreateZeroNode(TType(EbtInt));
        constructOffsetIvecArguments->push_back(zeroNode);

        offsetNode = TIntermAggregate::CreateConstructor(texCoordNode->getType(),
                                                         constructOffsetIvecArguments);
        offsetNode->setLine(texCoordNode->getLine());
    }
    else
    {
        offsetNode = sequence->at(3)->getAsTyped();
    }

    // Position+offset
    TIntermBinary *add = new TIntermBinary(EOpAdd, texCoordNode, offsetNode);
    add->setLine(texCoordNode->getLine());
    texelFetchArguments->push_back(add);

    // lod
    texelFetchArguments->push_back(sequence->at(2));

    ASSERT(texelFetchArguments->size() == 3u);

    TIntermTyped *texelFetchNode = CreateBuiltInFunctionCallNode("texelFetch", texelFetchArguments,
                                                                 *symbolTable, shaderVersion);
    texelFetchNode->setLine(node->getLine());

    // Replace the old node by this new node.
    queueReplacement(texelFetchNode, OriginalNode::IS_DROPPED);
    mFound = true;
    return false;
}

}  // anonymous namespace

bool RewriteTexelFetchOffset(TCompiler *compiler,
                             TIntermNode *root,
                             const TSymbolTable &symbolTable,
                             int shaderVersion)
{
    // texelFetchOffset is only valid in GLSL 3.0 and later.
    if (shaderVersion < 300)
        return true;

    return Traverser::Apply(compiler, root, symbolTable, shaderVersion);
}

}  // namespace sh
