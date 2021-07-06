//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PruneEmptyCases.cpp: The PruneEmptyCases function prunes cases that are followed by nothing from
// the AST.

#include "compiler/translator/tree_ops/PruneEmptyCases.h"

#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

bool AreEmptyBlocks(TIntermSequence *statements);

bool IsEmptyBlock(TIntermNode *node)
{
    TIntermBlock *asBlock = node->getAsBlock();
    if (asBlock)
    {
        return AreEmptyBlocks(asBlock->getSequence());
    }
    // Empty declarations should have already been pruned, otherwise they would need to be handled
    // here. Note that declarations for struct types do contain a nameless child node.
    ASSERT(node->getAsDeclarationNode() == nullptr ||
           !node->getAsDeclarationNode()->getSequence()->empty());
    // Pure literal statements should also already be pruned.
    ASSERT(node->getAsConstantUnion() == nullptr);
    return false;
}

// Return true if all statements in "statements" consist only of empty blocks and no-op statements.
// Returns true also if there are no statements.
bool AreEmptyBlocks(TIntermSequence *statements)
{
    for (size_t i = 0u; i < statements->size(); ++i)
    {
        if (!IsEmptyBlock(statements->at(i)))
        {
            return false;
        }
    }
    return true;
}

class PruneEmptyCasesTraverser : private TIntermTraverser
{
  public:
    ANGLE_NO_DISCARD static bool apply(TCompiler *compiler, TIntermBlock *root);

  private:
    PruneEmptyCasesTraverser();
    bool visitSwitch(Visit visit, TIntermSwitch *node) override;
};

bool PruneEmptyCasesTraverser::apply(TCompiler *compiler, TIntermBlock *root)
{
    PruneEmptyCasesTraverser prune;
    root->traverse(&prune);
    return prune.updateTree(compiler, root);
}

PruneEmptyCasesTraverser::PruneEmptyCasesTraverser() : TIntermTraverser(true, false, false) {}

bool PruneEmptyCasesTraverser::visitSwitch(Visit visit, TIntermSwitch *node)
{
    // This may mutate the statementList, but that's okay, since traversal has not yet reached
    // there.
    TIntermBlock *statementList = node->getStatementList();
    TIntermSequence *statements = statementList->getSequence();

    // Iterate block children in reverse order. Cases that are only followed by other cases or empty
    // blocks are marked for pruning.
    size_t i                       = statements->size();
    size_t lastNoOpInStatementList = i;
    while (i > 0)
    {
        --i;
        TIntermNode *statement = statements->at(i);
        if (statement->getAsCaseNode() || IsEmptyBlock(statement))
        {
            lastNoOpInStatementList = i;
        }
        else
        {
            break;
        }
    }
    if (lastNoOpInStatementList == 0)
    {
        // Remove the entire switch statement, extracting the init expression if needed.
        TIntermTyped *init = node->getInit();
        if (init->hasSideEffects())
        {
            queueReplacement(init, OriginalNode::IS_DROPPED);
        }
        else
        {
            TIntermSequence emptyReplacement;
            ASSERT(getParentNode()->getAsBlock());
            mMultiReplacements.push_back(NodeReplaceWithMultipleEntry(getParentNode()->getAsBlock(),
                                                                      node, emptyReplacement));
        }
        return false;
    }
    if (lastNoOpInStatementList < statements->size())
    {
        statements->erase(statements->begin() + lastNoOpInStatementList, statements->end());
    }

    return true;
}

}  // namespace

bool PruneEmptyCases(TCompiler *compiler, TIntermBlock *root)
{
    return PruneEmptyCasesTraverser::apply(compiler, root);
}

}  // namespace sh
