//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WrapSwitchStatementsInBlocks.cpp: Wrap switch statements in blocks and declare all switch-scoped
// variables there to make the AST compatible with HLSL output.
//
// switch (init)
// {
//     case 0:
//         float f;
//     default:
//         f = 1.0;
// }
//
// becomes
//
// {
//     float f;
//     switch (init)
//     {
//         case 0:
//         default:
//             f = 1.0;
//     }
// }

#include "compiler/translator/tree_ops/WrapSwitchStatementsInBlocks.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class WrapSwitchStatementsInBlocksTraverser : public TIntermTraverser
{
  public:
    WrapSwitchStatementsInBlocksTraverser() : TIntermTraverser(true, false, false) {}

    bool visitSwitch(Visit visit, TIntermSwitch *node) override;
};

bool WrapSwitchStatementsInBlocksTraverser::visitSwitch(Visit, TIntermSwitch *node)
{
    std::vector<TIntermDeclaration *> declarations;
    TIntermSequence *statementList = node->getStatementList()->getSequence();
    for (TIntermNode *statement : *statementList)
    {
        TIntermDeclaration *asDeclaration = statement->getAsDeclarationNode();
        if (asDeclaration)
        {
            declarations.push_back(asDeclaration);
        }
    }
    if (declarations.empty())
    {
        // We don't need to wrap the switch if it doesn't contain declarations as its direct
        // descendants.
        return true;
    }

    TIntermBlock *wrapperBlock = new TIntermBlock();
    for (TIntermDeclaration *declaration : declarations)
    {
        // SeparateDeclarations should have already been run.
        ASSERT(declaration->getSequence()->size() == 1);

        TIntermDeclaration *declarationInBlock = new TIntermDeclaration();
        TIntermSymbol *declaratorAsSymbol = declaration->getSequence()->at(0)->getAsSymbolNode();
        if (declaratorAsSymbol)
        {
            // This is a simple declaration like: "float f;"
            // Remove the declaration from inside the switch and put it in the wrapping block.
            TIntermSequence emptyReplacement;
            mMultiReplacements.push_back(NodeReplaceWithMultipleEntry(
                node->getStatementList(), declaration, emptyReplacement));

            declarationInBlock->appendDeclarator(declaratorAsSymbol->deepCopy());
            // The declaration can't be the last statement inside the switch since unused variables
            // should already have been pruned.
            ASSERT(declaration != statementList->back());
        }
        else
        {
            // This is an init declaration like: "float f = 0.0;"
            // Change the init declaration inside the switch into an assignment and put a plain
            // declaration in the wrapping block.
            TIntermBinary *declaratorAsBinary =
                declaration->getSequence()->at(0)->getAsBinaryNode();
            ASSERT(declaratorAsBinary);

            TIntermBinary *initAssignment = new TIntermBinary(
                EOpAssign, declaratorAsBinary->getLeft(), declaratorAsBinary->getRight());

            queueReplacementWithParent(node->getStatementList(), declaration, initAssignment,
                                       OriginalNode::IS_DROPPED);

            declarationInBlock->appendDeclarator(declaratorAsBinary->getLeft()->deepCopy());
        }
        wrapperBlock->appendStatement(declarationInBlock);
    }

    wrapperBlock->appendStatement(node);
    queueReplacement(wrapperBlock, OriginalNode::BECOMES_CHILD);

    // Should be fine to process multiple switch statements, even nesting ones in the same
    // traversal.
    return true;
}

}  // anonymous namespace

// Wrap switch statements in the AST into blocks when needed.
bool WrapSwitchStatementsInBlocks(TCompiler *compiler, TIntermBlock *root)
{
    WrapSwitchStatementsInBlocksTraverser traverser;
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
