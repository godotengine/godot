//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RewriteDoWhile.cpp: rewrites do-while loops using another equivalent
// construct.

#include "compiler/translator/tree_ops/RewriteDoWhile.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// An AST traverser that rewrites loops of the form
//   do {
//     CODE;
//   } while (CONDITION)
//
// to loops of the form
//   bool temp = false;
//   while (true) {
//     if (temp) {
//       if (!CONDITION) {
//         break;
//       }
//     }
//     temp = true;
//     CODE;
//   }
//
// The reason we don't use a simpler form, with for example just (temp && !CONDITION) in the
// while condition, is that short-circuit is often badly supported by driver shader compiler.
// The double if has the same effect, but forces shader compilers to behave.
//
// TODO(cwallez) when UnfoldShortCircuitIntoIf handles loops correctly, revisit this as we might
// be able to use while (temp || CONDITION) with temp initially set to true then run
// UnfoldShortCircuitIntoIf
class DoWhileRewriter : public TIntermTraverser
{
  public:
    DoWhileRewriter(TSymbolTable *symbolTable) : TIntermTraverser(true, false, false, symbolTable)
    {}

    bool visitBlock(Visit, TIntermBlock *node) override
    {
        // A well-formed AST can only have do-while inside TIntermBlock. By doing a prefix traversal
        // we are able to replace the do-while in the sequence directly as the content of the
        // do-while will be traversed later.

        TIntermSequence *statements = node->getSequence();

        // The statements vector will have new statements inserted when we encounter a do-while,
        // which prevents us from using a range-based for loop. Using the usual i++ works, as
        // the (two) new statements inserted replace the statement at the current position.
        for (size_t i = 0; i < statements->size(); i++)
        {
            TIntermNode *statement = (*statements)[i];
            TIntermLoop *loop      = statement->getAsLoopNode();

            if (loop == nullptr || loop->getType() != ELoopDoWhile)
            {
                continue;
            }

            // Found a loop to change.
            const TType *boolType = StaticType::Get<EbtBool, EbpUndefined, EvqTemporary, 1, 1>();
            TVariable *conditionVariable = CreateTempVariable(mSymbolTable, boolType);

            // bool temp = false;
            TIntermDeclaration *tempDeclaration =
                CreateTempInitDeclarationNode(conditionVariable, CreateBoolNode(false));

            // temp = true;
            TIntermBinary *assignTrue =
                CreateTempAssignmentNode(conditionVariable, CreateBoolNode(true));

            // if (temp) {
            //   if (!CONDITION) {
            //     break;
            //   }
            // }
            TIntermIfElse *breakIf = nullptr;
            {
                TIntermBranch *breakStatement = new TIntermBranch(EOpBreak, nullptr);

                TIntermBlock *breakBlock = new TIntermBlock();
                breakBlock->getSequence()->push_back(breakStatement);

                TIntermUnary *negatedCondition =
                    new TIntermUnary(EOpLogicalNot, loop->getCondition(), nullptr);

                TIntermIfElse *innerIf = new TIntermIfElse(negatedCondition, breakBlock, nullptr);

                TIntermBlock *innerIfBlock = new TIntermBlock();
                innerIfBlock->getSequence()->push_back(innerIf);

                breakIf = new TIntermIfElse(CreateTempSymbolNode(conditionVariable), innerIfBlock,
                                            nullptr);
            }

            // Assemble the replacement loops, reusing the do-while loop's body and inserting our
            // statements at the front.
            TIntermLoop *newLoop = nullptr;
            {
                TIntermBlock *body = loop->getBody();
                if (body == nullptr)
                {
                    body = new TIntermBlock();
                }
                auto sequence = body->getSequence();
                sequence->insert(sequence->begin(), assignTrue);
                sequence->insert(sequence->begin(), breakIf);

                newLoop = new TIntermLoop(ELoopWhile, nullptr, CreateBoolNode(true), nullptr, body);
            }

            TIntermSequence replacement;
            replacement.push_back(tempDeclaration);
            replacement.push_back(newLoop);

            node->replaceChildNodeWithMultiple(loop, replacement);
        }
        return true;
    }
};

}  // anonymous namespace

bool RewriteDoWhile(TCompiler *compiler, TIntermNode *root, TSymbolTable *symbolTable)
{
    DoWhileRewriter rewriter(symbolTable);

    root->traverse(&rewriter);

    return compiler->validateAST(root);
}

}  // namespace sh
