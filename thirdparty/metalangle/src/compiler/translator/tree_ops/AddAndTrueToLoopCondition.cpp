//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/AddAndTrueToLoopCondition.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// An AST traverser that rewrites for and while loops by replacing "condition" with
// "condition && true" to work around condition bug on Intel Mac.
class AddAndTrueToLoopConditionTraverser : public TIntermTraverser
{
  public:
    AddAndTrueToLoopConditionTraverser() : TIntermTraverser(true, false, false) {}

    bool visitLoop(Visit, TIntermLoop *loop) override
    {
        // do-while loop doesn't have this bug.
        if (loop->getType() != ELoopFor && loop->getType() != ELoopWhile)
        {
            return true;
        }

        // For loop may not have a condition.
        if (loop->getCondition() == nullptr)
        {
            return true;
        }

        // Constant true.
        TIntermTyped *trueValue = CreateBoolNode(true);

        // CONDITION && true.
        TIntermBinary *andOp = new TIntermBinary(EOpLogicalAnd, loop->getCondition(), trueValue);
        loop->setCondition(andOp);

        return true;
    }
};

}  // anonymous namespace

bool AddAndTrueToLoopCondition(TCompiler *compiler, TIntermNode *root)
{
    AddAndTrueToLoopConditionTraverser traverser;
    root->traverse(&traverser);
    return compiler->validateAST(root);
}

}  // namespace sh
