//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// BreakVariableAliasingInInnerLoops.h: To optimize simple assignments, the HLSL compiler frontend
//      may record a variable as aliasing another. Sometimes the alias information gets garbled
//      so we work around this issue by breaking the aliasing chain in inner loops.

#include "BreakVariableAliasingInInnerLoops.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

// A HLSL compiler developer gave us more details on the root cause and the workaround needed:
//     The root problem is that if the HLSL compiler is applying aliasing information even on
//     incomplete simulations (in this case, a single pass). The bug is triggered by an assignment
//     that comes from a series of assignments, possibly with swizzled or ternary operators with
//     known conditionals, where the source is before the loop.
//     So, a workaround is to add a +0 term to variables the first time they are assigned to in
//     an inner loop (if they are declared in an outside scope, otherwise there is no need).
//     This will break the aliasing chain.

// For simplicity here we add a +0 to any assignment that is in at least two nested loops. Because
// the bug only shows up with swizzles, and ternary assignment, whole array or whole structure
// assignment don't need a workaround.

namespace sh
{

namespace
{

class AliasingBreaker : public TIntermTraverser
{
  public:
    AliasingBreaker() : TIntermTraverser(true, false, true) {}

  protected:
    bool visitBinary(Visit visit, TIntermBinary *binary)
    {
        if (visit != PreVisit)
        {
            return false;
        }

        if (mLoopLevel < 2 || !binary->isAssignment())
        {
            return true;
        }

        TIntermTyped *B = binary->getRight();
        TType type      = B->getType();

        if (!type.isScalar() && !type.isVector() && !type.isMatrix())
        {
            return true;
        }

        if (type.isArray() || IsSampler(type.getBasicType()))
        {
            return true;
        }

        // We have a scalar / vector / matrix assignment with loop depth 2.
        // Transform it from
        //    A = B
        // to
        //    A = (B + typeof<B>(0));

        TIntermBinary *bPlusZero = new TIntermBinary(EOpAdd, B, CreateZeroNode(type));
        bPlusZero->setLine(B->getLine());

        binary->replaceChildNode(B, bPlusZero);

        return true;
    }

    bool visitLoop(Visit visit, TIntermLoop *loop)
    {
        if (visit == PreVisit)
        {
            mLoopLevel++;
        }
        else
        {
            ASSERT(mLoopLevel > 0);
            mLoopLevel--;
        }

        return true;
    }

  private:
    int mLoopLevel = 0;
};

}  // anonymous namespace

bool BreakVariableAliasingInInnerLoops(TCompiler *compiler, TIntermNode *root)
{
    AliasingBreaker breaker;
    root->traverse(&breaker);

    return compiler->validateAST(root);
}

}  // namespace sh
