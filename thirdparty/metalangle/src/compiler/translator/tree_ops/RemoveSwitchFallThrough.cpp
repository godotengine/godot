//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveSwitchFallThrough.cpp: Remove fall-through from switch statements.
// Note that it is unsafe to do further AST transformations on the AST generated
// by this function. It leaves duplicate nodes in the AST making replacements
// unreliable.

#include "compiler/translator/tree_ops/RemoveSwitchFallThrough.h"

#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class RemoveSwitchFallThroughTraverser : public TIntermTraverser
{
  public:
    static TIntermBlock *removeFallThrough(TIntermBlock *statementList,
                                           PerformanceDiagnostics *perfDiagnostics);

  private:
    RemoveSwitchFallThroughTraverser(TIntermBlock *statementList,
                                     PerformanceDiagnostics *perfDiagnostics);

    void visitSymbol(TIntermSymbol *node) override;
    void visitConstantUnion(TIntermConstantUnion *node) override;
    bool visitDeclaration(Visit, TIntermDeclaration *node) override;
    bool visitBinary(Visit, TIntermBinary *node) override;
    bool visitUnary(Visit, TIntermUnary *node) override;
    bool visitTernary(Visit visit, TIntermTernary *node) override;
    bool visitSwizzle(Visit, TIntermSwizzle *node) override;
    bool visitIfElse(Visit visit, TIntermIfElse *node) override;
    bool visitSwitch(Visit, TIntermSwitch *node) override;
    bool visitCase(Visit, TIntermCase *node) override;
    bool visitAggregate(Visit, TIntermAggregate *node) override;
    bool visitBlock(Visit, TIntermBlock *node) override;
    bool visitLoop(Visit, TIntermLoop *node) override;
    bool visitBranch(Visit, TIntermBranch *node) override;

    void outputSequence(TIntermSequence *sequence, size_t startIndex);
    void handlePreviousCase();

    TIntermBlock *mStatementList;
    TIntermBlock *mStatementListOut;
    bool mLastStatementWasBreak;
    TIntermBlock *mPreviousCase;
    std::vector<TIntermBlock *> mCasesSharingBreak;
    PerformanceDiagnostics *mPerfDiagnostics;
};

TIntermBlock *RemoveSwitchFallThroughTraverser::removeFallThrough(
    TIntermBlock *statementList,
    PerformanceDiagnostics *perfDiagnostics)
{
    RemoveSwitchFallThroughTraverser rm(statementList, perfDiagnostics);
    ASSERT(statementList);
    statementList->traverse(&rm);
    ASSERT(rm.mPreviousCase || statementList->getSequence()->empty());
    if (!rm.mLastStatementWasBreak && rm.mPreviousCase)
    {
        // Make sure that there's a branch at the end of the final case inside the switch statement.
        // This also ensures that any cases that fall through to the final case will get the break.
        TIntermBranch *finalBreak = new TIntermBranch(EOpBreak, nullptr);
        rm.mPreviousCase->getSequence()->push_back(finalBreak);
        rm.mLastStatementWasBreak = true;
    }
    rm.handlePreviousCase();
    return rm.mStatementListOut;
}

RemoveSwitchFallThroughTraverser::RemoveSwitchFallThroughTraverser(
    TIntermBlock *statementList,
    PerformanceDiagnostics *perfDiagnostics)
    : TIntermTraverser(true, false, false),
      mStatementList(statementList),
      mLastStatementWasBreak(false),
      mPreviousCase(nullptr),
      mPerfDiagnostics(perfDiagnostics)
{
    mStatementListOut = new TIntermBlock();
}

void RemoveSwitchFallThroughTraverser::visitSymbol(TIntermSymbol *node)
{
    // Note that this assumes that switch statements which don't begin by a case statement
    // have already been weeded out in validation.
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
}

void RemoveSwitchFallThroughTraverser::visitConstantUnion(TIntermConstantUnion *node)
{
    // Conditions of case labels are not traversed, so this is a constant statement like "0;".
    // These are no-ops so there's no need to add them back to the statement list. Should have
    // already been pruned out of the AST, in fact.
    UNREACHABLE();
}

bool RemoveSwitchFallThroughTraverser::visitDeclaration(Visit, TIntermDeclaration *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitBinary(Visit, TIntermBinary *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitUnary(Visit, TIntermUnary *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitTernary(Visit, TIntermTernary *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitSwizzle(Visit, TIntermSwizzle *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitIfElse(Visit, TIntermIfElse *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitSwitch(Visit, TIntermSwitch *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    // Don't go into nested switch statements
    return false;
}

void RemoveSwitchFallThroughTraverser::outputSequence(TIntermSequence *sequence, size_t startIndex)
{
    for (size_t i = startIndex; i < sequence->size(); ++i)
    {
        mStatementListOut->getSequence()->push_back(sequence->at(i));
    }
}

void RemoveSwitchFallThroughTraverser::handlePreviousCase()
{
    if (mPreviousCase)
        mCasesSharingBreak.push_back(mPreviousCase);
    if (mLastStatementWasBreak)
    {
        for (size_t i = 0; i < mCasesSharingBreak.size(); ++i)
        {
            ASSERT(!mCasesSharingBreak.at(i)->getSequence()->empty());
            if (mCasesSharingBreak.at(i)->getSequence()->size() == 1)
            {
                // Fall-through is allowed in case the label has no statements.
                outputSequence(mCasesSharingBreak.at(i)->getSequence(), 0);
            }
            else
            {
                // Include all the statements that this case can fall through under the same label.
                if (mCasesSharingBreak.size() > i + 1u)
                {
                    mPerfDiagnostics->warning(mCasesSharingBreak.at(i)->getLine(),
                                              "Performance: non-empty fall-through cases in "
                                              "switch statements generate extra code.",
                                              "switch");
                }
                for (size_t j = i; j < mCasesSharingBreak.size(); ++j)
                {
                    size_t startIndex =
                        j > i ? 1 : 0;  // Add the label only from the first sequence.
                    outputSequence(mCasesSharingBreak.at(j)->getSequence(), startIndex);
                }
            }
        }
        mCasesSharingBreak.clear();
    }
    mLastStatementWasBreak = false;
    mPreviousCase          = nullptr;
}

bool RemoveSwitchFallThroughTraverser::visitCase(Visit, TIntermCase *node)
{
    handlePreviousCase();
    mPreviousCase = new TIntermBlock();
    mPreviousCase->getSequence()->push_back(node);
    mPreviousCase->setLine(node->getLine());
    // Don't traverse the condition of the case statement
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitAggregate(Visit, TIntermAggregate *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool DoesBlockAlwaysBreak(TIntermBlock *node)
{
    if (node->getSequence()->empty())
    {
        return false;
    }

    TIntermBlock *lastStatementAsBlock = node->getSequence()->back()->getAsBlock();
    if (lastStatementAsBlock)
    {
        return DoesBlockAlwaysBreak(lastStatementAsBlock);
    }

    TIntermBranch *lastStatementAsBranch = node->getSequence()->back()->getAsBranchNode();
    return lastStatementAsBranch != nullptr;
}

bool RemoveSwitchFallThroughTraverser::visitBlock(Visit, TIntermBlock *node)
{
    if (node != mStatementList)
    {
        mPreviousCase->getSequence()->push_back(node);
        mLastStatementWasBreak = DoesBlockAlwaysBreak(node);
        return false;
    }
    return true;
}

bool RemoveSwitchFallThroughTraverser::visitLoop(Visit, TIntermLoop *node)
{
    mPreviousCase->getSequence()->push_back(node);
    mLastStatementWasBreak = false;
    return false;
}

bool RemoveSwitchFallThroughTraverser::visitBranch(Visit, TIntermBranch *node)
{
    mPreviousCase->getSequence()->push_back(node);
    // TODO: Verify that accepting return or continue statements here doesn't cause problems.
    mLastStatementWasBreak = true;
    return false;
}

}  // anonymous namespace

TIntermBlock *RemoveSwitchFallThrough(TIntermBlock *statementList,
                                      PerformanceDiagnostics *perfDiagnostics)
{
    return RemoveSwitchFallThroughTraverser::removeFallThrough(statementList, perfDiagnostics);
}

}  // namespace sh
