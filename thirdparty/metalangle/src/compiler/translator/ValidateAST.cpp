//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/ValidateAST.h"

#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class ValidateAST : public TIntermTraverser
{
  public:
    static bool validate(TIntermNode *root,
                         TDiagnostics *diagnostics,
                         const ValidateASTOptions &options);

    void visitSymbol(TIntermSymbol *node) override;
    void visitConstantUnion(TIntermConstantUnion *node) override;
    bool visitSwizzle(Visit visit, TIntermSwizzle *node) override;
    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitUnary(Visit visit, TIntermUnary *node) override;
    bool visitTernary(Visit visit, TIntermTernary *node) override;
    bool visitIfElse(Visit visit, TIntermIfElse *node) override;
    bool visitSwitch(Visit visit, TIntermSwitch *node) override;
    bool visitCase(Visit visit, TIntermCase *node) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override;
    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitBlock(Visit visit, TIntermBlock *node) override;
    bool visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node) override;
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;
    bool visitLoop(Visit visit, TIntermLoop *node) override;
    bool visitBranch(Visit visit, TIntermBranch *node) override;
    void visitPreprocessorDirective(TIntermPreprocessorDirective *node) override;

  private:
    ValidateAST(TIntermNode *root, TDiagnostics *diagnostics, const ValidateASTOptions &options);

    // Visit as a generic node
    void visitNode(Visit visit, TIntermNode *node);

    void expectNonNullChildren(Visit visit, TIntermNode *node, size_t least_count);

    bool validateInternal();

    ValidateASTOptions mOptions;
    TDiagnostics *mDiagnostics;

    // For validateSingleParent:
    std::map<TIntermNode *, TIntermNode *> mParent;
    bool mSingleParentFailed = false;

    // For validateNullNodes
    bool mNullNodesFailed = false;
};

bool ValidateAST::validate(TIntermNode *root,
                           TDiagnostics *diagnostics,
                           const ValidateASTOptions &options)
{
    ValidateAST validate(root, diagnostics, options);
    root->traverse(&validate);
    return validate.validateInternal();
}

ValidateAST::ValidateAST(TIntermNode *root,
                         TDiagnostics *diagnostics,
                         const ValidateASTOptions &options)
    : TIntermTraverser(true, false, true, nullptr), mOptions(options), mDiagnostics(diagnostics)
{
    if (mOptions.validateSingleParent)
    {
        mParent[root] = nullptr;
    }
}

void ValidateAST::visitNode(Visit visit, TIntermNode *node)
{
    if (visit == PreVisit && mOptions.validateSingleParent)
    {
        size_t childCount = node->getChildCount();
        for (size_t i = 0; i < childCount; ++i)
        {
            TIntermNode *child = node->getChildNode(i);
            if (mParent.find(child) != mParent.end())
            {
                // If child is visited twice but through the same parent, the problem is in one of
                // the ancestors.
                if (mParent[child] != node)
                {
                    mDiagnostics->error(node->getLine(), "Found child with two parents",
                                        "<validateSingleParent>");
                    mSingleParentFailed = true;
                }
            }

            mParent[child] = node;
        }
    }
}

void ValidateAST::expectNonNullChildren(Visit visit, TIntermNode *node, size_t least_count)
{
    if (visit == PreVisit && mOptions.validateNullNodes)
    {
        size_t childCount = node->getChildCount();
        if (childCount < least_count)
        {
            mDiagnostics->error(node->getLine(), "Too few children", "<validateNullNodes>");
            mNullNodesFailed = true;
        }

        for (size_t i = 0; i < childCount; ++i)
        {
            if (node->getChildNode(i) == nullptr)
            {
                mDiagnostics->error(node->getLine(), "Found nullptr child", "<validateNullNodes>");
                mNullNodesFailed = true;
            }
        }
    }
}

void ValidateAST::visitSymbol(TIntermSymbol *node)
{
    visitNode(PreVisit, node);
}

void ValidateAST::visitConstantUnion(TIntermConstantUnion *node)
{
    visitNode(PreVisit, node);
}

bool ValidateAST::visitSwizzle(Visit visit, TIntermSwizzle *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitBinary(Visit visit, TIntermBinary *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitUnary(Visit visit, TIntermUnary *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitTernary(Visit visit, TIntermTernary *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitIfElse(Visit visit, TIntermIfElse *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitSwitch(Visit visit, TIntermSwitch *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitCase(Visit visit, TIntermCase *node)
{
    visitNode(visit, node);
    return true;
}

void ValidateAST::visitFunctionPrototype(TIntermFunctionPrototype *node)
{
    visitNode(PreVisit, node);
}

bool ValidateAST::visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitAggregate(Visit visit, TIntermAggregate *node)
{
    visitNode(visit, node);
    expectNonNullChildren(visit, node, 0);
    return true;
}

bool ValidateAST::visitBlock(Visit visit, TIntermBlock *node)
{
    visitNode(visit, node);
    expectNonNullChildren(visit, node, 0);
    return true;
}

bool ValidateAST::visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    visitNode(visit, node);
    expectNonNullChildren(visit, node, 0);
    return true;
}

bool ValidateAST::visitLoop(Visit visit, TIntermLoop *node)
{
    visitNode(visit, node);
    return true;
}

bool ValidateAST::visitBranch(Visit visit, TIntermBranch *node)
{
    visitNode(visit, node);
    return true;
}

void ValidateAST::visitPreprocessorDirective(TIntermPreprocessorDirective *node)
{
    visitNode(PreVisit, node);
}

bool ValidateAST::validateInternal()
{
    return !mSingleParentFailed && !mNullNodesFailed;
}

}  // anonymous namespace

bool ValidateAST(TIntermNode *root, TDiagnostics *diagnostics, const ValidateASTOptions &options)
{
    return ValidateAST::validate(root, diagnostics, options);
}

}  // namespace sh
