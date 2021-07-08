//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/ValidateGlobalInitializer.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

const int kMaxAllowedTraversalDepth = 256;

class ValidateGlobalInitializerTraverser : public TIntermTraverser
{
  public:
    ValidateGlobalInitializerTraverser(int shaderVersion, bool isWebGL);

    void visitSymbol(TIntermSymbol *node) override;
    void visitConstantUnion(TIntermConstantUnion *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    bool visitBinary(Visit visit, TIntermBinary *node) override;
    bool visitUnary(Visit visit, TIntermUnary *node) override;

    bool isValid() const { return mIsValid && mMaxDepth < mMaxAllowedDepth; }
    bool issueWarning() const { return mIssueWarning; }

  private:
    int mShaderVersion;
    bool mIsWebGL;
    bool mIsValid;
    bool mIssueWarning;
};

void ValidateGlobalInitializerTraverser::visitSymbol(TIntermSymbol *node)
{
    // ESSL 1.00 section 4.3 (or ESSL 3.00 section 4.3):
    // Global initializers must be constant expressions.
    switch (node->getType().getQualifier())
    {
        case EvqConst:
            break;
        case EvqGlobal:
        case EvqTemporary:
        case EvqUniform:
            // We allow these cases to be compatible with legacy ESSL 1.00 content.
            // Implement stricter rules for ESSL 3.00 since there's no legacy content to deal
            // with.
            if ((mShaderVersion >= 300) || !mIsWebGL)
            {
                mIsValid = false;
            }
            else
            {
                mIssueWarning = true;
            }
            break;
        default:
            mIsValid = false;
    }
}

void ValidateGlobalInitializerTraverser::visitConstantUnion(TIntermConstantUnion *node)
{
    // Constant unions that are not constant expressions may result from folding a ternary
    // expression.
    switch (node->getType().getQualifier())
    {
        case EvqConst:
            break;
        case EvqTemporary:
            if ((mShaderVersion >= 300) || !mIsWebGL)
            {
                mIsValid = false;
            }
            else
            {
                mIssueWarning = true;
            }
            break;
        default:
            UNREACHABLE();
    }
}

bool ValidateGlobalInitializerTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    // Disallow calls to user-defined functions and texture lookup functions in global variable
    // initializers.
    // This is done simply by disabling all function calls - built-in math functions don't use
    // the function call ops.
    if (node->isFunctionCall())
    {
        mIsValid = false;
    }
    return true;
}

bool ValidateGlobalInitializerTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    if (node->isAssignment())
    {
        mIsValid = false;
    }
    return true;
}

bool ValidateGlobalInitializerTraverser::visitUnary(Visit visit, TIntermUnary *node)
{
    if (node->isAssignment())
    {
        mIsValid = false;
    }
    return true;
}

ValidateGlobalInitializerTraverser::ValidateGlobalInitializerTraverser(int shaderVersion,
                                                                       bool isWebGL)
    : TIntermTraverser(true, false, false, nullptr),
      mShaderVersion(shaderVersion),
      mIsWebGL(isWebGL),
      mIsValid(true),
      mIssueWarning(false)
{
    setMaxAllowedDepth(kMaxAllowedTraversalDepth);
}

}  // namespace

bool ValidateGlobalInitializer(TIntermTyped *initializer,
                               int shaderVersion,
                               bool isWebGL,
                               bool *warning)
{
    ValidateGlobalInitializerTraverser validate(shaderVersion, isWebGL);
    initializer->traverse(&validate);
    ASSERT(warning != nullptr);
    *warning = validate.issueWarning();
    return validate.isValid();
}

}  // namespace sh
