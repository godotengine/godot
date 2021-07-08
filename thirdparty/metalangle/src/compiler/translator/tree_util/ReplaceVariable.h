//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ReplaceVariable.h: Replace all references to a specific variable in the AST with references to
// another variable.

#ifndef COMPILER_TRANSLATOR_TREEUTIL_REPLACEVARIABLE_H_
#define COMPILER_TRANSLATOR_TREEUTIL_REPLACEVARIABLE_H_

#include "common/debug.h"

#include <stack>
#include <unordered_map>

namespace sh
{

class TCompiler;
class TFunction;
class TIntermAggregate;
class TIntermBlock;
class TIntermFunctionPrototype;
class TIntermNode;
class TIntermTyped;
class TSymbolTable;
class TVariable;

ANGLE_NO_DISCARD bool ReplaceVariable(TCompiler *compiler,
                                      TIntermBlock *root,
                                      const TVariable *toBeReplaced,
                                      const TVariable *replacement);
ANGLE_NO_DISCARD bool ReplaceVariableWithTyped(TCompiler *compiler,
                                               TIntermBlock *root,
                                               const TVariable *toBeReplaced,
                                               const TIntermTyped *replacement);

// A helper class to keep track of opaque variable re-typing during a pass.  Unlike the above
// functions, this can be used to replace all opaque variables of a certain type with another in a
// pass that possibly does other related transformations.  Only opaque variables are supported as
// replacing local variables is not supported.
//
// The class uses "old" to refer to the original type of the variable and "new" to refer to the type
// that will replace it.
//
// - replaceGlobalVariable(): Call to track a global variable that is replaced.
// - in TIntermTraverser::visitFunctionPrototype():
//   * Call visitFunctionPrototype().
//   * For every replaced parameter, call replaceFunctionParam().
//   * call convertFunctionPrototype() to convert the prototype based on the above replacements
//     and track the function with its replacement.
//   * Call replaceFunction() to track the function that is replaced.
// - In PreVisit of TIntermTraverser::visitAggregate():
//   * call preVisitAggregate()
// - In TIntermTraverser::visitSymbol():
//   * Replace non-function-call-argument symbols that refer to a global or function param with the
//     replacement (getVariableReplacement()).
//   * For function call arguments, call replaceFunctionCallArg() to track the replacement.
// - In PostVisit of TIntermTraverser::visitAggregate():
//   * Convert built-in functions per case.  Call convertASTFunction() for non built-in functions
//     for the replacement to be created.
//   * Call postVisitAggregate() when done.
//
class RetypeOpaqueVariablesHelper
{
  public:
    RetypeOpaqueVariablesHelper() {}
    ~RetypeOpaqueVariablesHelper() {}

    // Global variable handling:
    void replaceGlobalVariable(const TVariable *oldVar, TVariable *newVar)
    {
        ASSERT(mReplacedGlobalVariables.count(oldVar) == 0);
        mReplacedGlobalVariables[oldVar] = newVar;
    }
    TVariable *getVariableReplacement(const TVariable *oldVar) const
    {
        if (mReplacedGlobalVariables.count(oldVar) != 0)
        {
            return mReplacedGlobalVariables.at(oldVar);
        }
        else
        {
            // This function should only be called if the variable is expected to have been
            // replaced either way (as a global variable or a function parameter).
            ASSERT(mReplacedFunctionParams.count(oldVar) != 0);
            return mReplacedFunctionParams.at(oldVar);
        }
    }

    // Function parameters handling:
    void visitFunctionPrototype() { mReplacedFunctionParams.clear(); }
    void replaceFunctionParam(const TVariable *oldParam, TVariable *newParam)
    {
        ASSERT(mReplacedFunctionParams.count(oldParam) == 0);
        mReplacedFunctionParams[oldParam] = newParam;
    }
    TVariable *getFunctionParamReplacement(const TVariable *oldParam) const
    {
        ASSERT(mReplacedFunctionParams.count(oldParam) != 0);
        return mReplacedFunctionParams.at(oldParam);
    }

    // Function call arguments handling:
    void preVisitAggregate() { mReplacedFunctionCallArgs.emplace(); }
    bool isInAggregate() const { return !mReplacedFunctionCallArgs.empty(); }
    void postVisitAggregate() { mReplacedFunctionCallArgs.pop(); }
    void replaceFunctionCallArg(const TIntermNode *oldArg, TIntermTyped *newArg)
    {
        ASSERT(mReplacedFunctionCallArgs.top().count(oldArg) == 0);
        mReplacedFunctionCallArgs.top()[oldArg] = newArg;
    }
    TIntermTyped *getFunctionCallArgReplacement(const TIntermNode *oldArg) const
    {
        ASSERT(mReplacedFunctionCallArgs.top().count(oldArg) != 0);
        return mReplacedFunctionCallArgs.top().at(oldArg);
    }

    // Helper code conversion methods.
    TIntermFunctionPrototype *convertFunctionPrototype(TSymbolTable *symbolTable,
                                                       const TFunction *oldFunction);
    TIntermAggregate *convertASTFunction(TIntermAggregate *node);

  private:
    // A map from the old global variable to the new one.
    std::unordered_map<const TVariable *, TVariable *> mReplacedGlobalVariables;

    // A map from functions with old type parameters to one where that's replaced with the new type.
    std::unordered_map<const TFunction *, TFunction *> mReplacedFunctions;

    // A map from function old type parameters to their replacement new type parameter for the
    // current function definition.
    std::unordered_map<const TVariable *, TVariable *> mReplacedFunctionParams;

    // A map from function call old type arguments to their replacement for the current function
    // call.
    std::stack<std::unordered_map<const TIntermNode *, TIntermTyped *>> mReplacedFunctionCallArgs;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_REPLACEVARIABLE_H_
