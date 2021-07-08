//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ReplaceVariable.cpp: Replace all references to a specific variable in the AST with references to
// another variable.

#include "compiler/translator/tree_util/ReplaceVariable.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class ReplaceVariableTraverser : public TIntermTraverser
{
  public:
    ReplaceVariableTraverser(const TVariable *toBeReplaced, const TIntermTyped *replacement)
        : TIntermTraverser(true, false, false),
          mToBeReplaced(toBeReplaced),
          mReplacement(replacement)
    {}

    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == mToBeReplaced)
        {
            queueReplacement(mReplacement->deepCopy(), OriginalNode::IS_DROPPED);
        }
    }

  private:
    const TVariable *const mToBeReplaced;
    const TIntermTyped *const mReplacement;
};

}  // anonymous namespace

// Replaces every occurrence of a variable with another variable.
ANGLE_NO_DISCARD bool ReplaceVariable(TCompiler *compiler,
                                      TIntermBlock *root,
                                      const TVariable *toBeReplaced,
                                      const TVariable *replacement)
{
    ReplaceVariableTraverser traverser(toBeReplaced, new TIntermSymbol(replacement));
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

// Replaces every occurrence of a variable with a TIntermNode.
ANGLE_NO_DISCARD bool ReplaceVariableWithTyped(TCompiler *compiler,
                                               TIntermBlock *root,
                                               const TVariable *toBeReplaced,
                                               const TIntermTyped *replacement)
{
    ReplaceVariableTraverser traverser(toBeReplaced, replacement);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

TIntermFunctionPrototype *RetypeOpaqueVariablesHelper::convertFunctionPrototype(
    TSymbolTable *symbolTable,
    const TFunction *oldFunction)
{
    if (mReplacedFunctionParams.empty())
    {
        return nullptr;
    }

    // Create a new function prototype for replacement.
    TFunction *replacementFunction = new TFunction(
        symbolTable, oldFunction->name(), SymbolType::UserDefined,
        new TType(oldFunction->getReturnType()), oldFunction->isKnownToNotHaveSideEffects());
    for (size_t paramIndex = 0; paramIndex < oldFunction->getParamCount(); ++paramIndex)
    {
        const TVariable *param = oldFunction->getParam(paramIndex);
        TVariable *replacement = nullptr;
        auto replaced          = mReplacedFunctionParams.find(param);
        if (replaced != mReplacedFunctionParams.end())
        {
            replacement = replaced->second;
        }
        else
        {
            replacement = new TVariable(symbolTable, param->name(), new TType(param->getType()),
                                        SymbolType::UserDefined);
        }
        replacementFunction->addParameter(replacement);
    }
    mReplacedFunctions[oldFunction] = replacementFunction;

    TIntermFunctionPrototype *replacementPrototype =
        new TIntermFunctionPrototype(replacementFunction);

    return replacementPrototype;
}

TIntermAggregate *RetypeOpaqueVariablesHelper::convertASTFunction(TIntermAggregate *node)
{
    // See if the function needs replacement at all.
    const TFunction *function = node->getFunction();
    auto replacedFunction     = mReplacedFunctions.find(function);
    if (replacedFunction == mReplacedFunctions.end())
    {
        return nullptr;
    }

    // Arguments to this call are staged to be replaced at the same time.
    TFunction *substituteFunction        = replacedFunction->second;
    TIntermSequence *substituteArguments = new TIntermSequence;

    for (size_t paramIndex = 0; paramIndex < function->getParamCount(); ++paramIndex)
    {
        TIntermNode *param = node->getChildNode(paramIndex);

        TIntermNode *replacement = nullptr;
        auto replacedArg         = mReplacedFunctionCallArgs.top().find(param);
        if (replacedArg != mReplacedFunctionCallArgs.top().end())
        {
            replacement = replacedArg->second;
        }
        else
        {
            replacement = param->getAsTyped()->deepCopy();
        }
        substituteArguments->push_back(replacement);
    }

    return TIntermAggregate::CreateFunctionCall(*substituteFunction, substituteArguments);
}

}  // namespace sh
