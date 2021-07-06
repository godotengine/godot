//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ReplaceShadowingVariables.cpp: Replace all references to any variable in the AST that is
// a redefinition of a variable in a nested scope. This is a useful for ESSL 1.00 shaders
// where the spec section "4.2.3. Redeclaring Variables" states "However, a nested scope can
// override an outer scope's declaration of a particular variable name." This is changed in
// later spec versions, such as ESSL 3.20 spec which states "If [a variable] is declared as
// a parameter in a function definition, it is scoped until the end of that function
// definition. A function's parameter declarations and body together form a single scope."
//
// So this class is useful when translating from ESSL 1.00 shaders, where function body var
// redefinition is allowed, to later shader versions where it's not allowed.
//

#include "compiler/translator/tree_util/ReplaceShadowingVariables.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

#include <unordered_set>

namespace sh
{

namespace
{

// Custom struct to queue up any replacements until after AST traversal
struct DeferredReplacementBlock
{
    const TVariable *originalVariable;  // variable to be replaced
    TVariable *replacementVariable;     // variable to replace originalVar with
    TIntermBlock *functionBody;         // function body where replacement occurs
};

class ReplaceShadowingVariablesTraverser : public TIntermTraverser
{
  public:
    ReplaceShadowingVariablesTraverser(TSymbolTable *symbolTable)
        : TIntermTraverser(true, true, true),
          mSymbolTable(symbolTable),
          mParameterNames{},
          mFunctionBody(nullptr)
    {}

    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override
    {
        // In pre-visit of function, record params
        if (visit == PreVisit)
        {
            ASSERT(mParameterNames.size() == 0);
            const TFunction *func = node->getFunctionPrototype()->getFunction();
            // Grab all of the parameter names from the function prototype
            size_t paramCount = func->getParamCount();
            for (size_t i = 0; i < paramCount; ++i)
            {
                mParameterNames.emplace(std::string(func->getParam(i)->name().data()));
            }
            if (mParameterNames.size() > 0)
                mFunctionBody = node->getBody();
        }
        else if (visit == PostVisit)
        {
            // Clear data saved from function definition
            mParameterNames.clear();
            mFunctionBody = nullptr;
        }
        return true;
    }
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        if (visit == PreVisit && mParameterNames.size() != 0)
        {
            TIntermSequence *decls = node->getSequence();
            for (auto &declVector : *decls)
            {
                // no init case
                TIntermSymbol *symNode = declVector->getAsSymbolNode();
                if (symNode == nullptr)
                {
                    // init case
                    TIntermBinary *binaryNode = declVector->getAsBinaryNode();
                    ASSERT(binaryNode->getOp() == EOpInitialize);
                    symNode = binaryNode->getLeft()->getAsSymbolNode();
                }
                ASSERT(symNode != nullptr);
                std::string varName = std::string(symNode->variable().name().data());
                if (mParameterNames.count(varName) > 0)
                {
                    // We found a redefined var so queue replacement
                    mReplacements.emplace_back(DeferredReplacementBlock{
                        &symNode->variable(),
                        CreateTempVariable(mSymbolTable, &symNode->variable().getType()),
                        mFunctionBody});
                }
            }
        }
        return true;
    }
    // Perform replacement of vars for any deferred replacements that were identified
    ANGLE_NO_DISCARD bool executeReplacements(TCompiler *compiler)
    {
        for (DeferredReplacementBlock &replace : mReplacements)
        {
            if (!ReplaceVariable(compiler, replace.functionBody, replace.originalVariable,
                                 replace.replacementVariable))
            {
                return false;
            }
        }
        mReplacements.clear();
        return true;
    }

  private:
    TSymbolTable *mSymbolTable;
    std::unordered_set<std::string> mParameterNames;
    TIntermBlock *mFunctionBody;
    std::vector<DeferredReplacementBlock> mReplacements;
};

}  // anonymous namespace

// Replaces every occurrence of a variable with another variable.
ANGLE_NO_DISCARD bool ReplaceShadowingVariables(TCompiler *compiler,
                                                TIntermBlock *root,
                                                TSymbolTable *symbolTable)
{
    ReplaceShadowingVariablesTraverser traverser(symbolTable);
    root->traverse(&traverser);
    if (!traverser.executeReplacements(compiler))
    {
        return false;
    }
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
