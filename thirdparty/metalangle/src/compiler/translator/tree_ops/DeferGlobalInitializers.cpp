//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DeferGlobalInitializers is an AST traverser that moves global initializers into a separate
// function that is called in the beginning of main(). This enables initialization of globals with
// uniforms or non-constant globals, as allowed by the WebGL spec. Some initializers referencing
// non-constants may need to be unfolded into if statements in HLSL - this kind of steps should be
// done after DeferGlobalInitializers is run. Note that it's important that the function definition
// is at the end of the shader, as some globals may be declared after main().
//
// It can also initialize all uninitialized globals.
//

#include "compiler/translator/tree_ops/DeferGlobalInitializers.h"

#include <vector>

#include "compiler/translator/Compiler.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_ops/InitializeVariables.h"
#include "compiler/translator/tree_util/FindMain.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"

namespace sh
{

namespace
{

constexpr const ImmutableString kInitGlobalsString("initGlobals");

void GetDeferredInitializers(TIntermDeclaration *declaration,
                             bool initializeUninitializedGlobals,
                             bool canUseLoopsToInitialize,
                             bool highPrecisionSupported,
                             TIntermSequence *deferredInitializersOut,
                             std::vector<const TVariable *> *variablesToReplaceOut,
                             TSymbolTable *symbolTable)
{
    // SeparateDeclarations should have already been run.
    ASSERT(declaration->getSequence()->size() == 1);

    TIntermNode *declarator = declaration->getSequence()->back();
    TIntermBinary *init     = declarator->getAsBinaryNode();
    if (init)
    {
        TIntermSymbol *symbolNode = init->getLeft()->getAsSymbolNode();
        ASSERT(symbolNode);
        TIntermTyped *expression = init->getRight();

        if (expression->getQualifier() != EvqConst || !expression->hasConstantValue())
        {
            // For variables which are not constant, defer their real initialization until
            // after we initialize uniforms.
            // Deferral is done also in any cases where the variable can not be converted to a
            // constant union, since otherwise there's a chance that HLSL output will generate extra
            // statements from the initializer expression.

            // Change const global to a regular global if its initialization is deferred.
            // This can happen if ANGLE has not been able to fold the constant expression used
            // as an initializer.
            ASSERT(symbolNode->getQualifier() == EvqConst ||
                   symbolNode->getQualifier() == EvqGlobal);
            if (symbolNode->getQualifier() == EvqConst)
            {
                variablesToReplaceOut->push_back(&symbolNode->variable());
            }

            TIntermBinary *deferredInit =
                new TIntermBinary(EOpAssign, symbolNode->deepCopy(), init->getRight());
            deferredInitializersOut->push_back(deferredInit);

            // Remove the initializer from the global scope and just declare the global instead.
            declaration->replaceChildNode(init, symbolNode);
        }
    }
    else if (initializeUninitializedGlobals)
    {
        TIntermSymbol *symbolNode = declarator->getAsSymbolNode();
        ASSERT(symbolNode);

        // Ignore ANGLE internal variables and nameless declarations.
        if (symbolNode->variable().symbolType() == SymbolType::AngleInternal ||
            symbolNode->variable().symbolType() == SymbolType::Empty)
            return;

        if (symbolNode->getQualifier() == EvqGlobal)
        {
            TIntermSequence *initCode = CreateInitCode(symbolNode, canUseLoopsToInitialize,
                                                       highPrecisionSupported, symbolTable);
            deferredInitializersOut->insert(deferredInitializersOut->end(), initCode->begin(),
                                            initCode->end());
        }
    }
}

void InsertInitCallToMain(TIntermBlock *root,
                          TIntermSequence *deferredInitializers,
                          TSymbolTable *symbolTable)
{
    TIntermBlock *initGlobalsBlock = new TIntermBlock();
    initGlobalsBlock->getSequence()->swap(*deferredInitializers);

    TFunction *initGlobalsFunction =
        new TFunction(symbolTable, kInitGlobalsString, SymbolType::AngleInternal,
                      StaticType::GetBasic<EbtVoid>(), false);

    TIntermFunctionPrototype *initGlobalsFunctionPrototype =
        CreateInternalFunctionPrototypeNode(*initGlobalsFunction);
    root->getSequence()->insert(root->getSequence()->begin(), initGlobalsFunctionPrototype);
    TIntermFunctionDefinition *initGlobalsFunctionDefinition =
        CreateInternalFunctionDefinitionNode(*initGlobalsFunction, initGlobalsBlock);
    root->appendStatement(initGlobalsFunctionDefinition);

    TIntermAggregate *initGlobalsCall =
        TIntermAggregate::CreateFunctionCall(*initGlobalsFunction, new TIntermSequence());

    TIntermBlock *mainBody = FindMainBody(root);
    mainBody->getSequence()->insert(mainBody->getSequence()->begin(), initGlobalsCall);
}

}  // namespace

bool DeferGlobalInitializers(TCompiler *compiler,
                             TIntermBlock *root,
                             bool initializeUninitializedGlobals,
                             bool canUseLoopsToInitialize,
                             bool highPrecisionSupported,
                             TSymbolTable *symbolTable)
{
    TIntermSequence *deferredInitializers = new TIntermSequence();
    std::vector<const TVariable *> variablesToReplace;

    // Loop over all global statements and process the declarations. This is simpler than using a
    // traverser.
    for (TIntermNode *statement : *root->getSequence())
    {
        TIntermDeclaration *declaration = statement->getAsDeclarationNode();
        if (declaration)
        {
            GetDeferredInitializers(declaration, initializeUninitializedGlobals,
                                    canUseLoopsToInitialize, highPrecisionSupported,
                                    deferredInitializers, &variablesToReplace, symbolTable);
        }
    }

    // Add the function with initialization and the call to that.
    if (!deferredInitializers->empty())
    {
        InsertInitCallToMain(root, deferredInitializers, symbolTable);
    }

    // Replace constant variables with non-constant global variables.
    for (const TVariable *var : variablesToReplace)
    {
        TType *replacementType = new TType(var->getType());
        replacementType->setQualifier(EvqGlobal);
        TVariable *replacement =
            new TVariable(symbolTable, var->name(), replacementType, var->symbolType());
        if (!ReplaceVariable(compiler, root, var, replacement))
        {
            return false;
        }
    }

    return true;
}

}  // namespace sh
