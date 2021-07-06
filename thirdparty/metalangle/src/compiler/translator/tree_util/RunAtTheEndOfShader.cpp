//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RunAtTheEndOfShader.cpp: Add code to be run at the end of the shader. In case main() contains a
// return statement, this is done by replacing the main() function with another function that calls
// the old main, like this:
//
// void main() { body }
// =>
// void main0() { body }
// void main()
// {
//     main0();
//     codeToRun
// }
//
// This way the code will get run even if the return statement inside main is executed.
//

#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/FindMain.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

constexpr const ImmutableString kMainString("main");

class ContainsReturnTraverser : public TIntermTraverser
{
  public:
    ContainsReturnTraverser() : TIntermTraverser(true, false, false), mContainsReturn(false) {}

    bool visitBranch(Visit visit, TIntermBranch *node) override
    {
        if (node->getFlowOp() == EOpReturn)
        {
            mContainsReturn = true;
        }
        return false;
    }

    bool containsReturn() { return mContainsReturn; }

  private:
    bool mContainsReturn;
};

bool ContainsReturn(TIntermNode *node)
{
    ContainsReturnTraverser traverser;
    node->traverse(&traverser);
    return traverser.containsReturn();
}

void WrapMainAndAppend(TIntermBlock *root,
                       TIntermFunctionDefinition *main,
                       TIntermNode *codeToRun,
                       TSymbolTable *symbolTable)
{
    // Replace main() with main0() with the same body.
    TFunction *oldMain =
        new TFunction(symbolTable, kEmptyImmutableString, SymbolType::AngleInternal,
                      StaticType::GetBasic<EbtVoid>(), false);
    TIntermFunctionDefinition *oldMainDefinition =
        CreateInternalFunctionDefinitionNode(*oldMain, main->getBody());

    bool replaced = root->replaceChildNode(main, oldMainDefinition);
    ASSERT(replaced);

    // void main()
    TFunction *newMain = new TFunction(symbolTable, kMainString, SymbolType::UserDefined,
                                       StaticType::GetBasic<EbtVoid>(), false);
    TIntermFunctionPrototype *newMainProto = new TIntermFunctionPrototype(newMain);

    // {
    //     main0();
    //     codeToRun
    // }
    TIntermBlock *newMainBody = new TIntermBlock();
    TIntermAggregate *oldMainCall =
        TIntermAggregate::CreateFunctionCall(*oldMain, new TIntermSequence());
    newMainBody->appendStatement(oldMainCall);
    newMainBody->appendStatement(codeToRun);

    // Add the new main() to the root node.
    TIntermFunctionDefinition *newMainDefinition =
        new TIntermFunctionDefinition(newMainProto, newMainBody);
    root->appendStatement(newMainDefinition);
}

}  // anonymous namespace

bool RunAtTheEndOfShader(TCompiler *compiler,
                         TIntermBlock *root,
                         TIntermNode *codeToRun,
                         TSymbolTable *symbolTable)
{
    TIntermFunctionDefinition *main = FindMain(root);
    if (!ContainsReturn(main))
    {
        main->getBody()->appendStatement(codeToRun);
    }
    else
    {
        WrapMainAndAppend(root, main, codeToRun, symbolTable);
    }

    return compiler->validateAST(root);
}

}  // namespace sh
