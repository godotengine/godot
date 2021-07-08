//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/InitializeVariables.h"

#include "angle_gl.h"
#include "common/debug.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/FindMain.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

void AddArrayZeroInitSequence(const TIntermTyped *initializedNode,
                              bool canUseLoopsToInitialize,
                              bool highPrecisionSupported,
                              TIntermSequence *initSequenceOut,
                              TSymbolTable *symbolTable);

void AddStructZeroInitSequence(const TIntermTyped *initializedNode,
                               bool canUseLoopsToInitialize,
                               bool highPrecisionSupported,
                               TIntermSequence *initSequenceOut,
                               TSymbolTable *symbolTable);

TIntermBinary *CreateZeroInitAssignment(const TIntermTyped *initializedNode)
{
    TIntermTyped *zero = CreateZeroNode(initializedNode->getType());
    return new TIntermBinary(EOpAssign, initializedNode->deepCopy(), zero);
}

void AddZeroInitSequence(const TIntermTyped *initializedNode,
                         bool canUseLoopsToInitialize,
                         bool highPrecisionSupported,
                         TIntermSequence *initSequenceOut,
                         TSymbolTable *symbolTable)
{
    if (initializedNode->isArray())
    {
        AddArrayZeroInitSequence(initializedNode, canUseLoopsToInitialize, highPrecisionSupported,
                                 initSequenceOut, symbolTable);
    }
    else if (initializedNode->getType().isStructureContainingArrays() ||
             initializedNode->getType().isNamelessStruct())
    {
        AddStructZeroInitSequence(initializedNode, canUseLoopsToInitialize, highPrecisionSupported,
                                  initSequenceOut, symbolTable);
    }
    else
    {
        initSequenceOut->push_back(CreateZeroInitAssignment(initializedNode));
    }
}

void AddStructZeroInitSequence(const TIntermTyped *initializedNode,
                               bool canUseLoopsToInitialize,
                               bool highPrecisionSupported,
                               TIntermSequence *initSequenceOut,
                               TSymbolTable *symbolTable)
{
    ASSERT(initializedNode->getBasicType() == EbtStruct);
    const TStructure *structType = initializedNode->getType().getStruct();
    for (int i = 0; i < static_cast<int>(structType->fields().size()); ++i)
    {
        TIntermBinary *element = new TIntermBinary(EOpIndexDirectStruct,
                                                   initializedNode->deepCopy(), CreateIndexNode(i));
        // Structs can't be defined inside structs, so the type of a struct field can't be a
        // nameless struct.
        ASSERT(!element->getType().isNamelessStruct());
        AddZeroInitSequence(element, canUseLoopsToInitialize, highPrecisionSupported,
                            initSequenceOut, symbolTable);
    }
}

void AddArrayZeroInitStatementList(const TIntermTyped *initializedNode,
                                   bool canUseLoopsToInitialize,
                                   bool highPrecisionSupported,
                                   TIntermSequence *initSequenceOut,
                                   TSymbolTable *symbolTable)
{
    for (unsigned int i = 0; i < initializedNode->getOutermostArraySize(); ++i)
    {
        TIntermBinary *element =
            new TIntermBinary(EOpIndexDirect, initializedNode->deepCopy(), CreateIndexNode(i));
        AddZeroInitSequence(element, canUseLoopsToInitialize, highPrecisionSupported,
                            initSequenceOut, symbolTable);
    }
}

void AddArrayZeroInitForLoop(const TIntermTyped *initializedNode,
                             bool highPrecisionSupported,
                             TIntermSequence *initSequenceOut,
                             TSymbolTable *symbolTable)
{
    ASSERT(initializedNode->isArray());
    const TType *mediumpIndexType = StaticType::Get<EbtInt, EbpMedium, EvqTemporary, 1, 1>();
    const TType *highpIndexType   = StaticType::Get<EbtInt, EbpHigh, EvqTemporary, 1, 1>();
    TVariable *indexVariable =
        CreateTempVariable(symbolTable, highPrecisionSupported ? highpIndexType : mediumpIndexType);

    TIntermSymbol *indexSymbolNode = CreateTempSymbolNode(indexVariable);
    TIntermDeclaration *indexInit =
        CreateTempInitDeclarationNode(indexVariable, CreateZeroNode(indexVariable->getType()));
    TIntermConstantUnion *arraySizeNode = CreateIndexNode(initializedNode->getOutermostArraySize());
    TIntermBinary *indexSmallerThanSize =
        new TIntermBinary(EOpLessThan, indexSymbolNode->deepCopy(), arraySizeNode);
    TIntermUnary *indexIncrement =
        new TIntermUnary(EOpPreIncrement, indexSymbolNode->deepCopy(), nullptr);

    TIntermBlock *forLoopBody       = new TIntermBlock();
    TIntermSequence *forLoopBodySeq = forLoopBody->getSequence();

    TIntermBinary *element = new TIntermBinary(EOpIndexIndirect, initializedNode->deepCopy(),
                                               indexSymbolNode->deepCopy());
    AddZeroInitSequence(element, true, highPrecisionSupported, forLoopBodySeq, symbolTable);

    TIntermLoop *forLoop =
        new TIntermLoop(ELoopFor, indexInit, indexSmallerThanSize, indexIncrement, forLoopBody);
    initSequenceOut->push_back(forLoop);
}

void AddArrayZeroInitSequence(const TIntermTyped *initializedNode,
                              bool canUseLoopsToInitialize,
                              bool highPrecisionSupported,
                              TIntermSequence *initSequenceOut,
                              TSymbolTable *symbolTable)
{
    // The array elements are assigned one by one to keep the AST compatible with ESSL 1.00 which
    // doesn't have array assignment. We'll do this either with a for loop or just a list of
    // statements assigning to each array index. Note that it is important to have the array init in
    // the right order to workaround http://crbug.com/709317
    bool isSmallArray = initializedNode->getOutermostArraySize() <= 1u ||
                        (initializedNode->getBasicType() != EbtStruct &&
                         !initializedNode->getType().isArrayOfArrays() &&
                         initializedNode->getOutermostArraySize() <= 3u);
    if (initializedNode->getQualifier() == EvqFragData ||
        initializedNode->getQualifier() == EvqFragmentOut || isSmallArray ||
        !canUseLoopsToInitialize)
    {
        // Fragment outputs should not be indexed by non-constant indices.
        // Also it doesn't make sense to use loops to initialize very small arrays.
        AddArrayZeroInitStatementList(initializedNode, canUseLoopsToInitialize,
                                      highPrecisionSupported, initSequenceOut, symbolTable);
    }
    else
    {
        AddArrayZeroInitForLoop(initializedNode, highPrecisionSupported, initSequenceOut,
                                symbolTable);
    }
}

void InsertInitCode(TCompiler *compiler,
                    TIntermSequence *mainBody,
                    const InitVariableList &variables,
                    TSymbolTable *symbolTable,
                    int shaderVersion,
                    const TExtensionBehavior &extensionBehavior,
                    bool canUseLoopsToInitialize,
                    bool highPrecisionSupported)
{
    for (const auto &var : variables)
    {
        // Note that tempVariableName will reference a short-lived char array here - that's fine
        // since we're only using it to find symbols.
        ImmutableString tempVariableName(var.name.c_str(), var.name.length());

        TIntermTyped *initializedSymbol = nullptr;
        if (var.isBuiltIn())
        {
            initializedSymbol =
                ReferenceBuiltInVariable(tempVariableName, *symbolTable, shaderVersion);
            if (initializedSymbol->getQualifier() == EvqFragData &&
                !IsExtensionEnabled(extensionBehavior, TExtension::EXT_draw_buffers))
            {
                // If GL_EXT_draw_buffers is disabled, only the 0th index of gl_FragData can be
                // written to.
                // TODO(oetuaho): This is a bit hacky and would be better to remove, if we came up
                // with a good way to do it. Right now "gl_FragData" in symbol table is initialized
                // to have the array size of MaxDrawBuffers, and the initialization happens before
                // the shader sets the extensions it is using.
                initializedSymbol =
                    new TIntermBinary(EOpIndexDirect, initializedSymbol, CreateIndexNode(0));
            }
        }
        else
        {
            initializedSymbol = ReferenceGlobalVariable(tempVariableName, *symbolTable);
        }
        ASSERT(initializedSymbol != nullptr);

        TIntermSequence *initCode = CreateInitCode(initializedSymbol, canUseLoopsToInitialize,
                                                   highPrecisionSupported, symbolTable);
        mainBody->insert(mainBody->begin(), initCode->begin(), initCode->end());
    }
}

class InitializeLocalsTraverser : public TIntermTraverser
{
  public:
    InitializeLocalsTraverser(int shaderVersion,
                              TSymbolTable *symbolTable,
                              bool canUseLoopsToInitialize,
                              bool highPrecisionSupported)
        : TIntermTraverser(true, false, false, symbolTable),
          mShaderVersion(shaderVersion),
          mCanUseLoopsToInitialize(canUseLoopsToInitialize),
          mHighPrecisionSupported(highPrecisionSupported)
    {}

  protected:
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        for (TIntermNode *declarator : *node->getSequence())
        {
            if (!mInGlobalScope && !declarator->getAsBinaryNode())
            {
                TIntermSymbol *symbol = declarator->getAsSymbolNode();
                ASSERT(symbol);
                if (symbol->variable().symbolType() == SymbolType::Empty)
                {
                    continue;
                }

                // Arrays may need to be initialized one element at a time, since ESSL 1.00 does not
                // support array constructors or assigning arrays.
                bool arrayConstructorUnavailable =
                    (symbol->isArray() || symbol->getType().isStructureContainingArrays()) &&
                    mShaderVersion == 100;
                // Nameless struct constructors can't be referred to, so they also need to be
                // initialized one element at a time.
                // TODO(oetuaho): Check if it makes sense to initialize using a loop, even if we
                // could use an initializer. It could at least reduce code size for very large
                // arrays, but could hurt runtime performance.
                if (arrayConstructorUnavailable || symbol->getType().isNamelessStruct())
                {
                    // SimplifyLoopConditions should have been run so the parent node of this node
                    // should not be a loop.
                    ASSERT(getParentNode()->getAsLoopNode() == nullptr);
                    // SeparateDeclarations should have already been run, so we don't need to worry
                    // about further declarators in this declaration depending on the effects of
                    // this declarator.
                    ASSERT(node->getSequence()->size() == 1);
                    insertStatementsInParentBlock(
                        TIntermSequence(), *CreateInitCode(symbol, mCanUseLoopsToInitialize,
                                                           mHighPrecisionSupported, mSymbolTable));
                }
                else
                {
                    TIntermBinary *init =
                        new TIntermBinary(EOpInitialize, symbol, CreateZeroNode(symbol->getType()));
                    queueReplacementWithParent(node, symbol, init, OriginalNode::BECOMES_CHILD);
                }
            }
        }
        return false;
    }

  private:
    int mShaderVersion;
    bool mCanUseLoopsToInitialize;
    bool mHighPrecisionSupported;
};

}  // namespace

TIntermSequence *CreateInitCode(const TIntermTyped *initializedSymbol,
                                bool canUseLoopsToInitialize,
                                bool highPrecisionSupported,
                                TSymbolTable *symbolTable)
{
    TIntermSequence *initCode = new TIntermSequence();
    AddZeroInitSequence(initializedSymbol, canUseLoopsToInitialize, highPrecisionSupported,
                        initCode, symbolTable);
    return initCode;
}

bool InitializeUninitializedLocals(TCompiler *compiler,
                                   TIntermBlock *root,
                                   int shaderVersion,
                                   bool canUseLoopsToInitialize,
                                   bool highPrecisionSupported,
                                   TSymbolTable *symbolTable)
{
    InitializeLocalsTraverser traverser(shaderVersion, symbolTable, canUseLoopsToInitialize,
                                        highPrecisionSupported);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

bool InitializeVariables(TCompiler *compiler,
                         TIntermBlock *root,
                         const InitVariableList &vars,
                         TSymbolTable *symbolTable,
                         int shaderVersion,
                         const TExtensionBehavior &extensionBehavior,
                         bool canUseLoopsToInitialize,
                         bool highPrecisionSupported)
{
    TIntermBlock *body = FindMainBody(root);
    InsertInitCode(compiler, body->getSequence(), vars, symbolTable, shaderVersion,
                   extensionBehavior, canUseLoopsToInitialize, highPrecisionSupported);

    return compiler->validateAST(root);
}

}  // namespace sh
