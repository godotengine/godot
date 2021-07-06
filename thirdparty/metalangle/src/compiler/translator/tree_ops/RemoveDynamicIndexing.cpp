//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveDynamicIndexing is an AST traverser to remove dynamic indexing of non-SSBO vectors and
// matrices, replacing them with calls to functions that choose which component to return or write.
// We don't need to consider dynamic indexing in SSBO since it can be directly as part of the offset
// of RWByteAddressBuffer.
//

#include "compiler/translator/tree_ops/RemoveDynamicIndexing.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNodePatternMatcher.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

const TType *kIndexType = StaticType::Get<EbtInt, EbpHigh, EvqIn, 1, 1>();

constexpr const ImmutableString kBaseName("base");
constexpr const ImmutableString kIndexName("index");
constexpr const ImmutableString kValueName("value");

std::string GetIndexFunctionName(const TType &type, bool write)
{
    TInfoSinkBase nameSink;
    nameSink << "dyn_index_";
    if (write)
    {
        nameSink << "write_";
    }
    if (type.isMatrix())
    {
        nameSink << "mat" << type.getCols() << "x" << type.getRows();
    }
    else
    {
        switch (type.getBasicType())
        {
            case EbtInt:
                nameSink << "ivec";
                break;
            case EbtBool:
                nameSink << "bvec";
                break;
            case EbtUInt:
                nameSink << "uvec";
                break;
            case EbtFloat:
                nameSink << "vec";
                break;
            default:
                UNREACHABLE();
        }
        nameSink << type.getNominalSize();
    }
    return nameSink.str();
}

TIntermConstantUnion *CreateIntConstantNode(int i)
{
    TConstantUnion *constant = new TConstantUnion();
    constant->setIConst(i);
    return new TIntermConstantUnion(constant, TType(EbtInt, EbpHigh));
}

TIntermTyped *EnsureSignedInt(TIntermTyped *node)
{
    if (node->getBasicType() == EbtInt)
        return node;

    TIntermSequence *arguments = new TIntermSequence();
    arguments->push_back(node);
    return TIntermAggregate::CreateConstructor(TType(EbtInt), arguments);
}

TType *GetFieldType(const TType &indexedType)
{
    if (indexedType.isMatrix())
    {
        TType *fieldType = new TType(indexedType.getBasicType(), indexedType.getPrecision());
        fieldType->setPrimarySize(static_cast<unsigned char>(indexedType.getRows()));
        return fieldType;
    }
    else
    {
        return new TType(indexedType.getBasicType(), indexedType.getPrecision());
    }
}

const TType *GetBaseType(const TType &type, bool write)
{
    TType *baseType = new TType(type);
    // Conservatively use highp here, even if the indexed type is not highp. That way the code can't
    // end up using mediump version of an indexing function for a highp value, if both mediump and
    // highp values are being indexed in the shader. For HLSL precision doesn't matter, but in
    // principle this code could be used with multiple backends.
    baseType->setPrecision(EbpHigh);
    baseType->setQualifier(EvqInOut);
    if (!write)
        baseType->setQualifier(EvqIn);
    return baseType;
}

// Generate a read or write function for one field in a vector/matrix.
// Out-of-range indices are clamped. This is consistent with how ANGLE handles out-of-range
// indices in other places.
// Note that indices can be either int or uint. We create only int versions of the functions,
// and convert uint indices to int at the call site.
// read function example:
// float dyn_index_vec2(in vec2 base, in int index)
// {
//    switch(index)
//    {
//      case (0):
//        return base[0];
//      case (1):
//        return base[1];
//      default:
//        break;
//    }
//    if (index < 0)
//      return base[0];
//    return base[1];
// }
// write function example:
// void dyn_index_write_vec2(inout vec2 base, in int index, in float value)
// {
//    switch(index)
//    {
//      case (0):
//        base[0] = value;
//        return;
//      case (1):
//        base[1] = value;
//        return;
//      default:
//        break;
//    }
//    if (index < 0)
//    {
//      base[0] = value;
//      return;
//    }
//    base[1] = value;
// }
// Note that else is not used in above functions to avoid the RewriteElseBlocks transformation.
TIntermFunctionDefinition *GetIndexFunctionDefinition(const TType &type,
                                                      bool write,
                                                      const TFunction &func,
                                                      TSymbolTable *symbolTable)
{
    ASSERT(!type.isArray());

    int numCases = 0;
    if (type.isMatrix())
    {
        numCases = type.getCols();
    }
    else
    {
        numCases = type.getNominalSize();
    }

    std::string functionName                = GetIndexFunctionName(type, write);
    TIntermFunctionPrototype *prototypeNode = CreateInternalFunctionPrototypeNode(func);

    TIntermSymbol *baseParam  = new TIntermSymbol(func.getParam(0));
    TIntermSymbol *indexParam = new TIntermSymbol(func.getParam(1));
    TIntermSymbol *valueParam = nullptr;
    if (write)
    {
        valueParam = new TIntermSymbol(func.getParam(2));
    }

    TIntermBlock *statementList = new TIntermBlock();
    for (int i = 0; i < numCases; ++i)
    {
        TIntermCase *caseNode = new TIntermCase(CreateIntConstantNode(i));
        statementList->getSequence()->push_back(caseNode);

        TIntermBinary *indexNode =
            new TIntermBinary(EOpIndexDirect, baseParam->deepCopy(), CreateIndexNode(i));
        if (write)
        {
            TIntermBinary *assignNode =
                new TIntermBinary(EOpAssign, indexNode, valueParam->deepCopy());
            statementList->getSequence()->push_back(assignNode);
            TIntermBranch *returnNode = new TIntermBranch(EOpReturn, nullptr);
            statementList->getSequence()->push_back(returnNode);
        }
        else
        {
            TIntermBranch *returnNode = new TIntermBranch(EOpReturn, indexNode);
            statementList->getSequence()->push_back(returnNode);
        }
    }

    // Default case
    TIntermCase *defaultNode = new TIntermCase(nullptr);
    statementList->getSequence()->push_back(defaultNode);
    TIntermBranch *breakNode = new TIntermBranch(EOpBreak, nullptr);
    statementList->getSequence()->push_back(breakNode);

    TIntermSwitch *switchNode = new TIntermSwitch(indexParam->deepCopy(), statementList);

    TIntermBlock *bodyNode = new TIntermBlock();
    bodyNode->getSequence()->push_back(switchNode);

    TIntermBinary *cond =
        new TIntermBinary(EOpLessThan, indexParam->deepCopy(), CreateIntConstantNode(0));

    // Two blocks: one accesses (either reads or writes) the first element and returns,
    // the other accesses the last element.
    TIntermBlock *useFirstBlock = new TIntermBlock();
    TIntermBlock *useLastBlock  = new TIntermBlock();
    TIntermBinary *indexFirstNode =
        new TIntermBinary(EOpIndexDirect, baseParam->deepCopy(), CreateIndexNode(0));
    TIntermBinary *indexLastNode =
        new TIntermBinary(EOpIndexDirect, baseParam->deepCopy(), CreateIndexNode(numCases - 1));
    if (write)
    {
        TIntermBinary *assignFirstNode =
            new TIntermBinary(EOpAssign, indexFirstNode, valueParam->deepCopy());
        useFirstBlock->getSequence()->push_back(assignFirstNode);
        TIntermBranch *returnNode = new TIntermBranch(EOpReturn, nullptr);
        useFirstBlock->getSequence()->push_back(returnNode);

        TIntermBinary *assignLastNode =
            new TIntermBinary(EOpAssign, indexLastNode, valueParam->deepCopy());
        useLastBlock->getSequence()->push_back(assignLastNode);
    }
    else
    {
        TIntermBranch *returnFirstNode = new TIntermBranch(EOpReturn, indexFirstNode);
        useFirstBlock->getSequence()->push_back(returnFirstNode);

        TIntermBranch *returnLastNode = new TIntermBranch(EOpReturn, indexLastNode);
        useLastBlock->getSequence()->push_back(returnLastNode);
    }
    TIntermIfElse *ifNode = new TIntermIfElse(cond, useFirstBlock, nullptr);
    bodyNode->getSequence()->push_back(ifNode);
    bodyNode->getSequence()->push_back(useLastBlock);

    TIntermFunctionDefinition *indexingFunction =
        new TIntermFunctionDefinition(prototypeNode, bodyNode);
    return indexingFunction;
}

class RemoveDynamicIndexingTraverser : public TLValueTrackingTraverser
{
  public:
    RemoveDynamicIndexingTraverser(TSymbolTable *symbolTable,
                                   PerformanceDiagnostics *perfDiagnostics);

    bool visitBinary(Visit visit, TIntermBinary *node) override;

    void insertHelperDefinitions(TIntermNode *root);

    void nextIteration();

    bool usedTreeInsertion() const { return mUsedTreeInsertion; }

  protected:
    // Maps of types that are indexed to the indexing function ids used for them. Note that these
    // can not store multiple variants of the same type with different precisions - only one
    // precision gets stored.
    std::map<TType, TFunction *> mIndexedVecAndMatrixTypes;
    std::map<TType, TFunction *> mWrittenVecAndMatrixTypes;

    bool mUsedTreeInsertion;

    // When true, the traverser will remove side effects from any indexing expression.
    // This is done so that in code like
    //   V[j++][i]++.
    // where V is an array of vectors, j++ will only be evaluated once.
    bool mRemoveIndexSideEffectsInSubtree;

    PerformanceDiagnostics *mPerfDiagnostics;
};

RemoveDynamicIndexingTraverser::RemoveDynamicIndexingTraverser(
    TSymbolTable *symbolTable,
    PerformanceDiagnostics *perfDiagnostics)
    : TLValueTrackingTraverser(true, false, false, symbolTable),
      mUsedTreeInsertion(false),
      mRemoveIndexSideEffectsInSubtree(false),
      mPerfDiagnostics(perfDiagnostics)
{}

void RemoveDynamicIndexingTraverser::insertHelperDefinitions(TIntermNode *root)
{
    TIntermBlock *rootBlock = root->getAsBlock();
    ASSERT(rootBlock != nullptr);
    TIntermSequence insertions;
    for (auto &type : mIndexedVecAndMatrixTypes)
    {
        insertions.push_back(
            GetIndexFunctionDefinition(type.first, false, *type.second, mSymbolTable));
    }
    for (auto &type : mWrittenVecAndMatrixTypes)
    {
        insertions.push_back(
            GetIndexFunctionDefinition(type.first, true, *type.second, mSymbolTable));
    }
    rootBlock->insertChildNodes(0, insertions);
}

// Create a call to dyn_index_*() based on an indirect indexing op node
TIntermAggregate *CreateIndexFunctionCall(TIntermBinary *node,
                                          TIntermTyped *index,
                                          TFunction *indexingFunction)
{
    ASSERT(node->getOp() == EOpIndexIndirect);
    TIntermSequence *arguments = new TIntermSequence();
    arguments->push_back(node->getLeft());
    arguments->push_back(index);

    TIntermAggregate *indexingCall =
        TIntermAggregate::CreateFunctionCall(*indexingFunction, arguments);
    indexingCall->setLine(node->getLine());
    return indexingCall;
}

TIntermAggregate *CreateIndexedWriteFunctionCall(TIntermBinary *node,
                                                 TVariable *index,
                                                 TVariable *writtenValue,
                                                 TFunction *indexedWriteFunction)
{
    ASSERT(node->getOp() == EOpIndexIndirect);
    TIntermSequence *arguments = new TIntermSequence();
    // Deep copy the child nodes so that two pointers to the same node don't end up in the tree.
    arguments->push_back(node->getLeft()->deepCopy());
    arguments->push_back(CreateTempSymbolNode(index));
    arguments->push_back(CreateTempSymbolNode(writtenValue));

    TIntermAggregate *indexedWriteCall =
        TIntermAggregate::CreateFunctionCall(*indexedWriteFunction, arguments);
    indexedWriteCall->setLine(node->getLine());
    return indexedWriteCall;
}

bool RemoveDynamicIndexingTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    if (mUsedTreeInsertion)
        return false;

    if (node->getOp() == EOpIndexIndirect)
    {
        if (mRemoveIndexSideEffectsInSubtree)
        {
            ASSERT(node->getRight()->hasSideEffects());
            // In case we're just removing index side effects, convert
            //   v_expr[index_expr]
            // to this:
            //   int s0 = index_expr; v_expr[s0];
            // Now v_expr[s0] can be safely executed several times without unintended side effects.
            TIntermDeclaration *indexVariableDeclaration = nullptr;
            TVariable *indexVariable = DeclareTempVariable(mSymbolTable, node->getRight(),
                                                           EvqTemporary, &indexVariableDeclaration);
            insertStatementInParentBlock(indexVariableDeclaration);
            mUsedTreeInsertion = true;

            // Replace the index with the temp variable
            TIntermSymbol *tempIndex = CreateTempSymbolNode(indexVariable);
            queueReplacementWithParent(node, node->getRight(), tempIndex, OriginalNode::IS_DROPPED);
        }
        else if (IntermNodePatternMatcher::IsDynamicIndexingOfNonSSBOVectorOrMatrix(node))
        {
            mPerfDiagnostics->warning(node->getLine(),
                                      "Performance: dynamic indexing of vectors and "
                                      "matrices is emulated and can be slow.",
                                      "[]");
            bool write = isLValueRequiredHere();

#if defined(ANGLE_ENABLE_ASSERTS)
            // Make sure that IntermNodePatternMatcher is consistent with the slightly differently
            // implemented checks in this traverser.
            IntermNodePatternMatcher matcher(
                IntermNodePatternMatcher::kDynamicIndexingOfVectorOrMatrixInLValue);
            ASSERT(matcher.match(node, getParentNode(), isLValueRequiredHere()) == write);
#endif

            const TType &type = node->getLeft()->getType();
            ImmutableString indexingFunctionName(GetIndexFunctionName(type, false));
            TFunction *indexingFunction = nullptr;
            if (mIndexedVecAndMatrixTypes.find(type) == mIndexedVecAndMatrixTypes.end())
            {
                indexingFunction =
                    new TFunction(mSymbolTable, indexingFunctionName, SymbolType::AngleInternal,
                                  GetFieldType(type), true);
                indexingFunction->addParameter(new TVariable(
                    mSymbolTable, kBaseName, GetBaseType(type, false), SymbolType::AngleInternal));
                indexingFunction->addParameter(
                    new TVariable(mSymbolTable, kIndexName, kIndexType, SymbolType::AngleInternal));
                mIndexedVecAndMatrixTypes[type] = indexingFunction;
            }
            else
            {
                indexingFunction = mIndexedVecAndMatrixTypes[type];
            }

            if (write)
            {
                // Convert:
                //   v_expr[index_expr]++;
                // to this:
                //   int s0 = index_expr; float s1 = dyn_index(v_expr, s0); s1++;
                //   dyn_index_write(v_expr, s0, s1);
                // This works even if index_expr has some side effects.
                if (node->getLeft()->hasSideEffects())
                {
                    // If v_expr has side effects, those need to be removed before proceeding.
                    // Otherwise the side effects of v_expr would be evaluated twice.
                    // The only case where an l-value can have side effects is when it is
                    // indexing. For example, it can be V[j++] where V is an array of vectors.
                    mRemoveIndexSideEffectsInSubtree = true;
                    return true;
                }

                TIntermBinary *leftBinary = node->getLeft()->getAsBinaryNode();
                if (leftBinary != nullptr &&
                    IntermNodePatternMatcher::IsDynamicIndexingOfNonSSBOVectorOrMatrix(leftBinary))
                {
                    // This is a case like:
                    // mat2 m;
                    // m[a][b]++;
                    // Process the child node m[a] first.
                    return true;
                }

                // TODO(oetuaho@nvidia.com): This is not optimal if the expression using the value
                // only writes it and doesn't need the previous value. http://anglebug.com/1116

                TFunction *indexedWriteFunction = nullptr;
                if (mWrittenVecAndMatrixTypes.find(type) == mWrittenVecAndMatrixTypes.end())
                {
                    ImmutableString functionName(
                        GetIndexFunctionName(node->getLeft()->getType(), true));
                    indexedWriteFunction =
                        new TFunction(mSymbolTable, functionName, SymbolType::AngleInternal,
                                      StaticType::GetBasic<EbtVoid>(), false);
                    indexedWriteFunction->addParameter(new TVariable(mSymbolTable, kBaseName,
                                                                     GetBaseType(type, true),
                                                                     SymbolType::AngleInternal));
                    indexedWriteFunction->addParameter(new TVariable(
                        mSymbolTable, kIndexName, kIndexType, SymbolType::AngleInternal));
                    TType *valueType = GetFieldType(type);
                    valueType->setQualifier(EvqIn);
                    indexedWriteFunction->addParameter(new TVariable(
                        mSymbolTable, kValueName, static_cast<const TType *>(valueType),
                        SymbolType::AngleInternal));
                    mWrittenVecAndMatrixTypes[type] = indexedWriteFunction;
                }
                else
                {
                    indexedWriteFunction = mWrittenVecAndMatrixTypes[type];
                }

                TIntermSequence insertionsBefore;
                TIntermSequence insertionsAfter;

                // Store the index in a temporary signed int variable.
                // s0 = index_expr;
                TIntermTyped *indexInitializer               = EnsureSignedInt(node->getRight());
                TIntermDeclaration *indexVariableDeclaration = nullptr;
                TVariable *indexVariable                     = DeclareTempVariable(
                    mSymbolTable, indexInitializer, EvqTemporary, &indexVariableDeclaration);
                insertionsBefore.push_back(indexVariableDeclaration);

                // s1 = dyn_index(v_expr, s0);
                TIntermAggregate *indexingCall = CreateIndexFunctionCall(
                    node, CreateTempSymbolNode(indexVariable), indexingFunction);
                TIntermDeclaration *fieldVariableDeclaration = nullptr;
                TVariable *fieldVariable                     = DeclareTempVariable(
                    mSymbolTable, indexingCall, EvqTemporary, &fieldVariableDeclaration);
                insertionsBefore.push_back(fieldVariableDeclaration);

                // dyn_index_write(v_expr, s0, s1);
                TIntermAggregate *indexedWriteCall = CreateIndexedWriteFunctionCall(
                    node, indexVariable, fieldVariable, indexedWriteFunction);
                insertionsAfter.push_back(indexedWriteCall);
                insertStatementsInParentBlock(insertionsBefore, insertionsAfter);

                // replace the node with s1
                queueReplacement(CreateTempSymbolNode(fieldVariable), OriginalNode::IS_DROPPED);
                mUsedTreeInsertion = true;
            }
            else
            {
                // The indexed value is not being written, so we can simply convert
                //   v_expr[index_expr]
                // into
                //   dyn_index(v_expr, index_expr)
                // If the index_expr is unsigned, we'll convert it to signed.
                ASSERT(!mRemoveIndexSideEffectsInSubtree);
                TIntermAggregate *indexingCall = CreateIndexFunctionCall(
                    node, EnsureSignedInt(node->getRight()), indexingFunction);
                queueReplacement(indexingCall, OriginalNode::IS_DROPPED);
            }
        }
    }
    return !mUsedTreeInsertion;
}

void RemoveDynamicIndexingTraverser::nextIteration()
{
    mUsedTreeInsertion               = false;
    mRemoveIndexSideEffectsInSubtree = false;
}

}  // namespace

bool RemoveDynamicIndexing(TCompiler *compiler,
                           TIntermNode *root,
                           TSymbolTable *symbolTable,
                           PerformanceDiagnostics *perfDiagnostics)
{
    RemoveDynamicIndexingTraverser traverser(symbolTable, perfDiagnostics);
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (!traverser.updateTree(compiler, root))
        {
            return false;
        }
    } while (traverser.usedTreeInsertion());
    // TODO(oetuaho@nvidia.com): It might be nicer to add the helper definitions also in the middle
    // of traversal. Now the tree ends up in an inconsistent state in the middle, since there are
    // function call nodes with no corresponding definition nodes. This needs special handling in
    // TIntermLValueTrackingTraverser, and creates intricacies that are not easily apparent from a
    // superficial reading of the code.
    traverser.insertHelperDefinitions(root);
    return compiler->validateAST(root);
}

}  // namespace sh
