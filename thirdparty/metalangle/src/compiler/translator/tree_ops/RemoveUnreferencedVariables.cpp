//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveUnreferencedVariables.cpp:
//  Drop variables that are declared but never referenced in the AST. This avoids adding unnecessary
//  initialization code for them. Also removes unreferenced struct types.
//

#include "compiler/translator/tree_ops/RemoveUnreferencedVariables.h"

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class CollectVariableRefCountsTraverser : public TIntermTraverser
{
  public:
    CollectVariableRefCountsTraverser();

    using RefCountMap = std::unordered_map<int, unsigned int>;
    RefCountMap &getSymbolIdRefCounts() { return mSymbolIdRefCounts; }
    RefCountMap &getStructIdRefCounts() { return mStructIdRefCounts; }

    void visitSymbol(TIntermSymbol *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override;

  private:
    void incrementStructTypeRefCount(const TType &type);

    RefCountMap mSymbolIdRefCounts;

    // Structure reference counts are counted from symbols, constructors, function calls, function
    // return values and from interface block and structure fields. We need to track both function
    // calls and function return values since there's a compiler option not to prune unused
    // functions. The type of a constant union may also be a struct, but statements that are just a
    // constant union are always pruned, and if the constant union is used somehow it will get
    // counted by something else.
    RefCountMap mStructIdRefCounts;
};

CollectVariableRefCountsTraverser::CollectVariableRefCountsTraverser()
    : TIntermTraverser(true, false, false)
{}

void CollectVariableRefCountsTraverser::incrementStructTypeRefCount(const TType &type)
{
    if (type.isInterfaceBlock())
    {
        const auto *block = type.getInterfaceBlock();
        ASSERT(block);

        // We can end up incrementing ref counts of struct types referenced from an interface block
        // multiple times for the same block. This doesn't matter, because interface blocks can't be
        // pruned so we'll never do the reverse operation.
        for (const auto &field : block->fields())
        {
            ASSERT(!field->type()->isInterfaceBlock());
            incrementStructTypeRefCount(*field->type());
        }
        return;
    }

    const auto *structure = type.getStruct();
    if (structure != nullptr)
    {
        auto structIter = mStructIdRefCounts.find(structure->uniqueId().get());
        if (structIter == mStructIdRefCounts.end())
        {
            mStructIdRefCounts[structure->uniqueId().get()] = 1u;

            for (const auto &field : structure->fields())
            {
                incrementStructTypeRefCount(*field->type());
            }

            return;
        }
        ++(structIter->second);
    }
}

void CollectVariableRefCountsTraverser::visitSymbol(TIntermSymbol *node)
{
    incrementStructTypeRefCount(node->getType());

    auto iter = mSymbolIdRefCounts.find(node->uniqueId().get());
    if (iter == mSymbolIdRefCounts.end())
    {
        mSymbolIdRefCounts[node->uniqueId().get()] = 1u;
        return;
    }
    ++(iter->second);
}

bool CollectVariableRefCountsTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    // This tracks struct references in both function calls and constructors.
    incrementStructTypeRefCount(node->getType());
    return true;
}

void CollectVariableRefCountsTraverser::visitFunctionPrototype(TIntermFunctionPrototype *node)
{
    incrementStructTypeRefCount(node->getType());
    size_t paramCount = node->getFunction()->getParamCount();
    for (size_t i = 0; i < paramCount; ++i)
    {
        incrementStructTypeRefCount(node->getFunction()->getParam(i)->getType());
    }
}

// Traverser that removes all unreferenced variables on one traversal.
class RemoveUnreferencedVariablesTraverser : public TIntermTraverser
{
  public:
    RemoveUnreferencedVariablesTraverser(
        CollectVariableRefCountsTraverser::RefCountMap *symbolIdRefCounts,
        CollectVariableRefCountsTraverser::RefCountMap *structIdRefCounts,
        TSymbolTable *symbolTable);

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;
    void visitSymbol(TIntermSymbol *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;

    // Traverse loop and block nodes in reverse order. Note that this traverser does not track
    // parent block positions, so insertStatementInParentBlock is unusable!
    void traverseBlock(TIntermBlock *block) override;
    void traverseLoop(TIntermLoop *loop) override;

  private:
    void removeVariableDeclaration(TIntermDeclaration *node, TIntermTyped *declarator);
    void decrementStructTypeRefCount(const TType &type);

    CollectVariableRefCountsTraverser::RefCountMap *mSymbolIdRefCounts;
    CollectVariableRefCountsTraverser::RefCountMap *mStructIdRefCounts;
    bool mRemoveReferences;
};

RemoveUnreferencedVariablesTraverser::RemoveUnreferencedVariablesTraverser(
    CollectVariableRefCountsTraverser::RefCountMap *symbolIdRefCounts,
    CollectVariableRefCountsTraverser::RefCountMap *structIdRefCounts,
    TSymbolTable *symbolTable)
    : TIntermTraverser(true, false, true, symbolTable),
      mSymbolIdRefCounts(symbolIdRefCounts),
      mStructIdRefCounts(structIdRefCounts),
      mRemoveReferences(false)
{}

void RemoveUnreferencedVariablesTraverser::decrementStructTypeRefCount(const TType &type)
{
    auto *structure = type.getStruct();
    if (structure != nullptr)
    {
        ASSERT(mStructIdRefCounts->find(structure->uniqueId().get()) != mStructIdRefCounts->end());
        unsigned int structRefCount = --(*mStructIdRefCounts)[structure->uniqueId().get()];

        if (structRefCount == 0)
        {
            for (const auto &field : structure->fields())
            {
                decrementStructTypeRefCount(*field->type());
            }
        }
    }
}

void RemoveUnreferencedVariablesTraverser::removeVariableDeclaration(TIntermDeclaration *node,
                                                                     TIntermTyped *declarator)
{
    if (declarator->getType().isStructSpecifier() && !declarator->getType().isNamelessStruct())
    {
        unsigned int structId = declarator->getType().getStruct()->uniqueId().get();
        unsigned int structRefCountInThisDeclarator = 1u;
        if (declarator->getAsBinaryNode() &&
            declarator->getAsBinaryNode()->getRight()->getAsAggregate())
        {
            ASSERT(declarator->getAsBinaryNode()->getLeft()->getType().getStruct() ==
                   declarator->getType().getStruct());
            ASSERT(declarator->getAsBinaryNode()->getRight()->getType().getStruct() ==
                   declarator->getType().getStruct());
            structRefCountInThisDeclarator = 2u;
        }
        if ((*mStructIdRefCounts)[structId] > structRefCountInThisDeclarator)
        {
            // If this declaration declares a named struct type that is used elsewhere, we need to
            // keep it. We can still change the declarator though so that it doesn't declare an
            // unreferenced variable.

            // Note that since we're not removing the entire declaration, the struct's reference
            // count will end up being one less than the correct refcount. But since the struct
            // declaration is kept, the incorrect refcount can't cause any other problems.

            if (declarator->getAsSymbolNode() &&
                declarator->getAsSymbolNode()->variable().symbolType() == SymbolType::Empty)
            {
                // Already an empty declaration - nothing to do.
                return;
            }
            TVariable *emptyVariable =
                new TVariable(mSymbolTable, kEmptyImmutableString, new TType(declarator->getType()),
                              SymbolType::Empty);
            queueReplacementWithParent(node, declarator, new TIntermSymbol(emptyVariable),
                                       OriginalNode::IS_DROPPED);
            return;
        }
    }

    if (getParentNode()->getAsBlock())
    {
        TIntermSequence emptyReplacement;
        mMultiReplacements.push_back(
            NodeReplaceWithMultipleEntry(getParentNode()->getAsBlock(), node, emptyReplacement));
    }
    else
    {
        ASSERT(getParentNode()->getAsLoopNode());
        queueReplacement(nullptr, OriginalNode::IS_DROPPED);
    }
}

bool RemoveUnreferencedVariablesTraverser::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    if (visit == PreVisit)
    {
        // SeparateDeclarations should have already been run.
        ASSERT(node->getSequence()->size() == 1u);

        TIntermTyped *declarator = node->getSequence()->back()->getAsTyped();
        ASSERT(declarator);

        // We can only remove variables that are not a part of the shader interface.
        TQualifier qualifier = declarator->getQualifier();
        if (qualifier != EvqTemporary && qualifier != EvqGlobal && qualifier != EvqConst)
        {
            return true;
        }

        bool canRemoveVariable    = false;
        TIntermSymbol *symbolNode = declarator->getAsSymbolNode();
        if (symbolNode != nullptr)
        {
            canRemoveVariable = (*mSymbolIdRefCounts)[symbolNode->uniqueId().get()] == 1u ||
                                symbolNode->variable().symbolType() == SymbolType::Empty;
        }
        TIntermBinary *initNode = declarator->getAsBinaryNode();
        if (initNode != nullptr)
        {
            ASSERT(initNode->getLeft()->getAsSymbolNode());
            int symbolId = initNode->getLeft()->getAsSymbolNode()->uniqueId().get();
            canRemoveVariable =
                (*mSymbolIdRefCounts)[symbolId] == 1u && !initNode->getRight()->hasSideEffects();
        }

        if (canRemoveVariable)
        {
            removeVariableDeclaration(node, declarator);
            mRemoveReferences = true;
        }
        return true;
    }
    ASSERT(visit == PostVisit);
    mRemoveReferences = false;
    return true;
}

void RemoveUnreferencedVariablesTraverser::visitSymbol(TIntermSymbol *node)
{
    if (mRemoveReferences)
    {
        ASSERT(mSymbolIdRefCounts->find(node->uniqueId().get()) != mSymbolIdRefCounts->end());
        --(*mSymbolIdRefCounts)[node->uniqueId().get()];

        decrementStructTypeRefCount(node->getType());
    }
}

bool RemoveUnreferencedVariablesTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    if (visit == PreVisit && mRemoveReferences)
    {
        decrementStructTypeRefCount(node->getType());
    }
    return true;
}

void RemoveUnreferencedVariablesTraverser::traverseBlock(TIntermBlock *node)
{
    // We traverse blocks in reverse order.  This way reference counts can be decremented when
    // removing initializers, and variables that become unused when initializers are removed can be
    // removed on the same traversal.

    ScopedNodeInTraversalPath addToPath(this, node);

    bool visit = true;

    TIntermSequence *sequence = node->getSequence();

    if (preVisit)
        visit = visitBlock(PreVisit, node);

    if (visit)
    {
        for (auto iter = sequence->rbegin(); iter != sequence->rend(); ++iter)
        {
            (*iter)->traverse(this);
            if (visit && inVisit)
            {
                if ((iter + 1) != sequence->rend())
                    visit = visitBlock(InVisit, node);
            }
        }
    }

    if (visit && postVisit)
        visitBlock(PostVisit, node);
}

void RemoveUnreferencedVariablesTraverser::traverseLoop(TIntermLoop *node)
{
    // We traverse loops in reverse order as well. The loop body gets traversed before the init
    // node.

    ScopedNodeInTraversalPath addToPath(this, node);

    bool visit = true;

    if (preVisit)
        visit = visitLoop(PreVisit, node);

    if (visit)
    {
        // We don't need to traverse loop expressions or conditions since they can't be declarations
        // in the AST (loops which have a declaration in their condition get transformed in the
        // parsing stage).
        ASSERT(node->getExpression() == nullptr ||
               node->getExpression()->getAsDeclarationNode() == nullptr);
        ASSERT(node->getCondition() == nullptr ||
               node->getCondition()->getAsDeclarationNode() == nullptr);

        if (node->getBody())
            node->getBody()->traverse(this);

        if (node->getInit())
            node->getInit()->traverse(this);
    }

    if (visit && postVisit)
        visitLoop(PostVisit, node);
}

}  // namespace

bool RemoveUnreferencedVariables(TCompiler *compiler, TIntermBlock *root, TSymbolTable *symbolTable)
{
    CollectVariableRefCountsTraverser collector;
    root->traverse(&collector);
    RemoveUnreferencedVariablesTraverser traverser(&collector.getSymbolIdRefCounts(),
                                                   &collector.getStructIdRefCounts(), symbolTable);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
