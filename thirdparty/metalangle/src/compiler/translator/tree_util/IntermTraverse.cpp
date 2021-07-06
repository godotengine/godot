//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_util/IntermTraverse.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNode_util.h"

namespace sh
{

// Traverse the intermediate representation tree, and call a node type specific visit function for
// each node. Traversal is done recursively through the node member function traverse(). Nodes with
// children can have their whole subtree skipped if preVisit is turned on and the type specific
// function returns false.
template <typename T>
void TIntermTraverser::traverse(T *node)
{
    ScopedNodeInTraversalPath addToPath(this, node);
    if (!addToPath.isWithinDepthLimit())
        return;

    bool visit = true;

    // Visit the node before children if pre-visiting.
    if (preVisit)
        visit = node->visit(PreVisit, this);

    if (visit)
    {
        size_t childIndex = 0;
        size_t childCount = node->getChildCount();

        while (childIndex < childCount && visit)
        {
            node->getChildNode(childIndex)->traverse(this);
            if (inVisit && childIndex != childCount - 1)
            {
                visit = node->visit(InVisit, this);
            }
            ++childIndex;
        }

        if (visit && postVisit)
            node->visit(PostVisit, this);
    }
}

// Instantiate template for RewriteAtomicFunctionExpressions, in case this gets inlined thus not
// exported from the TU.
template void TIntermTraverser::traverse(TIntermNode *);

void TIntermNode::traverse(TIntermTraverser *it)
{
    it->traverse(this);
}

void TIntermSymbol::traverse(TIntermTraverser *it)
{
    TIntermTraverser::ScopedNodeInTraversalPath addToPath(it, this);
    it->visitSymbol(this);
}

void TIntermConstantUnion::traverse(TIntermTraverser *it)
{
    TIntermTraverser::ScopedNodeInTraversalPath addToPath(it, this);
    it->visitConstantUnion(this);
}

void TIntermFunctionPrototype::traverse(TIntermTraverser *it)
{
    TIntermTraverser::ScopedNodeInTraversalPath addToPath(it, this);
    it->visitFunctionPrototype(this);
}

void TIntermBinary::traverse(TIntermTraverser *it)
{
    it->traverseBinary(this);
}

void TIntermUnary::traverse(TIntermTraverser *it)
{
    it->traverseUnary(this);
}

void TIntermFunctionDefinition::traverse(TIntermTraverser *it)
{
    it->traverseFunctionDefinition(this);
}

void TIntermBlock::traverse(TIntermTraverser *it)
{
    it->traverseBlock(this);
}

void TIntermAggregate::traverse(TIntermTraverser *it)
{
    it->traverseAggregate(this);
}

void TIntermLoop::traverse(TIntermTraverser *it)
{
    it->traverseLoop(this);
}

void TIntermPreprocessorDirective::traverse(TIntermTraverser *it)
{
    it->visitPreprocessorDirective(this);
}

bool TIntermSymbol::visit(Visit visit, TIntermTraverser *it)
{
    it->visitSymbol(this);
    return false;
}

bool TIntermConstantUnion::visit(Visit visit, TIntermTraverser *it)
{
    it->visitConstantUnion(this);
    return false;
}

bool TIntermFunctionPrototype::visit(Visit visit, TIntermTraverser *it)
{
    it->visitFunctionPrototype(this);
    return false;
}

bool TIntermFunctionDefinition::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitFunctionDefinition(visit, this);
}

bool TIntermUnary::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitUnary(visit, this);
}

bool TIntermSwizzle::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitSwizzle(visit, this);
}

bool TIntermBinary::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitBinary(visit, this);
}

bool TIntermTernary::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitTernary(visit, this);
}

bool TIntermAggregate::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitAggregate(visit, this);
}

bool TIntermDeclaration::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitDeclaration(visit, this);
}

bool TIntermInvariantDeclaration::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitInvariantDeclaration(visit, this);
}

bool TIntermBlock::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitBlock(visit, this);
}

bool TIntermIfElse::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitIfElse(visit, this);
}

bool TIntermLoop::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitLoop(visit, this);
}

bool TIntermBranch::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitBranch(visit, this);
}

bool TIntermSwitch::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitSwitch(visit, this);
}

bool TIntermCase::visit(Visit visit, TIntermTraverser *it)
{
    return it->visitCase(visit, this);
}

bool TIntermPreprocessorDirective::visit(Visit visit, TIntermTraverser *it)
{
    it->visitPreprocessorDirective(this);
    return false;
}

TIntermTraverser::TIntermTraverser(bool preVisit,
                                   bool inVisit,
                                   bool postVisit,
                                   TSymbolTable *symbolTable)
    : preVisit(preVisit),
      inVisit(inVisit),
      postVisit(postVisit),
      mMaxDepth(0),
      mMaxAllowedDepth(std::numeric_limits<int>::max()),
      mInGlobalScope(true),
      mSymbolTable(symbolTable)
{
    // Only enabling inVisit is not supported.
    ASSERT(!(inVisit && !preVisit && !postVisit));
}

TIntermTraverser::~TIntermTraverser() {}

void TIntermTraverser::setMaxAllowedDepth(int depth)
{
    mMaxAllowedDepth = depth;
}

const TIntermBlock *TIntermTraverser::getParentBlock() const
{
    if (!mParentBlockStack.empty())
    {
        return mParentBlockStack.back().node;
    }
    return nullptr;
}

void TIntermTraverser::pushParentBlock(TIntermBlock *node)
{
    mParentBlockStack.push_back(ParentBlock(node, 0));
}

void TIntermTraverser::incrementParentBlockPos()
{
    ++mParentBlockStack.back().pos;
}

void TIntermTraverser::popParentBlock()
{
    ASSERT(!mParentBlockStack.empty());
    mParentBlockStack.pop_back();
}

void TIntermTraverser::insertStatementsInParentBlock(const TIntermSequence &insertions)
{
    TIntermSequence emptyInsertionsAfter;
    insertStatementsInParentBlock(insertions, emptyInsertionsAfter);
}

void TIntermTraverser::insertStatementsInParentBlock(const TIntermSequence &insertionsBefore,
                                                     const TIntermSequence &insertionsAfter)
{
    ASSERT(!mParentBlockStack.empty());
    ParentBlock &parentBlock = mParentBlockStack.back();
    if (mPath.back() == parentBlock.node)
    {
        ASSERT(mParentBlockStack.size() >= 2u);
        // The current node is a block node, so the parent block is not the topmost one in the block
        // stack, but the one below that.
        parentBlock = mParentBlockStack.at(mParentBlockStack.size() - 2u);
    }
    NodeInsertMultipleEntry insert(parentBlock.node, parentBlock.pos, insertionsBefore,
                                   insertionsAfter);
    mInsertions.push_back(insert);
}

void TIntermTraverser::insertStatementInParentBlock(TIntermNode *statement)
{
    TIntermSequence insertions;
    insertions.push_back(statement);
    insertStatementsInParentBlock(insertions);
}

void TIntermTraverser::insertStatementsInBlockAtPosition(TIntermBlock *parent,
                                                         size_t position,
                                                         const TIntermSequence &insertionsBefore,
                                                         const TIntermSequence &insertionsAfter)
{
    ASSERT(parent);
    ASSERT(position >= 0);
    ASSERT(position < parent->getChildCount());

    mInsertions.emplace_back(parent, position, insertionsBefore, insertionsAfter);
}

void TLValueTrackingTraverser::setInFunctionCallOutParameter(bool inOutParameter)
{
    mInFunctionCallOutParameter = inOutParameter;
}

bool TLValueTrackingTraverser::isInFunctionCallOutParameter() const
{
    return mInFunctionCallOutParameter;
}

void TIntermTraverser::traverseBinary(TIntermBinary *node)
{
    traverse(node);
}

void TLValueTrackingTraverser::traverseBinary(TIntermBinary *node)
{
    ScopedNodeInTraversalPath addToPath(this, node);
    if (!addToPath.isWithinDepthLimit())
        return;

    bool visit = true;

    // visit the node before children if pre-visiting.
    if (preVisit)
        visit = node->visit(PreVisit, this);

    // Visit the children, in the right order.
    if (visit)
    {
        if (node->isAssignment())
        {
            ASSERT(!isLValueRequiredHere());
            setOperatorRequiresLValue(true);
        }

        node->getLeft()->traverse(this);

        if (node->isAssignment())
            setOperatorRequiresLValue(false);

        if (inVisit)
            visit = node->visit(InVisit, this);

        if (visit)
        {
            // Some binary operations like indexing can be inside an expression which must be an
            // l-value.
            bool parentOperatorRequiresLValue     = operatorRequiresLValue();
            bool parentInFunctionCallOutParameter = isInFunctionCallOutParameter();

            // Index is not required to be an l-value even when the surrounding expression is
            // required to be an l-value.
            TOperator op = node->getOp();
            if (op == EOpIndexDirect || op == EOpIndexDirectInterfaceBlock ||
                op == EOpIndexDirectStruct || op == EOpIndexIndirect)
            {
                setOperatorRequiresLValue(false);
                setInFunctionCallOutParameter(false);
            }

            node->getRight()->traverse(this);

            setOperatorRequiresLValue(parentOperatorRequiresLValue);
            setInFunctionCallOutParameter(parentInFunctionCallOutParameter);

            // Visit the node after the children, if requested and the traversal
            // hasn't been cancelled yet.
            if (postVisit)
                visit = node->visit(PostVisit, this);
        }
    }
}

void TIntermTraverser::traverseUnary(TIntermUnary *node)
{
    traverse(node);
}

void TLValueTrackingTraverser::traverseUnary(TIntermUnary *node)
{
    ScopedNodeInTraversalPath addToPath(this, node);
    if (!addToPath.isWithinDepthLimit())
        return;

    bool visit = true;

    if (preVisit)
        visit = node->visit(PreVisit, this);

    if (visit)
    {
        ASSERT(!operatorRequiresLValue());
        switch (node->getOp())
        {
            case EOpPostIncrement:
            case EOpPostDecrement:
            case EOpPreIncrement:
            case EOpPreDecrement:
                setOperatorRequiresLValue(true);
                break;
            default:
                break;
        }

        node->getOperand()->traverse(this);

        setOperatorRequiresLValue(false);

        if (postVisit)
            visit = node->visit(PostVisit, this);
    }
}

// Traverse a function definition node. This keeps track of global scope.
void TIntermTraverser::traverseFunctionDefinition(TIntermFunctionDefinition *node)
{
    ScopedNodeInTraversalPath addToPath(this, node);
    if (!addToPath.isWithinDepthLimit())
        return;

    bool visit = true;

    if (preVisit)
        visit = node->visit(PreVisit, this);

    if (visit)
    {
        node->getFunctionPrototype()->traverse(this);
        if (inVisit)
            visit = node->visit(InVisit, this);
        if (visit)
        {
            mInGlobalScope = false;
            node->getBody()->traverse(this);
            mInGlobalScope = true;
            if (postVisit)
                visit = node->visit(PostVisit, this);
        }
    }
}

// Traverse a block node. This keeps track of the position of traversed child nodes within the block
// so that nodes may be inserted before or after them.
void TIntermTraverser::traverseBlock(TIntermBlock *node)
{
    ScopedNodeInTraversalPath addToPath(this, node);
    if (!addToPath.isWithinDepthLimit())
        return;

    pushParentBlock(node);

    bool visit = true;

    TIntermSequence *sequence = node->getSequence();

    if (preVisit)
        visit = node->visit(PreVisit, this);

    if (visit)
    {
        for (auto *child : *sequence)
        {
            if (visit)
            {
                child->traverse(this);
                if (inVisit)
                {
                    if (child != sequence->back())
                        visit = node->visit(InVisit, this);
                }

                incrementParentBlockPos();
            }
        }

        if (visit && postVisit)
            visit = node->visit(PostVisit, this);
    }

    popParentBlock();
}

void TIntermTraverser::traverseAggregate(TIntermAggregate *node)
{
    traverse(node);
}

bool TIntermTraverser::CompareInsertion(const NodeInsertMultipleEntry &a,
                                        const NodeInsertMultipleEntry &b)
{
    if (a.parent != b.parent)
    {
        return a.parent < b.parent;
    }
    return a.position < b.position;
}

bool TIntermTraverser::updateTree(TCompiler *compiler, TIntermNode *node)
{
    // Sort the insertions so that insertion position is increasing and same position insertions are
    // not reordered. The insertions are processed in reverse order so that multiple insertions to
    // the same parent node are handled correctly.
    std::stable_sort(mInsertions.begin(), mInsertions.end(), CompareInsertion);
    for (size_t ii = 0; ii < mInsertions.size(); ++ii)
    {
        // If two insertions are to the same position, insert them in the order they were specified.
        // The std::stable_sort call above will automatically guarantee this.
        const NodeInsertMultipleEntry &insertion = mInsertions[mInsertions.size() - ii - 1];
        ASSERT(insertion.parent);
        if (!insertion.insertionsAfter.empty())
        {
            bool inserted = insertion.parent->insertChildNodes(insertion.position + 1,
                                                               insertion.insertionsAfter);
            ASSERT(inserted);
        }
        if (!insertion.insertionsBefore.empty())
        {
            bool inserted =
                insertion.parent->insertChildNodes(insertion.position, insertion.insertionsBefore);
            ASSERT(inserted);
        }
    }
    for (size_t ii = 0; ii < mReplacements.size(); ++ii)
    {
        const NodeUpdateEntry &replacement = mReplacements[ii];
        ASSERT(replacement.parent);
        bool replaced =
            replacement.parent->replaceChildNode(replacement.original, replacement.replacement);
        ASSERT(replaced);

        if (!replacement.originalBecomesChildOfReplacement)
        {
            // In AST traversing, a parent is visited before its children.
            // After we replace a node, if its immediate child is to
            // be replaced, we need to make sure we don't update the replaced
            // node; instead, we update the replacement node.
            for (size_t jj = ii + 1; jj < mReplacements.size(); ++jj)
            {
                NodeUpdateEntry &replacement2 = mReplacements[jj];
                if (replacement2.parent == replacement.original)
                    replacement2.parent = replacement.replacement;
            }
        }
    }
    for (size_t ii = 0; ii < mMultiReplacements.size(); ++ii)
    {
        const NodeReplaceWithMultipleEntry &replacement = mMultiReplacements[ii];
        ASSERT(replacement.parent);
        bool replaced = replacement.parent->replaceChildNodeWithMultiple(replacement.original,
                                                                         replacement.replacements);
        ASSERT(replaced);
    }

    clearReplacementQueue();

    return compiler->validateAST(node);
}

void TIntermTraverser::clearReplacementQueue()
{
    mReplacements.clear();
    mMultiReplacements.clear();
    mInsertions.clear();
}

void TIntermTraverser::queueReplacement(TIntermNode *replacement, OriginalNode originalStatus)
{
    queueReplacementWithParent(getParentNode(), mPath.back(), replacement, originalStatus);
}

void TIntermTraverser::queueReplacementWithParent(TIntermNode *parent,
                                                  TIntermNode *original,
                                                  TIntermNode *replacement,
                                                  OriginalNode originalStatus)
{
    bool originalBecomesChild = (originalStatus == OriginalNode::BECOMES_CHILD);
    mReplacements.push_back(NodeUpdateEntry(parent, original, replacement, originalBecomesChild));
}

TLValueTrackingTraverser::TLValueTrackingTraverser(bool preVisit,
                                                   bool inVisit,
                                                   bool postVisit,
                                                   TSymbolTable *symbolTable)
    : TIntermTraverser(preVisit, inVisit, postVisit, symbolTable),
      mOperatorRequiresLValue(false),
      mInFunctionCallOutParameter(false)
{
    ASSERT(symbolTable);
}

void TLValueTrackingTraverser::traverseAggregate(TIntermAggregate *node)
{
    ScopedNodeInTraversalPath addToPath(this, node);
    if (!addToPath.isWithinDepthLimit())
        return;

    bool visit = true;

    TIntermSequence *sequence = node->getSequence();

    if (preVisit)
        visit = node->visit(PreVisit, this);

    if (visit)
    {
        size_t paramIndex = 0u;
        for (auto *child : *sequence)
        {
            if (visit)
            {
                if (node->getFunction())
                {
                    // Both built-ins and user defined functions should have the function symbol
                    // set.
                    ASSERT(paramIndex < node->getFunction()->getParamCount());
                    TQualifier qualifier =
                        node->getFunction()->getParam(paramIndex)->getType().getQualifier();
                    setInFunctionCallOutParameter(qualifier == EvqOut || qualifier == EvqInOut);
                    ++paramIndex;
                }
                else
                {
                    ASSERT(node->isConstructor());
                }
                child->traverse(this);
                if (inVisit)
                {
                    if (child != sequence->back())
                        visit = node->visit(InVisit, this);
                }
            }
        }
        setInFunctionCallOutParameter(false);

        if (visit && postVisit)
            visit = node->visit(PostVisit, this);
    }
}

void TIntermTraverser::traverseLoop(TIntermLoop *node)
{
    traverse(node);
}
}  // namespace sh
