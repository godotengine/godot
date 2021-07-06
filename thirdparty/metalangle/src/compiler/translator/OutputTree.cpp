//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

void OutputFunction(TInfoSinkBase &out, const char *str, const TFunction *func)
{
    const char *internal =
        (func->symbolType() == SymbolType::AngleInternal) ? " (internal function)" : "";
    out << str << internal << ": " << func->name() << " (symbol id " << func->uniqueId().get()
        << ")";
}

// Two purposes:
// 1.  Show an example of how to iterate tree.  Functions can also directly call traverse() on
//     children themselves to have finer grained control over the process than shown here, though
//     that's not recommended if it can be avoided.
// 2.  Print out a text based description of the tree.

// The traverser subclass is used to carry along data from node to node in the traversal.
class TOutputTraverser : public TIntermTraverser
{
  public:
    TOutputTraverser(TInfoSinkBase &out)
        : TIntermTraverser(true, false, false), mOut(out), mIndentDepth(0)
    {}

  protected:
    void visitSymbol(TIntermSymbol *) override;
    void visitConstantUnion(TIntermConstantUnion *) override;
    bool visitSwizzle(Visit visit, TIntermSwizzle *node) override;
    bool visitBinary(Visit visit, TIntermBinary *) override;
    bool visitUnary(Visit visit, TIntermUnary *) override;
    bool visitTernary(Visit visit, TIntermTernary *node) override;
    bool visitIfElse(Visit visit, TIntermIfElse *node) override;
    bool visitSwitch(Visit visit, TIntermSwitch *node) override;
    bool visitCase(Visit visit, TIntermCase *node) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override;
    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *) override;
    bool visitBlock(Visit visit, TIntermBlock *) override;
    bool visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node) override;
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;
    bool visitLoop(Visit visit, TIntermLoop *) override;
    bool visitBranch(Visit visit, TIntermBranch *) override;

    int getCurrentIndentDepth() const { return mIndentDepth + getCurrentTraversalDepth(); }

    TInfoSinkBase &mOut;
    int mIndentDepth;
};

//
// Helper functions for printing, not part of traversing.
//
void OutputTreeText(TInfoSinkBase &out, TIntermNode *node, const int depth)
{
    int i;

    out.location(node->getLine().first_file, node->getLine().first_line);

    for (i = 0; i < depth; ++i)
        out << "  ";
}

//
// The rest of the file are the traversal functions.  The last one
// is the one that starts the traversal.
//
// Return true from interior nodes to have the external traversal
// continue on to children.  If you process children yourself,
// return false.
//

void TOutputTraverser::visitSymbol(TIntermSymbol *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    if (node->variable().symbolType() == SymbolType::Empty)
    {
        mOut << "''";
    }
    else
    {
        mOut << "'" << node->getName() << "' ";
    }
    mOut << "(symbol id " << node->uniqueId().get() << ") ";
    mOut << "(" << node->getType() << ")";
    mOut << "\n";
}

bool TOutputTraverser::visitSwizzle(Visit visit, TIntermSwizzle *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "vector swizzle (";
    node->writeOffsetsAsXYZW(&mOut);
    mOut << ")";

    mOut << " (" << node->getType() << ")";
    mOut << "\n";
    return true;
}

bool TOutputTraverser::visitBinary(Visit visit, TIntermBinary *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    switch (node->getOp())
    {
        case EOpComma:
            mOut << "comma";
            break;
        case EOpAssign:
            mOut << "move second child to first child";
            break;
        case EOpInitialize:
            mOut << "initialize first child with second child";
            break;
        case EOpAddAssign:
            mOut << "add second child into first child";
            break;
        case EOpSubAssign:
            mOut << "subtract second child into first child";
            break;
        case EOpMulAssign:
            mOut << "multiply second child into first child";
            break;
        case EOpVectorTimesMatrixAssign:
            mOut << "matrix mult second child into first child";
            break;
        case EOpVectorTimesScalarAssign:
            mOut << "vector scale second child into first child";
            break;
        case EOpMatrixTimesScalarAssign:
            mOut << "matrix scale second child into first child";
            break;
        case EOpMatrixTimesMatrixAssign:
            mOut << "matrix mult second child into first child";
            break;
        case EOpDivAssign:
            mOut << "divide second child into first child";
            break;
        case EOpIModAssign:
            mOut << "modulo second child into first child";
            break;
        case EOpBitShiftLeftAssign:
            mOut << "bit-wise shift first child left by second child";
            break;
        case EOpBitShiftRightAssign:
            mOut << "bit-wise shift first child right by second child";
            break;
        case EOpBitwiseAndAssign:
            mOut << "bit-wise and second child into first child";
            break;
        case EOpBitwiseXorAssign:
            mOut << "bit-wise xor second child into first child";
            break;
        case EOpBitwiseOrAssign:
            mOut << "bit-wise or second child into first child";
            break;

        case EOpIndexDirect:
            mOut << "direct index";
            break;
        case EOpIndexIndirect:
            mOut << "indirect index";
            break;
        case EOpIndexDirectStruct:
            mOut << "direct index for structure";
            break;
        case EOpIndexDirectInterfaceBlock:
            mOut << "direct index for interface block";
            break;

        case EOpAdd:
            mOut << "add";
            break;
        case EOpSub:
            mOut << "subtract";
            break;
        case EOpMul:
            mOut << "component-wise multiply";
            break;
        case EOpDiv:
            mOut << "divide";
            break;
        case EOpIMod:
            mOut << "modulo";
            break;
        case EOpBitShiftLeft:
            mOut << "bit-wise shift left";
            break;
        case EOpBitShiftRight:
            mOut << "bit-wise shift right";
            break;
        case EOpBitwiseAnd:
            mOut << "bit-wise and";
            break;
        case EOpBitwiseXor:
            mOut << "bit-wise xor";
            break;
        case EOpBitwiseOr:
            mOut << "bit-wise or";
            break;

        case EOpEqual:
            mOut << "Compare Equal";
            break;
        case EOpNotEqual:
            mOut << "Compare Not Equal";
            break;
        case EOpLessThan:
            mOut << "Compare Less Than";
            break;
        case EOpGreaterThan:
            mOut << "Compare Greater Than";
            break;
        case EOpLessThanEqual:
            mOut << "Compare Less Than or Equal";
            break;
        case EOpGreaterThanEqual:
            mOut << "Compare Greater Than or Equal";
            break;

        case EOpVectorTimesScalar:
            mOut << "vector-scale";
            break;
        case EOpVectorTimesMatrix:
            mOut << "vector-times-matrix";
            break;
        case EOpMatrixTimesVector:
            mOut << "matrix-times-vector";
            break;
        case EOpMatrixTimesScalar:
            mOut << "matrix-scale";
            break;
        case EOpMatrixTimesMatrix:
            mOut << "matrix-multiply";
            break;

        case EOpLogicalOr:
            mOut << "logical-or";
            break;
        case EOpLogicalXor:
            mOut << "logical-xor";
            break;
        case EOpLogicalAnd:
            mOut << "logical-and";
            break;
        default:
            mOut << "<unknown op>";
    }

    mOut << " (" << node->getType() << ")";

    mOut << "\n";

    // Special handling for direct indexes. Because constant
    // unions are not aware they are struct indexes, treat them
    // here where we have that contextual knowledge.
    if (node->getOp() == EOpIndexDirectStruct || node->getOp() == EOpIndexDirectInterfaceBlock)
    {
        node->getLeft()->traverse(this);

        TIntermConstantUnion *intermConstantUnion = node->getRight()->getAsConstantUnion();
        ASSERT(intermConstantUnion);

        OutputTreeText(mOut, intermConstantUnion, getCurrentIndentDepth() + 1);

        // The following code finds the field name from the constant union
        const TConstantUnion *constantUnion   = intermConstantUnion->getConstantValue();
        const TStructure *structure           = node->getLeft()->getType().getStruct();
        const TInterfaceBlock *interfaceBlock = node->getLeft()->getType().getInterfaceBlock();
        ASSERT(structure || interfaceBlock);

        const TFieldList &fields = structure ? structure->fields() : interfaceBlock->fields();

        const TField *field = fields[constantUnion->getIConst()];

        mOut << constantUnion->getIConst() << " (field '" << field->name() << "')";

        mOut << "\n";

        return false;
    }

    return true;
}

bool TOutputTraverser::visitUnary(Visit visit, TIntermUnary *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    switch (node->getOp())
    {
        // Give verbose names for ops that have special syntax and some built-in functions that are
        // easy to confuse with others, but mostly use GLSL names for functions.
        case EOpNegative:
            mOut << "Negate value";
            break;
        case EOpPositive:
            mOut << "Positive sign";
            break;
        case EOpLogicalNot:
            mOut << "negation";
            break;
        case EOpBitwiseNot:
            mOut << "bit-wise not";
            break;

        case EOpPostIncrement:
            mOut << "Post-Increment";
            break;
        case EOpPostDecrement:
            mOut << "Post-Decrement";
            break;
        case EOpPreIncrement:
            mOut << "Pre-Increment";
            break;
        case EOpPreDecrement:
            mOut << "Pre-Decrement";
            break;

        case EOpArrayLength:
            mOut << "Array length";
            break;

        case EOpLogicalNotComponentWise:
            mOut << "component-wise not";
            break;

        default:
            mOut << GetOperatorString(node->getOp());
            break;
    }

    mOut << " (" << node->getType() << ")";

    mOut << "\n";

    return true;
}

bool TOutputTraverser::visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "Function Definition:\n";
    return true;
}

bool TOutputTraverser::visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "Invariant Declaration:\n";
    return true;
}

void TOutputTraverser::visitFunctionPrototype(TIntermFunctionPrototype *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());
    OutputFunction(mOut, "Function Prototype", node->getFunction());
    mOut << " (" << node->getType() << ")";
    mOut << "\n";
    size_t paramCount = node->getFunction()->getParamCount();
    for (size_t i = 0; i < paramCount; ++i)
    {
        const TVariable *param = node->getFunction()->getParam(i);
        OutputTreeText(mOut, node, getCurrentIndentDepth() + 1);
        mOut << "parameter: " << param->name() << " (" << param->getType() << ")";
    }
}

bool TOutputTraverser::visitAggregate(Visit visit, TIntermAggregate *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    if (node->getOp() == EOpNull)
    {
        mOut.prefix(SH_ERROR);
        mOut << "node is still EOpNull!\n";
        return true;
    }

    // Give verbose names for some built-in functions that are easy to confuse with others, but
    // mostly use GLSL names for functions.
    switch (node->getOp())
    {
        case EOpCallFunctionInAST:
            OutputFunction(mOut, "Call an user-defined function", node->getFunction());
            break;
        case EOpCallInternalRawFunction:
            OutputFunction(mOut, "Call an internal function with raw implementation",
                           node->getFunction());
            break;
        case EOpCallBuiltInFunction:
            OutputFunction(mOut, "Call a built-in function", node->getFunction());
            break;

        case EOpConstruct:
            // The type of the constructor will be printed below.
            mOut << "Construct";
            break;

        case EOpEqualComponentWise:
            mOut << "component-wise equal";
            break;
        case EOpNotEqualComponentWise:
            mOut << "component-wise not equal";
            break;
        case EOpLessThanComponentWise:
            mOut << "component-wise less than";
            break;
        case EOpGreaterThanComponentWise:
            mOut << "component-wise greater than";
            break;
        case EOpLessThanEqualComponentWise:
            mOut << "component-wise less than or equal";
            break;
        case EOpGreaterThanEqualComponentWise:
            mOut << "component-wise greater than or equal";
            break;

        case EOpDot:
            mOut << "dot product";
            break;
        case EOpCross:
            mOut << "cross product";
            break;
        case EOpMulMatrixComponentWise:
            mOut << "component-wise multiply";
            break;

        default:
            mOut << GetOperatorString(node->getOp());
            break;
    }

    mOut << " (" << node->getType() << ")";

    mOut << "\n";

    return true;
}

bool TOutputTraverser::visitBlock(Visit visit, TIntermBlock *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "Code block\n";

    return true;
}

bool TOutputTraverser::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "Declaration\n";

    return true;
}

bool TOutputTraverser::visitTernary(Visit visit, TIntermTernary *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    mOut << "Ternary selection";
    mOut << " (" << node->getType() << ")\n";

    ++mIndentDepth;

    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "Condition\n";
    node->getCondition()->traverse(this);

    OutputTreeText(mOut, node, getCurrentIndentDepth());
    if (node->getTrueExpression())
    {
        mOut << "true case\n";
        node->getTrueExpression()->traverse(this);
    }
    if (node->getFalseExpression())
    {
        OutputTreeText(mOut, node, getCurrentIndentDepth());
        mOut << "false case\n";
        node->getFalseExpression()->traverse(this);
    }

    --mIndentDepth;

    return false;
}

bool TOutputTraverser::visitIfElse(Visit visit, TIntermIfElse *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    mOut << "If test\n";

    ++mIndentDepth;

    OutputTreeText(mOut, node, getCurrentIndentDepth());
    mOut << "Condition\n";
    node->getCondition()->traverse(this);

    OutputTreeText(mOut, node, getCurrentIndentDepth());
    if (node->getTrueBlock())
    {
        mOut << "true case\n";
        node->getTrueBlock()->traverse(this);
    }
    else
    {
        mOut << "true case is null\n";
    }

    if (node->getFalseBlock())
    {
        OutputTreeText(mOut, node, getCurrentIndentDepth());
        mOut << "false case\n";
        node->getFalseBlock()->traverse(this);
    }

    --mIndentDepth;

    return false;
}

bool TOutputTraverser::visitSwitch(Visit visit, TIntermSwitch *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    mOut << "Switch\n";

    return true;
}

bool TOutputTraverser::visitCase(Visit visit, TIntermCase *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    if (node->getCondition() == nullptr)
    {
        mOut << "Default\n";
    }
    else
    {
        mOut << "Case\n";
    }

    return true;
}

void TOutputTraverser::visitConstantUnion(TIntermConstantUnion *node)
{
    size_t size = node->getType().getObjectSize();

    for (size_t i = 0; i < size; i++)
    {
        OutputTreeText(mOut, node, getCurrentIndentDepth());
        switch (node->getConstantValue()[i].getType())
        {
            case EbtBool:
                if (node->getConstantValue()[i].getBConst())
                    mOut << "true";
                else
                    mOut << "false";

                mOut << " ("
                     << "const bool"
                     << ")";
                mOut << "\n";
                break;
            case EbtFloat:
                mOut << node->getConstantValue()[i].getFConst();
                mOut << " (const float)\n";
                break;
            case EbtInt:
                mOut << node->getConstantValue()[i].getIConst();
                mOut << " (const int)\n";
                break;
            case EbtUInt:
                mOut << node->getConstantValue()[i].getUConst();
                mOut << " (const uint)\n";
                break;
            case EbtYuvCscStandardEXT:
                mOut << getYuvCscStandardEXTString(
                    node->getConstantValue()[i].getYuvCscStandardEXTConst());
                mOut << " (const yuvCscStandardEXT)\n";
                break;
            default:
                mOut.prefix(SH_ERROR);
                mOut << "Unknown constant\n";
                break;
        }
    }
}

bool TOutputTraverser::visitLoop(Visit visit, TIntermLoop *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    mOut << "Loop with condition ";
    if (node->getType() == ELoopDoWhile)
        mOut << "not ";
    mOut << "tested first\n";

    ++mIndentDepth;

    OutputTreeText(mOut, node, getCurrentIndentDepth());
    if (node->getCondition())
    {
        mOut << "Loop Condition\n";
        node->getCondition()->traverse(this);
    }
    else
    {
        mOut << "No loop condition\n";
    }

    OutputTreeText(mOut, node, getCurrentIndentDepth());
    if (node->getBody())
    {
        mOut << "Loop Body\n";
        node->getBody()->traverse(this);
    }
    else
    {
        mOut << "No loop body\n";
    }

    if (node->getExpression())
    {
        OutputTreeText(mOut, node, getCurrentIndentDepth());
        mOut << "Loop Terminal Expression\n";
        node->getExpression()->traverse(this);
    }

    --mIndentDepth;

    return false;
}

bool TOutputTraverser::visitBranch(Visit visit, TIntermBranch *node)
{
    OutputTreeText(mOut, node, getCurrentIndentDepth());

    switch (node->getFlowOp())
    {
        case EOpKill:
            mOut << "Branch: Kill";
            break;
        case EOpBreak:
            mOut << "Branch: Break";
            break;
        case EOpContinue:
            mOut << "Branch: Continue";
            break;
        case EOpReturn:
            mOut << "Branch: Return";
            break;
        default:
            mOut << "Branch: Unknown Branch";
            break;
    }

    if (node->getExpression())
    {
        mOut << " with expression\n";
        ++mIndentDepth;
        node->getExpression()->traverse(this);
        --mIndentDepth;
    }
    else
    {
        mOut << "\n";
    }

    return false;
}

}  // anonymous namespace

void OutputTree(TIntermNode *root, TInfoSinkBase &out)
{
    TOutputTraverser it(out);
    ASSERT(root);
    root->traverse(&it);
}

}  // namespace sh
