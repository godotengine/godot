//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/ValidateLimitations.h"

#include "angle_gl.h"
#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/ParseContext.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

int GetLoopSymbolId(TIntermLoop *loop)
{
    // Here we assume all the operations are valid, because the loop node is
    // already validated before this call.
    TIntermSequence *declSeq = loop->getInit()->getAsDeclarationNode()->getSequence();
    TIntermBinary *declInit  = (*declSeq)[0]->getAsBinaryNode();
    TIntermSymbol *symbol    = declInit->getLeft()->getAsSymbolNode();

    return symbol->uniqueId().get();
}

// Traverses a node to check if it represents a constant index expression.
// Definition:
// constant-index-expressions are a superset of constant-expressions.
// Constant-index-expressions can include loop indices as defined in
// GLSL ES 1.0 spec, Appendix A, section 4.
// The following are constant-index-expressions:
// - Constant expressions
// - Loop indices as defined in section 4
// - Expressions composed of both of the above
class ValidateConstIndexExpr : public TIntermTraverser
{
  public:
    ValidateConstIndexExpr(const std::vector<int> &loopSymbols)
        : TIntermTraverser(true, false, false), mValid(true), mLoopSymbolIds(loopSymbols)
    {}

    // Returns true if the parsed node represents a constant index expression.
    bool isValid() const { return mValid; }

    void visitSymbol(TIntermSymbol *symbol) override
    {
        // Only constants and loop indices are allowed in a
        // constant index expression.
        if (mValid)
        {
            bool isLoopSymbol = std::find(mLoopSymbolIds.begin(), mLoopSymbolIds.end(),
                                          symbol->uniqueId().get()) != mLoopSymbolIds.end();
            mValid            = (symbol->getQualifier() == EvqConst) || isLoopSymbol;
        }
    }

  private:
    bool mValid;
    const std::vector<int> mLoopSymbolIds;
};

// Traverses intermediate tree to ensure that the shader does not exceed the
// minimum functionality mandated in GLSL 1.0 spec, Appendix A.
class ValidateLimitationsTraverser : public TLValueTrackingTraverser
{
  public:
    ValidateLimitationsTraverser(sh::GLenum shaderType,
                                 TSymbolTable *symbolTable,
                                 TDiagnostics *diagnostics);

    void visitSymbol(TIntermSymbol *node) override;
    bool visitBinary(Visit, TIntermBinary *) override;
    bool visitLoop(Visit, TIntermLoop *) override;

  private:
    void error(TSourceLoc loc, const char *reason, const char *token);
    void error(TSourceLoc loc, const char *reason, const ImmutableString &token);

    bool isLoopIndex(TIntermSymbol *symbol);
    bool validateLoopType(TIntermLoop *node);

    bool validateForLoopHeader(TIntermLoop *node);
    // If valid, return the index symbol id; Otherwise, return -1.
    int validateForLoopInit(TIntermLoop *node);
    bool validateForLoopCond(TIntermLoop *node, int indexSymbolId);
    bool validateForLoopExpr(TIntermLoop *node, int indexSymbolId);

    // Returns true if indexing does not exceed the minimum functionality
    // mandated in GLSL 1.0 spec, Appendix A, Section 5.
    bool isConstExpr(TIntermNode *node);
    bool isConstIndexExpr(TIntermNode *node);
    bool validateIndexing(TIntermBinary *node);

    sh::GLenum mShaderType;
    TDiagnostics *mDiagnostics;
    std::vector<int> mLoopSymbolIds;
};

ValidateLimitationsTraverser::ValidateLimitationsTraverser(sh::GLenum shaderType,
                                                           TSymbolTable *symbolTable,
                                                           TDiagnostics *diagnostics)
    : TLValueTrackingTraverser(true, false, false, symbolTable),
      mShaderType(shaderType),
      mDiagnostics(diagnostics)
{
    ASSERT(diagnostics);
}

void ValidateLimitationsTraverser::visitSymbol(TIntermSymbol *node)
{
    if (isLoopIndex(node) && isLValueRequiredHere())
    {
        error(node->getLine(),
              "Loop index cannot be statically assigned to within the body of the loop",
              node->getName());
    }
}

bool ValidateLimitationsTraverser::visitBinary(Visit, TIntermBinary *node)
{
    // Check indexing.
    switch (node->getOp())
    {
        case EOpIndexDirect:
        case EOpIndexIndirect:
            validateIndexing(node);
            break;
        default:
            break;
    }
    return true;
}

bool ValidateLimitationsTraverser::visitLoop(Visit, TIntermLoop *node)
{
    if (!validateLoopType(node))
        return false;

    if (!validateForLoopHeader(node))
        return false;

    TIntermNode *body = node->getBody();
    if (body != nullptr)
    {
        mLoopSymbolIds.push_back(GetLoopSymbolId(node));
        body->traverse(this);
        mLoopSymbolIds.pop_back();
    }

    // The loop is fully processed - no need to visit children.
    return false;
}

void ValidateLimitationsTraverser::error(TSourceLoc loc, const char *reason, const char *token)
{
    mDiagnostics->error(loc, reason, token);
}

void ValidateLimitationsTraverser::error(TSourceLoc loc,
                                         const char *reason,
                                         const ImmutableString &token)
{
    error(loc, reason, token.data());
}

bool ValidateLimitationsTraverser::isLoopIndex(TIntermSymbol *symbol)
{
    return std::find(mLoopSymbolIds.begin(), mLoopSymbolIds.end(), symbol->uniqueId().get()) !=
           mLoopSymbolIds.end();
}

bool ValidateLimitationsTraverser::validateLoopType(TIntermLoop *node)
{
    TLoopType type = node->getType();
    if (type == ELoopFor)
        return true;

    // Reject while and do-while loops.
    error(node->getLine(), "This type of loop is not allowed", type == ELoopWhile ? "while" : "do");
    return false;
}

bool ValidateLimitationsTraverser::validateForLoopHeader(TIntermLoop *node)
{
    ASSERT(node->getType() == ELoopFor);

    //
    // The for statement has the form:
    //    for ( init-declaration ; condition ; expression ) statement
    //
    int indexSymbolId = validateForLoopInit(node);
    if (indexSymbolId < 0)
        return false;
    if (!validateForLoopCond(node, indexSymbolId))
        return false;
    if (!validateForLoopExpr(node, indexSymbolId))
        return false;

    return true;
}

int ValidateLimitationsTraverser::validateForLoopInit(TIntermLoop *node)
{
    TIntermNode *init = node->getInit();
    if (init == nullptr)
    {
        error(node->getLine(), "Missing init declaration", "for");
        return -1;
    }

    //
    // init-declaration has the form:
    //     type-specifier identifier = constant-expression
    //
    TIntermDeclaration *decl = init->getAsDeclarationNode();
    if (decl == nullptr)
    {
        error(init->getLine(), "Invalid init declaration", "for");
        return -1;
    }
    // To keep things simple do not allow declaration list.
    TIntermSequence *declSeq = decl->getSequence();
    if (declSeq->size() != 1)
    {
        error(decl->getLine(), "Invalid init declaration", "for");
        return -1;
    }
    TIntermBinary *declInit = (*declSeq)[0]->getAsBinaryNode();
    if ((declInit == nullptr) || (declInit->getOp() != EOpInitialize))
    {
        error(decl->getLine(), "Invalid init declaration", "for");
        return -1;
    }
    TIntermSymbol *symbol = declInit->getLeft()->getAsSymbolNode();
    if (symbol == nullptr)
    {
        error(declInit->getLine(), "Invalid init declaration", "for");
        return -1;
    }
    // The loop index has type int or float.
    TBasicType type = symbol->getBasicType();
    if ((type != EbtInt) && (type != EbtUInt) && (type != EbtFloat))
    {
        error(symbol->getLine(), "Invalid type for loop index", getBasicString(type));
        return -1;
    }
    // The loop index is initialized with constant expression.
    if (!isConstExpr(declInit->getRight()))
    {
        error(declInit->getLine(), "Loop index cannot be initialized with non-constant expression",
              symbol->getName());
        return -1;
    }

    return symbol->uniqueId().get();
}

bool ValidateLimitationsTraverser::validateForLoopCond(TIntermLoop *node, int indexSymbolId)
{
    TIntermNode *cond = node->getCondition();
    if (cond == nullptr)
    {
        error(node->getLine(), "Missing condition", "for");
        return false;
    }
    //
    // condition has the form:
    //     loop_index relational_operator constant_expression
    //
    TIntermBinary *binOp = cond->getAsBinaryNode();
    if (binOp == nullptr)
    {
        error(node->getLine(), "Invalid condition", "for");
        return false;
    }
    // Loop index should be to the left of relational operator.
    TIntermSymbol *symbol = binOp->getLeft()->getAsSymbolNode();
    if (symbol == nullptr)
    {
        error(binOp->getLine(), "Invalid condition", "for");
        return false;
    }
    if (symbol->uniqueId().get() != indexSymbolId)
    {
        error(symbol->getLine(), "Expected loop index", symbol->getName());
        return false;
    }
    // Relational operator is one of: > >= < <= == or !=.
    switch (binOp->getOp())
    {
        case EOpEqual:
        case EOpNotEqual:
        case EOpLessThan:
        case EOpGreaterThan:
        case EOpLessThanEqual:
        case EOpGreaterThanEqual:
            break;
        default:
            error(binOp->getLine(), "Invalid relational operator",
                  GetOperatorString(binOp->getOp()));
            break;
    }
    // Loop index must be compared with a constant.
    if (!isConstExpr(binOp->getRight()))
    {
        error(binOp->getLine(), "Loop index cannot be compared with non-constant expression",
              symbol->getName());
        return false;
    }

    return true;
}

bool ValidateLimitationsTraverser::validateForLoopExpr(TIntermLoop *node, int indexSymbolId)
{
    TIntermNode *expr = node->getExpression();
    if (expr == nullptr)
    {
        error(node->getLine(), "Missing expression", "for");
        return false;
    }

    // for expression has one of the following forms:
    //     loop_index++
    //     loop_index--
    //     loop_index += constant_expression
    //     loop_index -= constant_expression
    //     ++loop_index
    //     --loop_index
    // The last two forms are not specified in the spec, but I am assuming
    // its an oversight.
    TIntermUnary *unOp   = expr->getAsUnaryNode();
    TIntermBinary *binOp = unOp ? nullptr : expr->getAsBinaryNode();

    TOperator op          = EOpNull;
    TIntermSymbol *symbol = nullptr;
    if (unOp != nullptr)
    {
        op     = unOp->getOp();
        symbol = unOp->getOperand()->getAsSymbolNode();
    }
    else if (binOp != nullptr)
    {
        op     = binOp->getOp();
        symbol = binOp->getLeft()->getAsSymbolNode();
    }

    // The operand must be loop index.
    if (symbol == nullptr)
    {
        error(expr->getLine(), "Invalid expression", "for");
        return false;
    }
    if (symbol->uniqueId().get() != indexSymbolId)
    {
        error(symbol->getLine(), "Expected loop index", symbol->getName());
        return false;
    }

    // The operator is one of: ++ -- += -=.
    switch (op)
    {
        case EOpPostIncrement:
        case EOpPostDecrement:
        case EOpPreIncrement:
        case EOpPreDecrement:
            ASSERT((unOp != nullptr) && (binOp == nullptr));
            break;
        case EOpAddAssign:
        case EOpSubAssign:
            ASSERT((unOp == nullptr) && (binOp != nullptr));
            break;
        default:
            error(expr->getLine(), "Invalid operator", GetOperatorString(op));
            return false;
    }

    // Loop index must be incremented/decremented with a constant.
    if (binOp != nullptr)
    {
        if (!isConstExpr(binOp->getRight()))
        {
            error(binOp->getLine(), "Loop index cannot be modified by non-constant expression",
                  symbol->getName());
            return false;
        }
    }

    return true;
}

bool ValidateLimitationsTraverser::isConstExpr(TIntermNode *node)
{
    ASSERT(node != nullptr);
    return node->getAsConstantUnion() != nullptr && node->getAsTyped()->getQualifier() == EvqConst;
}

bool ValidateLimitationsTraverser::isConstIndexExpr(TIntermNode *node)
{
    ASSERT(node != nullptr);

    ValidateConstIndexExpr validate(mLoopSymbolIds);
    node->traverse(&validate);
    return validate.isValid();
}

bool ValidateLimitationsTraverser::validateIndexing(TIntermBinary *node)
{
    ASSERT((node->getOp() == EOpIndexDirect) || (node->getOp() == EOpIndexIndirect));

    bool valid          = true;
    TIntermTyped *index = node->getRight();
    // The index expession must be a constant-index-expression unless
    // the operand is a uniform in a vertex shader.
    TIntermTyped *operand = node->getLeft();
    bool skip = (mShaderType == GL_VERTEX_SHADER) && (operand->getQualifier() == EvqUniform);
    if (!skip && !isConstIndexExpr(index))
    {
        error(index->getLine(), "Index expression must be constant", "[]");
        valid = false;
    }
    return valid;
}

}  // namespace

bool ValidateLimitations(TIntermNode *root,
                         GLenum shaderType,
                         TSymbolTable *symbolTable,
                         TDiagnostics *diagnostics)
{
    ValidateLimitationsTraverser validate(shaderType, symbolTable, diagnostics);
    root->traverse(&validate);
    return diagnostics->numErrors() == 0;
}

}  // namespace sh
