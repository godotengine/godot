//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

//
// Definition of the in-memory high-level intermediate representation
// of shaders.  This is a tree that parser creates.
//
// Nodes in the tree are defined as a hierarchy of classes derived from
// TIntermNode. Each is a node in a tree.  There is no preset branching factor;
// each node can have it's own type of list of children.
//

#ifndef COMPILER_TRANSLATOR_INTERMNODE_H_
#define COMPILER_TRANSLATOR_INTERMNODE_H_

#include "GLSLANG/ShaderLang.h"

#include <algorithm>
#include <queue>

#include "common/angleutils.h"
#include "compiler/translator/Common.h"
#include "compiler/translator/ConstantUnion.h"
#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Operator.h"
#include "compiler/translator/SymbolUniqueId.h"
#include "compiler/translator/Types.h"
#include "compiler/translator/tree_util/Visit.h"

namespace sh
{

class TDiagnostics;

class TIntermTraverser;
class TIntermAggregate;
class TIntermBlock;
class TIntermInvariantDeclaration;
class TIntermDeclaration;
class TIntermFunctionPrototype;
class TIntermFunctionDefinition;
class TIntermSwizzle;
class TIntermBinary;
class TIntermUnary;
class TIntermConstantUnion;
class TIntermTernary;
class TIntermIfElse;
class TIntermSwitch;
class TIntermCase;
class TIntermTyped;
class TIntermSymbol;
class TIntermLoop;
class TInfoSink;
class TInfoSinkBase;
class TIntermBranch;
class TIntermPreprocessorDirective;

class TSymbolTable;
class TFunction;
class TVariable;

//
// Base class for the tree nodes
//
class TIntermNode : angle::NonCopyable
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    TIntermNode()
    {
        // TODO: Move this to TSourceLoc constructor
        // after getting rid of TPublicType.
        mLine.first_file = mLine.last_file = 0;
        mLine.first_line = mLine.last_line = 0;
    }
    virtual ~TIntermNode() {}

    const TSourceLoc &getLine() const { return mLine; }
    void setLine(const TSourceLoc &l) { mLine = l; }

    virtual void traverse(TIntermTraverser *it);
    virtual bool visit(Visit visit, TIntermTraverser *it) = 0;

    virtual TIntermTyped *getAsTyped() { return nullptr; }
    virtual TIntermConstantUnion *getAsConstantUnion() { return nullptr; }
    virtual TIntermFunctionDefinition *getAsFunctionDefinition() { return nullptr; }
    virtual TIntermAggregate *getAsAggregate() { return nullptr; }
    virtual TIntermBlock *getAsBlock() { return nullptr; }
    virtual TIntermFunctionPrototype *getAsFunctionPrototypeNode() { return nullptr; }
    virtual TIntermInvariantDeclaration *getAsInvariantDeclarationNode() { return nullptr; }
    virtual TIntermDeclaration *getAsDeclarationNode() { return nullptr; }
    virtual TIntermSwizzle *getAsSwizzleNode() { return nullptr; }
    virtual TIntermBinary *getAsBinaryNode() { return nullptr; }
    virtual TIntermUnary *getAsUnaryNode() { return nullptr; }
    virtual TIntermTernary *getAsTernaryNode() { return nullptr; }
    virtual TIntermIfElse *getAsIfElseNode() { return nullptr; }
    virtual TIntermSwitch *getAsSwitchNode() { return nullptr; }
    virtual TIntermCase *getAsCaseNode() { return nullptr; }
    virtual TIntermSymbol *getAsSymbolNode() { return nullptr; }
    virtual TIntermLoop *getAsLoopNode() { return nullptr; }
    virtual TIntermBranch *getAsBranchNode() { return nullptr; }
    virtual TIntermPreprocessorDirective *getAsPreprocessorDirective() { return nullptr; }

    virtual TIntermNode *deepCopy() const = 0;

    virtual size_t getChildCount() const                  = 0;
    virtual TIntermNode *getChildNode(size_t index) const = 0;
    // Replace a child node. Return true if |original| is a child
    // node and it is replaced; otherwise, return false.
    virtual bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) = 0;

  protected:
    TSourceLoc mLine;
};

//
// This is just to help yacc.
//
struct TIntermNodePair
{
    TIntermNode *node1;
    TIntermNode *node2;
};

//
// Intermediate class for nodes that have a type.
//
class TIntermTyped : public TIntermNode
{
  public:
    TIntermTyped() {}

    virtual TIntermTyped *deepCopy() const override = 0;

    TIntermTyped *getAsTyped() override { return this; }

    virtual TIntermTyped *fold(TDiagnostics *diagnostics) { return this; }

    // getConstantValue() returns the constant value that this node represents, if any. It
    // should only be used after nodes have been replaced with their folded versions returned
    // from fold(). hasConstantValue() returns true if getConstantValue() will return a value.
    virtual bool hasConstantValue() const;
    virtual const TConstantUnion *getConstantValue() const;

    // True if executing the expression represented by this node affects state, like values of
    // variables. False if the executing the expression only computes its return value without
    // affecting state. May return true conservatively.
    virtual bool hasSideEffects() const = 0;

    virtual const TType &getType() const = 0;

    TBasicType getBasicType() const { return getType().getBasicType(); }
    TQualifier getQualifier() const { return getType().getQualifier(); }
    TPrecision getPrecision() const { return getType().getPrecision(); }
    TMemoryQualifier getMemoryQualifier() const { return getType().getMemoryQualifier(); }
    int getCols() const { return getType().getCols(); }
    int getRows() const { return getType().getRows(); }
    int getNominalSize() const { return getType().getNominalSize(); }
    int getSecondarySize() const { return getType().getSecondarySize(); }

    bool isInterfaceBlock() const { return getType().isInterfaceBlock(); }
    bool isMatrix() const { return getType().isMatrix(); }
    bool isArray() const { return getType().isArray(); }
    bool isVector() const { return getType().isVector(); }
    bool isScalar() const { return getType().isScalar(); }
    bool isScalarInt() const { return getType().isScalarInt(); }
    const char *getBasicString() const { return getType().getBasicString(); }

    unsigned int getOutermostArraySize() const { return getType().getOutermostArraySize(); }

  protected:
    TIntermTyped(const TIntermTyped &node);
};

//
// Handle for, do-while, and while loops.
//
enum TLoopType
{
    ELoopFor,
    ELoopWhile,
    ELoopDoWhile
};

class TIntermLoop : public TIntermNode
{
  public:
    TIntermLoop(TLoopType type,
                TIntermNode *init,
                TIntermTyped *cond,
                TIntermTyped *expr,
                TIntermBlock *body);

    TIntermLoop *getAsLoopNode() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TLoopType getType() const { return mType; }
    TIntermNode *getInit() { return mInit; }
    TIntermTyped *getCondition() { return mCond; }
    TIntermTyped *getExpression() { return mExpr; }
    TIntermBlock *getBody() { return mBody; }

    void setInit(TIntermNode *init) { mInit = init; }
    void setCondition(TIntermTyped *condition) { mCond = condition; }
    void setExpression(TIntermTyped *expression) { mExpr = expression; }
    void setBody(TIntermBlock *body) { mBody = body; }

    virtual TIntermLoop *deepCopy() const override { return new TIntermLoop(*this); }

  protected:
    TLoopType mType;
    TIntermNode *mInit;   // for-loop initialization
    TIntermTyped *mCond;  // loop exit condition
    TIntermTyped *mExpr;  // for-loop expression
    TIntermBlock *mBody;  // loop body

  private:
    TIntermLoop(const TIntermLoop &);
};

//
// Handle break, continue, return, and kill.
//
class TIntermBranch : public TIntermNode
{
  public:
    TIntermBranch(TOperator op, TIntermTyped *e) : mFlowOp(op), mExpression(e) {}

    TIntermBranch *getAsBranchNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TOperator getFlowOp() { return mFlowOp; }
    TIntermTyped *getExpression() { return mExpression; }

    virtual TIntermBranch *deepCopy() const override { return new TIntermBranch(*this); }

  protected:
    TOperator mFlowOp;
    TIntermTyped *mExpression;  // zero except for "return exp;" statements

  private:
    TIntermBranch(const TIntermBranch &);
};

// Nodes that correspond to variable symbols in the source code. These may be regular variables or
// interface block instances. In declarations that only declare a struct type but no variables, a
// TIntermSymbol node with an empty variable is used to store the type.
class TIntermSymbol : public TIntermTyped
{
  public:
    TIntermSymbol(const TVariable *variable);

    TIntermTyped *deepCopy() const override { return new TIntermSymbol(*this); }

    bool hasConstantValue() const override;
    const TConstantUnion *getConstantValue() const override;

    bool hasSideEffects() const override { return false; }

    const TType &getType() const override;

    const TSymbolUniqueId &uniqueId() const;
    ImmutableString getName() const;
    const TVariable &variable() const { return *mVariable; }

    TIntermSymbol *getAsSymbolNode() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *, TIntermNode *) override { return false; }

  private:
    TIntermSymbol(const TIntermSymbol &) = default;  // Note: not deleted, just private!

    const TVariable *const mVariable;  // Guaranteed to be non-null
};

// A typed expression that is not just representing a symbol table symbol.
class TIntermExpression : public TIntermTyped
{
  public:
    TIntermExpression(const TType &t);

    const TType &getType() const override { return mType; }

  protected:
    TType *getTypePointer() { return &mType; }
    void setType(const TType &t) { mType = t; }
    void setTypePreservePrecision(const TType &t);

    TIntermExpression(const TIntermExpression &node) = default;

    TType mType;
};

// Constant folded node.
// Note that nodes may be constant folded and not be constant expressions with the EvqConst
// qualifier. This happens for example when the following expression is processed:
// "true ? 1.0 : non_constant"
// Other nodes than TIntermConstantUnion may also be constant expressions.
//
class TIntermConstantUnion : public TIntermExpression
{
  public:
    TIntermConstantUnion(const TConstantUnion *unionPointer, const TType &type)
        : TIntermExpression(type), mUnionArrayPointer(unionPointer)
    {
        ASSERT(unionPointer);
    }

    TIntermTyped *deepCopy() const override { return new TIntermConstantUnion(*this); }

    bool hasConstantValue() const override;
    const TConstantUnion *getConstantValue() const override;

    bool hasSideEffects() const override { return false; }

    int getIConst(size_t index) const
    {
        return mUnionArrayPointer ? mUnionArrayPointer[index].getIConst() : 0;
    }
    unsigned int getUConst(size_t index) const
    {
        return mUnionArrayPointer ? mUnionArrayPointer[index].getUConst() : 0;
    }
    float getFConst(size_t index) const
    {
        return mUnionArrayPointer ? mUnionArrayPointer[index].getFConst() : 0.0f;
    }
    bool getBConst(size_t index) const
    {
        return mUnionArrayPointer ? mUnionArrayPointer[index].getBConst() : false;
    }
    bool isZero(size_t index) const
    {
        return mUnionArrayPointer ? mUnionArrayPointer[index].isZero() : false;
    }

    TIntermConstantUnion *getAsConstantUnion() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *, TIntermNode *) override { return false; }

    TConstantUnion *foldUnaryNonComponentWise(TOperator op);
    TConstantUnion *foldUnaryComponentWise(TOperator op, TDiagnostics *diagnostics);

    static const TConstantUnion *FoldBinary(TOperator op,
                                            const TConstantUnion *leftArray,
                                            const TType &leftType,
                                            const TConstantUnion *rightArray,
                                            const TType &rightType,
                                            TDiagnostics *diagnostics,
                                            const TSourceLoc &line);

    static const TConstantUnion *FoldIndexing(const TType &type,
                                              const TConstantUnion *constArray,
                                              int index);
    static TConstantUnion *FoldAggregateBuiltIn(TIntermAggregate *aggregate,
                                                TDiagnostics *diagnostics);
    static bool IsFloatDivision(TBasicType t1, TBasicType t2);

  protected:
    // Same data may be shared between multiple constant unions, so it can't be modified.
    const TConstantUnion *mUnionArrayPointer;

  private:
    typedef float (*FloatTypeUnaryFunc)(float);
    void foldFloatTypeUnary(const TConstantUnion &parameter,
                            FloatTypeUnaryFunc builtinFunc,
                            TConstantUnion *result) const;

    TIntermConstantUnion(const TIntermConstantUnion &node);  // Note: not deleted, just private!
};

//
// Intermediate class for node types that hold operators.
//
class TIntermOperator : public TIntermExpression
{
  public:
    TOperator getOp() const { return mOp; }

    bool isAssignment() const;
    bool isMultiplication() const;
    bool isConstructor() const;

    // Returns true for calls mapped to EOpCall*, false for built-ins that have their own specific
    // ops.
    bool isFunctionCall() const;

    bool hasSideEffects() const override { return isAssignment(); }

  protected:
    TIntermOperator(TOperator op) : TIntermExpression(TType(EbtFloat, EbpUndefined)), mOp(op) {}
    TIntermOperator(TOperator op, const TType &type) : TIntermExpression(type), mOp(op) {}

    TIntermOperator(const TIntermOperator &) = default;

    const TOperator mOp;
};

// Node for vector swizzles.
class TIntermSwizzle : public TIntermExpression
{
  public:
    // This constructor determines the type of the node based on the operand.
    TIntermSwizzle(TIntermTyped *operand, const TVector<int> &swizzleOffsets);

    TIntermTyped *deepCopy() const override { return new TIntermSwizzle(*this); }

    TIntermSwizzle *getAsSwizzleNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    bool hasSideEffects() const override { return mOperand->hasSideEffects(); }

    TIntermTyped *getOperand() { return mOperand; }
    void writeOffsetsAsXYZW(TInfoSinkBase *out) const;

    const TVector<int> &getSwizzleOffsets() { return mSwizzleOffsets; }

    bool hasDuplicateOffsets() const;
    void setHasFoldedDuplicateOffsets(bool hasFoldedDuplicateOffsets);
    bool offsetsMatch(int offset) const;

    TIntermTyped *fold(TDiagnostics *diagnostics) override;

  protected:
    TIntermTyped *mOperand;
    TVector<int> mSwizzleOffsets;
    bool mHasFoldedDuplicateOffsets;

  private:
    void promote();

    TIntermSwizzle(const TIntermSwizzle &node);  // Note: not deleted, just private!
};

//
// Nodes for all the basic binary math operators.
//
class TIntermBinary : public TIntermOperator
{
  public:
    // This constructor determines the type of the binary node based on the operands and op.
    TIntermBinary(TOperator op, TIntermTyped *left, TIntermTyped *right);
    // Comma qualifier depends on the shader version, so use this to create comma nodes:
    static TIntermBinary *CreateComma(TIntermTyped *left, TIntermTyped *right, int shaderVersion);

    TIntermTyped *deepCopy() const override { return new TIntermBinary(*this); }

    bool hasConstantValue() const override;
    const TConstantUnion *getConstantValue() const override;

    static TOperator GetMulOpBasedOnOperands(const TType &left, const TType &right);
    static TOperator GetMulAssignOpBasedOnOperands(const TType &left, const TType &right);

    TIntermBinary *getAsBinaryNode() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    bool hasSideEffects() const override
    {
        return isAssignment() || mLeft->hasSideEffects() || mRight->hasSideEffects();
    }

    TIntermTyped *getLeft() const { return mLeft; }
    TIntermTyped *getRight() const { return mRight; }
    TIntermTyped *fold(TDiagnostics *diagnostics) override;

    void setAddIndexClamp() { mAddIndexClamp = true; }
    bool getAddIndexClamp() const { return mAddIndexClamp; }

    // This method is only valid for EOpIndexDirectStruct. It returns the name of the field.
    const ImmutableString &getIndexStructFieldName() const;

  protected:
    TIntermTyped *mLeft;
    TIntermTyped *mRight;

    // If set to true, wrap any EOpIndexIndirect with a clamp to bounds.
    bool mAddIndexClamp;

  private:
    void promote();

    static TQualifier GetCommaQualifier(int shaderVersion,
                                        const TIntermTyped *left,
                                        const TIntermTyped *right);

    TIntermBinary(const TIntermBinary &node);  // Note: not deleted, just private!
};

//
// Nodes for unary math operators.
//
class TIntermUnary : public TIntermOperator
{
  public:
    TIntermUnary(TOperator op, TIntermTyped *operand, const TFunction *function);

    TIntermTyped *deepCopy() const override { return new TIntermUnary(*this); }

    TIntermUnary *getAsUnaryNode() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    bool hasSideEffects() const override { return isAssignment() || mOperand->hasSideEffects(); }

    TIntermTyped *getOperand() { return mOperand; }
    TIntermTyped *fold(TDiagnostics *diagnostics) override;

    const TFunction *getFunction() const { return mFunction; }

    void setUseEmulatedFunction() { mUseEmulatedFunction = true; }
    bool getUseEmulatedFunction() { return mUseEmulatedFunction; }

  protected:
    TIntermTyped *mOperand;

    // If set to true, replace the built-in function call with an emulated one
    // to work around driver bugs.
    bool mUseEmulatedFunction;

    const TFunction *const mFunction;

  private:
    void promote();

    TIntermUnary(const TIntermUnary &node);  // note: not deleted, just private!
};

typedef TVector<TIntermNode *> TIntermSequence;
typedef TVector<int> TQualifierList;

// Interface for node classes that have an arbitrarily sized set of children.
class TIntermAggregateBase
{
  public:
    virtual ~TIntermAggregateBase() {}

    virtual TIntermSequence *getSequence()             = 0;
    virtual const TIntermSequence *getSequence() const = 0;

    bool replaceChildNodeWithMultiple(TIntermNode *original, const TIntermSequence &replacements);
    bool insertChildNodes(TIntermSequence::size_type position, const TIntermSequence &insertions);

  protected:
    TIntermAggregateBase() {}

    bool replaceChildNodeInternal(TIntermNode *original, TIntermNode *replacement);
};

//
// Nodes that operate on an arbitrary sized set of children.
//
class TIntermAggregate : public TIntermOperator, public TIntermAggregateBase
{
  public:
    static TIntermAggregate *CreateFunctionCall(const TFunction &func, TIntermSequence *arguments);

    static TIntermAggregate *CreateRawFunctionCall(const TFunction &func,
                                                   TIntermSequence *arguments);

    // This covers all built-in function calls - whether they are associated with an op or not.
    static TIntermAggregate *CreateBuiltInFunctionCall(const TFunction &func,
                                                       TIntermSequence *arguments);
    static TIntermAggregate *CreateConstructor(const TType &type, TIntermSequence *arguments);
    ~TIntermAggregate() {}

    // Note: only supported for nodes that can be a part of an expression.
    TIntermTyped *deepCopy() const override { return new TIntermAggregate(*this); }

    TIntermAggregate *shallowCopy() const;

    bool hasConstantValue() const override;
    const TConstantUnion *getConstantValue() const override;

    TIntermAggregate *getAsAggregate() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    bool hasSideEffects() const override;

    TIntermTyped *fold(TDiagnostics *diagnostics) override;

    TIntermSequence *getSequence() override { return &mArguments; }
    const TIntermSequence *getSequence() const override { return &mArguments; }

    void setUseEmulatedFunction() { mUseEmulatedFunction = true; }
    bool getUseEmulatedFunction() { return mUseEmulatedFunction; }

    // Returns true if changing parameter precision may affect the return value.
    bool gotPrecisionFromChildren() const { return mGotPrecisionFromChildren; }

    const TFunction *getFunction() const { return mFunction; }

    // Get the function name to display to the user in an error message.
    const char *functionName() const;

  protected:
    TIntermSequence mArguments;

    // If set to true, replace the built-in function call with an emulated one
    // to work around driver bugs. Only for calls mapped to ops other than EOpCall*.
    bool mUseEmulatedFunction;

    bool mGotPrecisionFromChildren;

    const TFunction *const mFunction;

  private:
    TIntermAggregate(const TFunction *func,
                     const TType &type,
                     TOperator op,
                     TIntermSequence *arguments);

    TIntermAggregate(const TIntermAggregate &node);  // note: not deleted, just private!

    void setPrecisionAndQualifier();

    bool areChildrenConstQualified();

    void setPrecisionFromChildren();

    void setPrecisionForBuiltInOp();

    // Returns true if precision was set according to special rules for this built-in.
    bool setPrecisionForSpecialBuiltInOp();

    // Used for built-in functions under EOpCallBuiltInFunction. The function name in the symbol
    // info needs to be set before calling this.
    void setBuiltInFunctionPrecision();
};

// A list of statements. Either the root node which contains declarations and function definitions,
// or a block that can be marked with curly braces {}.
class TIntermBlock : public TIntermNode, public TIntermAggregateBase
{
  public:
    TIntermBlock() : TIntermNode() {}
    ~TIntermBlock() {}

    TIntermBlock *getAsBlock() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    // Only intended for initially building the block.
    void appendStatement(TIntermNode *statement);
    void insertStatement(size_t insertPosition, TIntermNode *statement);

    TIntermSequence *getSequence() override { return &mStatements; }
    const TIntermSequence *getSequence() const override { return &mStatements; }

    TIntermBlock *deepCopy() const override { return new TIntermBlock(*this); }

  protected:
    TIntermSequence mStatements;

  private:
    TIntermBlock(const TIntermBlock &);
};

// Function prototype. May be in the AST either as a function prototype declaration or as a part of
// a function definition. The type of the node is the function return type.
class TIntermFunctionPrototype : public TIntermTyped
{
  public:
    TIntermFunctionPrototype(const TFunction *function);
    ~TIntermFunctionPrototype() {}

    TIntermFunctionPrototype *getAsFunctionPrototypeNode() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    const TType &getType() const override;

    TIntermTyped *deepCopy() const override
    {
        UNREACHABLE();
        return nullptr;
    }
    bool hasSideEffects() const override
    {
        UNREACHABLE();
        return true;
    }

    const TFunction *getFunction() const { return mFunction; }

  protected:
    const TFunction *const mFunction;
};

// Node for function definitions. The prototype child node stores the function header including
// parameters, and the body child node stores the function body.
class TIntermFunctionDefinition : public TIntermNode
{
  public:
    TIntermFunctionDefinition(TIntermFunctionPrototype *prototype, TIntermBlock *body)
        : TIntermNode(), mPrototype(prototype), mBody(body)
    {
        ASSERT(prototype != nullptr);
        ASSERT(body != nullptr);
    }

    TIntermFunctionDefinition *getAsFunctionDefinition() override { return this; }
    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TIntermFunctionPrototype *getFunctionPrototype() const { return mPrototype; }
    TIntermBlock *getBody() const { return mBody; }

    const TFunction *getFunction() const { return mPrototype->getFunction(); }

    TIntermNode *deepCopy() const override
    {
        UNREACHABLE();
        return nullptr;
    }

  private:
    TIntermFunctionPrototype *mPrototype;
    TIntermBlock *mBody;
};

// Struct, interface block or variable declaration. Can contain multiple variable declarators.
class TIntermDeclaration : public TIntermNode, public TIntermAggregateBase
{
  public:
    TIntermDeclaration() : TIntermNode() {}
    ~TIntermDeclaration() {}

    TIntermDeclaration *getAsDeclarationNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    // Only intended for initially building the declaration.
    // The declarator node should be either TIntermSymbol or TIntermBinary with op set to
    // EOpInitialize.
    void appendDeclarator(TIntermTyped *declarator);

    TIntermSequence *getSequence() override { return &mDeclarators; }
    const TIntermSequence *getSequence() const override { return &mDeclarators; }

    TIntermNode *deepCopy() const override
    {
        UNREACHABLE();
        return nullptr;
    }

  protected:
    TIntermSequence mDeclarators;
};

// Specialized declarations for attributing invariance.
class TIntermInvariantDeclaration : public TIntermNode
{
  public:
    TIntermInvariantDeclaration(TIntermSymbol *symbol, const TSourceLoc &line);

    virtual TIntermInvariantDeclaration *getAsInvariantDeclarationNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    TIntermSymbol *getSymbol() { return mSymbol; }

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TIntermInvariantDeclaration *deepCopy() const override
    {
        return new TIntermInvariantDeclaration(*this);
    }

  private:
    TIntermSymbol *mSymbol;

    TIntermInvariantDeclaration(const TIntermInvariantDeclaration &);
};

// For ternary operators like a ? b : c.
class TIntermTernary : public TIntermExpression
{
  public:
    TIntermTernary(TIntermTyped *cond, TIntermTyped *trueExpression, TIntermTyped *falseExpression);

    TIntermTernary *getAsTernaryNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TIntermTyped *getCondition() const { return mCondition; }
    TIntermTyped *getTrueExpression() const { return mTrueExpression; }
    TIntermTyped *getFalseExpression() const { return mFalseExpression; }

    TIntermTyped *deepCopy() const override { return new TIntermTernary(*this); }

    bool hasSideEffects() const override
    {
        return mCondition->hasSideEffects() || mTrueExpression->hasSideEffects() ||
               mFalseExpression->hasSideEffects();
    }

    TIntermTyped *fold(TDiagnostics *diagnostics) override;

  private:
    TIntermTernary(const TIntermTernary &node);  // Note: not deleted, just private!

    static TQualifier DetermineQualifier(TIntermTyped *cond,
                                         TIntermTyped *trueExpression,
                                         TIntermTyped *falseExpression);

    TIntermTyped *mCondition;
    TIntermTyped *mTrueExpression;
    TIntermTyped *mFalseExpression;
};

class TIntermIfElse : public TIntermNode
{
  public:
    TIntermIfElse(TIntermTyped *cond, TIntermBlock *trueB, TIntermBlock *falseB);

    TIntermIfElse *getAsIfElseNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TIntermTyped *getCondition() const { return mCondition; }
    TIntermBlock *getTrueBlock() const { return mTrueBlock; }
    TIntermBlock *getFalseBlock() const { return mFalseBlock; }

    TIntermIfElse *deepCopy() const override { return new TIntermIfElse(*this); }

  protected:
    TIntermTyped *mCondition;
    TIntermBlock *mTrueBlock;
    TIntermBlock *mFalseBlock;

  private:
    TIntermIfElse(const TIntermIfElse &);
};

//
// Switch statement.
//
class TIntermSwitch : public TIntermNode
{
  public:
    TIntermSwitch(TIntermTyped *init, TIntermBlock *statementList);

    TIntermSwitch *getAsSwitchNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    TIntermTyped *getInit() { return mInit; }
    TIntermBlock *getStatementList() { return mStatementList; }

    // Must be called with a non-null statementList.
    void setStatementList(TIntermBlock *statementList);

    TIntermSwitch *deepCopy() const override { return new TIntermSwitch(*this); }

  protected:
    TIntermTyped *mInit;
    TIntermBlock *mStatementList;

  private:
    TIntermSwitch(const TIntermSwitch &);
};

//
// Case label.
//
class TIntermCase : public TIntermNode
{
  public:
    TIntermCase(TIntermTyped *condition) : TIntermNode(), mCondition(condition) {}

    TIntermCase *getAsCaseNode() override { return this; }
    bool visit(Visit visit, TIntermTraverser *it) final;

    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;
    bool replaceChildNode(TIntermNode *original, TIntermNode *replacement) override;

    bool hasCondition() const { return mCondition != nullptr; }
    TIntermTyped *getCondition() const { return mCondition; }

    TIntermCase *deepCopy() const override { return new TIntermCase(*this); }

  protected:
    TIntermTyped *mCondition;

  private:
    TIntermCase(const TIntermCase &);
};

//
// Preprocessor Directive.
//  #ifdef, #define, #if, #endif, etc.
//

enum class PreprocessorDirective
{
    Define,
    Ifdef,
    If,
    Endif,
};

class TIntermPreprocessorDirective final : public TIntermNode
{
  public:
    // This could also take an ImmutableString as an argument.
    TIntermPreprocessorDirective(PreprocessorDirective directive, ImmutableString command);
    ~TIntermPreprocessorDirective() final;

    void traverse(TIntermTraverser *it) final;
    bool visit(Visit visit, TIntermTraverser *it) final;
    bool replaceChildNode(TIntermNode *, TIntermNode *) final { return false; }

    TIntermPreprocessorDirective *getAsPreprocessorDirective() final { return this; }
    size_t getChildCount() const final;
    TIntermNode *getChildNode(size_t index) const final;

    PreprocessorDirective getDirective() const { return mDirective; }
    const ImmutableString &getCommand() const { return mCommand; }

    TIntermPreprocessorDirective *deepCopy() const override
    {
        return new TIntermPreprocessorDirective(*this);
    }

  private:
    PreprocessorDirective mDirective;
    ImmutableString mCommand;

    TIntermPreprocessorDirective(const TIntermPreprocessorDirective &);
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_INTERMNODE_H_
