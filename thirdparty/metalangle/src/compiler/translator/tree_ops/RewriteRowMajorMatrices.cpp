//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteRowMajorMatrices: Rewrite row-major matrices as column-major.
//

#include "compiler/translator/tree_ops/RewriteRowMajorMatrices.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"

namespace sh
{
namespace
{
// Only structs with matrices are tracked.  If layout(row_major) is applied to a struct that doesn't
// have matrices, it's silently dropped.  This is also used to avoid creating duplicates for inner
// structs that don't have matrices.
struct StructConversionData
{
    // The converted struct with every matrix transposed.
    TStructure *convertedStruct = nullptr;

    // The copy-from and copy-to functions copying from a struct to its converted version and back.
    TFunction *copyFromOriginal = nullptr;
    TFunction *copyToOriginal   = nullptr;
};

bool DoesFieldContainRowMajorMatrix(const TField *field, bool isBlockRowMajor)
{
    TLayoutMatrixPacking matrixPacking = field->type()->getLayoutQualifier().matrixPacking;

    // The field is row major if either explicitly specified as such, or if it inherits it from the
    // block layout qualifier.
    if (matrixPacking == EmpColumnMajor || (matrixPacking == EmpUnspecified && !isBlockRowMajor))
    {
        return false;
    }

    // The field is qualified with row_major, but if it's not a matrix or a struct containing
    // matrices, that's a useless qualifier.
    const TType *type = field->type();
    return type->isMatrix() || type->isStructureContainingMatrices();
}

TField *DuplicateField(const TField *field)
{
    return new TField(new TType(*field->type()), field->name(), field->line(), field->symbolType());
}

void SetColumnMajor(TType *type)
{
    TLayoutQualifier layoutQualifier = type->getLayoutQualifier();
    layoutQualifier.matrixPacking    = EmpColumnMajor;
    type->setLayoutQualifier(layoutQualifier);
}

TType *TransposeMatrixType(const TType *type)
{
    TType *newType = new TType(*type);

    SetColumnMajor(newType);

    newType->setPrimarySize(static_cast<unsigned char>(type->getRows()));
    newType->setSecondarySize(static_cast<unsigned char>(type->getCols()));

    return newType;
}

void CopyArraySizes(const TType *from, TType *to)
{
    if (from->isArray())
    {
        to->makeArrays(*from->getArraySizes());
    }
}

// Determine if the node is an index node (array index or struct field selection).  For the purposes
// of this transformation, swizzle nodes are considered index nodes too.
bool IsIndexNode(TIntermNode *node, TIntermNode *child)
{
    if (node->getAsSwizzleNode())
    {
        return true;
    }

    TIntermBinary *binaryNode = node->getAsBinaryNode();
    if (binaryNode == nullptr || child != binaryNode->getLeft())
    {
        return false;
    }

    TOperator op = binaryNode->getOp();

    return op == EOpIndexDirect || op == EOpIndexDirectInterfaceBlock ||
           op == EOpIndexDirectStruct || op == EOpIndexIndirect;
}

TIntermSymbol *CopyToTempVariable(TSymbolTable *symbolTable,
                                  TIntermTyped *node,
                                  TIntermSequence *prependStatements)
{
    TVariable *temp              = CreateTempVariable(symbolTable, &node->getType());
    TIntermDeclaration *tempDecl = CreateTempInitDeclarationNode(temp, node);
    prependStatements->push_back(tempDecl);

    return new TIntermSymbol(temp);
}

TIntermAggregate *CreateStructCopyCall(const TFunction *copyFunc, TIntermTyped *expression)
{
    return TIntermAggregate::CreateFunctionCall(*copyFunc, new TIntermSequence({expression}));
}

TIntermTyped *CreateTransposeCall(TSymbolTable *symbolTable, TIntermTyped *expression)
{
    return CreateBuiltInFunctionCallNode("transpose", new TIntermSequence({expression}),
                                         *symbolTable, 300);
}

TOperator GetIndex(TSymbolTable *symbolTable,
                   TIntermNode *node,
                   TIntermSequence *indices,
                   TIntermSequence *prependStatements)
{
    // Swizzle nodes are converted EOpIndexDirect for simplicity, with one index per swizzle
    // channel.
    TIntermSwizzle *asSwizzle = node->getAsSwizzleNode();
    if (asSwizzle)
    {
        for (int channel : asSwizzle->getSwizzleOffsets())
        {
            indices->push_back(CreateIndexNode(channel));
        }
        return EOpIndexDirect;
    }

    TIntermBinary *binaryNode = node->getAsBinaryNode();
    ASSERT(binaryNode);

    TOperator op = binaryNode->getOp();
    ASSERT(op == EOpIndexDirect || op == EOpIndexDirectInterfaceBlock ||
           op == EOpIndexDirectStruct || op == EOpIndexIndirect);

    TIntermTyped *rhs = binaryNode->getRight()->deepCopy();
    if (rhs->getAsConstantUnion() == nullptr)
    {
        rhs = CopyToTempVariable(symbolTable, rhs, prependStatements);
    }

    indices->push_back(rhs);
    return op;
}

TIntermTyped *ReplicateIndexNode(TSymbolTable *symbolTable,
                                 TIntermNode *node,
                                 TIntermTyped *lhs,
                                 TIntermSequence *indices)
{
    TIntermSwizzle *asSwizzle = node->getAsSwizzleNode();
    if (asSwizzle)
    {
        return new TIntermSwizzle(lhs, asSwizzle->getSwizzleOffsets());
    }

    TIntermBinary *binaryNode = node->getAsBinaryNode();
    ASSERT(binaryNode);

    ASSERT(indices->size() == 1);
    TIntermTyped *rhs = indices->front()->getAsTyped();

    return new TIntermBinary(binaryNode->getOp(), lhs, rhs);
}

TOperator GetIndexOp(TIntermNode *node)
{
    return node->getAsConstantUnion() ? EOpIndexDirect : EOpIndexIndirect;
}

bool IsConvertedField(TIntermTyped *indexNode,
                      const std::unordered_map<const TField *, bool> &convertedFields)
{
    TIntermBinary *asBinary = indexNode->getAsBinaryNode();
    if (asBinary == nullptr)
    {
        return false;
    }

    if (asBinary->getOp() != EOpIndexDirectInterfaceBlock)
    {
        return false;
    }

    const TInterfaceBlock *interfaceBlock = asBinary->getLeft()->getType().getInterfaceBlock();
    ASSERT(interfaceBlock);

    TIntermConstantUnion *fieldIndexNode = asBinary->getRight()->getAsConstantUnion();
    ASSERT(fieldIndexNode);
    ASSERT(fieldIndexNode->getConstantValue() != nullptr);

    int fieldIndex      = fieldIndexNode->getConstantValue()->getIConst();
    const TField *field = interfaceBlock->fields()[fieldIndex];

    return convertedFields.count(field) > 0 && convertedFields.at(field);
}

// A helper class to transform expressions of array type.  Iterates over every element of the
// array.
class TransformArrayHelper
{
  public:
    TransformArrayHelper(TIntermTyped *baseExpression)
        : mBaseExpression(baseExpression),
          mBaseExpressionType(baseExpression->getType()),
          mArrayIndices(mBaseExpressionType.getArraySizes()->size(), 0)
    {}

    TIntermTyped *getNextElement(TIntermTyped *valueExpression, TIntermTyped **valueElementOut)
    {
        const TVector<unsigned int> *arraySizes = mBaseExpressionType.getArraySizes();

        // If the last index overflows, element enumeration is done.
        if (mArrayIndices.back() >= arraySizes->back())
        {
            return nullptr;
        }

        TIntermTyped *element = getCurrentElement(mBaseExpression);
        if (valueExpression)
        {
            *valueElementOut = getCurrentElement(valueExpression);
        }

        incrementIndices(arraySizes);
        return element;
    }

    void accumulateForRead(TSymbolTable *symbolTable,
                           TIntermTyped *transformedElement,
                           TIntermSequence *prependStatements)
    {
        TIntermTyped *temp = CopyToTempVariable(symbolTable, transformedElement, prependStatements);
        mReadTransformConstructorArgs.push_back(temp);
    }

    TIntermTyped *constructReadTransformExpression()
    {
        const TVector<unsigned int> &arraySizes = *mBaseExpressionType.getArraySizes();
        TIntermTyped *firstElement = mReadTransformConstructorArgs.front()->getAsTyped();
        const TType &baseType      = firstElement->getType();

        // If N dimensions, acc[0] == size[0] and acc[i] == size[i] * acc[i-1].
        // The last value is unused, and is not present.
        TVector<unsigned int> accumulatedArraySizes(arraySizes.size() - 1);

        accumulatedArraySizes[0] = arraySizes[0];
        for (size_t index = 1; index + 1 < arraySizes.size(); ++index)
        {
            accumulatedArraySizes[index] = accumulatedArraySizes[index - 1] * arraySizes[index];
        }

        return constructReadTransformExpressionHelper(arraySizes, accumulatedArraySizes, baseType,
                                                      0);
    }

  private:
    TIntermTyped *getCurrentElement(TIntermTyped *expression)
    {
        TIntermTyped *element = expression->deepCopy();
        for (auto it = mArrayIndices.rbegin(); it != mArrayIndices.rend(); ++it)
        {
            unsigned int index = *it;
            element            = new TIntermBinary(EOpIndexDirect, element, CreateIndexNode(index));
        }
        return element;
    }

    void incrementIndices(const TVector<unsigned int> *arraySizes)
    {
        // Assume mArrayIndices is an N digit number, where digit i is in the range
        // [0, arraySizes[i]).  This function increments this number.  Last digit is the most
        // significant digit.
        for (size_t digitIndex = 0; digitIndex < arraySizes->size(); ++digitIndex)
        {
            ++mArrayIndices[digitIndex];
            if (mArrayIndices[digitIndex] < (*arraySizes)[digitIndex])
            {
                break;
            }
            if (digitIndex + 1 != arraySizes->size())
            {
                // This digit has now overflown and is reset to 0, carry will be added to the next
                // digit.  The most significant digit will keep the overflow though, to make it
                // clear we have exhausted the range.
                mArrayIndices[digitIndex] = 0;
            }
        }
    }

    TIntermTyped *constructReadTransformExpressionHelper(
        const TVector<unsigned int> arraySizes,
        const TVector<unsigned int> accumulatedArraySizes,
        const TType &baseType,
        size_t elementsOffset)
    {
        ASSERT(!arraySizes.empty());

        TType *transformedType = new TType(baseType);
        transformedType->makeArrays(arraySizes);

        // If one dimensional, create the constructor with the given elements.
        if (arraySizes.size() == 1)
        {
            ASSERT(accumulatedArraySizes.size() == 0);

            auto sliceStart = mReadTransformConstructorArgs.begin() + elementsOffset;
            TIntermSequence slice(sliceStart, sliceStart + arraySizes[0]);

            return TIntermAggregate::CreateConstructor(*transformedType, &slice);
        }

        // If not, create constructors for every column recursively.
        TVector<unsigned int> subArraySizes(arraySizes.begin(), arraySizes.end() - 1);
        TVector<unsigned int> subArrayAccumulatedSizes(accumulatedArraySizes.begin(),
                                                       accumulatedArraySizes.end() - 1);

        TIntermSequence constructorArgs;
        unsigned int colStride = accumulatedArraySizes.back();
        for (size_t col = 0; col < arraySizes.back(); ++col)
        {
            size_t colElementsOffset = elementsOffset + col * colStride;

            constructorArgs.push_back(constructReadTransformExpressionHelper(
                subArraySizes, subArrayAccumulatedSizes, baseType, colElementsOffset));
        }

        return TIntermAggregate::CreateConstructor(*transformedType, &constructorArgs);
    }

    TIntermTyped *mBaseExpression;
    const TType &mBaseExpressionType;
    TVector<unsigned int> mArrayIndices;

    TIntermSequence mReadTransformConstructorArgs;
};

// Traverser that:
//
// 1. Converts |layout(row_major) matCxR M| to |layout(column_major) matRxC Mt|.
// 2. Converts |layout(row_major) S s| to |layout(column_major) St st|, where S is a struct that
//    contains matrices, and St is a new struct with the transformation in 1 applied to matrix
//    members (recursively).
// 3. When read from, the following transformations are applied:
//
//            M       -> transpose(Mt)
//            M[c]    -> gvecN(Mt[0][c], Mt[1][c], ..., Mt[N-1][c])
//            M[c][r] -> Mt[r][c]
//            M[c].yz -> gvec2(Mt[1][c], Mt[2][c])
//            MArr    -> MType[D1]..[DN](transpose(MtArr[0]...[0]), ...)
//            s       -> copy_St_to_S(st)
//            sArr    -> SType[D1]...[DN](copy_St_to_S(stArr[0]..[0]), ...)
//            (matrix reads through struct are transformed similarly to M)
//
// 4. When written to, the following transformations are applied:
//
//      M = exp       -> Mt = transpose(exp)
//      M[c] = exp    -> temp = exp
//                       Mt[0][c] = temp[0]
//                       Mt[1][c] = temp[1]
//                       ...
//                       Mt[N-1][c] = temp[N-1]
//      M[c][r] = exp -> Mt[r][c] = exp
//      M[c].yz = exp -> temp = exp
//                       Mt[1][c] = temp[0]
//                       Mt[2][c] = temp[1]
//      MArr = exp    -> temp = exp
//                       Mt = MtType[D1]..[DN](temp([0]...[0]), ...)
//      s = exp       -> st = copy_S_to_St(exp)
//      sArr = exp    -> temp = exp
//                       St = StType[D1]...[DN](copy_S_to_St(temp[0]..[0]), ...)
//      (matrix writes through struct are transformed similarly to M)
//
// 5. If any of the above is passed to an `inout` parameter, both transformations are applied:
//
//            f(M[c]) -> temp = gvecN(Mt[0][c], Mt[1][c], ..., Mt[N-1][c])
//                       f(temp)
//                       Mt[0][c] = temp[0]
//                       Mt[1][c] = temp[1]
//                       ...
//                       Mt[N-1][c] = temp[N-1]
//
//               f(s) -> temp = copy_St_to_S(st)
//                       f(temp)
//                       st = copy_S_to_St(temp)
//
//    If passed to an `out` parameter, the `temp` parameter is simply not initialized.
//
// 6. If the expression leading to the matrix or struct has array subscripts, temp values are
//    created for them to avoid duplicating side effects.
//
class RewriteRowMajorMatricesTraverser : public TIntermTraverser
{
  public:
    RewriteRowMajorMatricesTraverser(TCompiler *compiler, TSymbolTable *symbolTable)
        : TIntermTraverser(true, true, true, symbolTable),
          mCompiler(compiler),
          mStructMapOut(&mOuterPass.structMap),
          mInterfaceBlockMapIn(mOuterPass.interfaceBlockMap),
          mInterfaceBlockFieldConvertedIn(mOuterPass.interfaceBlockFieldConverted),
          mCopyFunctionDefinitionsOut(&mOuterPass.copyFunctionDefinitions),
          mOuterTraverser(nullptr),
          mInnerPassRoot(nullptr),
          mIsProcessingInnerPassSubtree(false)
    {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        // No need to process declarations in inner passes.
        if (mInnerPassRoot != nullptr)
        {
            return true;
        }

        if (visit != PreVisit)
        {
            return true;
        }

        const TIntermSequence &sequence = *(node->getSequence());

        TIntermTyped *variable = sequence.front()->getAsTyped();
        const TType &type      = variable->getType();

        // If it's a struct declaration that has matrices, remember it.  If a row-major instance
        // of it is created, it will have to be converted.
        if (type.isStructSpecifier() && type.isStructureContainingMatrices())
        {
            const TStructure *structure = type.getStruct();
            ASSERT(structure);

            ASSERT(mOuterPass.structMap.count(structure) == 0);

            StructConversionData structData;
            mOuterPass.structMap[structure] = structData;

            return false;
        }

        // If it's an interface block, it may have to be converted if it contains any row-major
        // fields.
        if (type.isInterfaceBlock() && type.getInterfaceBlock()->containsMatrices())
        {
            const TInterfaceBlock *block = type.getInterfaceBlock();
            ASSERT(block);
            bool isBlockRowMajor = type.getLayoutQualifier().matrixPacking == EmpRowMajor;

            const TFieldList &fields = block->fields();
            bool anyRowMajor         = isBlockRowMajor;

            for (const TField *field : fields)
            {
                if (DoesFieldContainRowMajorMatrix(field, isBlockRowMajor))
                {
                    anyRowMajor = true;
                    break;
                }
            }

            if (anyRowMajor)
            {
                convertInterfaceBlock(node);
            }

            return false;
        }

        return true;
    }

    void visitSymbol(TIntermSymbol *symbol) override
    {
        // If in inner pass, only process if the symbol is under that root.
        if (mInnerPassRoot != nullptr && !mIsProcessingInnerPassSubtree)
        {
            return;
        }

        const TVariable *symbolVariable = &symbol->variable();

        // If the symbol doesn't need to be replaced, there's nothing to do.
        if (mInterfaceBlockMapIn.count(symbolVariable) == 0)
        {
            return;
        }

        transformExpression(symbol);
    }

    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        if (node == mInnerPassRoot)
        {
            // We only want to process the right-hand side of an assignment in inner passes.  When
            // visit is InVisit, the left-hand side is already processed, and the right-hand side is
            // next.  Set a flag to mark this duration.
            mIsProcessingInnerPassSubtree = visit == InVisit;
        }

        return true;
    }

    TIntermSequence *getStructCopyFunctions() { return &mOuterPass.copyFunctionDefinitions; }

  private:
    typedef std::unordered_map<const TStructure *, StructConversionData> StructMap;
    typedef std::unordered_map<const TVariable *, TVariable *> InterfaceBlockMap;
    typedef std::unordered_map<const TField *, bool> InterfaceBlockFieldConverted;

    RewriteRowMajorMatricesTraverser(
        TSymbolTable *symbolTable,
        RewriteRowMajorMatricesTraverser *outerTraverser,
        const InterfaceBlockMap &interfaceBlockMap,
        const InterfaceBlockFieldConverted &interfaceBlockFieldConverted,
        StructMap *structMap,
        TIntermSequence *copyFunctionDefinitions,
        TIntermBinary *innerPassRoot)
        : TIntermTraverser(true, true, true, symbolTable),
          mStructMapOut(structMap),
          mInterfaceBlockMapIn(interfaceBlockMap),
          mInterfaceBlockFieldConvertedIn(interfaceBlockFieldConverted),
          mCopyFunctionDefinitionsOut(copyFunctionDefinitions),
          mOuterTraverser(outerTraverser),
          mInnerPassRoot(innerPassRoot),
          mIsProcessingInnerPassSubtree(false)
    {}

    void convertInterfaceBlock(TIntermDeclaration *node)
    {
        ASSERT(mInnerPassRoot == nullptr);

        const TIntermSequence &sequence = *(node->getSequence());

        TIntermTyped *variableNode   = sequence.front()->getAsTyped();
        const TType &type            = variableNode->getType();
        const TInterfaceBlock *block = type.getInterfaceBlock();
        ASSERT(block);

        bool isBlockRowMajor = type.getLayoutQualifier().matrixPacking == EmpRowMajor;

        // Recreate the struct with its row-major fields converted to column-major equivalents.
        TIntermSequence newDeclarations;

        TFieldList *newFields = new TFieldList;
        for (const TField *field : block->fields())
        {
            TField *newField = nullptr;

            if (DoesFieldContainRowMajorMatrix(field, isBlockRowMajor))
            {
                newField = convertField(field, &newDeclarations);

                // Remember that this field was converted.
                mOuterPass.interfaceBlockFieldConverted[field] = true;
            }
            else
            {
                newField = DuplicateField(field);
            }

            newFields->push_back(newField);
        }

        // Create a new interface block with these fields.
        TLayoutQualifier blockLayoutQualifier = type.getLayoutQualifier();
        blockLayoutQualifier.matrixPacking    = EmpColumnMajor;

        TInterfaceBlock *newInterfaceBlock =
            new TInterfaceBlock(mSymbolTable, block->name(), newFields, blockLayoutQualifier,
                                block->symbolType(), block->extension());

        // Create a new declaration with the new type.  Declarations are separated at this point,
        // so there should be only one variable here.
        ASSERT(sequence.size() == 1);

        TType *newInterfaceBlockType =
            new TType(newInterfaceBlock, type.getQualifier(), blockLayoutQualifier);

        TIntermDeclaration *newDeclaration = new TIntermDeclaration;
        const TVariable *variable          = &variableNode->getAsSymbolNode()->variable();

        const TType *newType = newInterfaceBlockType;
        if (type.isArray())
        {
            TType *newArrayType = new TType(*newType);
            CopyArraySizes(&type, newArrayType);
            newType = newArrayType;
        }

        // If the interface block variable itself is temp, use an empty name.
        bool variableIsTemp = variable->symbolType() == SymbolType::Empty;
        const ImmutableString &variableName =
            variableIsTemp ? kEmptyImmutableString : variable->name();

        TVariable *newVariable = new TVariable(mSymbolTable, variableName, newType,
                                               variable->symbolType(), variable->extension());

        newDeclaration->appendDeclarator(new TIntermSymbol(newVariable));

        mOuterPass.interfaceBlockMap[variable] = newVariable;

        newDeclarations.push_back(newDeclaration);

        // Replace the interface block definition with the new one, prepending any new struct
        // definitions.
        mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), node, newDeclarations);
    }

    void convertStruct(const TStructure *structure, TIntermSequence *newDeclarations)
    {
        ASSERT(mInnerPassRoot == nullptr);

        ASSERT(mOuterPass.structMap.count(structure) != 0);
        StructConversionData *structData = &mOuterPass.structMap[structure];

        if (structData->convertedStruct)
        {
            return;
        }

        TFieldList *newFields = new TFieldList;
        for (const TField *field : structure->fields())
        {
            newFields->push_back(convertField(field, newDeclarations));
        }

        // Create unique names for the converted structs.  We can't leave them nameless and have
        // a name autogenerated similar to temp variables, as nameless structs exist.  A fake
        // variable is created for the sole purpose of generating a temp name.
        TVariable *newStructTypeName =
            new TVariable(mSymbolTable, kEmptyImmutableString, StaticType::GetBasic<EbtUInt>(),
                          SymbolType::Empty);

        TStructure *newStruct = new TStructure(mSymbolTable, newStructTypeName->name(), newFields,
                                               SymbolType::AngleInternal);
        TType *newType        = new TType(newStruct, true);
        TVariable *newStructVar =
            new TVariable(mSymbolTable, kEmptyImmutableString, newType, SymbolType::Empty);

        TIntermDeclaration *structDecl = new TIntermDeclaration;
        structDecl->appendDeclarator(new TIntermSymbol(newStructVar));

        newDeclarations->push_back(structDecl);

        structData->convertedStruct = newStruct;
    }

    TField *convertField(const TField *field, TIntermSequence *newDeclarations)
    {
        ASSERT(mInnerPassRoot == nullptr);

        TField *newField = nullptr;

        const TType *fieldType = field->type();
        TType *newType         = nullptr;

        if (fieldType->isStructureContainingMatrices())
        {
            // If the field is a struct instance, convert the struct and replace the field
            // with an instance of the new struct.
            const TStructure *fieldTypeStruct = fieldType->getStruct();
            convertStruct(fieldTypeStruct, newDeclarations);

            StructConversionData &structData = mOuterPass.structMap[fieldTypeStruct];
            newType                          = new TType(structData.convertedStruct, false);
            SetColumnMajor(newType);
            CopyArraySizes(fieldType, newType);
        }
        else if (fieldType->isMatrix())
        {
            // If the field is a matrix, transpose the matrix and replace the field with
            // that, removing the matrix packing qualifier.
            newType = TransposeMatrixType(fieldType);
        }

        if (newType)
        {
            newField = new TField(newType, field->name(), field->line(), field->symbolType());
        }
        else
        {
            newField = DuplicateField(field);
        }

        return newField;
    }

    void determineAccess(TIntermNode *expression,
                         TIntermNode *accessor,
                         bool *isReadOut,
                         bool *isWriteOut)
    {
        // If passing to a function, look at whether the parameter is in, out or inout.
        TIntermAggregate *functionCall = accessor->getAsAggregate();

        if (functionCall)
        {
            TIntermSequence *arguments = functionCall->getSequence();
            for (size_t argIndex = 0; argIndex < arguments->size(); ++argIndex)
            {
                if ((*arguments)[argIndex] == expression)
                {
                    TQualifier qualifier = EvqIn;

                    // If the aggregate is not a function call, it's a constructor, and so every
                    // argument is an input.
                    const TFunction *function = functionCall->getFunction();
                    if (function)
                    {
                        const TVariable *param = function->getParam(argIndex);
                        qualifier              = param->getType().getQualifier();
                    }

                    *isReadOut  = qualifier != EvqOut;
                    *isWriteOut = qualifier == EvqOut || qualifier == EvqInOut;
                    break;
                }
            }
            return;
        }

        TIntermBinary *assignment = accessor->getAsBinaryNode();
        if (assignment && IsAssignment(assignment->getOp()))
        {
            // If expression is on the right of assignment, it's being read from.
            *isReadOut = assignment->getRight() == expression;
            // If it's on the left of assignment, it's being written to.
            *isWriteOut = assignment->getLeft() == expression;
            return;
        }

        // Any other usage is a read.
        *isReadOut  = true;
        *isWriteOut = false;
    }

    void transformExpression(TIntermSymbol *symbol)
    {
        // Walk up the parent chain while the nodes are EOpIndex* (whether array indexing or struct
        // field selection) or swizzle and construct the replacement expression.  This traversal can
        // lead to one of the following possibilities:
        //
        // - a.b[N].etc.s (struct, or struct array): copy function should be declared and used,
        // - a.b[N].etc.M (matrix or matrix array): transpose() should be used,
        // - a.b[N].etc.M[c] (a column): each element in column needs to be handled separately,
        // - a.b[N].etc.M[c].yz (multiple elements): similar to whole column, but a subset of
        //   elements,
        // - a.b[N].etc.M[c][r] (an element): single element to handle.
        // - a.b[N].etc.x (not struct or matrix): not modified
        //
        // primaryIndex will contain c, if any.  secondaryIndices will contain {0, ..., R-1}
        // (if no [r] or swizzle), {r} (if [r]), or {1, 2} (corresponding to .yz) if any.
        //
        // In all cases, the base symbol is replaced.  |baseExpression| will contain everything up
        // to (and not including) the last index/swizzle operations, i.e. a.b[N].etc.s/M/x.  Any
        // non constant array subscript is assigned to a temp variable to avoid duplicating side
        // effects.
        //
        // ---
        //
        // NOTE that due to the use of insertStatementsInParentBlock, cases like this will be
        // mistranslated, and this bug is likely present in most transformations that use this
        // feature:
        //
        //     if (x == 1 && a.b[x = 2].etc.M = value)
        //
        // which will translate to:
        //
        //     temp = (x = 2)
        //     if (x == 1 && a.b[temp].etc.M = transpose(value))
        //
        // See http://anglebug.com/3829.
        //
        TIntermTyped *baseExpression =
            new TIntermSymbol(mInterfaceBlockMapIn.at(&symbol->variable()));
        const TStructure *structure = nullptr;

        TIntermNode *primaryIndex = nullptr;
        TIntermSequence secondaryIndices;

        // In some cases, it is necessary to prepend or append statements.  Those are captured in
        // |prependStatements| and |appendStatements|.
        TIntermSequence prependStatements;
        TIntermSequence appendStatements;

        // If the expression is neither a struct or matrix, no modification is necessary.
        // If it's a struct that doesn't have matrices, again there's no transformation necessary.
        // If it's an interface block matrix field that didn't need to be transposed, no
        // transpformation is necessary.
        //
        // In all these cases, |baseExpression| contains all of the original expression.
        bool requiresTransformation = false;

        uint32_t accessorIndex         = 0;
        TIntermTyped *previousAncestor = symbol;
        while (IsIndexNode(getAncestorNode(accessorIndex), previousAncestor))
        {
            TIntermTyped *ancestor = getAncestorNode(accessorIndex)->getAsTyped();
            ASSERT(ancestor);

            const TType &previousAncestorType = previousAncestor->getType();

            TIntermSequence indices;
            TOperator op = GetIndex(mSymbolTable, ancestor, &indices, &prependStatements);

            bool opIsIndex     = op == EOpIndexDirect || op == EOpIndexIndirect;
            bool isArrayIndex  = opIsIndex && previousAncestorType.isArray();
            bool isMatrixIndex = opIsIndex && previousAncestorType.isMatrix();

            // If it's a direct index in a matrix, it's the primary index.
            bool isMatrixPrimarySubscript = isMatrixIndex && !isArrayIndex;
            ASSERT(!isMatrixPrimarySubscript ||
                   (primaryIndex == nullptr && secondaryIndices.empty()));
            // If primary index is seen and the ancestor is still an index, it must be a direct
            // index as the secondary one.  Note that if primaryIndex is set, there can only ever be
            // one more parent of interest, and that's subscripting the second dimension.
            bool isMatrixSecondarySubscript = primaryIndex != nullptr;
            ASSERT(!isMatrixSecondarySubscript || (opIsIndex && !isArrayIndex));

            if (requiresTransformation && isMatrixPrimarySubscript)
            {
                ASSERT(indices.size() == 1);
                primaryIndex = indices.front();

                // Default the secondary indices to include every row.  If there's a secondary
                // subscript provided, it will override this.
                int rows = previousAncestorType.getRows();
                for (int r = 0; r < rows; ++r)
                {
                    secondaryIndices.push_back(CreateIndexNode(r));
                }
            }
            else if (isMatrixSecondarySubscript)
            {
                ASSERT(requiresTransformation);

                secondaryIndices = indices;

                // Indices after this point are not interesting.  There can't actually be any other
                // index nodes other than desktop GLSL's swizzles on scalars, like M[1][2].yyy.
                ++accessorIndex;
                break;
            }
            else
            {
                // Replicate the expression otherwise.
                baseExpression =
                    ReplicateIndexNode(mSymbolTable, ancestor, baseExpression, &indices);

                const TType &ancestorType = ancestor->getType();
                structure                 = ancestorType.getStruct();

                requiresTransformation =
                    requiresTransformation ||
                    IsConvertedField(ancestor, mInterfaceBlockFieldConvertedIn);

                // If we reach a point where the expression is neither a matrix-containing struct
                // nor a matrix, there's no transformation required.  This can happen if we decend
                // through a struct marked with row-major but arrive at a member that doesn't
                // include a matrix.
                if (!ancestorType.isMatrix() && !ancestorType.isStructureContainingMatrices())
                {
                    requiresTransformation = false;
                }
            }

            previousAncestor = ancestor;
            ++accessorIndex;
        }

        TIntermNode *originalExpression =
            accessorIndex == 0 ? symbol : getAncestorNode(accessorIndex - 1);
        TIntermNode *accessor = getAncestorNode(accessorIndex);

        if (!requiresTransformation)
        {
            ASSERT(primaryIndex == nullptr);
            queueReplacementWithParent(accessor, originalExpression, baseExpression,
                                       OriginalNode::IS_DROPPED);

            RewriteRowMajorMatricesTraverser *traverser = mOuterTraverser ? mOuterTraverser : this;
            traverser->insertStatementsInParentBlock(prependStatements, appendStatements);
            return;
        }

        ASSERT(structure == nullptr || primaryIndex == nullptr);
        ASSERT(structure != nullptr || baseExpression->getType().isMatrix());

        // At the end, we can determine if the expression is being read from or written to (or both,
        // if sent as an inout parameter to a function).  For the sake of the transformation, the
        // left-hand side of operations like += can be treated as "written to", without necessarily
        // "read from".
        bool isRead  = false;
        bool isWrite = false;

        determineAccess(originalExpression, accessor, &isRead, &isWrite);

        ASSERT(isRead || isWrite);

        TIntermTyped *readExpression = nullptr;
        if (isRead)
        {
            readExpression = transformReadExpression(
                baseExpression, primaryIndex, &secondaryIndices, structure, &prependStatements);

            // If both read from and written to (i.e. passed to inout parameter), store the
            // expression in a temp variable and pass that to the function.
            if (isWrite)
            {
                readExpression =
                    CopyToTempVariable(mSymbolTable, readExpression, &prependStatements);
            }

            // Replace the original expression with the transformed one.  Read transformations
            // always generate a single expression that can be used in place of the original (as
            // oppposed to write transformations that can generate multiple statements).
            queueReplacementWithParent(accessor, originalExpression, readExpression,
                                       OriginalNode::IS_DROPPED);
        }

        TIntermSequence postTransformPrependStatements;
        TIntermSequence *writeStatements = &appendStatements;
        TOperator assignmentOperator     = EOpAssign;

        if (isWrite)
        {
            TIntermTyped *valueExpression = readExpression;

            if (!valueExpression)
            {
                // If there's already a read expression, this was an inout parameter and
                // |valueExpression| will contain the temp variable that was passed to the function
                // instead.
                //
                // If not, then the modification is either through being passed as an out parameter
                // to a function, or an assignment.  In the former case, create a temp variable to
                // be passed to the function.  In the latter case, create a temp variable that holds
                // the right hand side expression.
                //
                // In either case, use that temp value as the value to assign to |baseExpression|.

                TVariable *temp =
                    CreateTempVariable(mSymbolTable, &originalExpression->getAsTyped()->getType());
                TIntermDeclaration *tempDecl = nullptr;

                valueExpression = new TIntermSymbol(temp);

                TIntermBinary *assignment = accessor->getAsBinaryNode();
                if (assignment)
                {
                    assignmentOperator = assignment->getOp();
                    ASSERT(IsAssignment(assignmentOperator));

                    // We are converting the assignment to the left-hand side of an expression in
                    // the form M=exp.  A subexpression of exp itself could require a
                    // transformation.  This complicates things as there would be two replacements:
                    //
                    // - Replace M=exp with temp (because the return value of the assignment could
                    //   be used)
                    // - Replace exp with exp2, where parent is M=exp
                    //
                    // The second replacement however is ineffective as the whole of M=exp is
                    // already transformed.  What's worse, M=exp is transformed without taking exp's
                    // transformations into account.  To address this issue, this same traverser is
                    // called on the right-hand side expression, with a special flag such that it
                    // only processes that expression.
                    //
                    RewriteRowMajorMatricesTraverser *outerTraverser =
                        mOuterTraverser ? mOuterTraverser : this;
                    RewriteRowMajorMatricesTraverser rhsTraverser(
                        mSymbolTable, outerTraverser, mInterfaceBlockMapIn,
                        mInterfaceBlockFieldConvertedIn, mStructMapOut, mCopyFunctionDefinitionsOut,
                        assignment);
                    getRootNode()->traverse(&rhsTraverser);
                    bool valid = rhsTraverser.updateTree(mCompiler, getRootNode());
                    ASSERT(valid);

                    tempDecl = CreateTempInitDeclarationNode(temp, assignment->getRight());

                    // Replace the whole assignment expression with the right-hand side as a read
                    // expression, in case the result of the assignment is used.  For example, this
                    // transforms:
                    //
                    //     if ((M += exp) == X)
                    //     {
                    //         // use M
                    //     }
                    //
                    // to:
                    //
                    //     temp = exp;
                    //     M += transform(temp);
                    //     if (transform(M) == X)
                    //     {
                    //         // use M
                    //     }
                    //
                    // Note that in this case the assignment to M must be prepended in the parent
                    // block.  In contrast, when sent to a function, the assignment to M should be
                    // done after the current function call is done.
                    //
                    // If the read from M itself (to replace assigmnet) needs to generate extra
                    // statements, they should be appended after the statements that write to M.
                    // These statements are stored in postTransformPrependStatements and appended to
                    // prependStatements in the end.
                    //
                    writeStatements = &prependStatements;

                    TIntermTyped *assignmentResultExpression = transformReadExpression(
                        baseExpression->deepCopy(), primaryIndex, &secondaryIndices, structure,
                        &postTransformPrependStatements);

                    // Replace the whole assignment, instead of just the right hand side.
                    TIntermNode *accessorParent = getAncestorNode(accessorIndex + 1);
                    queueReplacementWithParent(accessorParent, accessor, assignmentResultExpression,
                                               OriginalNode::IS_DROPPED);
                }
                else
                {
                    tempDecl = CreateTempDeclarationNode(temp);

                    // Replace the write expression (a function call argument) with the temp
                    // variable.
                    queueReplacementWithParent(accessor, originalExpression, valueExpression,
                                               OriginalNode::IS_DROPPED);
                }
                prependStatements.push_back(tempDecl);
            }

            if (isRead)
            {
                baseExpression = baseExpression->deepCopy();
            }
            transformWriteExpression(baseExpression, primaryIndex, &secondaryIndices, structure,
                                     valueExpression, assignmentOperator, writeStatements);
        }

        prependStatements.insert(prependStatements.end(), postTransformPrependStatements.begin(),
                                 postTransformPrependStatements.end());

        RewriteRowMajorMatricesTraverser *traverser = mOuterTraverser ? mOuterTraverser : this;
        traverser->insertStatementsInParentBlock(prependStatements, appendStatements);
    }

    TIntermTyped *transformReadExpression(TIntermTyped *baseExpression,
                                          TIntermNode *primaryIndex,
                                          TIntermSequence *secondaryIndices,
                                          const TStructure *structure,
                                          TIntermSequence *prependStatements)
    {
        const TType &baseExpressionType = baseExpression->getType();

        if (structure)
        {
            ASSERT(primaryIndex == nullptr && secondaryIndices->empty());
            ASSERT(mStructMapOut->count(structure) != 0);
            ASSERT((*mStructMapOut)[structure].convertedStruct != nullptr);

            // Declare copy-from-converted-to-original-struct function (if not already).
            declareStructCopyToOriginal(structure);

            const TFunction *copyToOriginal = (*mStructMapOut)[structure].copyToOriginal;

            if (baseExpressionType.isArray())
            {
                // If base expression is an array, transform every element.
                TransformArrayHelper transformHelper(baseExpression);

                TIntermTyped *element = nullptr;
                while ((element = transformHelper.getNextElement(nullptr, nullptr)) != nullptr)
                {
                    TIntermTyped *transformedElement =
                        CreateStructCopyCall(copyToOriginal, element);
                    transformHelper.accumulateForRead(mSymbolTable, transformedElement,
                                                      prependStatements);
                }
                return transformHelper.constructReadTransformExpression();
            }
            else
            {
                // If not reading an array, the result is simply a call to this function with the
                // base expression.
                return CreateStructCopyCall(copyToOriginal, baseExpression);
            }
        }

        // If not indexed, the result is transpose(exp)
        if (primaryIndex == nullptr)
        {
            ASSERT(secondaryIndices->empty());

            if (baseExpressionType.isArray())
            {
                // If array, transpose every element.
                TransformArrayHelper transformHelper(baseExpression);

                TIntermTyped *element = nullptr;
                while ((element = transformHelper.getNextElement(nullptr, nullptr)) != nullptr)
                {
                    TIntermTyped *transformedElement = CreateTransposeCall(mSymbolTable, element);
                    transformHelper.accumulateForRead(mSymbolTable, transformedElement,
                                                      prependStatements);
                }
                return transformHelper.constructReadTransformExpression();
            }
            else
            {
                return CreateTransposeCall(mSymbolTable, baseExpression);
            }
        }

        // If indexed the result is a vector (or just one element) where the primary and secondary
        // indices are swapped.
        ASSERT(!secondaryIndices->empty());

        TOperator primaryIndexOp          = GetIndexOp(primaryIndex);
        TIntermTyped *primaryIndexAsTyped = primaryIndex->getAsTyped();

        TIntermSequence transposedColumn;
        for (TIntermNode *secondaryIndex : *secondaryIndices)
        {
            TOperator secondaryIndexOp          = GetIndexOp(secondaryIndex);
            TIntermTyped *secondaryIndexAsTyped = secondaryIndex->getAsTyped();

            TIntermBinary *colIndexed = new TIntermBinary(
                secondaryIndexOp, baseExpression->deepCopy(), secondaryIndexAsTyped->deepCopy());
            TIntermBinary *colRowIndexed =
                new TIntermBinary(primaryIndexOp, colIndexed, primaryIndexAsTyped->deepCopy());

            transposedColumn.push_back(colRowIndexed);
        }

        if (secondaryIndices->size() == 1)
        {
            // If only one element, return that directly.
            return transposedColumn.front()->getAsTyped();
        }

        // Otherwise create a constructor with the appropriate dimension.
        TType *vecType = new TType(baseExpressionType.getBasicType(), secondaryIndices->size());
        return TIntermAggregate::CreateConstructor(*vecType, &transposedColumn);
    }

    void transformWriteExpression(TIntermTyped *baseExpression,
                                  TIntermNode *primaryIndex,
                                  TIntermSequence *secondaryIndices,
                                  const TStructure *structure,
                                  TIntermTyped *valueExpression,
                                  TOperator assignmentOperator,
                                  TIntermSequence *writeStatements)
    {
        const TType &baseExpressionType = baseExpression->getType();

        if (structure)
        {
            ASSERT(primaryIndex == nullptr && secondaryIndices->empty());
            ASSERT(mStructMapOut->count(structure) != 0);
            ASSERT((*mStructMapOut)[structure].convertedStruct != nullptr);

            // Declare copy-to-converted-from-original-struct function (if not already).
            declareStructCopyFromOriginal(structure);

            // The result is call to this function with the value expression assigned to base
            // expression.
            const TFunction *copyFromOriginal = (*mStructMapOut)[structure].copyFromOriginal;

            if (baseExpressionType.isArray())
            {
                // If array, assign every element.
                TransformArrayHelper transformHelper(baseExpression);

                TIntermTyped *element      = nullptr;
                TIntermTyped *valueElement = nullptr;
                while ((element = transformHelper.getNextElement(valueExpression, &valueElement)) !=
                       nullptr)
                {
                    TIntermTyped *functionCall =
                        CreateStructCopyCall(copyFromOriginal, valueElement);
                    writeStatements->push_back(new TIntermBinary(EOpAssign, element, functionCall));
                }
            }
            else
            {
                TIntermTyped *functionCall =
                    CreateStructCopyCall(copyFromOriginal, valueExpression->deepCopy());
                writeStatements->push_back(
                    new TIntermBinary(EOpAssign, baseExpression, functionCall));
            }

            return;
        }

        // If not indexed, the result is transpose(exp)
        if (primaryIndex == nullptr)
        {
            ASSERT(secondaryIndices->empty());

            if (baseExpressionType.isArray())
            {
                // If array, assign every element.
                TransformArrayHelper transformHelper(baseExpression);

                TIntermTyped *element      = nullptr;
                TIntermTyped *valueElement = nullptr;
                while ((element = transformHelper.getNextElement(valueExpression, &valueElement)) !=
                       nullptr)
                {
                    TIntermTyped *valueTransposed = CreateTransposeCall(mSymbolTable, valueElement);
                    writeStatements->push_back(
                        new TIntermBinary(EOpAssign, element, valueTransposed));
                }
            }
            else
            {
                TIntermTyped *valueTransposed =
                    CreateTransposeCall(mSymbolTable, valueExpression->deepCopy());
                writeStatements->push_back(
                    new TIntermBinary(assignmentOperator, baseExpression, valueTransposed));
            }

            return;
        }

        // If indexed, create one assignment per secondary index.  If the right-hand side is a
        // scalar, it's used with every assignment.  If it's a vector, the assignment is
        // per-component.  The right-hand side cannot be a matrix as that would imply left-hand
        // side being a matrix too, which is covered above where |primaryIndex == nullptr|.
        ASSERT(!secondaryIndices->empty());

        bool isValueExpressionScalar = valueExpression->getType().getNominalSize() == 1;
        ASSERT(isValueExpressionScalar || valueExpression->getType().getNominalSize() ==
                                              static_cast<int>(secondaryIndices->size()));

        TOperator primaryIndexOp          = GetIndexOp(primaryIndex);
        TIntermTyped *primaryIndexAsTyped = primaryIndex->getAsTyped();

        for (TIntermNode *secondaryIndex : *secondaryIndices)
        {
            TOperator secondaryIndexOp          = GetIndexOp(secondaryIndex);
            TIntermTyped *secondaryIndexAsTyped = secondaryIndex->getAsTyped();

            TIntermBinary *colIndexed = new TIntermBinary(
                secondaryIndexOp, baseExpression->deepCopy(), secondaryIndexAsTyped->deepCopy());
            TIntermBinary *colRowIndexed =
                new TIntermBinary(primaryIndexOp, colIndexed, primaryIndexAsTyped->deepCopy());

            TIntermTyped *valueExpressionIndexed = valueExpression->deepCopy();
            if (!isValueExpressionScalar)
            {
                valueExpressionIndexed = new TIntermBinary(secondaryIndexOp, valueExpressionIndexed,
                                                           secondaryIndexAsTyped->deepCopy());
            }

            writeStatements->push_back(
                new TIntermBinary(assignmentOperator, colRowIndexed, valueExpressionIndexed));
        }
    }

    const TFunction *getCopyStructFieldFunction(const TType *fromFieldType,
                                                const TType *toFieldType,
                                                bool isCopyToOriginal)
    {
        ASSERT(fromFieldType->getStruct());
        ASSERT(toFieldType->getStruct());

        // If copying from or to the original struct, the "to" field struct could require
        // conversion to or from the "from" field struct.  |isCopyToOriginal| tells us if we
        // should expect to find toField or fromField in mStructMapOut, if true or false
        // respectively.
        const TFunction *fieldCopyFunction = nullptr;
        if (isCopyToOriginal)
        {
            const TStructure *toFieldStruct = toFieldType->getStruct();

            auto iter = mStructMapOut->find(toFieldStruct);
            if (iter != mStructMapOut->end())
            {
                declareStructCopyToOriginal(toFieldStruct);
                fieldCopyFunction = iter->second.copyToOriginal;
            }
        }
        else
        {
            const TStructure *fromFieldStruct = fromFieldType->getStruct();

            auto iter = mStructMapOut->find(fromFieldStruct);
            if (iter != mStructMapOut->end())
            {
                declareStructCopyFromOriginal(fromFieldStruct);
                fieldCopyFunction = iter->second.copyFromOriginal;
            }
        }

        return fieldCopyFunction;
    }

    void addFieldCopy(TIntermBlock *body,
                      TIntermTyped *to,
                      TIntermTyped *from,
                      bool isCopyToOriginal)
    {
        const TType &fromType = from->getType();
        const TType &toType   = to->getType();

        TIntermTyped *rhs = from;

        if (fromType.getStruct())
        {
            const TFunction *fieldCopyFunction =
                getCopyStructFieldFunction(&fromType, &toType, isCopyToOriginal);

            if (fieldCopyFunction)
            {
                rhs = CreateStructCopyCall(fieldCopyFunction, from);
            }
        }
        else if (fromType.isMatrix())
        {
            rhs = CreateTransposeCall(mSymbolTable, from);
        }

        body->appendStatement(new TIntermBinary(EOpAssign, to, rhs));
    }

    TFunction *declareStructCopy(const TStructure *from,
                                 const TStructure *to,
                                 bool isCopyToOriginal)
    {
        TType *fromType = new TType(from, true);
        TType *toType   = new TType(to, true);

        // Create the parameter and return value variables.
        TVariable *fromVar = new TVariable(mSymbolTable, ImmutableString("from"), fromType,
                                           SymbolType::AngleInternal);
        TVariable *toVar =
            new TVariable(mSymbolTable, ImmutableString("to"), toType, SymbolType::AngleInternal);

        TIntermSymbol *fromSymbol = new TIntermSymbol(fromVar);
        TIntermSymbol *toSymbol   = new TIntermSymbol(toVar);

        // Create the function body as statements are generated.
        TIntermBlock *body = new TIntermBlock;

        // Declare the result variable.
        TIntermDeclaration *toDecl = new TIntermDeclaration();
        toDecl->appendDeclarator(toSymbol);
        body->appendStatement(toDecl);

        // Iterate over fields of the struct and copy one by one, transposing the matrices.  If a
        // struct is encountered that requires a transformation, this function is recursively
        // called.  As a result, it is important that the copy functions are placed in the code in
        // order.
        const TFieldList &fromFields = from->fields();
        const TFieldList &toFields   = to->fields();
        ASSERT(fromFields.size() == toFields.size());

        for (size_t fieldIndex = 0; fieldIndex < fromFields.size(); ++fieldIndex)
        {
            TIntermTyped *fieldIndexNode = CreateIndexNode(static_cast<int>(fieldIndex));

            TIntermTyped *fromField =
                new TIntermBinary(EOpIndexDirectStruct, fromSymbol->deepCopy(), fieldIndexNode);
            TIntermTyped *toField = new TIntermBinary(EOpIndexDirectStruct, toSymbol->deepCopy(),
                                                      fieldIndexNode->deepCopy());

            const TType *fromFieldType = fromFields[fieldIndex]->type();
            bool isStructOrMatrix      = fromFieldType->getStruct() || fromFieldType->isMatrix();

            if (fromFieldType->isArray() && isStructOrMatrix)
            {
                // If struct or matrix array, we need to copy element by element.
                TransformArrayHelper transformHelper(toField);

                TIntermTyped *toElement   = nullptr;
                TIntermTyped *fromElement = nullptr;
                while ((toElement = transformHelper.getNextElement(fromField, &fromElement)) !=
                       nullptr)
                {
                    addFieldCopy(body, toElement, fromElement, isCopyToOriginal);
                }
            }
            else
            {
                addFieldCopy(body, toField, fromField, isCopyToOriginal);
            }
        }

        // Add return statement.
        body->appendStatement(new TIntermBranch(EOpReturn, toSymbol->deepCopy()));

        // Declare the function
        TFunction *copyFunction = new TFunction(mSymbolTable, kEmptyImmutableString,
                                                SymbolType::AngleInternal, toType, true);
        copyFunction->addParameter(fromVar);

        TIntermFunctionDefinition *functionDef =
            CreateInternalFunctionDefinitionNode(*copyFunction, body);
        mCopyFunctionDefinitionsOut->push_back(functionDef);

        return copyFunction;
    }

    void declareStructCopyFromOriginal(const TStructure *structure)
    {
        StructConversionData *structData = &(*mStructMapOut)[structure];
        if (structData->copyFromOriginal)
        {
            return;
        }

        structData->copyFromOriginal =
            declareStructCopy(structure, structData->convertedStruct, false);
    }

    void declareStructCopyToOriginal(const TStructure *structure)
    {
        StructConversionData *structData = &(*mStructMapOut)[structure];
        if (structData->copyToOriginal)
        {
            return;
        }

        structData->copyToOriginal =
            declareStructCopy(structData->convertedStruct, structure, true);
    }

    TCompiler *mCompiler;

    // This traverser can call itself to transform a subexpression before moving on.  However, it
    // needs to accumulate conversion functions in inner passes.  The fields below marked with Out
    // or In are inherited from the outer pass (for inner passes), or point to storage fields in
    // mOuterPass (for the outer pass).  The latter should not be used by the inner passes as they
    // would be empty, so they are placed inside a struct to make them explicit.
    struct
    {
        StructMap structMap;
        InterfaceBlockMap interfaceBlockMap;
        InterfaceBlockFieldConverted interfaceBlockFieldConverted;
        TIntermSequence copyFunctionDefinitions;
    } mOuterPass;

    // A map from structures with matrices to their converted version.
    StructMap *mStructMapOut;
    // A map from interface block instances with row-major matrices to their converted variable.
    const InterfaceBlockMap &mInterfaceBlockMapIn;
    // A map from interface block fields to whether they need to be converted.  If a field was
    // already column-major, it shouldn't be transposed.
    const InterfaceBlockFieldConverted &mInterfaceBlockFieldConvertedIn;

    TIntermSequence *mCopyFunctionDefinitionsOut;

    // If set, it's an inner pass and this will point to the outer pass traverser.  All statement
    // insertions are stored in the outer traverser and applied at once in the end.  This prevents
    // the inner passes from adding statements which invalidates the outer traverser's statement
    // position tracking.
    RewriteRowMajorMatricesTraverser *mOuterTraverser;

    // If set, it's an inner pass that should only process the right-hand side of this particular
    // node.
    TIntermBinary *mInnerPassRoot;
    bool mIsProcessingInnerPassSubtree;
};

}  // anonymous namespace

bool RewriteRowMajorMatrices(TCompiler *compiler, TIntermBlock *root, TSymbolTable *symbolTable)
{
    RewriteRowMajorMatricesTraverser traverser(compiler, symbolTable);
    root->traverse(&traverser);
    if (!traverser.updateTree(compiler, root))
    {
        return false;
    }

    size_t firstFunctionIndex = FindFirstFunctionDefinitionIndex(root);
    root->insertChildNodes(firstFunctionIndex, *traverser.getStructCopyFunctions());

    return compiler->validateAST(root);
}
}  // namespace sh
