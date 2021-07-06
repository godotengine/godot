//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteAtomicCounters: Emulate atomic counter buffers with storage buffers.
//

#include "compiler/translator/tree_ops/RewriteAtomicCounters.h"

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
constexpr ImmutableString kAtomicCounterTypeName  = ImmutableString("ANGLE_atomic_uint");
constexpr ImmutableString kAtomicCounterBlockName = ImmutableString("ANGLEAtomicCounters");
constexpr ImmutableString kAtomicCounterVarName   = ImmutableString("atomicCounters");
constexpr ImmutableString kAtomicCounterFieldName = ImmutableString("counters");

// DeclareAtomicCountersBuffer adds a storage buffer array that's used with atomic counters.
const TVariable *DeclareAtomicCountersBuffers(TIntermBlock *root, TSymbolTable *symbolTable)
{
    // Define `uint counters[];` as the only field in the interface block.
    TFieldList *fieldList = new TFieldList;
    TType *counterType    = new TType(EbtUInt);
    counterType->makeArray(0);

    TField *countersField =
        new TField(counterType, kAtomicCounterFieldName, TSourceLoc(), SymbolType::AngleInternal);

    fieldList->push_back(countersField);

    TMemoryQualifier coherentMemory = TMemoryQualifier::Create();
    coherentMemory.coherent         = true;

    // There are a maximum of 8 atomic counter buffers per IMPLEMENTATION_MAX_ATOMIC_COUNTER_BUFFERS
    // in libANGLE/Constants.h.
    constexpr uint32_t kMaxAtomicCounterBuffers = 8;

    // Define a storage block "ANGLEAtomicCounters" with instance name "atomicCounters".
    return DeclareInterfaceBlock(root, symbolTable, fieldList, EvqBuffer, coherentMemory,
                                 kMaxAtomicCounterBuffers, kAtomicCounterBlockName,
                                 kAtomicCounterVarName);
}

TIntermConstantUnion *CreateUIntConstant(uint32_t value)
{
    TType *constantType = new TType(*StaticType::GetBasic<EbtUInt, 1>());
    constantType->setQualifier(EvqConst);

    TConstantUnion *constantValue = new TConstantUnion;
    constantValue->setUConst(value);
    return new TIntermConstantUnion(constantValue, *constantType);
}

TIntermTyped *CreateAtomicCounterConstant(TType *atomicCounterType,
                                          uint32_t binding,
                                          uint32_t offset)
{
    ASSERT(atomicCounterType->getBasicType() == EbtStruct);

    TIntermSequence *arguments = new TIntermSequence();
    arguments->push_back(CreateUIntConstant(binding));
    arguments->push_back(CreateUIntConstant(offset));

    return TIntermAggregate::CreateConstructor(*atomicCounterType, arguments);
}

TIntermBinary *CreateAtomicCounterRef(const TVariable *atomicCounters,
                                      const TIntermTyped *bindingOffset,
                                      const TIntermTyped *bufferOffsets)
{
    // The atomic counters storage buffer declaration looks as such:
    //
    // layout(...) buffer ANGLEAtomicCounters
    // {
    //     uint counters[];
    // } atomicCounters[N];
    //
    // Where N is large enough to accommodate atomic counter buffer bindings used in the shader.
    //
    // Given an ANGLEAtomicCounter variable (which is a struct of {binding, offset}), we need to
    // return:
    //
    // atomicCounters[binding].counters[offset]
    //
    // The offset itself is the provided one plus an offset given through uniforms.

    TIntermSymbol *atomicCountersRef = new TIntermSymbol(atomicCounters);

    TIntermConstantUnion *bindingFieldRef  = CreateIndexNode(0);
    TIntermConstantUnion *offsetFieldRef   = CreateIndexNode(1);
    TIntermConstantUnion *countersFieldRef = CreateIndexNode(0);

    // Create references to bindingOffset.binding and bindingOffset.offset.
    TIntermBinary *binding =
        new TIntermBinary(EOpIndexDirectStruct, bindingOffset->deepCopy(), bindingFieldRef);
    TIntermBinary *offset =
        new TIntermBinary(EOpIndexDirectStruct, bindingOffset->deepCopy(), offsetFieldRef);

    // Create reference to atomicCounters[bindingOffset.binding]
    TIntermBinary *countersBlock = new TIntermBinary(EOpIndexDirect, atomicCountersRef, binding);

    // Create reference to atomicCounters[bindingOffset.binding].counters
    TIntermBinary *counters =
        new TIntermBinary(EOpIndexDirectInterfaceBlock, countersBlock, countersFieldRef);

    // Create bufferOffsets[binding / 4].  Each uint in bufferOffsets contains offsets for 4
    // bindings.
    TIntermBinary *bindingDivFour =
        new TIntermBinary(EOpDiv, binding->deepCopy(), CreateUIntConstant(4));
    TIntermBinary *bufferOffsetUint =
        new TIntermBinary(EOpIndexDirect, bufferOffsets->deepCopy(), bindingDivFour);

    // Create (binding % 4) * 8
    TIntermBinary *bindingModFour =
        new TIntermBinary(EOpIMod, binding->deepCopy(), CreateUIntConstant(4));
    TIntermBinary *bufferOffsetShift =
        new TIntermBinary(EOpMul, bindingModFour, CreateUIntConstant(8));

    // Create bufferOffsets[binding / 4] >> ((binding % 4) * 8) & 0xFF
    TIntermBinary *bufferOffsetShifted =
        new TIntermBinary(EOpBitShiftRight, bufferOffsetUint, bufferOffsetShift);
    TIntermBinary *bufferOffset =
        new TIntermBinary(EOpBitwiseAnd, bufferOffsetShifted, CreateUIntConstant(0xFF));

    // return atomicCounters[bindingOffset.binding].counters[bindingOffset.offset + bufferOffset]
    offset = new TIntermBinary(EOpAdd, offset, bufferOffset);
    return new TIntermBinary(EOpIndexDirect, counters, offset);
}

// Traverser that:
//
// 1. Converts the |atomic_uint| types to |{uint,uint}| for binding and offset.
// 2. Substitutes the |uniform atomic_uint| declarations with a global declaration that holds the
//    binding and offset.
// 3. Substitutes |atomicVar[n]| with |buffer[binding].counters[offset + n]|.
class RewriteAtomicCountersTraverser : public TIntermTraverser
{
  public:
    RewriteAtomicCountersTraverser(TSymbolTable *symbolTable,
                                   const TVariable *atomicCounters,
                                   const TIntermTyped *acbBufferOffsets)
        : TIntermTraverser(true, true, true, symbolTable),
          mAtomicCounters(atomicCounters),
          mAcbBufferOffsets(acbBufferOffsets),
          mAtomicCounterType(nullptr),
          mAtomicCounterTypeConst(nullptr),
          mAtomicCounterTypeDeclaration(nullptr)
    {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        if (visit != PreVisit)
        {
            return true;
        }

        const TIntermSequence &sequence = *(node->getSequence());

        TIntermTyped *variable = sequence.front()->getAsTyped();
        const TType &type      = variable->getType();
        bool isAtomicCounter   = type.getQualifier() == EvqUniform && type.isAtomicCounter();

        if (isAtomicCounter)
        {
            // Atomic counters cannot have initializers, so the declaration must necessarily be a
            // symbol.
            TIntermSymbol *samplerVariable = variable->getAsSymbolNode();
            ASSERT(samplerVariable != nullptr);

            declareAtomicCounter(&samplerVariable->variable(), node);
            return false;
        }

        return true;
    }

    void visitFunctionPrototype(TIntermFunctionPrototype *node) override
    {
        const TFunction *function = node->getFunction();
        // Go over the parameters and replace the atomic arguments with a uint type.
        mRetyper.visitFunctionPrototype();
        for (size_t paramIndex = 0; paramIndex < function->getParamCount(); ++paramIndex)
        {
            const TVariable *param = function->getParam(paramIndex);
            TVariable *replacement = convertFunctionParameter(node, param);
            if (replacement)
            {
                mRetyper.replaceFunctionParam(param, replacement);
            }
        }

        TIntermFunctionPrototype *replacementPrototype =
            mRetyper.convertFunctionPrototype(mSymbolTable, function);
        if (replacementPrototype)
        {
            queueReplacement(replacementPrototype, OriginalNode::IS_DROPPED);
        }
    }

    bool visitAggregate(Visit visit, TIntermAggregate *node) override
    {
        if (visit == PreVisit)
        {
            mRetyper.preVisitAggregate();
        }

        if (visit != PostVisit)
        {
            return true;
        }

        if (node->getOp() == EOpCallBuiltInFunction)
        {
            convertBuiltinFunction(node);
        }
        else if (node->getOp() == EOpCallFunctionInAST)
        {
            TIntermAggregate *substituteCall = mRetyper.convertASTFunction(node);
            if (substituteCall)
            {
                queueReplacement(substituteCall, OriginalNode::IS_DROPPED);
            }
        }
        mRetyper.postVisitAggregate();

        return true;
    }

    void visitSymbol(TIntermSymbol *symbol) override
    {
        const TVariable *symbolVariable = &symbol->variable();

        if (!symbol->getType().isAtomicCounter())
        {
            return;
        }

        // The symbol is either referencing a global atomic counter, or is a function parameter.  In
        // either case, it could be an array.  The are the following possibilities:
        //
        //     layout(..) uniform atomic_uint ac;
        //     layout(..) uniform atomic_uint acArray[N];
        //
        //     void func(inout atomic_uint c)
        //     {
        //         otherFunc(c);
        //     }
        //
        //     void funcArray(inout atomic_uint cArray[N])
        //     {
        //         otherFuncArray(cArray);
        //         otherFunc(cArray[n]);
        //     }
        //
        //     void funcGlobal()
        //     {
        //         func(ac);
        //         func(acArray[n]);
        //         funcArray(acArray);
        //         atomicIncrement(ac);
        //         atomicIncrement(acArray[n]);
        //     }
        //
        // This should translate to:
        //
        //     buffer ANGLEAtomicCounters
        //     {
        //         uint counters[];
        //     } atomicCounters;
        //
        //     struct ANGLEAtomicCounter
        //     {
        //         uint binding;
        //         uint offset;
        //     };
        //     const ANGLEAtomicCounter ac = {<binding>, <offset>};
        //     const ANGLEAtomicCounter acArray = {<binding>, <offset>};
        //
        //     void func(inout ANGLEAtomicCounter c)
        //     {
        //         otherFunc(c);
        //     }
        //
        //     void funcArray(inout uint cArray)
        //     {
        //         otherFuncArray(cArray);
        //         otherFunc({cArray.binding, cArray.offset + n});
        //     }
        //
        //     void funcGlobal()
        //     {
        //         func(ac);
        //         func(acArray+n);
        //         funcArray(acArray);
        //         atomicAdd(atomicCounters[ac.binding]counters[ac.offset]);
        //         atomicAdd(atomicCounters[ac.binding]counters[ac.offset+n]);
        //     }
        //
        // In all cases, the argument transformation is stored in mRetyper.  In the function call's
        // PostVisit, if it's a builtin, the look up in |atomicCounters.counters| is done as well as
        // the builtin function change.  Otherwise, the transformed argument is passed on as is.
        //

        TIntermTyped *bindingOffset =
            new TIntermSymbol(mRetyper.getVariableReplacement(symbolVariable));
        ASSERT(bindingOffset != nullptr);

        TIntermNode *argument = convertFunctionArgument(symbol, &bindingOffset);

        if (mRetyper.isInAggregate())
        {
            mRetyper.replaceFunctionCallArg(argument, bindingOffset);
        }
        else
        {
            // If there's a stray ac[i] lying around, just delete it.  This can happen if the shader
            // uses ac[i].length(), which in RemoveArrayLengthMethod() will result in an ineffective
            // statement that's just ac[i]; (similarly for a stray ac;, it doesn't have to be
            // subscripted).  Note that the subscript could have side effects, but the
            // convertFunctionArgument above has already generated code that includes the subscript
            // (and therefore its side-effect).
            TIntermBlock *block = nullptr;
            for (uint32_t ancestorIndex = 0; block == nullptr; ++ancestorIndex)
            {
                block = getAncestorNode(ancestorIndex)->getAsBlock();
            }

            TIntermSequence emptySequence;
            mMultiReplacements.emplace_back(block, argument, emptySequence);
        }
    }

    TIntermDeclaration *getAtomicCounterTypeDeclaration() { return mAtomicCounterTypeDeclaration; }

  private:
    void declareAtomicCounter(const TVariable *atomicCounterVar, TIntermDeclaration *node)
    {
        // Create a global variable that contains the binding and offset of this atomic counter
        // declaration.
        if (mAtomicCounterType == nullptr)
        {
            declareAtomicCounterType();
        }
        ASSERT(mAtomicCounterTypeConst);

        TVariable *bindingOffset = new TVariable(mSymbolTable, atomicCounterVar->name(),
                                                 mAtomicCounterTypeConst, SymbolType::UserDefined);

        const TType &atomicCounterType = atomicCounterVar->getType();
        uint32_t offset                = atomicCounterType.getLayoutQualifier().offset;
        uint32_t binding               = atomicCounterType.getLayoutQualifier().binding;

        ASSERT(offset % 4 == 0);
        TIntermTyped *bindingOffsetInitValue =
            CreateAtomicCounterConstant(mAtomicCounterTypeConst, binding, offset / 4);

        TIntermSymbol *bindingOffsetSymbol = new TIntermSymbol(bindingOffset);
        TIntermBinary *bindingOffsetInit =
            new TIntermBinary(EOpInitialize, bindingOffsetSymbol, bindingOffsetInitValue);

        TIntermDeclaration *bindingOffsetDeclaration = new TIntermDeclaration();
        bindingOffsetDeclaration->appendDeclarator(bindingOffsetInit);

        // Replace the atomic_uint declaration with the binding/offset declaration.
        TIntermSequence replacement;
        replacement.push_back(bindingOffsetDeclaration);
        mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), node, replacement);

        // Remember the binding/offset variable.
        mRetyper.replaceGlobalVariable(atomicCounterVar, bindingOffset);
    }

    void declareAtomicCounterType()
    {
        ASSERT(mAtomicCounterType == nullptr);

        TFieldList *fields = new TFieldList();
        fields->push_back(new TField(new TType(EbtUInt, EbpUndefined, EvqGlobal, 1, 1),
                                     ImmutableString("binding"), TSourceLoc(),
                                     SymbolType::AngleInternal));
        fields->push_back(new TField(new TType(EbtUInt, EbpUndefined, EvqGlobal, 1, 1),
                                     ImmutableString("arrayIndex"), TSourceLoc(),
                                     SymbolType::AngleInternal));
        TStructure *atomicCounterTypeStruct =
            new TStructure(mSymbolTable, kAtomicCounterTypeName, fields, SymbolType::AngleInternal);
        mAtomicCounterType = new TType(atomicCounterTypeStruct, false);

        mAtomicCounterTypeDeclaration = new TIntermDeclaration;
        TVariable *emptyVariable      = new TVariable(mSymbolTable, kEmptyImmutableString,
                                                 mAtomicCounterType, SymbolType::Empty);
        mAtomicCounterTypeDeclaration->appendDeclarator(new TIntermSymbol(emptyVariable));

        // Keep a const variant around as well.
        mAtomicCounterTypeConst = new TType(*mAtomicCounterType);
        mAtomicCounterTypeConst->setQualifier(EvqConst);
    }

    TVariable *convertFunctionParameter(TIntermNode *parent, const TVariable *param)
    {
        if (!param->getType().isAtomicCounter())
        {
            return nullptr;
        }
        if (mAtomicCounterType == nullptr)
        {
            declareAtomicCounterType();
        }

        const TType *paramType = &param->getType();
        TType *newType =
            paramType->getQualifier() == EvqConst ? mAtomicCounterTypeConst : mAtomicCounterType;

        TVariable *replacementVar =
            new TVariable(mSymbolTable, param->name(), newType, SymbolType::UserDefined);

        return replacementVar;
    }

    TIntermTyped *convertFunctionArgumentHelper(
        const TVector<unsigned int> &runningArraySizeProducts,
        TIntermTyped *flattenedSubscript,
        uint32_t depth,
        uint32_t *subscriptCountOut)
    {
        std::string prefix(depth, ' ');
        TIntermNode *parent = getAncestorNode(depth);
        ASSERT(parent);

        TIntermBinary *arrayExpression = parent->getAsBinaryNode();
        if (!arrayExpression)
        {
            // If the parent is not an array subscript operation, we have reached the end of the
            // subscript chain.  Note the depth that's traversed so the corresponding node can be
            // taken as the function argument.
            *subscriptCountOut = depth;
            return flattenedSubscript;
        }

        ASSERT(arrayExpression->getOp() == EOpIndexDirect ||
               arrayExpression->getOp() == EOpIndexIndirect);

        // Assume i = n - depth.  Get Pi.  See comment in convertFunctionArgument.
        ASSERT(depth < runningArraySizeProducts.size());
        uint32_t thisDimensionSize =
            runningArraySizeProducts[runningArraySizeProducts.size() - 1 - depth];

        // Get Ii.
        TIntermTyped *thisDimensionOffset = arrayExpression->getRight();

        TIntermConstantUnion *subscriptAsConstant = thisDimensionOffset->getAsConstantUnion();
        const bool subscriptIsZero = subscriptAsConstant && subscriptAsConstant->isZero(0);

        // If Ii is zero, don't need to add Ii*Pi; that's zero.
        if (!subscriptIsZero)
        {
            thisDimensionOffset = thisDimensionOffset->deepCopy();

            // If Pi is 1, don't multiply.  Just accumulate Ii.
            if (thisDimensionSize != 1)
            {
                thisDimensionOffset = new TIntermBinary(EOpMul, thisDimensionOffset,
                                                        CreateUIntConstant(thisDimensionSize));
            }

            // Accumulate with the previous running offset, if any.
            if (flattenedSubscript)
            {
                flattenedSubscript =
                    new TIntermBinary(EOpAdd, flattenedSubscript, thisDimensionOffset);
            }
            else
            {
                flattenedSubscript = thisDimensionOffset;
            }
        }

        // Note: GLSL only allows 2 nested levels of arrays, so this recursion is bounded.
        return convertFunctionArgumentHelper(runningArraySizeProducts, flattenedSubscript,
                                             depth + 1, subscriptCountOut);
    }

    TIntermNode *convertFunctionArgument(TIntermNode *symbol, TIntermTyped **bindingOffset)
    {
        // Assume a general case of array declaration with N dimensions:
        //
        //     atomic_uint ac[Dn]..[D2][D1];
        //
        // Let's define
        //
        //     Pn = D(n-1)*...*D2*D1
        //
        // In that case, we have:
        //
        //     ac[In]         = ac + In*Pn
        //     ac[In][I(n-1)] = ac + In*Pn + I(n-1)*P(n-1)
        //     ac[In]...[Ii]  = ac + In*Pn + ... + Ii*Pi
        //
        // We have just visited a symbol; ac.  Walking the parent chain, we will visit the
        // expressions in the above order (ac, ac[In], ac[In][I(n-1)], ...).  We therefore can
        // simply walk the parent chain and accumulate Ii*Pi to obtain the offset from the base of
        // ac.

        TIntermSymbol *argumentAsSymbol = symbol->getAsSymbolNode();
        ASSERT(argumentAsSymbol);

        const TVector<unsigned int> *arraySizes = argumentAsSymbol->getType().getArraySizes();

        // Calculate Pi
        TVector<unsigned int> runningArraySizeProducts;
        if (arraySizes && arraySizes->size() > 0)
        {
            runningArraySizeProducts.resize(arraySizes->size());
            uint32_t runningProduct = 1;
            for (size_t dimension = 0; dimension < arraySizes->size(); ++dimension)
            {
                runningArraySizeProducts[dimension] = runningProduct;
                runningProduct *= (*arraySizes)[dimension];
            }
        }

        // Walk the parent chain and accumulate Ii*Pi
        uint32_t subscriptCount = 0;
        TIntermTyped *flattenedSubscript =
            convertFunctionArgumentHelper(runningArraySizeProducts, nullptr, 0, &subscriptCount);

        // Find the function argument, which is either in the form of ac (i.e. there are no
        // subscripts, in which case that's the function argument), or ac[In]...[Ii] (in which case
        // the function argument is the (n-i)th ancestor of ac.
        //
        // Note that this is the case because no other operation is allowed on ac other than
        // subscript.
        TIntermNode *argument = subscriptCount == 0 ? symbol : getAncestorNode(subscriptCount - 1);
        ASSERT(argument != nullptr);

        // If not subscripted, keep the argument as-is.
        if (flattenedSubscript == nullptr)
        {
            return argument;
        }

        // Copy the atomic counter binding/offset constant and modify it by adding the array
        // subscript to its offset field.
        TVariable *modified              = CreateTempVariable(mSymbolTable, mAtomicCounterType);
        TIntermDeclaration *modifiedDecl = CreateTempInitDeclarationNode(modified, *bindingOffset);

        TIntermSymbol *modifiedSymbol    = new TIntermSymbol(modified);
        TConstantUnion *offsetFieldIndex = new TConstantUnion;
        offsetFieldIndex->setIConst(1);
        TIntermConstantUnion *offsetFieldRef =
            new TIntermConstantUnion(offsetFieldIndex, *StaticType::GetBasic<EbtUInt>());
        TIntermBinary *offsetField =
            new TIntermBinary(EOpIndexDirectStruct, modifiedSymbol, offsetFieldRef);

        TIntermBinary *modifiedOffset =
            new TIntermBinary(EOpAddAssign, offsetField, flattenedSubscript);

        TIntermSequence *modifySequence = new TIntermSequence({modifiedDecl, modifiedOffset});
        insertStatementsInParentBlock(*modifySequence);

        *bindingOffset = modifiedSymbol->deepCopy();

        return argument;
    }

    void convertBuiltinFunction(TIntermAggregate *node)
    {
        // If the function is |memoryBarrierAtomicCounter|, simply replace it with
        // |memoryBarrierBuffer|.
        if (node->getFunction()->name() == "memoryBarrierAtomicCounter")
        {
            TIntermTyped *substituteCall = CreateBuiltInFunctionCallNode(
                "memoryBarrierBuffer", new TIntermSequence, *mSymbolTable, 310);
            queueReplacement(substituteCall, OriginalNode::IS_DROPPED);
            return;
        }

        // If it's an |atomicCounter*| function, replace the function with an |atomic*| equivalent.
        if (!node->getFunction()->isAtomicCounterFunction())
        {
            return;
        }

        const ImmutableString &functionName = node->getFunction()->name();
        TIntermSequence *arguments          = node->getSequence();

        // Note: atomicAdd(0) is used for atomic reads.
        uint32_t valueChange                = 0;
        constexpr char kAtomicAddFunction[] = "atomicAdd";
        bool isDecrement                    = false;

        if (functionName == "atomicCounterIncrement")
        {
            valueChange = 1;
        }
        else if (functionName == "atomicCounterDecrement")
        {
            // uint values are required to wrap around, so 0xFFFFFFFFu is used as -1.
            valueChange = std::numeric_limits<uint32_t>::max();
            static_assert(static_cast<uint32_t>(-1) == std::numeric_limits<uint32_t>::max(),
                          "uint32_t max is not -1");

            isDecrement = true;
        }
        else
        {
            ASSERT(functionName == "atomicCounter");
        }

        const TIntermNode *param = (*arguments)[0];

        TIntermTyped *bindingOffset = mRetyper.getFunctionCallArgReplacement(param);

        TIntermSequence *substituteArguments = new TIntermSequence;
        substituteArguments->push_back(
            CreateAtomicCounterRef(mAtomicCounters, bindingOffset, mAcbBufferOffsets));
        substituteArguments->push_back(CreateUIntConstant(valueChange));

        TIntermTyped *substituteCall = CreateBuiltInFunctionCallNode(
            kAtomicAddFunction, substituteArguments, *mSymbolTable, 310);

        // Note that atomicCounterDecrement returns the *new* value instead of the prior value,
        // unlike atomicAdd.  So we need to do a -1 on the result as well.
        if (isDecrement)
        {
            substituteCall = new TIntermBinary(EOpSub, substituteCall, CreateUIntConstant(1));
        }

        queueReplacement(substituteCall, OriginalNode::IS_DROPPED);
    }

    const TVariable *mAtomicCounters;
    const TIntermTyped *mAcbBufferOffsets;

    RetypeOpaqueVariablesHelper mRetyper;

    TType *mAtomicCounterType;
    TType *mAtomicCounterTypeConst;

    // Stored to be put at the top of the shader after the pass.
    TIntermDeclaration *mAtomicCounterTypeDeclaration;
};

}  // anonymous namespace

bool RewriteAtomicCounters(TCompiler *compiler,
                           TIntermBlock *root,
                           TSymbolTable *symbolTable,
                           const TIntermTyped *acbBufferOffsets)
{
    const TVariable *atomicCounters = DeclareAtomicCountersBuffers(root, symbolTable);

    RewriteAtomicCountersTraverser traverser(symbolTable, atomicCounters, acbBufferOffsets);
    root->traverse(&traverser);
    if (!traverser.updateTree(compiler, root))
    {
        return false;
    }

    TIntermDeclaration *atomicCounterTypeDeclaration = traverser.getAtomicCounterTypeDeclaration();
    if (atomicCounterTypeDeclaration)
    {
        root->getSequence()->insert(root->getSequence()->begin(), atomicCounterTypeDeclaration);
    }

    return compiler->validateAST(root);
}
}  // namespace sh
