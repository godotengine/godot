//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteStructSamplers: Extract structs from samplers.
//

#include "compiler/translator/tree_ops/RewriteStructSamplers.h"

#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{
namespace
{
// Helper method to get the sampler extracted struct type of a parameter.
TType *GetStructSamplerParameterType(TSymbolTable *symbolTable, const TVariable &param)
{
    const TStructure *structure = param.getType().getStruct();
    const TSymbol *structSymbol = symbolTable->findUserDefined(structure->name());
    ASSERT(structSymbol && structSymbol->isStruct());
    const TStructure *structVar = static_cast<const TStructure *>(structSymbol);
    TType *structType           = new TType(structVar, false);

    if (param.getType().isArray())
    {
        structType->makeArrays(*param.getType().getArraySizes());
    }

    ASSERT(!structType->isStructureContainingSamplers());

    return structType;
}

TIntermSymbol *ReplaceTypeOfSymbolNode(TIntermSymbol *symbolNode, TSymbolTable *symbolTable)
{
    const TVariable &oldVariable = symbolNode->variable();

    TType *newType = GetStructSamplerParameterType(symbolTable, oldVariable);

    TVariable *newVariable =
        new TVariable(oldVariable.uniqueId(), oldVariable.name(), oldVariable.symbolType(),
                      oldVariable.extension(), newType);
    return new TIntermSymbol(newVariable);
}

TIntermTyped *ReplaceTypeOfTypedStructNode(TIntermTyped *argument, TSymbolTable *symbolTable)
{
    TIntermSymbol *asSymbol = argument->getAsSymbolNode();
    if (asSymbol)
    {
        ASSERT(asSymbol->getType().getStruct());
        return ReplaceTypeOfSymbolNode(asSymbol, symbolTable);
    }

    TIntermTyped *replacement = argument->deepCopy();
    TIntermBinary *binary     = replacement->getAsBinaryNode();
    ASSERT(binary);

    while (binary)
    {
        ASSERT(binary->getOp() == EOpIndexDirectStruct || binary->getOp() == EOpIndexDirect);

        asSymbol = binary->getLeft()->getAsSymbolNode();

        if (asSymbol)
        {
            ASSERT(asSymbol->getType().getStruct());
            TIntermSymbol *newSymbol = ReplaceTypeOfSymbolNode(asSymbol, symbolTable);
            binary->replaceChildNode(binary->getLeft(), newSymbol);
            return replacement;
        }

        binary = binary->getLeft()->getAsBinaryNode();
    }

    UNREACHABLE();
    return nullptr;
}

void GenerateArrayStrides(const std::vector<size_t> &arraySizes,
                          std::vector<size_t> *arrayStridesOut)
{
    auto &strides = *arrayStridesOut;

    ASSERT(strides.empty());
    strides.reserve(arraySizes.size() + 1);

    size_t currentStride = 1;
    strides.push_back(1);
    for (auto it = arraySizes.rbegin(); it != arraySizes.rend(); ++it)
    {
        currentStride *= *it;
        strides.push_back(currentStride);
    }
}

// This returns an expression representing the correct index using the array
// index operations in node.
static TIntermTyped *GetIndexExpressionFromTypedNode(TIntermTyped *node,
                                                     const std::vector<size_t> &strides,
                                                     TIntermTyped *offset)
{
    TIntermTyped *result      = offset;
    TIntermTyped *currentNode = node;

    auto it = strides.end();
    --it;
    // If this is being used as an argument, not all indices may be present;
    // count how many indices are there.
    while (currentNode->getAsBinaryNode())
    {
        TIntermBinary *asBinary = currentNode->getAsBinaryNode();

        switch (asBinary->getOp())
        {
            case EOpIndexDirectStruct:
                break;

            case EOpIndexDirect:
            case EOpIndexIndirect:
                --it;
                break;

            default:
                UNREACHABLE();
                break;
        }

        currentNode = asBinary->getLeft();
    }

    currentNode = node;

    while (currentNode->getAsBinaryNode())
    {
        TIntermBinary *asBinary = currentNode->getAsBinaryNode();

        switch (asBinary->getOp())
        {
            case EOpIndexDirectStruct:
                break;

            case EOpIndexDirect:
            case EOpIndexIndirect:
            {
                TIntermBinary *multiply =
                    new TIntermBinary(EOpMul, CreateIndexNode(static_cast<int>(*it++)),
                                      asBinary->getRight()->deepCopy());
                result = new TIntermBinary(EOpAdd, result, multiply);
                break;
            }

            default:
                UNREACHABLE();
                break;
        }

        currentNode = asBinary->getLeft();
    }

    return result;
}

// Structures for keeping track of function instantiations.

// An instantiation is keyed by the flattened sizes of the sampler arrays.
typedef std::vector<size_t> Instantiation;

struct InstantiationHash
{
    size_t operator()(const Instantiation &v) const noexcept
    {
        std::hash<size_t> hasher;
        size_t seed = 0;
        for (size_t x : v)
        {
            seed ^= hasher(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Map from each function to a "set" of instantiations.
// We store a TFunction for each instantiation as its value.
typedef std::map<ImmutableString, std::unordered_map<Instantiation, TFunction *, InstantiationHash>>
    FunctionInstantiations;

typedef std::unordered_map<const TFunction *, const TFunction *> FunctionMap;

// Generates a new function from the given function using the given
// instantiation; generatedInstantiations can be null.
TFunction *GenerateFunctionFromArguments(const TFunction *function,
                                         const TIntermSequence *arguments,
                                         TSymbolTable *symbolTable,
                                         FunctionInstantiations *functionInstantiations,
                                         FunctionMap *functionMap,
                                         const FunctionInstantiations *generatedInstantiations)
{
    // Collect sizes of array arguments.
    Instantiation instantiation;
    for (TIntermNode *node : *arguments)
    {
        const TType &type = node->getAsTyped()->getType();
        if (type.isArray() && type.isSampler())
        {
            ASSERT(type.getNumArraySizes() == 1);
            instantiation.push_back((*type.getArraySizes())[0]);
        }
    }

    if (generatedInstantiations)
    {
        auto it1 = generatedInstantiations->find(function->name());
        if (it1 != generatedInstantiations->end())
        {
            const auto &map = it1->second;
            auto it2        = map.find(instantiation);
            if (it2 != map.end())
            {
                return it2->second;
            }
        }
    }

    TFunction **newFunction = &(*functionInstantiations)[function->name()][instantiation];

    if (!*newFunction)
    {
        *newFunction =
            new TFunction(symbolTable, kEmptyImmutableString, SymbolType::AngleInternal,
                          &function->getReturnType(), function->isKnownToNotHaveSideEffects());
        (*functionMap)[*newFunction] = function;
        // Insert parameters from updated function.
        TFunction *updatedFunction = symbolTable->findUserDefinedFunction(function->name());
        size_t paramCount          = updatedFunction->getParamCount();
        auto it                    = instantiation.begin();
        for (size_t paramIndex = 0; paramIndex < paramCount; ++paramIndex)
        {
            const TVariable *param = updatedFunction->getParam(paramIndex);
            const TType &paramType = param->getType();
            if (paramType.isArray() && paramType.isSampler())
            {
                TType *replacementType = new TType(paramType);
                size_t arraySize       = *it++;
                replacementType->setArraySize(0, static_cast<unsigned int>(arraySize));
                param =
                    new TVariable(symbolTable, param->name(), replacementType, param->symbolType());
            }
            (*newFunction)->addParameter(param);
        }
    }
    return *newFunction;
}

class ArrayTraverser
{
  public:
    ArrayTraverser() { mCumulativeArraySizeStack.push_back(1); }

    void enterArray(const TType &arrayType)
    {
        if (!arrayType.isArray())
            return;
        size_t currentArraySize = mCumulativeArraySizeStack.back();
        const auto &arraySizes  = *arrayType.getArraySizes();
        for (auto it = arraySizes.rbegin(); it != arraySizes.rend(); ++it)
        {
            unsigned int arraySize = *it;
            currentArraySize *= arraySize;
            mArraySizeStack.push_back(arraySize);
            mCumulativeArraySizeStack.push_back(currentArraySize);
        }
    }

    void exitArray(const TType &arrayType)
    {
        if (!arrayType.isArray())
            return;
        mArraySizeStack.resize(mArraySizeStack.size() - arrayType.getNumArraySizes());
        mCumulativeArraySizeStack.resize(mCumulativeArraySizeStack.size() -
                                         arrayType.getNumArraySizes());
    }

  protected:
    std::vector<size_t> mArraySizeStack;
    // The first element is 1; each successive element is the previous
    // multiplied by the size of the next nested array in the current sampler.
    // For example, with sampler2D foo[3][6], we would have {1, 3, 18}.
    std::vector<size_t> mCumulativeArraySizeStack;
};

struct VariableExtraData
{
    // The value consists of strides, starting from the outermost array.
    // For example, with sampler2D foo[3][6], we would have {1, 6, 18}.
    std::unordered_map<const TVariable *, std::vector<size_t>> arrayStrideMap;
    // For each generated array parameter, holds the offset parameter.
    std::unordered_map<const TVariable *, const TVariable *> paramOffsetMap;
};

class Traverser final : public TIntermTraverser, public ArrayTraverser
{
  public:
    explicit Traverser(TSymbolTable *symbolTable)
        : TIntermTraverser(true, false, true, symbolTable), mRemovedUniformsCount(0)
    {
        mSymbolTable->push();
    }

    ~Traverser() override { mSymbolTable->pop(); }

    int removedUniformsCount() const { return mRemovedUniformsCount; }

    // Each struct sampler declaration is stripped of its samplers. New uniforms are added for each
    // stripped struct sampler. Flattens all arrays, including default uniforms.
    bool visitDeclaration(Visit visit, TIntermDeclaration *decl) override
    {
        if (visit != PreVisit)
            return true;

        if (!mInGlobalScope)
        {
            return true;
        }

        const TIntermSequence &sequence = *(decl->getSequence());
        TIntermTyped *declarator        = sequence.front()->getAsTyped();
        const TType &type               = declarator->getType();

        if (type.isStructureContainingSamplers())
        {
            TIntermSequence *newSequence = new TIntermSequence;

            if (type.isStructSpecifier())
            {
                stripStructSpecifierSamplers(type.getStruct(), newSequence);
            }
            else
            {
                TIntermSymbol *asSymbol = declarator->getAsSymbolNode();
                ASSERT(asSymbol);
                const TVariable &variable = asSymbol->variable();
                ASSERT(variable.symbolType() != SymbolType::Empty);
                extractStructSamplerUniforms(decl, variable, type.getStruct(), newSequence);
            }

            mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), decl, *newSequence);
        }

        if (type.isSampler() && type.isArray())
        {
            TIntermSequence *newSequence = new TIntermSequence;
            TIntermSymbol *asSymbol      = declarator->getAsSymbolNode();
            ASSERT(asSymbol);
            const TVariable &variable = asSymbol->variable();
            ASSERT(variable.symbolType() != SymbolType::Empty);
            extractSampler(variable.name(), variable.symbolType(), variable.getType(), newSequence,
                           0);
            mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), decl, *newSequence);
        }

        return true;
    }

    // Each struct sampler reference is replaced with a reference to the new extracted sampler.
    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        if (visit != PreVisit)
            return true;
        // If the node isn't a sampler or if this isn't the outermost access,
        // continue.
        if (!node->getType().isSampler() || node->getType().isArray())
        {
            return true;
        }

        if (node->getOp() == EOpIndexDirect || node->getOp() == EOpIndexIndirect ||
            node->getOp() == EOpIndexDirectStruct)
        {
            ImmutableString newName = GetStructSamplerNameFromTypedNode(node);
            const TVariable *samplerReplacement =
                static_cast<const TVariable *>(mSymbolTable->findUserDefined(newName));
            ASSERT(samplerReplacement);

            TIntermTyped *replacement = new TIntermSymbol(samplerReplacement);

            if (replacement->isArray())
            {
                // Add in an indirect index if contained in an array
                const auto &strides = mVariableExtraData.arrayStrideMap[samplerReplacement];
                ASSERT(!strides.empty());
                if (strides.size() > 1)
                {
                    auto it = mVariableExtraData.paramOffsetMap.find(samplerReplacement);

                    TIntermTyped *offset =
                        it == mVariableExtraData.paramOffsetMap.end()
                            ? static_cast<TIntermTyped *>(CreateIndexNode(0))
                            : static_cast<TIntermTyped *>(new TIntermSymbol(it->second));

                    TIntermTyped *index = GetIndexExpressionFromTypedNode(node, strides, offset);
                    replacement         = new TIntermBinary(EOpIndexIndirect, replacement, index);
                }
            }

            queueReplacement(replacement, OriginalNode::IS_DROPPED);
            return true;
        }

        return true;
    }

    // In we are passing references to structs containing samplers we must new additional
    // arguments. For each extracted struct sampler a new argument is added. This chains to nested
    // structs.
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override
    {
        const TFunction *function = node->getFunction();

        if (!function->hasSamplerInStructOrArrayParams())
        {
            return;
        }

        const TSymbol *foundFunction = mSymbolTable->findUserDefined(function->name());
        if (foundFunction)
        {
            ASSERT(foundFunction->isFunction());
            function = static_cast<const TFunction *>(foundFunction);
        }
        else
        {
            TFunction *newFunction = createStructSamplerFunction(function);
            mSymbolTable->declareUserDefinedFunction(newFunction, true);
            function = newFunction;
        }

        ASSERT(!function->hasSamplerInStructOrArrayOfArrayParams());
        TIntermFunctionPrototype *newProto = new TIntermFunctionPrototype(function);
        queueReplacement(newProto, OriginalNode::IS_DROPPED);
    }

    // We insert a new scope for each function definition so we can track the new parameters.
    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override
    {
        if (visit == PreVisit)
        {
            mSymbolTable->push();
        }
        else
        {
            ASSERT(visit == PostVisit);
            mSymbolTable->pop();
        }
        return true;
    }

    // For function call nodes we pass references to the extracted struct samplers in that scope.
    bool visitAggregate(Visit visit, TIntermAggregate *node) override
    {
        if (visit != PreVisit)
            return true;

        if (!node->isFunctionCall())
            return true;

        const TFunction *function = node->getFunction();
        if (!function->hasSamplerInStructOrArrayParams())
            return true;

        ASSERT(node->getOp() == EOpCallFunctionInAST);
        TIntermSequence *newArguments = getStructSamplerArguments(function, node->getSequence());

        TFunction *newFunction = GenerateFunctionFromArguments(
            function, newArguments, mSymbolTable, &mFunctionInstantiations, &mFunctionMap, nullptr);

        TIntermAggregate *newCall =
            TIntermAggregate::CreateFunctionCall(*newFunction, newArguments);
        queueReplacement(newCall, OriginalNode::IS_DROPPED);
        return true;
    }

    FunctionInstantiations *getFunctionInstantiations() { return &mFunctionInstantiations; }

    std::unordered_map<const TFunction *, const TFunction *> *getFunctionMap()
    {
        return &mFunctionMap;
    }

  private:
    // This returns the name of a struct sampler reference. References are always TIntermBinary.
    static ImmutableString GetStructSamplerNameFromTypedNode(TIntermTyped *node)
    {
        std::string stringBuilder;

        TIntermTyped *currentNode = node;
        while (currentNode->getAsBinaryNode())
        {
            TIntermBinary *asBinary = currentNode->getAsBinaryNode();

            switch (asBinary->getOp())
            {
                case EOpIndexDirectStruct:
                {
                    stringBuilder.insert(0, asBinary->getIndexStructFieldName().data());
                    stringBuilder.insert(0, "_");
                    break;
                }

                case EOpIndexDirect:
                case EOpIndexIndirect:
                    break;

                default:
                    UNREACHABLE();
                    break;
            }

            currentNode = asBinary->getLeft();
        }

        const ImmutableString &variableName = currentNode->getAsSymbolNode()->variable().name();
        stringBuilder.insert(0, variableName.data());

        return stringBuilder;
    }

    // Removes all the struct samplers from a struct specifier.
    void stripStructSpecifierSamplers(const TStructure *structure, TIntermSequence *newSequence)
    {
        TFieldList *newFieldList = new TFieldList;
        ASSERT(structure->containsSamplers());

        for (const TField *field : structure->fields())
        {
            const TType &fieldType = *field->type();
            if (!fieldType.isSampler() && !isRemovedStructType(fieldType))
            {
                TType *newType = nullptr;

                if (fieldType.isStructureContainingSamplers())
                {
                    const TSymbol *structSymbol =
                        mSymbolTable->findUserDefined(fieldType.getStruct()->name());
                    ASSERT(structSymbol && structSymbol->isStruct());
                    const TStructure *fieldStruct = static_cast<const TStructure *>(structSymbol);
                    newType                       = new TType(fieldStruct, true);
                    if (fieldType.isArray())
                    {
                        newType->makeArrays(*fieldType.getArraySizes());
                    }
                }
                else
                {
                    newType = new TType(fieldType);
                }

                TField *newField =
                    new TField(newType, field->name(), field->line(), field->symbolType());
                newFieldList->push_back(newField);
            }
        }

        // Prune empty structs.
        if (newFieldList->empty())
        {
            mRemovedStructs.insert(structure->name());
            return;
        }

        TStructure *newStruct =
            new TStructure(mSymbolTable, structure->name(), newFieldList, structure->symbolType());
        TType *newStructType = new TType(newStruct, true);
        TVariable *newStructVar =
            new TVariable(mSymbolTable, kEmptyImmutableString, newStructType, SymbolType::Empty);
        TIntermSymbol *newStructRef = new TIntermSymbol(newStructVar);

        TIntermDeclaration *structDecl = new TIntermDeclaration;
        structDecl->appendDeclarator(newStructRef);

        newSequence->push_back(structDecl);

        mSymbolTable->declare(newStruct);

        mReplacedStructs[structure] = newStruct;
    }

    // Returns true if the type is a struct that was removed because we extracted all the members.
    bool isRemovedStructType(const TType &type) const
    {
        const TStructure *structure = type.getStruct();
        return (structure && (mRemovedStructs.count(structure->name()) > 0));
    }

    // Removes samplers from struct uniforms. For each sampler removed also adds a new globally
    // defined sampler uniform.
    void extractStructSamplerUniforms(TIntermDeclaration *oldDeclaration,
                                      const TVariable &variable,
                                      const TStructure *structure,
                                      TIntermSequence *newSequence)
    {
        ASSERT(structure->containsSamplers());

        size_t nonSamplerCount = 0;

        enterArray(variable.getType());

        for (const TField *field : structure->fields())
        {
            nonSamplerCount +=
                extractFieldSamplers(variable.name(), field, variable.getType(), newSequence);
        }

        if (nonSamplerCount > 0)
        {
            // Keep the old declaration with replaced type around if it has other members.
            std::unordered_map<const TStructure *, const TStructure *>::iterator ite =
                mReplacedStructs.find(structure);
            if (ite == mReplacedStructs.end())
            {
                newSequence->push_back(oldDeclaration);
            }
            else
            {
                // Replace the type of the declared variables.
                // NOTE(hqle): Remove this once merged with upstream. Since updated code from
                // upstream already deals with this.
                const TStructure *replacementStruct = ite->second;
                TIntermDeclaration *varDecl         = new TIntermDeclaration;
                for (TIntermNode *var : *(oldDeclaration->getSequence()))
                {
                    TIntermSymbol *declaratorSymbol = var->getAsSymbolNode();
                    ASSERT(declaratorSymbol);
                    const TVariable &declaratorVar = declaratorSymbol->variable();
                    const TType &oldType           = declaratorVar.getType();
                    TType *newStructType           = new TType(replacementStruct, true);
                    newStructType->setQualifier(oldType.getQualifier());
                    if (oldType.getArraySizes())
                    {
                        newStructType->makeArrays(*oldType.getArraySizes());
                    }
                    TVariable *newDeclaratorVar = new TVariable(
                        declaratorVar.uniqueId(), declaratorVar.name(), declaratorVar.symbolType(),
                        TExtension::UNDEFINED, newStructType);
                    TIntermSymbol *newDeclarator = new TIntermSymbol(newDeclaratorVar);
                    varDecl->appendDeclarator(newDeclarator);
                }

                newSequence->push_back(varDecl);
            }
        }
        else
        {
            mRemovedUniformsCount++;
        }

        exitArray(variable.getType());
    }

    // Extracts samplers from a field of a struct. Works with nested structs and arrays.
    size_t extractFieldSamplers(const ImmutableString &prefix,
                                const TField *field,
                                const TType &containingType,
                                TIntermSequence *newSequence)
    {
        return extractFieldSamplersImpl(prefix, field, newSequence);
    }

    // Extracts samplers from a field of a struct. Works with nested structs and arrays.
    size_t extractFieldSamplersImpl(const ImmutableString &prefix,
                                    const TField *field,
                                    TIntermSequence *newSequence)
    {
        size_t nonSamplerCount = 0;

        const TType &fieldType = *field->type();
        if (fieldType.isSampler() || fieldType.isStructureContainingSamplers())
        {
            ImmutableStringBuilder stringBuilder(prefix.length() + field->name().length() + 1);
            stringBuilder << prefix << "_" << field->name();
            ImmutableString newPrefix(stringBuilder);

            if (fieldType.isSampler())
            {
                extractSampler(newPrefix, SymbolType::AngleInternal, fieldType, newSequence, 0);
            }
            else
            {
                enterArray(fieldType);
                const TStructure *structure = fieldType.getStruct();
                for (const TField *nestedField : structure->fields())
                {
                    nonSamplerCount +=
                        extractFieldSamplers(newPrefix, nestedField, fieldType, newSequence);
                }
                exitArray(fieldType);
            }
        }
        else
        {
            nonSamplerCount++;
        }

        return nonSamplerCount;
    }

    // Extracts a sampler from a struct. Declares the new extracted sampler.
    void extractSampler(const ImmutableString &newName,
                        SymbolType symbolType,
                        const TType &fieldType,
                        TIntermSequence *newSequence,
                        size_t arrayLevel)
    {
        enterArray(fieldType);

        TType *newType = new TType(fieldType);
        while (newType->isArray())
        {
            newType->toArrayElementType();
        }
        if (!mArraySizeStack.empty())
        {
            newType->makeArray(static_cast<unsigned int>(mCumulativeArraySizeStack.back()));
        }
        newType->setQualifier(EvqUniform);
        TVariable *newVariable = new TVariable(mSymbolTable, newName, newType, symbolType);
        TIntermSymbol *newRef  = new TIntermSymbol(newVariable);

        TIntermDeclaration *samplerDecl = new TIntermDeclaration;
        samplerDecl->appendDeclarator(newRef);

        newSequence->push_back(samplerDecl);

        // TODO(syoussefi): Use a SymbolType::Empty name instead of generating a name as currently
        // done.  There is no guarantee that these generated names cannot clash.  Create a mapping
        // from the previous name to the name assigned to the SymbolType::Empty variable so
        // ShaderVariable::mappedName can be updated post-transformation.
        // http://anglebug.com/4301
        if (symbolType == SymbolType::AngleInternal)
        {
            mSymbolTable->declareInternal(newVariable);
        }
        else
        {
            mSymbolTable->declare(newVariable);
        }

        GenerateArrayStrides(mArraySizeStack, &mVariableExtraData.arrayStrideMap[newVariable]);

        exitArray(fieldType);
    }

    // Returns the chained name of a sampler uniform field.
    static ImmutableString GetFieldName(const ImmutableString &paramName, const TField *field)
    {
        ImmutableStringBuilder nameBuilder(paramName.length() + 1 + field->name().length());
        nameBuilder << paramName << "_";
        nameBuilder << field->name();

        return nameBuilder;
    }

    // A pattern that visits every parameter of a function call. Uses different handlers for struct
    // parameters, struct sampler parameters, and non-struct parameters.
    class StructSamplerFunctionVisitor : angle::NonCopyable, public ArrayTraverser
    {
      public:
        StructSamplerFunctionVisitor()          = default;
        virtual ~StructSamplerFunctionVisitor() = default;

        virtual void traverse(const TFunction *function)
        {
            size_t paramCount = function->getParamCount();

            for (size_t paramIndex = 0; paramIndex < paramCount; ++paramIndex)
            {
                const TVariable *param = function->getParam(paramIndex);
                const TType &paramType = param->getType();

                if (paramType.isStructureContainingSamplers())
                {
                    const ImmutableString &baseName = getNameFromIndex(function, paramIndex);
                    if (traverseStructContainingSamplers(baseName, paramType, paramIndex))
                    {
                        visitStructParam(function, paramIndex);
                    }
                }
                else if (paramType.isArray() && paramType.isSampler())
                {
                    const ImmutableString &paramName = getNameFromIndex(function, paramIndex);
                    traverseLeafSampler(paramName, paramType, paramIndex);
                }
                else
                {
                    visitNonStructParam(function, paramIndex);
                }
            }
        }

        virtual ImmutableString getNameFromIndex(const TFunction *function, size_t paramIndex) = 0;
        // Also includes samplers in arrays of arrays.
        virtual void visitSamplerInStructParam(const ImmutableString &name,
                                               const TType *type,
                                               size_t paramIndex)                              = 0;
        virtual void visitStructParam(const TFunction *function, size_t paramIndex)            = 0;
        virtual void visitNonStructParam(const TFunction *function, size_t paramIndex)         = 0;

      private:
        bool traverseStructContainingSamplers(const ImmutableString &baseName,
                                              const TType &structType,
                                              size_t paramIndex)
        {
            bool hasNonSamplerFields    = false;
            const TStructure *structure = structType.getStruct();
            enterArray(structType);
            for (const TField *field : structure->fields())
            {
                if (field->type()->isStructureContainingSamplers() || field->type()->isSampler())
                {
                    if (traverseSamplerInStruct(baseName, structType, field, paramIndex))
                    {
                        hasNonSamplerFields = true;
                    }
                }
                else
                {
                    hasNonSamplerFields = true;
                }
            }
            exitArray(structType);
            return hasNonSamplerFields;
        }

        bool traverseSamplerInStruct(const ImmutableString &baseName,
                                     const TType &baseType,
                                     const TField *field,
                                     size_t paramIndex)
        {
            bool hasNonSamplerParams = false;

            if (field->type()->isStructureContainingSamplers())
            {
                ImmutableString name = GetFieldName(baseName, field);
                hasNonSamplerParams =
                    traverseStructContainingSamplers(name, *field->type(), paramIndex);
            }
            else
            {
                ASSERT(field->type()->isSampler());
                ImmutableString name = GetFieldName(baseName, field);
                traverseLeafSampler(name, *field->type(), paramIndex);
            }

            return hasNonSamplerParams;
        }

        void traverseLeafSampler(const ImmutableString &samplerName,
                                 const TType &samplerType,
                                 size_t paramIndex)
        {
            enterArray(samplerType);
            visitSamplerInStructParam(samplerName, &samplerType, paramIndex);
            exitArray(samplerType);
            return;
        }
    };

    // A visitor that replaces functions with struct sampler references. The struct sampler
    // references are expanded to include new fields for the structs.
    class CreateStructSamplerFunctionVisitor final : public StructSamplerFunctionVisitor
    {
      public:
        CreateStructSamplerFunctionVisitor(TSymbolTable *symbolTable, VariableExtraData *extraData)
            : mSymbolTable(symbolTable), mNewFunction(nullptr), mExtraData(extraData)
        {}

        ImmutableString getNameFromIndex(const TFunction *function, size_t paramIndex) override
        {
            const TVariable *param = function->getParam(paramIndex);
            return param->name();
        }

        void traverse(const TFunction *function) override
        {
            mNewFunction =
                new TFunction(mSymbolTable, function->name(), function->symbolType(),
                              &function->getReturnType(), function->isKnownToNotHaveSideEffects());

            StructSamplerFunctionVisitor::traverse(function);
        }

        void visitSamplerInStructParam(const ImmutableString &name,
                                       const TType *type,
                                       size_t paramIndex) override
        {
            if (mArraySizeStack.size() > 0)
            {
                TType *newType = new TType(*type);
                newType->toArrayBaseType();
                newType->makeArray(static_cast<unsigned int>(mCumulativeArraySizeStack.back()));
                type = newType;
            }
            TVariable *fieldSampler =
                new TVariable(mSymbolTable, name, type, SymbolType::AngleInternal);
            mNewFunction->addParameter(fieldSampler);
            mSymbolTable->declareInternal(fieldSampler);
            if (mArraySizeStack.size() > 0)
            {
                // Also declare an offset parameter.
                const TType *intType     = StaticType::GetBasic<EbtInt>();
                TVariable *samplerOffset = new TVariable(mSymbolTable, kEmptyImmutableString,
                                                         intType, SymbolType::AngleInternal);
                mNewFunction->addParameter(samplerOffset);
                GenerateArrayStrides(mArraySizeStack, &mExtraData->arrayStrideMap[fieldSampler]);
                mExtraData->paramOffsetMap[fieldSampler] = samplerOffset;
            }
        }

        void visitStructParam(const TFunction *function, size_t paramIndex) override
        {
            const TVariable *param = function->getParam(paramIndex);
            TType *structType      = GetStructSamplerParameterType(mSymbolTable, *param);
            TVariable *newParam =
                new TVariable(mSymbolTable, param->name(), structType, param->symbolType());
            mNewFunction->addParameter(newParam);
        }

        void visitNonStructParam(const TFunction *function, size_t paramIndex) override
        {
            const TVariable *param = function->getParam(paramIndex);
            mNewFunction->addParameter(param);
        }

        TFunction *getNewFunction() const { return mNewFunction; }

      private:
        TSymbolTable *mSymbolTable;
        TFunction *mNewFunction;
        VariableExtraData *mExtraData;
    };

    TFunction *createStructSamplerFunction(const TFunction *function)
    {
        CreateStructSamplerFunctionVisitor visitor(mSymbolTable, &mVariableExtraData);
        visitor.traverse(function);
        return visitor.getNewFunction();
    }

    // A visitor that replaces function calls with expanded struct sampler parameters.
    class GetSamplerArgumentsVisitor final : public StructSamplerFunctionVisitor
    {
      public:
        GetSamplerArgumentsVisitor(TSymbolTable *symbolTable,
                                   const TIntermSequence *arguments,
                                   VariableExtraData *extraData)
            : mSymbolTable(symbolTable),
              mArguments(arguments),
              mNewArguments(new TIntermSequence),
              mExtraData(extraData)
        {}

        ImmutableString getNameFromIndex(const TFunction *function, size_t paramIndex) override
        {
            TIntermTyped *argument = (*mArguments)[paramIndex]->getAsTyped();
            return GetStructSamplerNameFromTypedNode(argument);
        }

        void visitSamplerInStructParam(const ImmutableString &name,
                                       const TType *type,
                                       size_t paramIndex) override
        {
            const TVariable *argSampler =
                static_cast<const TVariable *>(mSymbolTable->findUserDefined(name));
            ASSERT(argSampler);

            TIntermTyped *argument = (*mArguments)[paramIndex]->getAsTyped();

            auto it = mExtraData->paramOffsetMap.find(argSampler);
            TIntermTyped *argOffset =
                it == mExtraData->paramOffsetMap.end()
                    ? static_cast<TIntermTyped *>(CreateIndexNode(0))
                    : static_cast<TIntermTyped *>(new TIntermSymbol(it->second));

            TIntermTyped *finalOffset = GetIndexExpressionFromTypedNode(
                argument, mExtraData->arrayStrideMap[argSampler], argOffset);

            TIntermSymbol *argSymbol = new TIntermSymbol(argSampler);

            // If we have a regular sampler inside a struct (possibly an array
            // of structs), handle this case separately.
            if (!type->isArray() && mArraySizeStack.size() == 0)
            {
                if (argSampler->getType().isArray())
                {
                    TIntermTyped *argIndex =
                        new TIntermBinary(EOpIndexIndirect, argSymbol, finalOffset);
                    mNewArguments->push_back(argIndex);
                }
                else
                {
                    mNewArguments->push_back(argSymbol);
                }
                return;
            }

            mNewArguments->push_back(argSymbol);

            mNewArguments->push_back(finalOffset);
            // If array, we need to calculate the offset based on what indices
            // are present in the argument.
        }

        void visitStructParam(const TFunction *function, size_t paramIndex) override
        {
            // The tree structure of the parameter is modified to point to the new type. This leaves
            // the tree in a consistent state.
            TIntermTyped *argument    = (*mArguments)[paramIndex]->getAsTyped();
            TIntermTyped *replacement = ReplaceTypeOfTypedStructNode(argument, mSymbolTable);
            mNewArguments->push_back(replacement);
        }

        void visitNonStructParam(const TFunction *function, size_t paramIndex) override
        {
            TIntermTyped *argument = (*mArguments)[paramIndex]->getAsTyped();
            mNewArguments->push_back(argument);
        }

        TIntermSequence *getNewArguments() const { return mNewArguments; }

      private:
        TSymbolTable *mSymbolTable;
        const TIntermSequence *mArguments;
        TIntermSequence *mNewArguments;
        VariableExtraData *mExtraData;
    };

    TIntermSequence *getStructSamplerArguments(const TFunction *function,
                                               const TIntermSequence *arguments)
    {
        GetSamplerArgumentsVisitor visitor(mSymbolTable, arguments, &mVariableExtraData);
        visitor.traverse(function);
        return visitor.getNewArguments();
    }

    int mRemovedUniformsCount;
    std::set<ImmutableString> mRemovedStructs;
    std::unordered_map<const TStructure *, const TStructure *> mReplacedStructs;
    FunctionInstantiations mFunctionInstantiations;
    FunctionMap mFunctionMap;
    VariableExtraData mVariableExtraData;
};

class MonomorphizeTraverser final : public TIntermTraverser
{
  public:
    typedef std::unordered_map<const TVariable *, const TVariable *> VariableReplacementMap;

    explicit MonomorphizeTraverser(
        TCompiler *compiler,
        TSymbolTable *symbolTable,
        FunctionInstantiations *functionInstantiations,
        std::unordered_map<const TFunction *, const TFunction *> *functionMap)
        : TIntermTraverser(true, false, true, symbolTable),
          mFunctionInstantiations(*functionInstantiations),
          mFunctionMap(functionMap),
          mCompiler(compiler),
          mSubpassesSucceeded(true)
    {}

    void switchToPending()
    {
        mFunctionInstantiations.clear();
        mFunctionInstantiations.swap(mPendingInstantiations);
    }

    bool hasPending()
    {
        if (mPendingInstantiations.empty())
            return false;
        for (auto &entry : mPendingInstantiations)
        {
            if (!entry.second.empty())
            {
                return true;
            }
        }
        return false;
    }

    bool subpassesSucceeded() { return mSubpassesSucceeded; }

    void visitFunctionPrototype(TIntermFunctionPrototype *node) override
    {
        mReplacementPrototypes.clear();
        const TFunction *function = node->getFunction();

        auto &generatedMap = mGeneratedInstantiations[function->name()];

        auto it = mFunctionInstantiations.find(function->name());
        if (it == mFunctionInstantiations.end())
            return;
        for (const auto &instantiation : it->second)
        {
            TFunction *replacementFunction = instantiation.second;
            mReplacementPrototypes.push_back(new TIntermFunctionPrototype(replacementFunction));
            generatedMap[instantiation.first] = replacementFunction;
        }
        if (!mInFunctionDefinition)
        {
            insertStatementsInParentBlock(mReplacementPrototypes);
        }
    }

    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override
    {
        mInFunctionDefinition = visit == PreVisit;
        if (visit != PostVisit)
            return true;
        TIntermSequence replacements;
        const TFunction *function = node->getFunction();
        size_t numParameters      = function->getParamCount();

        for (TIntermNode *replacementNode : mReplacementPrototypes)
        {
            TIntermFunctionPrototype *replacementPrototype =
                replacementNode->getAsFunctionPrototypeNode();
            const TFunction *replacementFunction = replacementPrototype->getFunction();

            // Replace function parameters with correct array sizes.
            VariableReplacementMap variableReplacementMap;
            ASSERT(replacementPrototype->getFunction()->getParamCount() == numParameters);
            for (size_t i = 0; i < numParameters; i++)
            {
                const TVariable *origParam = function->getParam(i);
                const TVariable *newParam  = replacementFunction->getParam(i);
                if (origParam != newParam)
                {
                    variableReplacementMap[origParam] = newParam;
                }
            }

            TIntermBlock *body = node->getBody()->deepCopy();
            ReplaceVariablesTraverser replaceVariables(mSymbolTable, &variableReplacementMap);
            body->traverse(&replaceVariables);
            mSubpassesSucceeded &= replaceVariables.updateTree(mCompiler, body);
            CollectNewInstantiationsTraverser collectNewInstantiations(
                mSymbolTable, &mPendingInstantiations, &mGeneratedInstantiations, mFunctionMap);
            body->traverse(&collectNewInstantiations);
            mSubpassesSucceeded &= collectNewInstantiations.updateTree(mCompiler, body);
            replacements.push_back(new TIntermFunctionDefinition(replacementPrototype, body));
        }
        insertStatementsInParentBlock(replacements);
        return true;
    }

  private:
    bool mInFunctionDefinition;
    FunctionInstantiations mFunctionInstantiations;
    // Set of already-generated instantiations.
    FunctionInstantiations mGeneratedInstantiations;
    // New instantiations caused by other instantiations.
    FunctionInstantiations mPendingInstantiations;
    std::unordered_map<const TFunction *, const TFunction *> *mFunctionMap;
    TIntermSequence mReplacementPrototypes;
    TCompiler *mCompiler;
    bool mSubpassesSucceeded;

    class ReplaceVariablesTraverser : public TIntermTraverser
    {
      public:
        explicit ReplaceVariablesTraverser(TSymbolTable *symbolTable,
                                           VariableReplacementMap *variableReplacementMap)
            : TIntermTraverser(true, false, false, symbolTable),
              mVariableReplacementMap(variableReplacementMap)
        {}

        void visitSymbol(TIntermSymbol *node) override
        {
            const TVariable *variable = &node->variable();
            auto it                   = mVariableReplacementMap->find(variable);
            if (it != mVariableReplacementMap->end())
            {
                queueReplacement(new TIntermSymbol(it->second), OriginalNode::IS_DROPPED);
            }
        }

      private:
        VariableReplacementMap *mVariableReplacementMap;
    };

    class CollectNewInstantiationsTraverser : public TIntermTraverser
    {
      public:
        explicit CollectNewInstantiationsTraverser(
            TSymbolTable *symbolTable,
            FunctionInstantiations *pendingInstantiations,
            FunctionInstantiations *generatedInstantiations,
            std::unordered_map<const TFunction *, const TFunction *> *functionMap)
            : TIntermTraverser(true, false, false, symbolTable),
              mPendingInstantiations(pendingInstantiations),
              mGeneratedInstantiations(generatedInstantiations),
              mFunctionMap(functionMap)
        {}

        bool visitAggregate(Visit visit, TIntermAggregate *node) override
        {
            if (!node->isFunctionCall())
                return true;
            const TFunction *function = node->getFunction();
            const TFunction *oldFunction;
            {
                auto it = mFunctionMap->find(function);
                if (it == mFunctionMap->end())
                    return true;
                oldFunction = it->second;
            }
            ASSERT(node->getOp() == EOpCallFunctionInAST);
            TIntermSequence *arguments = node->getSequence();
            TFunction *newFunction     = GenerateFunctionFromArguments(
                oldFunction, arguments, mSymbolTable, mPendingInstantiations, mFunctionMap,
                mGeneratedInstantiations);
            queueReplacement(TIntermAggregate::CreateFunctionCall(*newFunction, arguments),
                             OriginalNode::IS_DROPPED);
            return true;
        }

      private:
        FunctionInstantiations *mPendingInstantiations;
        FunctionInstantiations *mGeneratedInstantiations;
        std::unordered_map<const TFunction *, const TFunction *> *mFunctionMap;
    };
};
}  // anonymous namespace

bool RewriteStructSamplers(TCompiler *compiler,
                           TIntermBlock *root,
                           TSymbolTable *symbolTable,
                           int *removedUniformsCountOut)
{
    Traverser rewriteStructSamplers(symbolTable);
    root->traverse(&rewriteStructSamplers);
    if (!rewriteStructSamplers.updateTree(compiler, root))
    {
        return false;
    }
    *removedUniformsCountOut = rewriteStructSamplers.removedUniformsCount();

    if (rewriteStructSamplers.getFunctionInstantiations()->empty())
    {
        return true;
    }

    MonomorphizeTraverser monomorphizeFunctions(compiler, symbolTable,
                                                rewriteStructSamplers.getFunctionInstantiations(),
                                                rewriteStructSamplers.getFunctionMap());
    root->traverse(&monomorphizeFunctions);
    if (!monomorphizeFunctions.subpassesSucceeded())
    {
        return false;
    }
    if (!monomorphizeFunctions.updateTree(compiler, root))
    {
        return false;
    }

    // Generate instantiations caused by other instantiations.
    while (monomorphizeFunctions.hasPending())
    {
        monomorphizeFunctions.switchToPending();
        root->traverse(&monomorphizeFunctions);
        if (!monomorphizeFunctions.subpassesSucceeded())
        {
            return false;
        }
        if (!monomorphizeFunctions.updateTree(compiler, root))
        {
            return false;
        }
    }

    return true;
}
}  // namespace sh
