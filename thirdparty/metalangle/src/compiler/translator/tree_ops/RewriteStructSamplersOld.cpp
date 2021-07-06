//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteStructSamplers: Extract structs from samplers.
//

#include "compiler/translator/tree_ops/RewriteStructSamplers.h"

#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/SymbolTable.h"
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

// Maximum string size of a hex unsigned int.
constexpr size_t kHexSize = ImmutableStringBuilder::GetHexCharCount<unsigned int>();

class Traverser final : public TIntermTraverser
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
    // stripped struct sampler.
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

        return true;
    }

    // Each struct sampler reference is replaced with a reference to the new extracted sampler.
    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        if (visit != PreVisit)
            return true;

        if (node->getOp() == EOpIndexDirectStruct && node->getType().isSampler())
        {
            ImmutableString newName = GetStructSamplerNameFromTypedNode(node);
            const TVariable *samplerReplacement =
                static_cast<const TVariable *>(mSymbolTable->findUserDefined(newName));
            ASSERT(samplerReplacement);

            TIntermSymbol *replacement = new TIntermSymbol(samplerReplacement);

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

        if (!function->hasSamplerInStructOrArrayOfArrayParams())
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
        if (!function->hasSamplerInStructOrArrayOfArrayParams())
            return true;

        ASSERT(node->getOp() == EOpCallFunctionInAST);
        TFunction *newFunction        = mSymbolTable->findUserDefinedFunction(function->name());
        TIntermSequence *newArguments = getStructSamplerArguments(function, node->getSequence());

        TIntermAggregate *newCall =
            TIntermAggregate::CreateFunctionCall(*newFunction, newArguments);
        queueReplacement(newCall, OriginalNode::IS_DROPPED);
        return true;
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
                case EOpIndexDirect:
                {
                    const int index = asBinary->getRight()->getAsConstantUnion()->getIConst(0);
                    const std::string strInt = Str(index);
                    stringBuilder.insert(0, strInt);
                    stringBuilder.insert(0, "_");
                    break;
                }
                case EOpIndexDirectStruct:
                {
                    stringBuilder.insert(0, asBinary->getIndexStructFieldName().data());
                    stringBuilder.insert(0, "_");
                    break;
                }

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

        for (const TField *field : structure->fields())
        {
            nonSamplerCount +=
                extractFieldSamplers(variable.name(), field, variable.getType(), newSequence);
        }

        if (nonSamplerCount > 0)
        {
            // Keep the old declaration around if it has other members.
            newSequence->push_back(oldDeclaration);
        }
        else
        {
            mRemovedUniformsCount++;
        }
    }

    // Extracts samplers from a field of a struct. Works with nested structs and arrays.
    size_t extractFieldSamplers(const ImmutableString &prefix,
                                const TField *field,
                                const TType &containingType,
                                TIntermSequence *newSequence)
    {
        if (containingType.isArray())
        {
            size_t nonSamplerCount = 0;

            // Name the samplers internally as varName_<index>_fieldName
            const TVector<unsigned int> &arraySizes = *containingType.getArraySizes();
            for (unsigned int arrayElement = 0; arrayElement < arraySizes[0]; ++arrayElement)
            {
                ImmutableStringBuilder stringBuilder(prefix.length() + kHexSize + 1);
                stringBuilder << prefix << "_";
                stringBuilder.appendHex(arrayElement);
                nonSamplerCount = extractFieldSamplersImpl(stringBuilder, field, newSequence);
            }

            return nonSamplerCount;
        }

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
                extractSampler(newPrefix, fieldType, newSequence);
            }
            else
            {
                const TStructure *structure = fieldType.getStruct();
                for (const TField *nestedField : structure->fields())
                {
                    nonSamplerCount +=
                        extractFieldSamplers(newPrefix, nestedField, fieldType, newSequence);
                }
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
                        const TType &fieldType,
                        TIntermSequence *newSequence) const
    {
        TType *newType = new TType(fieldType);
        newType->setQualifier(EvqUniform);
        TVariable *newVariable =
            new TVariable(mSymbolTable, newName, newType, SymbolType::AngleInternal);
        TIntermSymbol *newRef = new TIntermSymbol(newVariable);

        TIntermDeclaration *samplerDecl = new TIntermDeclaration;
        samplerDecl->appendDeclarator(newRef);

        newSequence->push_back(samplerDecl);

        mSymbolTable->declareInternal(newVariable);
    }

    // Returns the chained name of a sampler uniform field.
    static ImmutableString GetFieldName(const ImmutableString &paramName,
                                        const TField *field,
                                        unsigned arrayIndex)
    {
        ImmutableStringBuilder nameBuilder(paramName.length() + kHexSize + 2 +
                                           field->name().length());
        nameBuilder << paramName << "_";

        if (arrayIndex < std::numeric_limits<unsigned>::max())
        {
            nameBuilder.appendHex(arrayIndex);
            nameBuilder << "_";
        }
        nameBuilder << field->name();

        return nameBuilder;
    }

    // A pattern that visits every parameter of a function call. Uses different handlers for struct
    // parameters, struct sampler parameters, and non-struct parameters.
    class StructSamplerFunctionVisitor : angle::NonCopyable
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
                    if (traverseStructContainingSamplers(baseName, paramType))
                    {
                        visitStructParam(function, paramIndex);
                    }
                }
                else
                {
                    visitNonStructParam(function, paramIndex);
                }
            }
        }

        virtual ImmutableString getNameFromIndex(const TFunction *function, size_t paramIndex) = 0;
        virtual void visitSamplerInStructParam(const ImmutableString &name,
                                               const TField *field)                            = 0;
        virtual void visitStructParam(const TFunction *function, size_t paramIndex)            = 0;
        virtual void visitNonStructParam(const TFunction *function, size_t paramIndex)         = 0;

      private:
        bool traverseStructContainingSamplers(const ImmutableString &baseName,
                                              const TType &structType)
        {
            bool hasNonSamplerFields    = false;
            const TStructure *structure = structType.getStruct();
            for (const TField *field : structure->fields())
            {
                if (field->type()->isStructureContainingSamplers() || field->type()->isSampler())
                {
                    if (traverseSamplerInStruct(baseName, structType, field))
                    {
                        hasNonSamplerFields = true;
                    }
                }
                else
                {
                    hasNonSamplerFields = true;
                }
            }
            return hasNonSamplerFields;
        }

        bool traverseSamplerInStruct(const ImmutableString &baseName,
                                     const TType &baseType,
                                     const TField *field)
        {
            bool hasNonSamplerParams = false;

            if (baseType.isArray())
            {
                const TVector<unsigned int> &arraySizes = *baseType.getArraySizes();
                ASSERT(arraySizes.size() == 1);

                for (unsigned int arrayIndex = 0; arrayIndex < arraySizes[0]; ++arrayIndex)
                {
                    ImmutableString name = GetFieldName(baseName, field, arrayIndex);

                    if (field->type()->isStructureContainingSamplers())
                    {
                        if (traverseStructContainingSamplers(name, *field->type()))
                        {
                            hasNonSamplerParams = true;
                        }
                    }
                    else
                    {
                        ASSERT(field->type()->isSampler());
                        visitSamplerInStructParam(name, field);
                    }
                }
            }
            else if (field->type()->isStructureContainingSamplers())
            {
                ImmutableString name =
                    GetFieldName(baseName, field, std::numeric_limits<unsigned>::max());
                hasNonSamplerParams = traverseStructContainingSamplers(name, *field->type());
            }
            else
            {
                ASSERT(field->type()->isSampler());
                ImmutableString name =
                    GetFieldName(baseName, field, std::numeric_limits<unsigned>::max());
                visitSamplerInStructParam(name, field);
            }

            return hasNonSamplerParams;
        }
    };

    // A visitor that replaces functions with struct sampler references. The struct sampler
    // references are expanded to include new fields for the structs.
    class CreateStructSamplerFunctionVisitor final : public StructSamplerFunctionVisitor
    {
      public:
        CreateStructSamplerFunctionVisitor(TSymbolTable *symbolTable)
            : mSymbolTable(symbolTable), mNewFunction(nullptr)
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

        void visitSamplerInStructParam(const ImmutableString &name, const TField *field) override
        {
            TVariable *fieldSampler =
                new TVariable(mSymbolTable, name, field->type(), SymbolType::AngleInternal);
            mNewFunction->addParameter(fieldSampler);
            mSymbolTable->declareInternal(fieldSampler);
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
    };

    TFunction *createStructSamplerFunction(const TFunction *function) const
    {
        CreateStructSamplerFunctionVisitor visitor(mSymbolTable);
        visitor.traverse(function);
        return visitor.getNewFunction();
    }

    // A visitor that replaces function calls with expanded struct sampler parameters.
    class GetSamplerArgumentsVisitor final : public StructSamplerFunctionVisitor
    {
      public:
        GetSamplerArgumentsVisitor(TSymbolTable *symbolTable, const TIntermSequence *arguments)
            : mSymbolTable(symbolTable), mArguments(arguments), mNewArguments(new TIntermSequence)
        {}

        ImmutableString getNameFromIndex(const TFunction *function, size_t paramIndex) override
        {
            TIntermTyped *argument = (*mArguments)[paramIndex]->getAsTyped();
            return GetStructSamplerNameFromTypedNode(argument);
        }

        void visitSamplerInStructParam(const ImmutableString &name, const TField *field) override
        {
            TVariable *argSampler =
                new TVariable(mSymbolTable, name, field->type(), SymbolType::AngleInternal);
            TIntermSymbol *argSymbol = new TIntermSymbol(argSampler);
            mNewArguments->push_back(argSymbol);
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
    };

    TIntermSequence *getStructSamplerArguments(const TFunction *function,
                                               const TIntermSequence *arguments) const
    {
        GetSamplerArgumentsVisitor visitor(mSymbolTable, arguments);
        visitor.traverse(function);
        return visitor.getNewArguments();
    }

    int mRemovedUniformsCount;
    std::set<ImmutableString> mRemovedStructs;
};
}  // anonymous namespace

bool RewriteStructSamplersOld(TCompiler *compiler,
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
    return true;
}
}  // namespace sh
