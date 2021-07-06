//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Symbol.cpp: Symbols representing variables, functions, structures and interface blocks.
//

#if defined(_MSC_VER)
#    pragma warning(disable : 4718)
#endif

#include "compiler/translator/Symbol.h"

#include "compiler/translator/ImmutableStringBuilder.h"
#include "compiler/translator/SymbolTable.h"

namespace sh
{

namespace
{

constexpr const ImmutableString kMainName("main");
constexpr const ImmutableString kImageLoadName("imageLoad");
constexpr const ImmutableString kImageStoreName("imageStore");
constexpr const ImmutableString kImageSizeName("imageSize");
constexpr const ImmutableString kAtomicCounterName("atomicCounter");

static const char kFunctionMangledNameSeparator = '(';

}  // anonymous namespace

TSymbol::TSymbol(TSymbolTable *symbolTable,
                 const ImmutableString &name,
                 SymbolType symbolType,
                 SymbolClass symbolClass,
                 TExtension extension)
    : mName(name),
      mUniqueId(symbolTable->nextUniqueId()),
      mSymbolType(symbolType),
      mExtension(extension),
      mSymbolClass(symbolClass)
{
    ASSERT(mSymbolType == SymbolType::BuiltIn || mExtension == TExtension::UNDEFINED);
    ASSERT(mName != "" || mSymbolType == SymbolType::AngleInternal ||
           mSymbolType == SymbolType::Empty);
}

ImmutableString TSymbol::name() const
{
    if (!mName.empty())
    {
        return mName;
    }
    // This can be called for nameless function parameters in HLSL.
    ASSERT(mSymbolType == SymbolType::AngleInternal ||
           (mSymbolType == SymbolType::Empty && isVariable()));
    int uniqueId = mUniqueId.get();
    ImmutableStringBuilder symbolNameOut(sizeof(uniqueId) * 2u + 1u);
    symbolNameOut << 's';
    symbolNameOut.appendHex(mUniqueId.get());
    return symbolNameOut;
}

ImmutableString TSymbol::getMangledName() const
{
    if (mSymbolClass == SymbolClass::Function)
    {
        // We do this instead of using proper virtual functions so that we can better support
        // constexpr symbols.
        return static_cast<const TFunction *>(this)->getFunctionMangledName();
    }
    ASSERT(mSymbolType != SymbolType::Empty);
    return name();
}

TVariable::TVariable(TSymbolTable *symbolTable,
                     const ImmutableString &name,
                     const TType *type,
                     SymbolType symbolType,
                     TExtension extension)
    : TSymbol(symbolTable, name, symbolType, SymbolClass::Variable, extension),
      mType(type),
      unionArray(nullptr)
{
    ASSERT(mType);
    ASSERT(name.empty() || symbolType != SymbolType::Empty);
}

TStructure::TStructure(TSymbolTable *symbolTable,
                       const ImmutableString &name,
                       const TFieldList *fields,
                       SymbolType symbolType)
    : TSymbol(symbolTable, name, symbolType, SymbolClass::Struct), TFieldListCollection(fields)
{}

TStructure::TStructure(const TSymbolUniqueId &id,
                       const ImmutableString &name,
                       TExtension extension,
                       const TFieldList *fields)
    : TSymbol(id, name, SymbolType::BuiltIn, extension, SymbolClass::Struct),
      TFieldListCollection(fields)
{}

void TStructure::createSamplerSymbols(const char *namePrefix,
                                      const TString &apiNamePrefix,
                                      TVector<const TVariable *> *outputSymbols,
                                      TMap<const TVariable *, TString> *outputSymbolsToAPINames,
                                      TSymbolTable *symbolTable) const
{
    ASSERT(containsSamplers());
    for (const auto *field : *mFields)
    {
        const TType *fieldType = field->type();
        if (IsSampler(fieldType->getBasicType()) || fieldType->isStructureContainingSamplers())
        {
            std::stringstream fieldName = sh::InitializeStream<std::stringstream>();
            fieldName << namePrefix << "_" << field->name();
            TString fieldApiName = apiNamePrefix + ".";
            fieldApiName += field->name().data();
            fieldType->createSamplerSymbols(ImmutableString(fieldName.str()), fieldApiName,
                                            outputSymbols, outputSymbolsToAPINames, symbolTable);
        }
    }
}

void TStructure::setName(const ImmutableString &name)
{
    ImmutableString *mutableName = const_cast<ImmutableString *>(&mName);
    *mutableName                 = name;
}

TInterfaceBlock::TInterfaceBlock(TSymbolTable *symbolTable,
                                 const ImmutableString &name,
                                 const TFieldList *fields,
                                 const TLayoutQualifier &layoutQualifier,
                                 SymbolType symbolType,
                                 TExtension extension)
    : TSymbol(symbolTable, name, symbolType, SymbolClass::InterfaceBlock, extension),
      TFieldListCollection(fields),
      mBlockStorage(layoutQualifier.blockStorage),
      mBinding(layoutQualifier.binding)
{
    ASSERT(name != nullptr);
}

TInterfaceBlock::TInterfaceBlock(const TSymbolUniqueId &id,
                                 const ImmutableString &name,
                                 TExtension extension,
                                 const TFieldList *fields)
    : TSymbol(id, name, SymbolType::BuiltIn, extension, SymbolClass::InterfaceBlock),
      TFieldListCollection(fields),
      mBlockStorage(EbsUnspecified),
      mBinding(0)
{}

TFunction::TFunction(TSymbolTable *symbolTable,
                     const ImmutableString &name,
                     SymbolType symbolType,
                     const TType *retType,
                     bool knownToNotHaveSideEffects)
    : TSymbol(symbolTable, name, symbolType, SymbolClass::Function, TExtension::UNDEFINED),
      mParametersVector(new TParamVector()),
      mParameters(nullptr),
      mParamCount(0u),
      returnType(retType),
      mMangledName(""),
      mOp(EOpNull),
      defined(false),
      mHasPrototypeDeclaration(false),
      mKnownToNotHaveSideEffects(knownToNotHaveSideEffects)
{
    // Functions with an empty name are not allowed.
    ASSERT(symbolType != SymbolType::Empty);
    ASSERT(name != nullptr || symbolType == SymbolType::AngleInternal);
}

void TFunction::addParameter(const TVariable *p)
{
    ASSERT(mParametersVector);
    mParametersVector->push_back(p);
    mParameters  = mParametersVector->data();
    mParamCount  = mParametersVector->size();
    mMangledName = kEmptyImmutableString;
}

void TFunction::shareParameters(const TFunction &parametersSource)
{
    mParametersVector = nullptr;
    mParameters       = parametersSource.mParameters;
    mParamCount       = parametersSource.mParamCount;
    ASSERT(parametersSource.name() == name());
    mMangledName = parametersSource.mMangledName;
}

ImmutableString TFunction::buildMangledName() const
{
    std::string newName(name().data(), name().length());
    newName += kFunctionMangledNameSeparator;

    for (size_t i = 0u; i < mParamCount; ++i)
    {
        newName += mParameters[i]->getType().getMangledName();
    }
    return ImmutableString(newName);
}

bool TFunction::isMain() const
{
    return symbolType() == SymbolType::UserDefined && name() == kMainName;
}

bool TFunction::isImageFunction() const
{
    return symbolType() == SymbolType::BuiltIn &&
           (name() == kImageSizeName || name() == kImageLoadName || name() == kImageStoreName);
}

bool TFunction::isAtomicCounterFunction() const
{
    return SymbolType() == SymbolType::BuiltIn && name().beginsWith(kAtomicCounterName);
}

bool TFunction::hasSamplerInStructOrArrayParams() const
{
    for (size_t paramIndex = 0; paramIndex < mParamCount; ++paramIndex)
    {
        const TVariable *param = getParam(paramIndex);
        if (param->getType().isStructureContainingSamplers() ||
            (param->getType().isArray() && param->getType().isSampler()))
        {
            return true;
        }
    }

    return false;
}

bool TFunction::hasSamplerInStructOrArrayOfArrayParams() const
{
    for (size_t paramIndex = 0; paramIndex < mParamCount; ++paramIndex)
    {
        const TVariable *param = getParam(paramIndex);
        if (param->getType().isStructureContainingSamplers() ||
            (param->getType().isArrayOfArrays() && param->getType().isSampler()))
        {
            return true;
        }
    }

    return false;
}
}  // namespace sh
