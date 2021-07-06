//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#if defined(_MSC_VER)
#    pragma warning(disable : 4718)
#endif

#include "compiler/translator/Types.h"
#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/InfoSink.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/SymbolTable.h"

#include <algorithm>
#include <climits>

namespace sh
{

const char *getBasicString(TBasicType t)
{
    switch (t)
    {
        case EbtVoid:
            return "void";
        case EbtFloat:
            return "float";
        case EbtInt:
            return "int";
        case EbtUInt:
            return "uint";
        case EbtBool:
            return "bool";
        case EbtYuvCscStandardEXT:
            return "yuvCscStandardEXT";
        case EbtSampler2D:
            return "sampler2D";
        case EbtSampler3D:
            return "sampler3D";
        case EbtSamplerCube:
            return "samplerCube";
        case EbtSamplerExternalOES:
            return "samplerExternalOES";
        case EbtSamplerExternal2DY2YEXT:
            return "__samplerExternal2DY2YEXT";
        case EbtSampler2DRect:
            return "sampler2DRect";
        case EbtSampler2DArray:
            return "sampler2DArray";
        case EbtSampler2DMS:
            return "sampler2DMS";
        case EbtSampler2DMSArray:
            return "sampler2DMSArray";
        case EbtISampler2D:
            return "isampler2D";
        case EbtISampler3D:
            return "isampler3D";
        case EbtISamplerCube:
            return "isamplerCube";
        case EbtISampler2DArray:
            return "isampler2DArray";
        case EbtISampler2DMS:
            return "isampler2DMS";
        case EbtISampler2DMSArray:
            return "isampler2DMSArray";
        case EbtUSampler2D:
            return "usampler2D";
        case EbtUSampler3D:
            return "usampler3D";
        case EbtUSamplerCube:
            return "usamplerCube";
        case EbtUSampler2DArray:
            return "usampler2DArray";
        case EbtUSampler2DMS:
            return "usampler2DMS";
        case EbtUSampler2DMSArray:
            return "usampler2DMSArray";
        case EbtSampler2DShadow:
            return "sampler2DShadow";
        case EbtSamplerCubeShadow:
            return "samplerCubeShadow";
        case EbtSampler2DArrayShadow:
            return "sampler2DArrayShadow";
        case EbtStruct:
            return "structure";
        case EbtInterfaceBlock:
            return "interface block";
        case EbtImage2D:
            return "image2D";
        case EbtIImage2D:
            return "iimage2D";
        case EbtUImage2D:
            return "uimage2D";
        case EbtImage3D:
            return "image3D";
        case EbtIImage3D:
            return "iimage3D";
        case EbtUImage3D:
            return "uimage3D";
        case EbtImage2DArray:
            return "image2DArray";
        case EbtIImage2DArray:
            return "iimage2DArray";
        case EbtUImage2DArray:
            return "uimage2DArray";
        case EbtImageCube:
            return "imageCube";
        case EbtIImageCube:
            return "iimageCube";
        case EbtUImageCube:
            return "uimageCube";
        case EbtAtomicCounter:
            return "atomic_uint";
        default:
            UNREACHABLE();
            return "unknown type";
    }
}

// TType implementation.
TType::TType()
    : type(EbtVoid),
      precision(EbpUndefined),
      qualifier(EvqGlobal),
      invariant(false),
      memoryQualifier(TMemoryQualifier::Create()),
      layoutQualifier(TLayoutQualifier::Create()),
      primarySize(0),
      secondarySize(0),
      mArraySizes(nullptr),
      mInterfaceBlock(nullptr),
      mStructure(nullptr),
      mIsStructSpecifier(false),
      mMangledName(nullptr)
{}

TType::TType(TBasicType t, unsigned char ps, unsigned char ss)
    : type(t),
      precision(EbpUndefined),
      qualifier(EvqGlobal),
      invariant(false),
      memoryQualifier(TMemoryQualifier::Create()),
      layoutQualifier(TLayoutQualifier::Create()),
      primarySize(ps),
      secondarySize(ss),
      mArraySizes(nullptr),
      mInterfaceBlock(nullptr),
      mStructure(nullptr),
      mIsStructSpecifier(false),
      mMangledName(nullptr)
{}

TType::TType(TBasicType t, TPrecision p, TQualifier q, unsigned char ps, unsigned char ss)
    : type(t),
      precision(p),
      qualifier(q),
      invariant(false),
      memoryQualifier(TMemoryQualifier::Create()),
      layoutQualifier(TLayoutQualifier::Create()),
      primarySize(ps),
      secondarySize(ss),
      mArraySizes(nullptr),
      mInterfaceBlock(nullptr),
      mStructure(nullptr),
      mIsStructSpecifier(false),
      mMangledName(nullptr)
{}

TType::TType(const TPublicType &p)
    : type(p.getBasicType()),
      precision(p.precision),
      qualifier(p.qualifier),
      invariant(p.invariant),
      memoryQualifier(p.memoryQualifier),
      layoutQualifier(p.layoutQualifier),
      primarySize(p.getPrimarySize()),
      secondarySize(p.getSecondarySize()),
      mArraySizes(nullptr),
      mInterfaceBlock(nullptr),
      mStructure(nullptr),
      mIsStructSpecifier(false),
      mMangledName(nullptr)
{
    ASSERT(primarySize <= 4);
    ASSERT(secondarySize <= 4);
    if (p.isArray())
    {
        mArraySizes = new TVector<unsigned int>(*p.arraySizes);
    }
    if (p.getUserDef())
    {
        mStructure         = p.getUserDef();
        mIsStructSpecifier = p.isStructSpecifier();
    }
}

TType::TType(const TStructure *userDef, bool isStructSpecifier)
    : type(EbtStruct),
      precision(EbpUndefined),
      qualifier(EvqTemporary),
      invariant(false),
      memoryQualifier(TMemoryQualifier::Create()),
      layoutQualifier(TLayoutQualifier::Create()),
      primarySize(1),
      secondarySize(1),
      mArraySizes(nullptr),
      mInterfaceBlock(nullptr),
      mStructure(userDef),
      mIsStructSpecifier(isStructSpecifier),
      mMangledName(nullptr)
{}

TType::TType(const TInterfaceBlock *interfaceBlockIn,
             TQualifier qualifierIn,
             TLayoutQualifier layoutQualifierIn)
    : type(EbtInterfaceBlock),
      precision(EbpUndefined),
      qualifier(qualifierIn),
      invariant(false),
      memoryQualifier(TMemoryQualifier::Create()),
      layoutQualifier(layoutQualifierIn),
      primarySize(1),
      secondarySize(1),
      mArraySizes(nullptr),
      mInterfaceBlock(interfaceBlockIn),
      mStructure(0),
      mIsStructSpecifier(false),
      mMangledName(nullptr)
{}

TType::TType(const TType &t)
    : type(t.type),
      precision(t.precision),
      qualifier(t.qualifier),
      invariant(t.invariant),
      memoryQualifier(t.memoryQualifier),
      layoutQualifier(t.layoutQualifier),
      primarySize(t.primarySize),
      secondarySize(t.secondarySize),
      mArraySizes(t.mArraySizes ? new TVector<unsigned int>(*t.mArraySizes) : nullptr),
      mInterfaceBlock(t.mInterfaceBlock),
      mStructure(t.mStructure),
      mIsStructSpecifier(t.mIsStructSpecifier),
      mMangledName(t.mMangledName)
{}

TType &TType::operator=(const TType &t)
{
    type               = t.type;
    precision          = t.precision;
    qualifier          = t.qualifier;
    invariant          = t.invariant;
    memoryQualifier    = t.memoryQualifier;
    layoutQualifier    = t.layoutQualifier;
    primarySize        = t.primarySize;
    secondarySize      = t.secondarySize;
    mArraySizes        = t.mArraySizes ? new TVector<unsigned int>(*t.mArraySizes) : nullptr;
    mInterfaceBlock    = t.mInterfaceBlock;
    mStructure         = t.mStructure;
    mIsStructSpecifier = t.mIsStructSpecifier;
    mMangledName       = t.mMangledName;
    return *this;
}

bool TType::canBeConstructed() const
{
    switch (type)
    {
        case EbtFloat:
        case EbtInt:
        case EbtUInt:
        case EbtBool:
        case EbtStruct:
            return true;
        default:
            return false;
    }
}

const char *TType::getBuiltInTypeNameString() const
{
    if (isMatrix())
    {
        switch (getCols())
        {
            case 2:
                switch (getRows())
                {
                    case 2:
                        return "mat2";
                    case 3:
                        return "mat2x3";
                    case 4:
                        return "mat2x4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            case 3:
                switch (getRows())
                {
                    case 2:
                        return "mat3x2";
                    case 3:
                        return "mat3";
                    case 4:
                        return "mat3x4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            case 4:
                switch (getRows())
                {
                    case 2:
                        return "mat4x2";
                    case 3:
                        return "mat4x3";
                    case 4:
                        return "mat4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            default:
                UNREACHABLE();
                return nullptr;
        }
    }
    if (isVector())
    {
        switch (getBasicType())
        {
            case EbtFloat:
                switch (getNominalSize())
                {
                    case 2:
                        return "vec2";
                    case 3:
                        return "vec3";
                    case 4:
                        return "vec4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            case EbtInt:
                switch (getNominalSize())
                {
                    case 2:
                        return "ivec2";
                    case 3:
                        return "ivec3";
                    case 4:
                        return "ivec4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            case EbtBool:
                switch (getNominalSize())
                {
                    case 2:
                        return "bvec2";
                    case 3:
                        return "bvec3";
                    case 4:
                        return "bvec4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            case EbtUInt:
                switch (getNominalSize())
                {
                    case 2:
                        return "uvec2";
                    case 3:
                        return "uvec3";
                    case 4:
                        return "uvec4";
                    default:
                        UNREACHABLE();
                        return nullptr;
                }
            default:
                UNREACHABLE();
                return nullptr;
        }
    }
    ASSERT(getBasicType() != EbtStruct);
    ASSERT(getBasicType() != EbtInterfaceBlock);
    return getBasicString();
}

int TType::getDeepestStructNesting() const
{
    return mStructure ? mStructure->deepestNesting() : 0;
}

bool TType::isNamelessStruct() const
{
    return mStructure && mStructure->symbolType() == SymbolType::Empty;
}

bool TType::isStructureContainingArrays() const
{
    return mStructure ? mStructure->containsArrays() : false;
}

bool TType::isStructureContainingMatrices() const
{
    return mStructure ? mStructure->containsMatrices() : false;
}

bool TType::isStructureContainingType(TBasicType t) const
{
    return mStructure ? mStructure->containsType(t) : false;
}

bool TType::isStructureContainingSamplers() const
{
    return mStructure ? mStructure->containsSamplers() : false;
}

bool TType::canReplaceWithConstantUnion() const
{
    if (isArray())
    {
        return false;
    }
    if (!mStructure)
    {
        return true;
    }
    if (isStructureContainingArrays())
    {
        return false;
    }
    if (getObjectSize() > 16)
    {
        return false;
    }
    return true;
}

//
// Recursively generate mangled names.
//
const char *TType::buildMangledName() const
{
    TString mangledName(1, GetSizeMangledName(primarySize, secondarySize));

    TBasicMangledName typeName(type);
    char *basicMangledName = typeName.getName();
    static_assert(TBasicMangledName::mangledNameSize == 2, "Mangled name size is not 2");
    if (basicMangledName[0] != '{')
    {
        mangledName += basicMangledName[0];
        mangledName += basicMangledName[1];
    }
    else
    {
        ASSERT(type == EbtStruct || type == EbtInterfaceBlock);
        switch (type)
        {
            case EbtStruct:
                mangledName += "{s";
                if (mStructure->symbolType() != SymbolType::Empty)
                {
                    mangledName += mStructure->name().data();
                }
                mangledName += mStructure->mangledFieldList();
                mangledName += '}';
                break;
            case EbtInterfaceBlock:
                mangledName += "{i";
                mangledName += mInterfaceBlock->name().data();
                mangledName += mInterfaceBlock->mangledFieldList();
                mangledName += '}';
                break;
            default:
                UNREACHABLE();
                break;
        }
    }

    if (mArraySizes)
    {
        for (unsigned int arraySize : *mArraySizes)
        {
            char buf[20];
            snprintf(buf, sizeof(buf), "%d", arraySize);
            mangledName += '[';
            mangledName += buf;
            mangledName += ']';
        }
    }

    // Copy string contents into a pool-allocated buffer, so we never need to call delete.
    return AllocatePoolCharArray(mangledName.c_str(), mangledName.size());
}

size_t TType::getObjectSize() const
{
    size_t totalSize;

    if (getBasicType() == EbtStruct)
        totalSize = mStructure->objectSize();
    else
        totalSize = primarySize * secondarySize;

    if (totalSize == 0)
        return 0;

    if (mArraySizes)
    {
        for (size_t arraySize : *mArraySizes)
        {
            if (arraySize > INT_MAX / totalSize)
                totalSize = INT_MAX;
            else
                totalSize *= arraySize;
        }
    }

    return totalSize;
}

int TType::getLocationCount() const
{
    int count = 1;

    if (getBasicType() == EbtStruct)
    {
        count = mStructure->getLocationCount();
    }

    if (count == 0)
    {
        return 0;
    }

    if (mArraySizes)
    {
        for (unsigned int arraySize : *mArraySizes)
        {
            if (arraySize > static_cast<unsigned int>(std::numeric_limits<int>::max() / count))
            {
                count = std::numeric_limits<int>::max();
            }
            else
            {
                count *= static_cast<int>(arraySize);
            }
        }
    }

    return count;
}

unsigned int TType::getArraySizeProduct() const
{
    if (!mArraySizes)
        return 1u;

    unsigned int product = 1u;

    for (unsigned int arraySize : *mArraySizes)
    {
        product *= arraySize;
    }
    return product;
}

bool TType::isUnsizedArray() const
{
    if (!mArraySizes)
        return false;

    for (unsigned int arraySize : *mArraySizes)
    {
        if (arraySize == 0u)
        {
            return true;
        }
    }
    return false;
}

bool TType::sameNonArrayType(const TType &right) const
{
    return (type == right.type && primarySize == right.primarySize &&
            secondarySize == right.secondarySize && mStructure == right.mStructure);
}

bool TType::isElementTypeOf(const TType &arrayType) const
{
    if (!sameNonArrayType(arrayType))
    {
        return false;
    }
    if (arrayType.getNumArraySizes() != getNumArraySizes() + 1u)
    {
        return false;
    }
    if (isArray())
    {
        for (size_t i = 0; i < mArraySizes->size(); ++i)
        {
            if ((*mArraySizes)[i] != (*arrayType.mArraySizes)[i])
            {
                return false;
            }
        }
    }
    return true;
}

void TType::sizeUnsizedArrays(const TVector<unsigned int> *newArraySizes)
{
    size_t newArraySizesSize = newArraySizes ? newArraySizes->size() : 0;
    for (size_t i = 0u; i < getNumArraySizes(); ++i)
    {
        if ((*mArraySizes)[i] == 0)
        {
            if (i < newArraySizesSize)
            {
                ASSERT(newArraySizes != nullptr);
                (*mArraySizes)[i] = (*newArraySizes)[i];
            }
            else
            {
                (*mArraySizes)[i] = 1u;
            }
        }
    }
    invalidateMangledName();
}

void TType::sizeOutermostUnsizedArray(unsigned int arraySize)
{
    ASSERT(isArray());
    ASSERT(mArraySizes->back() == 0u);
    mArraySizes->back() = arraySize;
}

void TType::setBasicType(TBasicType t)
{
    if (type != t)
    {
        type = t;
        invalidateMangledName();
    }
}

void TType::setPrimarySize(unsigned char ps)
{
    if (primarySize != ps)
    {
        ASSERT(ps <= 4);
        primarySize = ps;
        invalidateMangledName();
    }
}

void TType::setSecondarySize(unsigned char ss)
{
    if (secondarySize != ss)
    {
        ASSERT(ss <= 4);
        secondarySize = ss;
        invalidateMangledName();
    }
}

void TType::makeArray(unsigned int s)
{
    if (!mArraySizes)
        mArraySizes = new TVector<unsigned int>();

    mArraySizes->push_back(s);
    invalidateMangledName();
}

void TType::makeArrays(const TVector<unsigned int> &sizes)
{
    if (!mArraySizes)
        mArraySizes = new TVector<unsigned int>();

    mArraySizes->insert(mArraySizes->end(), sizes.begin(), sizes.end());
    invalidateMangledName();
}

void TType::setArraySize(size_t arrayDimension, unsigned int s)
{
    ASSERT(mArraySizes != nullptr);
    ASSERT(arrayDimension < mArraySizes->size());
    if (mArraySizes->at(arrayDimension) != s)
    {
        (*mArraySizes)[arrayDimension] = s;
        invalidateMangledName();
    }
}

void TType::toArrayElementType()
{
    ASSERT(mArraySizes != nullptr);
    if (mArraySizes->size() > 0)
    {
        mArraySizes->pop_back();
        invalidateMangledName();
    }
}

void TType::toArrayBaseType()
{
    if (mArraySizes == nullptr)
    {
        return;
    }
    if (mArraySizes->size() > 0)
    {
        mArraySizes->clear();
    }
    invalidateMangledName();
}

void TType::setInterfaceBlock(const TInterfaceBlock *interfaceBlockIn)
{
    if (mInterfaceBlock != interfaceBlockIn)
    {
        mInterfaceBlock = interfaceBlockIn;
        invalidateMangledName();
    }
}

const char *TType::getMangledName() const
{
    if (mMangledName == nullptr)
    {
        mMangledName = buildMangledName();
    }

    return mMangledName;
}

void TType::realize()
{
    getMangledName();
}

void TType::invalidateMangledName()
{
    mMangledName = nullptr;
}

void TType::createSamplerSymbols(const ImmutableString &namePrefix,
                                 const TString &apiNamePrefix,
                                 TVector<const TVariable *> *outputSymbols,
                                 TMap<const TVariable *, TString> *outputSymbolsToAPINames,
                                 TSymbolTable *symbolTable) const
{
    if (isStructureContainingSamplers())
    {
        if (isArray())
        {
            TType elementType(*this);
            elementType.toArrayElementType();
            for (unsigned int arrayIndex = 0u; arrayIndex < getOutermostArraySize(); ++arrayIndex)
            {
                std::stringstream elementName = sh::InitializeStream<std::stringstream>();
                elementName << namePrefix << "_" << arrayIndex;
                TStringStream elementApiName;
                elementApiName << apiNamePrefix << "[" << arrayIndex << "]";
                elementType.createSamplerSymbols(ImmutableString(elementName.str()),
                                                 elementApiName.str(), outputSymbols,
                                                 outputSymbolsToAPINames, symbolTable);
            }
        }
        else
        {
            mStructure->createSamplerSymbols(namePrefix.data(), apiNamePrefix, outputSymbols,
                                             outputSymbolsToAPINames, symbolTable);
        }
        return;
    }

    ASSERT(IsSampler(type));
    TVariable *variable =
        new TVariable(symbolTable, namePrefix, new TType(*this), SymbolType::AngleInternal);
    outputSymbols->push_back(variable);
    if (outputSymbolsToAPINames)
    {
        (*outputSymbolsToAPINames)[variable] = apiNamePrefix;
    }
}

TFieldListCollection::TFieldListCollection(const TFieldList *fields)
    : mFields(fields), mObjectSize(0), mDeepestNesting(0)
{}

bool TFieldListCollection::containsArrays() const
{
    for (const auto *field : *mFields)
    {
        const TType *fieldType = field->type();
        if (fieldType->isArray() || fieldType->isStructureContainingArrays())
            return true;
    }
    return false;
}

bool TFieldListCollection::containsMatrices() const
{
    for (const auto *field : *mFields)
    {
        const TType *fieldType = field->type();
        if (fieldType->isMatrix() || fieldType->isStructureContainingMatrices())
            return true;
    }
    return false;
}

bool TFieldListCollection::containsType(TBasicType type) const
{
    for (const auto *field : *mFields)
    {
        const TType *fieldType = field->type();
        if (fieldType->getBasicType() == type || fieldType->isStructureContainingType(type))
            return true;
    }
    return false;
}

bool TFieldListCollection::containsSamplers() const
{
    for (const auto *field : *mFields)
    {
        const TType *fieldType = field->type();
        if (IsSampler(fieldType->getBasicType()) || fieldType->isStructureContainingSamplers())
            return true;
    }
    return false;
}

TString TFieldListCollection::buildMangledFieldList() const
{
    TString mangledName;
    for (const auto *field : *mFields)
    {
        mangledName += field->type()->getMangledName();
    }
    return mangledName;
}

size_t TFieldListCollection::calculateObjectSize() const
{
    size_t size = 0;
    for (const TField *field : *mFields)
    {
        size_t fieldSize = field->type()->getObjectSize();
        if (fieldSize > INT_MAX - size)
            size = INT_MAX;
        else
            size += fieldSize;
    }
    return size;
}

size_t TFieldListCollection::objectSize() const
{
    if (mObjectSize == 0)
        mObjectSize = calculateObjectSize();
    return mObjectSize;
}

int TFieldListCollection::getLocationCount() const
{
    int count = 0;
    for (const TField *field : *mFields)
    {
        int fieldCount = field->type()->getLocationCount();
        if (fieldCount > std::numeric_limits<int>::max() - count)
        {
            count = std::numeric_limits<int>::max();
        }
        else
        {
            count += fieldCount;
        }
    }
    return count;
}

int TFieldListCollection::deepestNesting() const
{
    if (mDeepestNesting == 0)
        mDeepestNesting = calculateDeepestNesting();
    return mDeepestNesting;
}

const TString &TFieldListCollection::mangledFieldList() const
{
    if (mMangledFieldList.empty())
        mMangledFieldList = buildMangledFieldList();
    return mMangledFieldList;
}

int TFieldListCollection::calculateDeepestNesting() const
{
    int maxNesting = 0;
    for (size_t i = 0; i < mFields->size(); ++i)
        maxNesting = std::max(maxNesting, (*mFields)[i]->type()->getDeepestStructNesting());
    return 1 + maxNesting;
}

// TPublicType implementation.
void TPublicType::initialize(const TTypeSpecifierNonArray &typeSpecifier, TQualifier q)
{
    typeSpecifierNonArray = typeSpecifier;
    layoutQualifier       = TLayoutQualifier::Create();
    memoryQualifier       = TMemoryQualifier::Create();
    qualifier             = q;
    invariant             = false;
    precision             = EbpUndefined;
    arraySizes            = nullptr;
}

void TPublicType::initializeBasicType(TBasicType basicType)
{
    typeSpecifierNonArray.type          = basicType;
    typeSpecifierNonArray.primarySize   = 1;
    typeSpecifierNonArray.secondarySize = 1;
    layoutQualifier                     = TLayoutQualifier::Create();
    memoryQualifier                     = TMemoryQualifier::Create();
    qualifier                           = EvqTemporary;
    invariant                           = false;
    precision                           = EbpUndefined;
    arraySizes                          = nullptr;
}

bool TPublicType::isStructureContainingArrays() const
{
    if (!typeSpecifierNonArray.userDef)
    {
        return false;
    }

    return typeSpecifierNonArray.userDef->containsArrays();
}

bool TPublicType::isStructureContainingType(TBasicType t) const
{
    if (!typeSpecifierNonArray.userDef)
    {
        return false;
    }

    return typeSpecifierNonArray.userDef->containsType(t);
}

void TPublicType::setArraySizes(TVector<unsigned int> *sizes)
{
    arraySizes = sizes;
}

bool TPublicType::isArray() const
{
    return arraySizes && !arraySizes->empty();
}

void TPublicType::clearArrayness()
{
    arraySizes = nullptr;
}

bool TPublicType::isAggregate() const
{
    return isArray() || typeSpecifierNonArray.isMatrix() || typeSpecifierNonArray.isVector();
}

}  // namespace sh
