//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// blocklayout.cpp:
//   Implementation for block layout classes and methods.
//

#include "compiler/translator/blocklayout.h"

#include "common/mathutil.h"
#include "common/utilities.h"
#include "compiler/translator/Common.h"

namespace sh
{

namespace
{
class BlockLayoutMapVisitor : public BlockEncoderVisitor
{
  public:
    BlockLayoutMapVisitor(BlockLayoutMap *blockInfoOut,
                          const std::string &instanceName,
                          BlockLayoutEncoder *encoder)
        : BlockEncoderVisitor(instanceName, instanceName, encoder), mInfoOut(blockInfoOut)
    {}

    void encodeVariable(const ShaderVariable &variable,
                        const BlockMemberInfo &variableInfo,
                        const std::string &name,
                        const std::string &mappedName) override
    {
        ASSERT(!gl::IsSamplerType(variable.type));
        if (!gl::IsOpaqueType(variable.type))
        {
            (*mInfoOut)[name] = variableInfo;
        }
    }

  private:
    BlockLayoutMap *mInfoOut;
};

template <typename VarT>
void GetInterfaceBlockInfo(const std::vector<VarT> &fields,
                           const std::string &prefix,
                           BlockLayoutEncoder *encoder,
                           bool inRowMajorLayout,
                           BlockLayoutMap *blockInfoOut)
{
    BlockLayoutMapVisitor visitor(blockInfoOut, prefix, encoder);
    TraverseShaderVariables(fields, inRowMajorLayout, &visitor);
}

void TraverseStructVariable(const ShaderVariable &variable,
                            bool isRowMajorLayout,
                            ShaderVariableVisitor *visitor)
{
    const std::vector<ShaderVariable> &fields = variable.fields;

    visitor->enterStructAccess(variable, isRowMajorLayout);
    TraverseShaderVariables(fields, isRowMajorLayout, visitor);
    visitor->exitStructAccess(variable, isRowMajorLayout);
}

void TraverseStructArrayVariable(const ShaderVariable &variable,
                                 bool inRowMajorLayout,
                                 ShaderVariableVisitor *visitor)
{
    visitor->enterArray(variable);

    // Nested arrays are processed starting from outermost (arrayNestingIndex 0u) and ending at the
    // innermost. We make a special case for unsized arrays.
    const unsigned int currentArraySize = variable.getNestedArraySize(0);
    unsigned int count                  = std::max(currentArraySize, 1u);
    for (unsigned int arrayElement = 0u; arrayElement < count; ++arrayElement)
    {
        visitor->enterArrayElement(variable, arrayElement);
        ShaderVariable elementVar = variable;
        elementVar.indexIntoArray(arrayElement);

        if (variable.arraySizes.size() > 1u)
        {
            TraverseStructArrayVariable(elementVar, inRowMajorLayout, visitor);
        }
        else
        {
            TraverseStructVariable(elementVar, inRowMajorLayout, visitor);
        }

        visitor->exitArrayElement(variable, arrayElement);
    }

    visitor->exitArray(variable);
}

void TraverseArrayOfArraysVariable(const ShaderVariable &variable,
                                   unsigned int arrayNestingIndex,
                                   bool isRowMajorMatrix,
                                   ShaderVariableVisitor *visitor)
{
    visitor->enterArray(variable);

    const unsigned int currentArraySize = variable.getNestedArraySize(arrayNestingIndex);
    unsigned int count                  = std::max(currentArraySize, 1u);
    for (unsigned int arrayElement = 0u; arrayElement < count; ++arrayElement)
    {
        visitor->enterArrayElement(variable, arrayElement);

        ShaderVariable elementVar = variable;
        elementVar.indexIntoArray(arrayElement);

        if (arrayNestingIndex + 2u < variable.arraySizes.size())
        {
            TraverseArrayOfArraysVariable(elementVar, arrayNestingIndex, isRowMajorMatrix, visitor);
        }
        else
        {
            if (gl::IsSamplerType(variable.type))
            {
                visitor->visitSampler(elementVar);
            }
            else
            {
                visitor->visitVariable(elementVar, isRowMajorMatrix);
            }
        }

        visitor->exitArrayElement(variable, arrayElement);
    }

    visitor->exitArray(variable);
}

std::string CollapseNameStack(const std::vector<std::string> &nameStack)
{
    std::stringstream strstr = sh::InitializeStream<std::stringstream>();
    for (const std::string &part : nameStack)
    {
        strstr << part;
    }
    return strstr.str();
}

size_t GetStd430BaseAlignment(GLenum variableType, bool isRowMajor)
{
    GLenum flippedType   = isRowMajor ? variableType : gl::TransposeMatrixType(variableType);
    size_t numComponents = static_cast<size_t>(gl::VariableColumnCount(flippedType));
    return ComponentAlignment(numComponents);
}

class BaseAlignmentVisitor : public ShaderVariableVisitor
{
  public:
    BaseAlignmentVisitor() = default;
    void visitVariable(const ShaderVariable &variable, bool isRowMajor) override
    {
        size_t baseAlignment = GetStd430BaseAlignment(variable.type, isRowMajor);
        mCurrentAlignment    = std::max(mCurrentAlignment, baseAlignment);
    }

    // This is in components rather than bytes.
    size_t getBaseAlignment() const { return mCurrentAlignment; }

  private:
    size_t mCurrentAlignment = 0;
};
}  // anonymous namespace

// BlockLayoutEncoder implementation.
BlockLayoutEncoder::BlockLayoutEncoder() : mCurrentOffset(0) {}

BlockMemberInfo BlockLayoutEncoder::encodeType(GLenum type,
                                               const std::vector<unsigned int> &arraySizes,
                                               bool isRowMajorMatrix)
{
    int arrayStride;
    int matrixStride;

    getBlockLayoutInfo(type, arraySizes, isRowMajorMatrix, &arrayStride, &matrixStride);

    const BlockMemberInfo memberInfo(static_cast<int>(mCurrentOffset * kBytesPerComponent),
                                     static_cast<int>(arrayStride * kBytesPerComponent),
                                     static_cast<int>(matrixStride * kBytesPerComponent),
                                     isRowMajorMatrix);

    advanceOffset(type, arraySizes, isRowMajorMatrix, arrayStride, matrixStride);

    return memberInfo;
}

size_t BlockLayoutEncoder::getShaderVariableSize(const ShaderVariable &structVar, bool isRowMajor)
{
    size_t currentOffset = mCurrentOffset;
    mCurrentOffset       = 0;
    BlockEncoderVisitor visitor("", "", this);
    enterAggregateType(structVar);
    TraverseShaderVariables(structVar.fields, isRowMajor, &visitor);
    exitAggregateType(structVar);
    size_t structVarSize = getCurrentOffset();
    mCurrentOffset       = currentOffset;
    return structVarSize;
}

// static
size_t BlockLayoutEncoder::GetBlockRegister(const BlockMemberInfo &info)
{
    return (info.offset / kBytesPerComponent) / kComponentsPerRegister;
}

// static
size_t BlockLayoutEncoder::GetBlockRegisterElement(const BlockMemberInfo &info)
{
    return (info.offset / kBytesPerComponent) % kComponentsPerRegister;
}

void BlockLayoutEncoder::align(size_t baseAlignment)
{
    mCurrentOffset = rx::roundUp<size_t>(mCurrentOffset, baseAlignment);
}

// DummyBlockEncoder implementation.
void DummyBlockEncoder::getBlockLayoutInfo(GLenum type,
                                           const std::vector<unsigned int> &arraySizes,
                                           bool isRowMajorMatrix,
                                           int *arrayStrideOut,
                                           int *matrixStrideOut)
{
    *arrayStrideOut  = 0;
    *matrixStrideOut = 0;
}

// Std140BlockEncoder implementation.
Std140BlockEncoder::Std140BlockEncoder() {}

void Std140BlockEncoder::enterAggregateType(const ShaderVariable &structVar)
{
    align(getBaseAlignment(structVar));
}

void Std140BlockEncoder::exitAggregateType(const ShaderVariable &structVar)
{
    align(getBaseAlignment(structVar));
}

void Std140BlockEncoder::getBlockLayoutInfo(GLenum type,
                                            const std::vector<unsigned int> &arraySizes,
                                            bool isRowMajorMatrix,
                                            int *arrayStrideOut,
                                            int *matrixStrideOut)
{
    // We assume we are only dealing with 4 byte components (no doubles or half-words currently)
    ASSERT(gl::VariableComponentSize(gl::VariableComponentType(type)) == kBytesPerComponent);

    size_t baseAlignment = 0;
    int matrixStride     = 0;
    int arrayStride      = 0;

    if (gl::IsMatrixType(type))
    {
        baseAlignment = getTypeBaseAlignment(type, isRowMajorMatrix);
        matrixStride  = static_cast<int>(getTypeBaseAlignment(type, isRowMajorMatrix));

        if (!arraySizes.empty())
        {
            const int numRegisters = gl::MatrixRegisterCount(type, isRowMajorMatrix);
            arrayStride =
                static_cast<int>(getTypeBaseAlignment(type, isRowMajorMatrix) * numRegisters);
        }
    }
    else if (!arraySizes.empty())
    {
        baseAlignment = static_cast<int>(getTypeBaseAlignment(type, false));
        arrayStride   = static_cast<int>(getTypeBaseAlignment(type, false));
    }
    else
    {
        const size_t numComponents = static_cast<size_t>(gl::VariableComponentCount(type));
        baseAlignment              = ComponentAlignment(numComponents);
    }

    mCurrentOffset = rx::roundUp(mCurrentOffset, baseAlignment);

    *matrixStrideOut = matrixStride;
    *arrayStrideOut  = arrayStride;
}

void Std140BlockEncoder::advanceOffset(GLenum type,
                                       const std::vector<unsigned int> &arraySizes,
                                       bool isRowMajorMatrix,
                                       int arrayStride,
                                       int matrixStride)
{
    if (!arraySizes.empty())
    {
        mCurrentOffset += arrayStride * gl::ArraySizeProduct(arraySizes);
    }
    else if (gl::IsMatrixType(type))
    {
        const int numRegisters = gl::MatrixRegisterCount(type, isRowMajorMatrix);
        mCurrentOffset += matrixStride * numRegisters;
    }
    else
    {
        mCurrentOffset += gl::VariableComponentCount(type);
    }
}

size_t Std140BlockEncoder::getBaseAlignment(const ShaderVariable &variable) const
{
    return kComponentsPerRegister;
}

size_t Std140BlockEncoder::getTypeBaseAlignment(GLenum type, bool isRowMajorMatrix) const
{
    return kComponentsPerRegister;
}

// Std430BlockEncoder implementation.
Std430BlockEncoder::Std430BlockEncoder() {}

size_t Std430BlockEncoder::getBaseAlignment(const ShaderVariable &shaderVar) const
{
    if (shaderVar.isStruct())
    {
        BaseAlignmentVisitor visitor;
        TraverseShaderVariables(shaderVar.fields, false, &visitor);
        return visitor.getBaseAlignment();
    }

    return GetStd430BaseAlignment(shaderVar.type, shaderVar.isRowMajorLayout);
}

size_t Std430BlockEncoder::getTypeBaseAlignment(GLenum type, bool isRowMajorMatrix) const
{
    return GetStd430BaseAlignment(type, isRowMajorMatrix);
}

void GetInterfaceBlockInfo(const std::vector<ShaderVariable> &fields,
                           const std::string &prefix,
                           BlockLayoutEncoder *encoder,
                           BlockLayoutMap *blockInfoOut)
{
    // Matrix packing is always recorded in individual fields, so they'll set the row major layout
    // flag to true if needed.
    GetInterfaceBlockInfo(fields, prefix, encoder, false, blockInfoOut);
}

void GetUniformBlockInfo(const std::vector<ShaderVariable> &uniforms,
                         const std::string &prefix,
                         BlockLayoutEncoder *encoder,
                         BlockLayoutMap *blockInfoOut)
{
    // Matrix packing is always recorded in individual fields, so they'll set the row major layout
    // flag to true if needed.
    GetInterfaceBlockInfo(uniforms, prefix, encoder, false, blockInfoOut);
}

// VariableNameVisitor implementation.
VariableNameVisitor::VariableNameVisitor(const std::string &namePrefix,
                                         const std::string &mappedNamePrefix)
{
    if (!namePrefix.empty())
    {
        mNameStack.push_back(namePrefix + ".");
    }

    if (!mappedNamePrefix.empty())
    {
        mMappedNameStack.push_back(mappedNamePrefix + ".");
    }
}

VariableNameVisitor::~VariableNameVisitor() = default;

void VariableNameVisitor::enterStruct(const ShaderVariable &structVar)
{
    mNameStack.push_back(structVar.name);
    mMappedNameStack.push_back(structVar.mappedName);
}

void VariableNameVisitor::exitStruct(const ShaderVariable &structVar)
{
    mNameStack.pop_back();
    mMappedNameStack.pop_back();
}

void VariableNameVisitor::enterStructAccess(const ShaderVariable &structVar, bool isRowMajor)
{
    mNameStack.push_back(".");
    mMappedNameStack.push_back(".");
}

void VariableNameVisitor::exitStructAccess(const ShaderVariable &structVar, bool isRowMajor)
{
    mNameStack.pop_back();
    mMappedNameStack.pop_back();
}

void VariableNameVisitor::enterArray(const ShaderVariable &arrayVar)
{
    if (!arrayVar.hasParentArrayIndex() && !arrayVar.isStruct())
    {
        mNameStack.push_back(arrayVar.name);
        mMappedNameStack.push_back(arrayVar.mappedName);
    }
    mArraySizeStack.push_back(arrayVar.getOutermostArraySize());
}

void VariableNameVisitor::exitArray(const ShaderVariable &arrayVar)
{
    if (!arrayVar.hasParentArrayIndex() && !arrayVar.isStruct())
    {
        mNameStack.pop_back();
        mMappedNameStack.pop_back();
    }
    mArraySizeStack.pop_back();
}

void VariableNameVisitor::enterArrayElement(const ShaderVariable &arrayVar,
                                            unsigned int arrayElement)
{
    std::stringstream strstr = sh::InitializeStream<std::stringstream>();
    strstr << "[" << arrayElement << "]";
    std::string elementString = strstr.str();
    mNameStack.push_back(elementString);
    mMappedNameStack.push_back(elementString);
}

void VariableNameVisitor::exitArrayElement(const ShaderVariable &arrayVar,
                                           unsigned int arrayElement)
{
    mNameStack.pop_back();
    mMappedNameStack.pop_back();
}

std::string VariableNameVisitor::collapseNameStack() const
{
    return CollapseNameStack(mNameStack);
}

std::string VariableNameVisitor::collapseMappedNameStack() const
{
    return CollapseNameStack(mMappedNameStack);
}

void VariableNameVisitor::visitSampler(const sh::ShaderVariable &sampler)
{
    if (!sampler.hasParentArrayIndex())
    {
        mNameStack.push_back(sampler.name);
        mMappedNameStack.push_back(sampler.mappedName);
    }

    std::string name       = collapseNameStack();
    std::string mappedName = collapseMappedNameStack();

    if (!sampler.hasParentArrayIndex())
    {
        mNameStack.pop_back();
        mMappedNameStack.pop_back();
    }

    visitNamedSampler(sampler, name, mappedName, mArraySizeStack);
}

void VariableNameVisitor::visitVariable(const ShaderVariable &variable, bool isRowMajor)
{
    if (!variable.hasParentArrayIndex())
    {
        mNameStack.push_back(variable.name);
        mMappedNameStack.push_back(variable.mappedName);
    }

    std::string name       = collapseNameStack();
    std::string mappedName = collapseMappedNameStack();

    if (!variable.hasParentArrayIndex())
    {
        mNameStack.pop_back();
        mMappedNameStack.pop_back();
    }

    visitNamedVariable(variable, isRowMajor, name, mappedName, mArraySizeStack);
}

// BlockEncoderVisitor implementation.
BlockEncoderVisitor::BlockEncoderVisitor(const std::string &namePrefix,
                                         const std::string &mappedNamePrefix,
                                         BlockLayoutEncoder *encoder)
    : VariableNameVisitor(namePrefix, mappedNamePrefix), mEncoder(encoder)
{}

BlockEncoderVisitor::~BlockEncoderVisitor() = default;

void BlockEncoderVisitor::enterStructAccess(const ShaderVariable &structVar, bool isRowMajor)
{
    mStructStackSize++;
    if (!mIsTopLevelArrayStrideReady)
    {
        size_t structSize = mEncoder->getShaderVariableSize(structVar, isRowMajor);
        mTopLevelArrayStride *= structSize;
        mIsTopLevelArrayStrideReady = true;
    }

    VariableNameVisitor::enterStructAccess(structVar, isRowMajor);
    mEncoder->enterAggregateType(structVar);
}

void BlockEncoderVisitor::exitStructAccess(const ShaderVariable &structVar, bool isRowMajor)
{
    mStructStackSize--;
    mEncoder->exitAggregateType(structVar);
    VariableNameVisitor::exitStructAccess(structVar, isRowMajor);
}

void BlockEncoderVisitor::enterArrayElement(const sh::ShaderVariable &arrayVar,
                                            unsigned int arrayElement)
{
    if (mStructStackSize == 0 && !arrayVar.hasParentArrayIndex())
    {
        // From the ES 3.1 spec "7.3.1.1 Naming Active Resources":
        // For an active shader storage block member declared as an array of an aggregate type,
        // an entry will be generated only for the first array element, regardless of its type.
        // Such block members are referred to as top-level arrays. If the block member is an
        // aggregate type, the enumeration rules are then applied recursively.
        if (arrayElement == 0)
        {
            mTopLevelArraySize          = arrayVar.getOutermostArraySize();
            mTopLevelArrayStride        = arrayVar.getInnerArraySizeProduct();
            mIsTopLevelArrayStrideReady = false;
        }
        else
        {
            mSkipEnabled = true;
        }
    }
    VariableNameVisitor::enterArrayElement(arrayVar, arrayElement);
}

void BlockEncoderVisitor::exitArrayElement(const sh::ShaderVariable &arrayVar,
                                           unsigned int arrayElement)
{
    if (mStructStackSize == 0 && !arrayVar.hasParentArrayIndex())
    {
        mTopLevelArraySize          = 1;
        mTopLevelArrayStride        = 0;
        mIsTopLevelArrayStrideReady = true;
        mSkipEnabled                = false;
    }
    VariableNameVisitor::exitArrayElement(arrayVar, arrayElement);
}

void BlockEncoderVisitor::visitNamedVariable(const ShaderVariable &variable,
                                             bool isRowMajor,
                                             const std::string &name,
                                             const std::string &mappedName,
                                             const std::vector<unsigned int> &arraySizes)
{
    std::vector<unsigned int> innermostArraySize;

    if (variable.isArray())
    {
        innermostArraySize.push_back(variable.getNestedArraySize(0));
    }
    BlockMemberInfo variableInfo =
        mEncoder->encodeType(variable.type, innermostArraySize, isRowMajor);
    if (!mIsTopLevelArrayStrideReady)
    {
        ASSERT(mTopLevelArrayStride);
        mTopLevelArrayStride *= variableInfo.arrayStride;
        mIsTopLevelArrayStrideReady = true;
    }
    variableInfo.topLevelArrayStride = mTopLevelArrayStride;
    encodeVariable(variable, variableInfo, name, mappedName);
}

void TraverseShaderVariable(const ShaderVariable &variable,
                            bool isRowMajorLayout,
                            ShaderVariableVisitor *visitor)
{
    bool rowMajorLayout = (isRowMajorLayout || variable.isRowMajorLayout);
    bool isRowMajor     = rowMajorLayout && gl::IsMatrixType(variable.type);

    if (variable.isStruct())
    {
        visitor->enterStruct(variable);
        if (variable.isArray())
        {
            TraverseStructArrayVariable(variable, rowMajorLayout, visitor);
        }
        else
        {
            TraverseStructVariable(variable, rowMajorLayout, visitor);
        }
        visitor->exitStruct(variable);
    }
    else if (variable.isArrayOfArrays())
    {
        TraverseArrayOfArraysVariable(variable, 0u, isRowMajor, visitor);
    }
    else if (gl::IsSamplerType(variable.type))
    {
        visitor->visitSampler(variable);
    }
    else
    {
        visitor->visitVariable(variable, isRowMajor);
    }
}
}  // namespace sh
