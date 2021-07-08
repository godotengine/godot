//
// Copyright 2010 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "libANGLE/Uniform.h"

#include <cstring>

namespace gl
{

ActiveVariable::ActiveVariable() {}

ActiveVariable::~ActiveVariable() {}

ActiveVariable::ActiveVariable(const ActiveVariable &rhs) = default;
ActiveVariable &ActiveVariable::operator=(const ActiveVariable &rhs) = default;

void ActiveVariable::setActive(ShaderType shaderType, bool used)
{
    ASSERT(shaderType != ShaderType::InvalidEnum);
    mActiveUseBits.set(shaderType, used);
}

void ActiveVariable::unionReferencesWith(const ActiveVariable &other)
{
    mActiveUseBits |= other.mActiveUseBits;
}

ShaderType ActiveVariable::getFirstShaderTypeWhereActive() const
{
    return static_cast<ShaderType>(ScanForward(mActiveUseBits.bits()));
}

GLuint ActiveVariable::activeShaderCount() const
{
    return static_cast<GLuint>(mActiveUseBits.count());
}

LinkedUniform::LinkedUniform()
    : typeInfo(nullptr), bufferIndex(-1), blockInfo(sh::kDefaultBlockMemberInfo)
{}

LinkedUniform::LinkedUniform(GLenum typeIn,
                             GLenum precisionIn,
                             const std::string &nameIn,
                             const std::vector<unsigned int> &arraySizesIn,
                             const int bindingIn,
                             const int offsetIn,
                             const int locationIn,
                             const int bufferIndexIn,
                             const sh::BlockMemberInfo &blockInfoIn)
    : typeInfo(&GetUniformTypeInfo(typeIn)), bufferIndex(bufferIndexIn), blockInfo(blockInfoIn)
{
    type       = typeIn;
    precision  = precisionIn;
    name       = nameIn;
    arraySizes = arraySizesIn;
    binding    = bindingIn;
    offset     = offsetIn;
    location   = locationIn;
    ASSERT(!isArrayOfArrays());
    ASSERT(!isArray() || !isStruct());
}

LinkedUniform::LinkedUniform(const sh::ShaderVariable &uniform)
    : sh::ShaderVariable(uniform),
      typeInfo(&GetUniformTypeInfo(type)),
      bufferIndex(-1),
      blockInfo(sh::kDefaultBlockMemberInfo)
{
    ASSERT(!isArrayOfArrays());
    ASSERT(!isArray() || !isStruct());
}

LinkedUniform::LinkedUniform(const LinkedUniform &uniform)
    : sh::ShaderVariable(uniform),
      ActiveVariable(uniform),
      typeInfo(uniform.typeInfo),
      bufferIndex(uniform.bufferIndex),
      blockInfo(uniform.blockInfo),
      outerArraySizes(uniform.outerArraySizes)
{}

LinkedUniform &LinkedUniform::operator=(const LinkedUniform &uniform)
{
    sh::ShaderVariable::operator=(uniform);
    ActiveVariable::operator=(uniform);
    typeInfo                = uniform.typeInfo;
    bufferIndex             = uniform.bufferIndex;
    blockInfo               = uniform.blockInfo;
    outerArraySizes         = uniform.outerArraySizes;
    return *this;
}

LinkedUniform::~LinkedUniform() {}

BufferVariable::BufferVariable()
    : bufferIndex(-1), blockInfo(sh::kDefaultBlockMemberInfo), topLevelArraySize(-1)
{}

BufferVariable::BufferVariable(GLenum typeIn,
                               GLenum precisionIn,
                               const std::string &nameIn,
                               const std::vector<unsigned int> &arraySizesIn,
                               const int bufferIndexIn,
                               const sh::BlockMemberInfo &blockInfoIn)
    : bufferIndex(bufferIndexIn), blockInfo(blockInfoIn), topLevelArraySize(-1)
{
    type       = typeIn;
    precision  = precisionIn;
    name       = nameIn;
    arraySizes = arraySizesIn;
}

BufferVariable::~BufferVariable() {}

ShaderVariableBuffer::ShaderVariableBuffer() : binding(0), dataSize(0) {}

ShaderVariableBuffer::ShaderVariableBuffer(const ShaderVariableBuffer &other) = default;

ShaderVariableBuffer::~ShaderVariableBuffer() {}

int ShaderVariableBuffer::numActiveVariables() const
{
    return static_cast<int>(memberIndexes.size());
}

InterfaceBlock::InterfaceBlock() : isArray(false), arrayElement(0) {}

InterfaceBlock::InterfaceBlock(const std::string &nameIn,
                               const std::string &mappedNameIn,
                               bool isArrayIn,
                               unsigned int arrayElementIn,
                               int bindingIn)
    : name(nameIn), mappedName(mappedNameIn), isArray(isArrayIn), arrayElement(arrayElementIn)
{
    binding = bindingIn;
}

std::string InterfaceBlock::nameWithArrayIndex() const
{
    std::stringstream fullNameStr;
    fullNameStr << name;
    if (isArray)
    {
        fullNameStr << "[" << arrayElement << "]";
    }

    return fullNameStr.str();
}

std::string InterfaceBlock::mappedNameWithArrayIndex() const
{
    std::stringstream fullNameStr;
    fullNameStr << mappedName;
    if (isArray)
    {
        fullNameStr << "[" << arrayElement << "]";
    }

    return fullNameStr.str();
}
}  // namespace gl
