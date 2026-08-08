//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLArgument.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDataType.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTensor.hpp"
#include "MTLTexture.hpp"

namespace MTL
{
class Argument;
class ArrayType;
class PointerType;
class StructMember;
class StructType;
class TensorExtents;
class TensorReferenceType;
class TextureReferenceType;
class Type;
_MTL_ENUM(NS::UInteger, IndexType) {
    IndexTypeUInt16 = 0,
    IndexTypeUInt32 = 1,
};

_MTL_ENUM(NS::Integer, BindingType) {
    BindingTypeBuffer = 0,
    BindingTypeThreadgroupMemory = 1,
    BindingTypeTexture = 2,
    BindingTypeSampler = 3,
    BindingTypeImageblockData = 16,
    BindingTypeImageblock = 17,
    BindingTypeVisibleFunctionTable = 24,
    BindingTypePrimitiveAccelerationStructure = 25,
    BindingTypeInstanceAccelerationStructure = 26,
    BindingTypeIntersectionFunctionTable = 27,
    BindingTypeObjectPayload = 34,
    BindingTypeTensor = 37,
};

_MTL_ENUM(NS::UInteger, ArgumentType) {
    ArgumentTypeBuffer = 0,
    ArgumentTypeThreadgroupMemory = 1,
    ArgumentTypeTexture = 2,
    ArgumentTypeSampler = 3,
    ArgumentTypeImageblockData = 16,
    ArgumentTypeImageblock = 17,
    ArgumentTypeVisibleFunctionTable = 24,
    ArgumentTypePrimitiveAccelerationStructure = 25,
    ArgumentTypeInstanceAccelerationStructure = 26,
    ArgumentTypeIntersectionFunctionTable = 27,
};

_MTL_ENUM(NS::UInteger, BindingAccess) {
    BindingAccessReadOnly = 0,
    BindingAccessReadWrite = 1,
    BindingAccessWriteOnly = 2,
    ArgumentAccessReadOnly = 0,
    ArgumentAccessReadWrite = 1,
    ArgumentAccessWriteOnly = 2,
};

class Type : public NS::Referencing<Type>
{
public:
    static Type* alloc();

    DataType     dataType() const;

    Type*        init();
};
class StructMember : public NS::Referencing<StructMember>
{
public:
    static StructMember*  alloc();

    NS::UInteger          argumentIndex() const;

    ArrayType*            arrayType();

    DataType              dataType() const;

    StructMember*         init();

    NS::String*           name() const;

    NS::UInteger          offset() const;

    PointerType*          pointerType();

    StructType*           structType();

    TensorReferenceType*  tensorReferenceType();

    TextureReferenceType* textureReferenceType();
};
class StructType : public NS::Referencing<StructType, Type>
{
public:
    static StructType* alloc();

    StructType*        init();

    StructMember*      memberByName(const NS::String* name);

    NS::Array*         members() const;
};
class ArrayType : public NS::Referencing<ArrayType, Type>
{
public:
    static ArrayType*     alloc();

    NS::UInteger          argumentIndexStride() const;

    NS::UInteger          arrayLength() const;

    ArrayType*            elementArrayType();

    PointerType*          elementPointerType();

    StructType*           elementStructType();

    TensorReferenceType*  elementTensorReferenceType();

    TextureReferenceType* elementTextureReferenceType();

    DataType              elementType() const;

    ArrayType*            init();

    NS::UInteger          stride() const;
};
class PointerType : public NS::Referencing<PointerType, Type>
{
public:
    BindingAccess       access() const;

    NS::UInteger        alignment() const;

    static PointerType* alloc();

    NS::UInteger        dataSize() const;

    ArrayType*          elementArrayType();

    bool                elementIsArgumentBuffer() const;

    StructType*         elementStructType();

    DataType            elementType() const;

    PointerType*        init();
};
class TextureReferenceType : public NS::Referencing<TextureReferenceType, Type>
{
public:
    BindingAccess                access() const;

    static TextureReferenceType* alloc();

    TextureReferenceType*        init();

    bool                         isDepthTexture() const;

    DataType                     textureDataType() const;

    TextureType                  textureType() const;
};
class TensorReferenceType : public NS::Referencing<TensorReferenceType, Type>
{
public:
    BindingAccess               access() const;

    static TensorReferenceType* alloc();

    TensorExtents*              dimensions() const;

    DataType                    indexType() const;

    TensorReferenceType*        init();

    TensorDataType              tensorDataType() const;
};
class Argument : public NS::Referencing<Argument>
{
public:
    BindingAccess access() const;

    [[deprecated("please use isActive instead")]]
    bool             active() const;

    static Argument* alloc();

    NS::UInteger     arrayLength() const;

    NS::UInteger     bufferAlignment() const;

    NS::UInteger     bufferDataSize() const;

    DataType         bufferDataType() const;

    PointerType*     bufferPointerType() const;

    StructType*      bufferStructType() const;

    NS::UInteger     index() const;

    Argument*        init();

    bool             isActive() const;

    bool             isDepthTexture() const;

    NS::String*      name() const;

    DataType         textureDataType() const;

    TextureType      textureType() const;

    NS::UInteger     threadgroupMemoryAlignment() const;

    NS::UInteger     threadgroupMemoryDataSize() const;

    ArgumentType     type() const;
};
class Binding : public NS::Referencing<Binding>
{
public:
    BindingAccess access() const;

    [[deprecated("please use isArgument instead")]]
    bool         argument() const;

    NS::UInteger index() const;

    bool         isArgument() const;

    bool         isUsed() const;

    NS::String*  name() const;

    BindingType  type() const;

    [[deprecated("please use isUsed instead")]]
    bool used() const;
};
class BufferBinding : public NS::Referencing<BufferBinding, Binding>
{
public:
    NS::UInteger bufferAlignment() const;

    NS::UInteger bufferDataSize() const;

    DataType     bufferDataType() const;

    PointerType* bufferPointerType() const;

    StructType*  bufferStructType() const;
};
class ThreadgroupBinding : public NS::Referencing<ThreadgroupBinding, Binding>
{
public:
    NS::UInteger threadgroupMemoryAlignment() const;

    NS::UInteger threadgroupMemoryDataSize() const;
};
class TextureBinding : public NS::Referencing<TextureBinding, Binding>
{
public:
    NS::UInteger arrayLength() const;

    [[deprecated("please use isDepthTexture instead")]]
    bool        depthTexture() const;
    bool        isDepthTexture() const;

    DataType    textureDataType() const;

    TextureType textureType() const;
};
class ObjectPayloadBinding : public NS::Referencing<ObjectPayloadBinding, Binding>
{
public:
    NS::UInteger objectPayloadAlignment() const;

    NS::UInteger objectPayloadDataSize() const;
};
class TensorBinding : public NS::Referencing<TensorBinding, Binding>
{
public:
    TensorExtents* dimensions() const;

    DataType       indexType() const;

    TensorDataType tensorDataType() const;
};

}
_MTL_INLINE MTL::Type* MTL::Type::alloc()
{
    return NS::Object::alloc<MTL::Type>(_MTL_PRIVATE_CLS(MTLType));
}

_MTL_INLINE MTL::DataType MTL::Type::dataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(dataType));
}

_MTL_INLINE MTL::Type* MTL::Type::init()
{
    return NS::Object::init<MTL::Type>();
}

_MTL_INLINE MTL::StructMember* MTL::StructMember::alloc()
{
    return NS::Object::alloc<MTL::StructMember>(_MTL_PRIVATE_CLS(MTLStructMember));
}

_MTL_INLINE NS::UInteger MTL::StructMember::argumentIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(argumentIndex));
}

_MTL_INLINE MTL::ArrayType* MTL::StructMember::arrayType()
{
    return Object::sendMessage<MTL::ArrayType*>(this, _MTL_PRIVATE_SEL(arrayType));
}

_MTL_INLINE MTL::DataType MTL::StructMember::dataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(dataType));
}

_MTL_INLINE MTL::StructMember* MTL::StructMember::init()
{
    return NS::Object::init<MTL::StructMember>();
}

_MTL_INLINE NS::String* MTL::StructMember::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE NS::UInteger MTL::StructMember::offset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(offset));
}

_MTL_INLINE MTL::PointerType* MTL::StructMember::pointerType()
{
    return Object::sendMessage<MTL::PointerType*>(this, _MTL_PRIVATE_SEL(pointerType));
}

_MTL_INLINE MTL::StructType* MTL::StructMember::structType()
{
    return Object::sendMessage<MTL::StructType*>(this, _MTL_PRIVATE_SEL(structType));
}

_MTL_INLINE MTL::TensorReferenceType* MTL::StructMember::tensorReferenceType()
{
    return Object::sendMessage<MTL::TensorReferenceType*>(this, _MTL_PRIVATE_SEL(tensorReferenceType));
}

_MTL_INLINE MTL::TextureReferenceType* MTL::StructMember::textureReferenceType()
{
    return Object::sendMessage<MTL::TextureReferenceType*>(this, _MTL_PRIVATE_SEL(textureReferenceType));
}

_MTL_INLINE MTL::StructType* MTL::StructType::alloc()
{
    return NS::Object::alloc<MTL::StructType>(_MTL_PRIVATE_CLS(MTLStructType));
}

_MTL_INLINE MTL::StructType* MTL::StructType::init()
{
    return NS::Object::init<MTL::StructType>();
}

_MTL_INLINE MTL::StructMember* MTL::StructType::memberByName(const NS::String* name)
{
    return Object::sendMessage<MTL::StructMember*>(this, _MTL_PRIVATE_SEL(memberByName_), name);
}

_MTL_INLINE NS::Array* MTL::StructType::members() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(members));
}

_MTL_INLINE MTL::ArrayType* MTL::ArrayType::alloc()
{
    return NS::Object::alloc<MTL::ArrayType>(_MTL_PRIVATE_CLS(MTLArrayType));
}

_MTL_INLINE NS::UInteger MTL::ArrayType::argumentIndexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(argumentIndexStride));
}

_MTL_INLINE NS::UInteger MTL::ArrayType::arrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(arrayLength));
}

_MTL_INLINE MTL::ArrayType* MTL::ArrayType::elementArrayType()
{
    return Object::sendMessage<MTL::ArrayType*>(this, _MTL_PRIVATE_SEL(elementArrayType));
}

_MTL_INLINE MTL::PointerType* MTL::ArrayType::elementPointerType()
{
    return Object::sendMessage<MTL::PointerType*>(this, _MTL_PRIVATE_SEL(elementPointerType));
}

_MTL_INLINE MTL::StructType* MTL::ArrayType::elementStructType()
{
    return Object::sendMessage<MTL::StructType*>(this, _MTL_PRIVATE_SEL(elementStructType));
}

_MTL_INLINE MTL::TensorReferenceType* MTL::ArrayType::elementTensorReferenceType()
{
    return Object::sendMessage<MTL::TensorReferenceType*>(this, _MTL_PRIVATE_SEL(elementTensorReferenceType));
}

_MTL_INLINE MTL::TextureReferenceType* MTL::ArrayType::elementTextureReferenceType()
{
    return Object::sendMessage<MTL::TextureReferenceType*>(this, _MTL_PRIVATE_SEL(elementTextureReferenceType));
}

_MTL_INLINE MTL::DataType MTL::ArrayType::elementType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(elementType));
}

_MTL_INLINE MTL::ArrayType* MTL::ArrayType::init()
{
    return NS::Object::init<MTL::ArrayType>();
}

_MTL_INLINE NS::UInteger MTL::ArrayType::stride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(stride));
}

_MTL_INLINE MTL::BindingAccess MTL::PointerType::access() const
{
    return Object::sendMessage<MTL::BindingAccess>(this, _MTL_PRIVATE_SEL(access));
}

_MTL_INLINE NS::UInteger MTL::PointerType::alignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(alignment));
}

_MTL_INLINE MTL::PointerType* MTL::PointerType::alloc()
{
    return NS::Object::alloc<MTL::PointerType>(_MTL_PRIVATE_CLS(MTLPointerType));
}

_MTL_INLINE NS::UInteger MTL::PointerType::dataSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(dataSize));
}

_MTL_INLINE MTL::ArrayType* MTL::PointerType::elementArrayType()
{
    return Object::sendMessage<MTL::ArrayType*>(this, _MTL_PRIVATE_SEL(elementArrayType));
}

_MTL_INLINE bool MTL::PointerType::elementIsArgumentBuffer() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(elementIsArgumentBuffer));
}

_MTL_INLINE MTL::StructType* MTL::PointerType::elementStructType()
{
    return Object::sendMessage<MTL::StructType*>(this, _MTL_PRIVATE_SEL(elementStructType));
}

_MTL_INLINE MTL::DataType MTL::PointerType::elementType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(elementType));
}

_MTL_INLINE MTL::PointerType* MTL::PointerType::init()
{
    return NS::Object::init<MTL::PointerType>();
}

_MTL_INLINE MTL::BindingAccess MTL::TextureReferenceType::access() const
{
    return Object::sendMessage<MTL::BindingAccess>(this, _MTL_PRIVATE_SEL(access));
}

_MTL_INLINE MTL::TextureReferenceType* MTL::TextureReferenceType::alloc()
{
    return NS::Object::alloc<MTL::TextureReferenceType>(_MTL_PRIVATE_CLS(MTLTextureReferenceType));
}

_MTL_INLINE MTL::TextureReferenceType* MTL::TextureReferenceType::init()
{
    return NS::Object::init<MTL::TextureReferenceType>();
}

_MTL_INLINE bool MTL::TextureReferenceType::isDepthTexture() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isDepthTexture));
}

_MTL_INLINE MTL::DataType MTL::TextureReferenceType::textureDataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(textureDataType));
}

_MTL_INLINE MTL::TextureType MTL::TextureReferenceType::textureType() const
{
    return Object::sendMessage<MTL::TextureType>(this, _MTL_PRIVATE_SEL(textureType));
}

_MTL_INLINE MTL::BindingAccess MTL::TensorReferenceType::access() const
{
    return Object::sendMessage<MTL::BindingAccess>(this, _MTL_PRIVATE_SEL(access));
}

_MTL_INLINE MTL::TensorReferenceType* MTL::TensorReferenceType::alloc()
{
    return NS::Object::alloc<MTL::TensorReferenceType>(_MTL_PRIVATE_CLS(MTLTensorReferenceType));
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorReferenceType::dimensions() const
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(dimensions));
}

_MTL_INLINE MTL::DataType MTL::TensorReferenceType::indexType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL::TensorReferenceType* MTL::TensorReferenceType::init()
{
    return NS::Object::init<MTL::TensorReferenceType>();
}

_MTL_INLINE MTL::TensorDataType MTL::TensorReferenceType::tensorDataType() const
{
    return Object::sendMessage<MTL::TensorDataType>(this, _MTL_PRIVATE_SEL(tensorDataType));
}

_MTL_INLINE MTL::BindingAccess MTL::Argument::access() const
{
    return Object::sendMessage<MTL::BindingAccess>(this, _MTL_PRIVATE_SEL(access));
}

_MTL_INLINE bool MTL::Argument::active() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isActive));
}

_MTL_INLINE MTL::Argument* MTL::Argument::alloc()
{
    return NS::Object::alloc<MTL::Argument>(_MTL_PRIVATE_CLS(MTLArgument));
}

_MTL_INLINE NS::UInteger MTL::Argument::arrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(arrayLength));
}

_MTL_INLINE NS::UInteger MTL::Argument::bufferAlignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferAlignment));
}

_MTL_INLINE NS::UInteger MTL::Argument::bufferDataSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferDataSize));
}

_MTL_INLINE MTL::DataType MTL::Argument::bufferDataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(bufferDataType));
}

_MTL_INLINE MTL::PointerType* MTL::Argument::bufferPointerType() const
{
    return Object::sendMessage<MTL::PointerType*>(this, _MTL_PRIVATE_SEL(bufferPointerType));
}

_MTL_INLINE MTL::StructType* MTL::Argument::bufferStructType() const
{
    return Object::sendMessage<MTL::StructType*>(this, _MTL_PRIVATE_SEL(bufferStructType));
}

_MTL_INLINE NS::UInteger MTL::Argument::index() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(index));
}

_MTL_INLINE MTL::Argument* MTL::Argument::init()
{
    return NS::Object::init<MTL::Argument>();
}

_MTL_INLINE bool MTL::Argument::isActive() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isActive));
}

_MTL_INLINE bool MTL::Argument::isDepthTexture() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isDepthTexture));
}

_MTL_INLINE NS::String* MTL::Argument::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE MTL::DataType MTL::Argument::textureDataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(textureDataType));
}

_MTL_INLINE MTL::TextureType MTL::Argument::textureType() const
{
    return Object::sendMessage<MTL::TextureType>(this, _MTL_PRIVATE_SEL(textureType));
}

_MTL_INLINE NS::UInteger MTL::Argument::threadgroupMemoryAlignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadgroupMemoryAlignment));
}

_MTL_INLINE NS::UInteger MTL::Argument::threadgroupMemoryDataSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadgroupMemoryDataSize));
}

_MTL_INLINE MTL::ArgumentType MTL::Argument::type() const
{
    return Object::sendMessage<MTL::ArgumentType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE MTL::BindingAccess MTL::Binding::access() const
{
    return Object::sendMessage<MTL::BindingAccess>(this, _MTL_PRIVATE_SEL(access));
}

_MTL_INLINE bool MTL::Binding::argument() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isArgument));
}

_MTL_INLINE NS::UInteger MTL::Binding::index() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(index));
}

_MTL_INLINE bool MTL::Binding::isArgument() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isArgument));
}

_MTL_INLINE bool MTL::Binding::isUsed() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isUsed));
}

_MTL_INLINE NS::String* MTL::Binding::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE MTL::BindingType MTL::Binding::type() const
{
    return Object::sendMessage<MTL::BindingType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE bool MTL::Binding::used() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isUsed));
}

_MTL_INLINE NS::UInteger MTL::BufferBinding::bufferAlignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferAlignment));
}

_MTL_INLINE NS::UInteger MTL::BufferBinding::bufferDataSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferDataSize));
}

_MTL_INLINE MTL::DataType MTL::BufferBinding::bufferDataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(bufferDataType));
}

_MTL_INLINE MTL::PointerType* MTL::BufferBinding::bufferPointerType() const
{
    return Object::sendMessage<MTL::PointerType*>(this, _MTL_PRIVATE_SEL(bufferPointerType));
}

_MTL_INLINE MTL::StructType* MTL::BufferBinding::bufferStructType() const
{
    return Object::sendMessage<MTL::StructType*>(this, _MTL_PRIVATE_SEL(bufferStructType));
}

_MTL_INLINE NS::UInteger MTL::ThreadgroupBinding::threadgroupMemoryAlignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadgroupMemoryAlignment));
}

_MTL_INLINE NS::UInteger MTL::ThreadgroupBinding::threadgroupMemoryDataSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadgroupMemoryDataSize));
}

_MTL_INLINE NS::UInteger MTL::TextureBinding::arrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(arrayLength));
}

_MTL_INLINE bool MTL::TextureBinding::depthTexture() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isDepthTexture));
}

_MTL_INLINE bool MTL::TextureBinding::isDepthTexture() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isDepthTexture));
}

_MTL_INLINE MTL::DataType MTL::TextureBinding::textureDataType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(textureDataType));
}

_MTL_INLINE MTL::TextureType MTL::TextureBinding::textureType() const
{
    return Object::sendMessage<MTL::TextureType>(this, _MTL_PRIVATE_SEL(textureType));
}

_MTL_INLINE NS::UInteger MTL::ObjectPayloadBinding::objectPayloadAlignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(objectPayloadAlignment));
}

_MTL_INLINE NS::UInteger MTL::ObjectPayloadBinding::objectPayloadDataSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(objectPayloadDataSize));
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorBinding::dimensions() const
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(dimensions));
}

_MTL_INLINE MTL::DataType MTL::TensorBinding::indexType() const
{
    return Object::sendMessage<MTL::DataType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL::TensorDataType MTL::TensorBinding::tensorDataType() const
{
    return Object::sendMessage<MTL::TensorDataType>(this, _MTL_PRIVATE_SEL(tensorDataType));
}
