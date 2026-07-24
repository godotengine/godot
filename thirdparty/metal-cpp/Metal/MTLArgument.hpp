#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class TensorExtents;
    enum DataType : NS::UInteger;
    enum TensorDataType : NS::Integer;
    enum TextureType : NS::UInteger;
}
namespace NS {
    class Array;
    class String;
}

namespace MTL
{

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
    ArgumentAccessReadOnly = BindingAccessReadOnly,
    ArgumentAccessReadWrite = BindingAccessReadWrite,
    ArgumentAccessWriteOnly = BindingAccessWriteOnly,
};


class Type;
class StructMember;
class StructType;
class ArrayType;
class PointerType;
class TextureReferenceType;
class TensorReferenceType;
class Argument;
class Binding;
class BufferBinding;
class ThreadgroupBinding;
class TextureBinding;
class ObjectPayloadBinding;
class TensorBinding;

class Type : public NS::Referencing<Type>
{
public:
    static Type* alloc();
    Type*        init() const;

    MTL::DataType dataType() const;

};

class StructMember : public NS::Referencing<StructMember>
{
public:
    static StructMember* alloc();
    StructMember*        init() const;

    NS::UInteger               argumentIndex() const;
    MTL::ArrayType*            arrayType();
    MTL::DataType              dataType() const;
    NS::String*                name() const;
    NS::UInteger               offset() const;
    MTL::PointerType*          pointerType();
    MTL::StructType*           structType();
    MTL::TensorReferenceType*  tensorReferenceType();
    MTL::TextureReferenceType* textureReferenceType();

};

class StructType : public NS::Referencing<StructType, MTL::Type>
{
public:
    static StructType* alloc();
    StructType*        init() const;

    MTL::StructMember* memberByName(NS::String* name);
    NS::Array*         members() const;

};

class ArrayType : public NS::Referencing<ArrayType, MTL::Type>
{
public:
    static ArrayType* alloc();
    ArrayType*        init() const;

    NS::UInteger               argumentIndexStride() const;
    NS::UInteger               arrayLength() const;
    MTL::ArrayType*            elementArrayType();
    MTL::PointerType*          elementPointerType();
    MTL::StructType*           elementStructType();
    MTL::TensorReferenceType*  elementTensorReferenceType();
    MTL::TextureReferenceType* elementTextureReferenceType();
    MTL::DataType              elementType() const;
    NS::UInteger               stride() const;

};

class PointerType : public NS::Referencing<PointerType, MTL::Type>
{
public:
    static PointerType* alloc();
    PointerType*        init() const;

    MTL::BindingAccess access() const;
    NS::UInteger       alignment() const;
    NS::UInteger       dataSize() const;
    MTL::ArrayType*    elementArrayType();
    bool               elementIsArgumentBuffer() const;
    MTL::StructType*   elementStructType();
    MTL::DataType      elementType() const;

};

class TextureReferenceType : public NS::Referencing<TextureReferenceType, MTL::Type>
{
public:
    static TextureReferenceType* alloc();
    TextureReferenceType*        init() const;

    MTL::BindingAccess access() const;
    bool               isDepthTexture() const;
    MTL::DataType      textureDataType() const;
    MTL::TextureType   textureType() const;

};

class TensorReferenceType : public NS::Referencing<TensorReferenceType, MTL::Type>
{
public:
    static TensorReferenceType* alloc();
    TensorReferenceType*        init() const;

    MTL::BindingAccess  access() const;
    MTL::TensorExtents* dimensions() const;
    MTL::DataType       indexType() const;
    MTL::TensorDataType tensorDataType() const;

};

class Argument : public NS::Referencing<Argument>
{
public:
    static Argument* alloc();
    Argument*        init() const;

    MTL::BindingAccess access() const;
    bool               active() const;
    NS::UInteger       arrayLength() const;
    NS::UInteger       bufferAlignment() const;
    NS::UInteger       bufferDataSize() const;
    MTL::DataType      bufferDataType() const;
    MTL::PointerType*  bufferPointerType() const;
    MTL::StructType*   bufferStructType() const;
    NS::UInteger       index() const;
    bool               isActive();
    bool               isDepthTexture() const;
    NS::String*        name() const;
    MTL::DataType      textureDataType() const;
    MTL::TextureType   textureType() const;
    NS::UInteger       threadgroupMemoryAlignment() const;
    NS::UInteger       threadgroupMemoryDataSize() const;
    MTL::ArgumentType  type() const;

};

class Binding : public NS::Referencing<Binding>
{
public:
    MTL::BindingAccess access() const;
    bool               argument() const;
    NS::UInteger       index() const;
    bool               isArgument();
    bool               isUsed();
    NS::String*        name() const;
    MTL::BindingType   type() const;
    bool               used() const;

};

class BufferBinding : public NS::Referencing<BufferBinding, MTL::Binding>
{
public:
    NS::UInteger      bufferAlignment() const;
    NS::UInteger      bufferDataSize() const;
    MTL::DataType     bufferDataType() const;
    MTL::PointerType* bufferPointerType() const;
    MTL::StructType*  bufferStructType() const;

};

class ThreadgroupBinding : public NS::Referencing<ThreadgroupBinding, MTL::Binding>
{
public:
    NS::UInteger threadgroupMemoryAlignment() const;
    NS::UInteger threadgroupMemoryDataSize() const;

};

class TextureBinding : public NS::Referencing<TextureBinding, MTL::Binding>
{
public:
    NS::UInteger     arrayLength() const;
    bool             depthTexture() const;
    bool             isDepthTexture();
    MTL::DataType    textureDataType() const;
    MTL::TextureType textureType() const;

};

class ObjectPayloadBinding : public NS::Referencing<ObjectPayloadBinding, MTL::Binding>
{
public:
    NS::UInteger objectPayloadAlignment() const;
    NS::UInteger objectPayloadDataSize() const;

};

class TensorBinding : public NS::Referencing<TensorBinding, MTL::Binding>
{
public:
    MTL::TensorExtents* dimensions() const;
    MTL::DataType       indexType() const;
    MTL::TensorDataType tensorDataType() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLType;
extern "C" void *OBJC_CLASS_$_MTLStructMember;
extern "C" void *OBJC_CLASS_$_MTLStructType;
extern "C" void *OBJC_CLASS_$_MTLArrayType;
extern "C" void *OBJC_CLASS_$_MTLPointerType;
extern "C" void *OBJC_CLASS_$_MTLTextureReferenceType;
extern "C" void *OBJC_CLASS_$_MTLTensorReferenceType;
extern "C" void *OBJC_CLASS_$_MTLArgument;
extern "C" void *OBJC_CLASS_$_MTLBinding;
extern "C" void *OBJC_CLASS_$_MTLBufferBinding;
extern "C" void *OBJC_CLASS_$_MTLThreadgroupBinding;
extern "C" void *OBJC_CLASS_$_MTLTextureBinding;
extern "C" void *OBJC_CLASS_$_MTLObjectPayloadBinding;
extern "C" void *OBJC_CLASS_$_MTLTensorBinding;

_MTL_INLINE MTL::Type* MTL::Type::alloc()
{
    return _MTL_msg_MTL__Typep_alloc((const void*)&OBJC_CLASS_$_MTLType, nullptr);
}

_MTL_INLINE MTL::Type* MTL::Type::init() const
{
    return _MTL_msg_MTL__Typep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::Type::dataType() const
{
    return _MTL_msg_MTL__DataType_dataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructMember* MTL::StructMember::alloc()
{
    return _MTL_msg_MTL__StructMemberp_alloc((const void*)&OBJC_CLASS_$_MTLStructMember, nullptr);
}

_MTL_INLINE MTL::StructMember* MTL::StructMember::init() const
{
    return _MTL_msg_MTL__StructMemberp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::StructMember::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::StructMember::offset() const
{
    return _MTL_msg_NS__UInteger_offset((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::StructMember::dataType() const
{
    return _MTL_msg_MTL__DataType_dataType((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::StructMember::argumentIndex() const
{
    return _MTL_msg_NS__UInteger_argumentIndex((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::StructMember::structType()
{
    return _MTL_msg_MTL__StructTypep_structType((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArrayType* MTL::StructMember::arrayType()
{
    return _MTL_msg_MTL__ArrayTypep_arrayType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureReferenceType* MTL::StructMember::textureReferenceType()
{
    return _MTL_msg_MTL__TextureReferenceTypep_textureReferenceType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PointerType* MTL::StructMember::pointerType()
{
    return _MTL_msg_MTL__PointerTypep_pointerType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorReferenceType* MTL::StructMember::tensorReferenceType()
{
    return _MTL_msg_MTL__TensorReferenceTypep_tensorReferenceType((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::StructType::alloc()
{
    return _MTL_msg_MTL__StructTypep_alloc((const void*)&OBJC_CLASS_$_MTLStructType, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::StructType::init() const
{
    return _MTL_msg_MTL__StructTypep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::StructType::members() const
{
    return _MTL_msg_NS__Arrayp_members((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructMember* MTL::StructType::memberByName(NS::String* name)
{
    return _MTL_msg_MTL__StructMemberp_memberByName__NS__Stringp((const void*)this, nullptr, name);
}

_MTL_INLINE MTL::ArrayType* MTL::ArrayType::alloc()
{
    return _MTL_msg_MTL__ArrayTypep_alloc((const void*)&OBJC_CLASS_$_MTLArrayType, nullptr);
}

_MTL_INLINE MTL::ArrayType* MTL::ArrayType::init() const
{
    return _MTL_msg_MTL__ArrayTypep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::ArrayType::elementType() const
{
    return _MTL_msg_MTL__DataType_elementType((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ArrayType::arrayLength() const
{
    return _MTL_msg_NS__UInteger_arrayLength((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ArrayType::stride() const
{
    return _MTL_msg_NS__UInteger_stride((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ArrayType::argumentIndexStride() const
{
    return _MTL_msg_NS__UInteger_argumentIndexStride((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::ArrayType::elementStructType()
{
    return _MTL_msg_MTL__StructTypep_elementStructType((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArrayType* MTL::ArrayType::elementArrayType()
{
    return _MTL_msg_MTL__ArrayTypep_elementArrayType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureReferenceType* MTL::ArrayType::elementTextureReferenceType()
{
    return _MTL_msg_MTL__TextureReferenceTypep_elementTextureReferenceType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PointerType* MTL::ArrayType::elementPointerType()
{
    return _MTL_msg_MTL__PointerTypep_elementPointerType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorReferenceType* MTL::ArrayType::elementTensorReferenceType()
{
    return _MTL_msg_MTL__TensorReferenceTypep_elementTensorReferenceType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PointerType* MTL::PointerType::alloc()
{
    return _MTL_msg_MTL__PointerTypep_alloc((const void*)&OBJC_CLASS_$_MTLPointerType, nullptr);
}

_MTL_INLINE MTL::PointerType* MTL::PointerType::init() const
{
    return _MTL_msg_MTL__PointerTypep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::PointerType::elementType() const
{
    return _MTL_msg_MTL__DataType_elementType((const void*)this, nullptr);
}

_MTL_INLINE MTL::BindingAccess MTL::PointerType::access() const
{
    return _MTL_msg_MTL__BindingAccess_access((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::PointerType::alignment() const
{
    return _MTL_msg_NS__UInteger_alignment((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::PointerType::dataSize() const
{
    return _MTL_msg_NS__UInteger_dataSize((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::PointerType::elementIsArgumentBuffer() const
{
    return _MTL_msg_bool_elementIsArgumentBuffer((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::PointerType::elementStructType()
{
    return _MTL_msg_MTL__StructTypep_elementStructType((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArrayType* MTL::PointerType::elementArrayType()
{
    return _MTL_msg_MTL__ArrayTypep_elementArrayType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureReferenceType* MTL::TextureReferenceType::alloc()
{
    return _MTL_msg_MTL__TextureReferenceTypep_alloc((const void*)&OBJC_CLASS_$_MTLTextureReferenceType, nullptr);
}

_MTL_INLINE MTL::TextureReferenceType* MTL::TextureReferenceType::init() const
{
    return _MTL_msg_MTL__TextureReferenceTypep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::TextureReferenceType::textureDataType() const
{
    return _MTL_msg_MTL__DataType_textureDataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureType MTL::TextureReferenceType::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE MTL::BindingAccess MTL::TextureReferenceType::access() const
{
    return _MTL_msg_MTL__BindingAccess_access((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::TextureReferenceType::isDepthTexture() const
{
    return _MTL_msg_bool_isDepthTexture((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorReferenceType* MTL::TensorReferenceType::alloc()
{
    return _MTL_msg_MTL__TensorReferenceTypep_alloc((const void*)&OBJC_CLASS_$_MTLTensorReferenceType, nullptr);
}

_MTL_INLINE MTL::TensorReferenceType* MTL::TensorReferenceType::init() const
{
    return _MTL_msg_MTL__TensorReferenceTypep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorDataType MTL::TensorReferenceType::tensorDataType() const
{
    return _MTL_msg_MTL__TensorDataType_tensorDataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::TensorReferenceType::indexType() const
{
    return _MTL_msg_MTL__DataType_indexType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorReferenceType::dimensions() const
{
    return _MTL_msg_MTL__TensorExtentsp_dimensions((const void*)this, nullptr);
}

_MTL_INLINE MTL::BindingAccess MTL::TensorReferenceType::access() const
{
    return _MTL_msg_MTL__BindingAccess_access((const void*)this, nullptr);
}

_MTL_INLINE MTL::Argument* MTL::Argument::alloc()
{
    return _MTL_msg_MTL__Argumentp_alloc((const void*)&OBJC_CLASS_$_MTLArgument, nullptr);
}

_MTL_INLINE MTL::Argument* MTL::Argument::init() const
{
    return _MTL_msg_MTL__Argumentp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Argument::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArgumentType MTL::Argument::type() const
{
    return _MTL_msg_MTL__ArgumentType_type((const void*)this, nullptr);
}

_MTL_INLINE MTL::BindingAccess MTL::Argument::access() const
{
    return _MTL_msg_MTL__BindingAccess_access((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Argument::index() const
{
    return _MTL_msg_NS__UInteger_index((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Argument::active() const
{
    return _MTL_msg_bool_active((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Argument::bufferAlignment() const
{
    return _MTL_msg_NS__UInteger_bufferAlignment((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Argument::bufferDataSize() const
{
    return _MTL_msg_NS__UInteger_bufferDataSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::Argument::bufferDataType() const
{
    return _MTL_msg_MTL__DataType_bufferDataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::Argument::bufferStructType() const
{
    return _MTL_msg_MTL__StructTypep_bufferStructType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PointerType* MTL::Argument::bufferPointerType() const
{
    return _MTL_msg_MTL__PointerTypep_bufferPointerType((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Argument::threadgroupMemoryAlignment() const
{
    return _MTL_msg_NS__UInteger_threadgroupMemoryAlignment((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Argument::threadgroupMemoryDataSize() const
{
    return _MTL_msg_NS__UInteger_threadgroupMemoryDataSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureType MTL::Argument::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::Argument::textureDataType() const
{
    return _MTL_msg_MTL__DataType_textureDataType((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Argument::isDepthTexture() const
{
    return _MTL_msg_bool_isDepthTexture((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Argument::arrayLength() const
{
    return _MTL_msg_NS__UInteger_arrayLength((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Argument::isActive()
{
    return _MTL_msg_bool_isActive((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Binding::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE MTL::BindingType MTL::Binding::type() const
{
    return _MTL_msg_MTL__BindingType_type((const void*)this, nullptr);
}

_MTL_INLINE MTL::BindingAccess MTL::Binding::access() const
{
    return _MTL_msg_MTL__BindingAccess_access((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Binding::index() const
{
    return _MTL_msg_NS__UInteger_index((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Binding::used() const
{
    return _MTL_msg_bool_used((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Binding::argument() const
{
    return _MTL_msg_bool_argument((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Binding::isUsed()
{
    return _MTL_msg_bool_isUsed((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Binding::isArgument()
{
    return _MTL_msg_bool_isArgument((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::BufferBinding::bufferAlignment() const
{
    return _MTL_msg_NS__UInteger_bufferAlignment((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::BufferBinding::bufferDataSize() const
{
    return _MTL_msg_NS__UInteger_bufferDataSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::BufferBinding::bufferDataType() const
{
    return _MTL_msg_MTL__DataType_bufferDataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::StructType* MTL::BufferBinding::bufferStructType() const
{
    return _MTL_msg_MTL__StructTypep_bufferStructType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PointerType* MTL::BufferBinding::bufferPointerType() const
{
    return _MTL_msg_MTL__PointerTypep_bufferPointerType((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ThreadgroupBinding::threadgroupMemoryAlignment() const
{
    return _MTL_msg_NS__UInteger_threadgroupMemoryAlignment((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ThreadgroupBinding::threadgroupMemoryDataSize() const
{
    return _MTL_msg_NS__UInteger_threadgroupMemoryDataSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureType MTL::TextureBinding::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::TextureBinding::textureDataType() const
{
    return _MTL_msg_MTL__DataType_textureDataType((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::TextureBinding::depthTexture() const
{
    return _MTL_msg_bool_depthTexture((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::TextureBinding::arrayLength() const
{
    return _MTL_msg_NS__UInteger_arrayLength((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::TextureBinding::isDepthTexture()
{
    return _MTL_msg_bool_isDepthTexture((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ObjectPayloadBinding::objectPayloadAlignment() const
{
    return _MTL_msg_NS__UInteger_objectPayloadAlignment((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ObjectPayloadBinding::objectPayloadDataSize() const
{
    return _MTL_msg_NS__UInteger_objectPayloadDataSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorDataType MTL::TensorBinding::tensorDataType() const
{
    return _MTL_msg_MTL__TensorDataType_tensorDataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::DataType MTL::TensorBinding::indexType() const
{
    return _MTL_msg_MTL__DataType_indexType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorBinding::dimensions() const
{
    return _MTL_msg_MTL__TensorExtentsp_dimensions((const void*)this, nullptr);
}
