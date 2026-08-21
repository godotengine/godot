#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResource.hpp"

namespace MTL {
    class Device;
    class Tensor;
    class TensorDescriptor;
    class Texture;
    class TextureDescriptor;
    enum BufferSparseTier : NS::Integer;
}
namespace NS {
    class Error;
    class String;
}

namespace MTL
{

class Buffer : public NS::Referencing<Buffer, MTL::Resource>
{
public:
    void                  addDebugMarker(NS::String* marker, NS::Range range);
    void *                contents();
    void                  didModifyRange(NS::Range range);
    MTL::GPUAddress       gpuAddress() const;
    NS::UInteger          length() const;
    MTL::Buffer*          newRemoteBufferView(MTL::Device* device);
    MTL::Tensor*          newTensor(MTL::TensorDescriptor* descriptor, NS::UInteger offset, NS::Error** error);
    MTL::Texture*         newTexture(MTL::TextureDescriptor* descriptor, NS::UInteger offset, NS::UInteger bytesPerRow);
    MTL::Buffer*          remoteStorageBuffer() const;
    void                  removeAllDebugMarkers();
    MTL::BufferSparseTier sparseBufferTier() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLBuffer;

_MTL_INLINE NS::UInteger MTL::Buffer::length() const
{
    return _MTL_msg_NS__UInteger_length((const void*)this, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::Buffer::remoteStorageBuffer() const
{
    return _MTL_msg_MTL__Bufferp_remoteStorageBuffer((const void*)this, nullptr);
}

_MTL_INLINE MTL::GPUAddress MTL::Buffer::gpuAddress() const
{
    return _MTL_msg_MTL__GPUAddress_gpuAddress((const void*)this, nullptr);
}

_MTL_INLINE MTL::BufferSparseTier MTL::Buffer::sparseBufferTier() const
{
    return _MTL_msg_MTL__BufferSparseTier_sparseBufferTier((const void*)this, nullptr);
}

_MTL_INLINE void * MTL::Buffer::contents()
{
    return _MTL_msg_voidp_contents((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Buffer::didModifyRange(NS::Range range)
{
    _MTL_msg_v_didModifyRange__NS__Range((const void*)this, nullptr, range);
}

_MTL_INLINE MTL::Texture* MTL::Buffer::newTexture(MTL::TextureDescriptor* descriptor, NS::UInteger offset, NS::UInteger bytesPerRow)
{
    return _MTL_msg_MTL__Texturep_newTextureWithDescriptor_offset_bytesPerRow__MTL__TextureDescriptorp_NS__UInteger_NS__UInteger((const void*)this, nullptr, descriptor, offset, bytesPerRow);
}

_MTL_INLINE MTL::Tensor* MTL::Buffer::newTensor(MTL::TensorDescriptor* descriptor, NS::UInteger offset, NS::Error** error)
{
    return _MTL_msg_MTL__Tensorp_newTensorWithDescriptor_offset_error__MTL__TensorDescriptorp_NS__UInteger_NS__Errorpp((const void*)this, nullptr, descriptor, offset, error);
}

_MTL_INLINE void MTL::Buffer::addDebugMarker(NS::String* marker, NS::Range range)
{
    _MTL_msg_v_addDebugMarker_range__NS__Stringp_NS__Range((const void*)this, nullptr, marker, range);
}

_MTL_INLINE void MTL::Buffer::removeAllDebugMarkers()
{
    _MTL_msg_v_removeAllDebugMarkers((const void*)this, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::Buffer::newRemoteBufferView(MTL::Device* device)
{
    return _MTL_msg_MTL__Bufferp_newRemoteBufferViewForDevice__MTL__Devicep((const void*)this, nullptr, device);
}
