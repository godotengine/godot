//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLBuffer.hpp
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
#include "MTLDefines.hpp"
#include "MTLGPUAddress.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLResource.hpp"

namespace MTL
{
class Buffer;
class Device;
class Tensor;
class TensorDescriptor;
class Texture;
class TextureDescriptor;

class Buffer : public NS::Referencing<Buffer, Resource>
{
public:
    void             addDebugMarker(const NS::String* marker, NS::Range range);

    void*            contents();

    void             didModifyRange(NS::Range range);

    GPUAddress       gpuAddress() const;

    NS::UInteger     length() const;

    Buffer*          newRemoteBufferViewForDevice(const MTL::Device* device);

    Tensor*          newTensor(const MTL::TensorDescriptor* descriptor, NS::UInteger offset, NS::Error** error);

    Texture*         newTexture(const MTL::TextureDescriptor* descriptor, NS::UInteger offset, NS::UInteger bytesPerRow);

    Buffer*          remoteStorageBuffer() const;

    void             removeAllDebugMarkers();

    BufferSparseTier sparseBufferTier() const;
};

}
_MTL_INLINE void MTL::Buffer::addDebugMarker(const NS::String* marker, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addDebugMarker_range_), marker, range);
}

_MTL_INLINE void* MTL::Buffer::contents()
{
    return Object::sendMessage<void*>(this, _MTL_PRIVATE_SEL(contents));
}

_MTL_INLINE void MTL::Buffer::didModifyRange(NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(didModifyRange_), range);
}

_MTL_INLINE MTL::GPUAddress MTL::Buffer::gpuAddress() const
{
    return Object::sendMessage<MTL::GPUAddress>(this, _MTL_PRIVATE_SEL(gpuAddress));
}

_MTL_INLINE NS::UInteger MTL::Buffer::length() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(length));
}

_MTL_INLINE MTL::Buffer* MTL::Buffer::newRemoteBufferViewForDevice(const MTL::Device* device)
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(newRemoteBufferViewForDevice_), device);
}

_MTL_INLINE MTL::Tensor* MTL::Buffer::newTensor(const MTL::TensorDescriptor* descriptor, NS::UInteger offset, NS::Error** error)
{
    return Object::sendMessage<MTL::Tensor*>(this, _MTL_PRIVATE_SEL(newTensorWithDescriptor_offset_error_), descriptor, offset, error);
}

_MTL_INLINE MTL::Texture* MTL::Buffer::newTexture(const MTL::TextureDescriptor* descriptor, NS::UInteger offset, NS::UInteger bytesPerRow)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureWithDescriptor_offset_bytesPerRow_), descriptor, offset, bytesPerRow);
}

_MTL_INLINE MTL::Buffer* MTL::Buffer::remoteStorageBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(remoteStorageBuffer));
}

_MTL_INLINE void MTL::Buffer::removeAllDebugMarkers()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeAllDebugMarkers));
}

_MTL_INLINE MTL::BufferSparseTier MTL::Buffer::sparseBufferTier() const
{
    return Object::sendMessage<MTL::BufferSparseTier>(this, _MTL_PRIVATE_SEL(sparseBufferTier));
}
