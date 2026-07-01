#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResourceViewPool.hpp"

namespace MTL {
    class Buffer;
    class Texture;
    class TextureDescriptor;
    class TextureViewDescriptor;
}

namespace MTL
{

class TextureViewPool : public NS::Referencing<TextureViewPool, MTL::ResourceViewPool>
{
public:
    MTL::ResourceID setTextureView(MTL::Texture* texture, NS::UInteger index);
    MTL::ResourceID setTextureView(MTL::Texture* texture, MTL::TextureViewDescriptor* descriptor, NS::UInteger index);
    MTL::ResourceID setTextureViewFromBuffer(MTL::Buffer* buffer, MTL::TextureDescriptor* descriptor, NS::UInteger offset, NS::UInteger bytesPerRow, NS::UInteger index);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLTextureViewPool;

_MTL_INLINE MTL::ResourceID MTL::TextureViewPool::setTextureView(MTL::Texture* texture, NS::UInteger index)
{
    return _MTL_msg_MTL__ResourceID_setTextureView_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE MTL::ResourceID MTL::TextureViewPool::setTextureView(MTL::Texture* texture, MTL::TextureViewDescriptor* descriptor, NS::UInteger index)
{
    return _MTL_msg_MTL__ResourceID_setTextureView_descriptor_atIndex__MTL__Texturep_MTL__TextureViewDescriptorp_NS__UInteger((const void*)this, nullptr, texture, descriptor, index);
}

_MTL_INLINE MTL::ResourceID MTL::TextureViewPool::setTextureViewFromBuffer(MTL::Buffer* buffer, MTL::TextureDescriptor* descriptor, NS::UInteger offset, NS::UInteger bytesPerRow, NS::UInteger index)
{
    return _MTL_msg_MTL__ResourceID_setTextureViewFromBuffer_descriptor_offset_bytesPerRow_atIndex__MTL__Bufferp_MTL__TextureDescriptorp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, descriptor, offset, bytesPerRow, index);
}
