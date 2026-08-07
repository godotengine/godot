#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLCommandEncoder.hpp"

namespace MTL {
    class Buffer;
    class Fence;
    class Texture;
}

namespace MTL
{

_MTL_ENUM(NS::UInteger, SparseTextureMappingMode) {
    SparseTextureMappingModeMap = 0,
    SparseTextureMappingModeUnmap = 1,
};


class ResourceStateCommandEncoder : public NS::Referencing<ResourceStateCommandEncoder, MTL::CommandEncoder>
{
public:
    void moveTextureMappingsFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void updateFence(MTL::Fence* fence);
    void updateTextureMapping(MTL::Texture* texture, MTL::SparseTextureMappingMode const mode, MTL::Region const region, NS::UInteger const mipLevel, NS::UInteger const slice);
    void updateTextureMapping(MTL::Texture* texture, MTL::SparseTextureMappingMode const mode, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);
    void updateTextureMappings(MTL::Texture* texture, MTL::SparseTextureMappingMode const mode, const MTL::Region * regions, const NS::UInteger * mipLevels, const NS::UInteger * slices, NS::UInteger numRegions);
    void wait(MTL::Fence* fence);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLResourceStateCommandEncoder;

_MTL_INLINE void MTL::ResourceStateCommandEncoder::updateTextureMappings(MTL::Texture* texture, MTL::SparseTextureMappingMode const mode, const MTL::Region * regions, const NS::UInteger * mipLevels, const NS::UInteger * slices, NS::UInteger numRegions)
{
    _MTL_msg_v_updateTextureMappings_mode_regions_mipLevels_slices_numRegions__MTL__Texturep_MTL__SparseTextureMappingModeconst_constMTL__Regionp_constNS__UIntegerp_constNS__UIntegerp_NS__UInteger((const void*)this, nullptr, texture, mode, regions, mipLevels, slices, numRegions);
}

_MTL_INLINE void MTL::ResourceStateCommandEncoder::updateTextureMapping(MTL::Texture* texture, MTL::SparseTextureMappingMode const mode, MTL::Region const region, NS::UInteger const mipLevel, NS::UInteger const slice)
{
    _MTL_msg_v_updateTextureMapping_mode_region_mipLevel_slice__MTL__Texturep_MTL__SparseTextureMappingModeconst_MTL__Regionconst_NS__UIntegerconst_NS__UIntegerconst((const void*)this, nullptr, texture, mode, region, mipLevel, slice);
}

_MTL_INLINE void MTL::ResourceStateCommandEncoder::updateTextureMapping(MTL::Texture* texture, MTL::SparseTextureMappingMode const mode, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_updateTextureMapping_mode_indirectBuffer_indirectBufferOffset__MTL__Texturep_MTL__SparseTextureMappingModeconst_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, texture, mode, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::ResourceStateCommandEncoder::updateFence(MTL::Fence* fence)
{
    _MTL_msg_v_updateFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::ResourceStateCommandEncoder::wait(MTL::Fence* fence)
{
    _MTL_msg_v_waitForFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::ResourceStateCommandEncoder::moveTextureMappingsFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    _MTL_msg_v_moveTextureMappingsFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}
