#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class AccelerationStructure;
    class Buffer;
    class ComputePipelineState;
    class DepthStencilState;
    class Device;
    class IndirectCommandBuffer;
    class IntersectionFunctionTable;
    class RenderPipelineState;
    class SamplerState;
    class Texture;
    class VisibleFunctionTable;
}
namespace NS {
    class String;
}

namespace MTL
{

class ArgumentEncoder : public NS::Referencing<ArgumentEncoder>
{
public:
    NS::UInteger          alignment() const;
    void *                constantData(NS::UInteger index);
    MTL::Device*          device() const;
    NS::UInteger          encodedLength() const;
    NS::String*           label() const;
    MTL::ArgumentEncoder* newArgumentEncoder(NS::UInteger index);
    void                  setAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger index);
    void                  setArgumentBuffer(MTL::Buffer* argumentBuffer, NS::UInteger offset);
    void                  setArgumentBuffer(MTL::Buffer* argumentBuffer, NS::UInteger startOffset, NS::UInteger arrayElement);
    void                  setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void                  setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void                  setComputePipelineState(MTL::ComputePipelineState* pipeline, NS::UInteger index);
    void                  setComputePipelineStates(const MTL::ComputePipelineState* const * pipelines, NS::Range range);
    void                  setDepthStencilState(MTL::DepthStencilState* depthStencilState, NS::UInteger index);
    void                  setDepthStencilStates(const MTL::DepthStencilState* const * depthStencilStates, NS::Range range);
    void                  setIndirectCommandBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::UInteger index);
    void                  setIndirectCommandBuffers(const MTL::IndirectCommandBuffer* const * buffers, NS::Range range);
    void                  setIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger index);
    void                  setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range);
    void                  setLabel(NS::String* label);
    void                  setRenderPipelineState(MTL::RenderPipelineState* pipeline, NS::UInteger index);
    void                  setRenderPipelineStates(const MTL::RenderPipelineState* const * pipelines, NS::Range range);
    void                  setSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void                  setSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void                  setTexture(MTL::Texture* texture, NS::UInteger index);
    void                  setTextures(const MTL::Texture* const * textures, NS::Range range);
    void                  setVisibleFunctionTable(MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger index);
    void                  setVisibleFunctionTables(const MTL::VisibleFunctionTable* const * visibleFunctionTables, NS::Range range);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLArgumentEncoder;

_MTL_INLINE MTL::Device* MTL::ArgumentEncoder::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::ArgumentEncoder::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentEncoder::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE NS::UInteger MTL::ArgumentEncoder::encodedLength() const
{
    return _MTL_msg_NS__UInteger_encodedLength((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ArgumentEncoder::alignment() const
{
    return _MTL_msg_NS__UInteger_alignment((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentEncoder::setArgumentBuffer(MTL::Buffer* argumentBuffer, NS::UInteger offset)
{
    _MTL_msg_v_setArgumentBuffer_offset__MTL__Bufferp_NS__UInteger((const void*)this, nullptr, argumentBuffer, offset);
}

_MTL_INLINE void MTL::ArgumentEncoder::setArgumentBuffer(MTL::Buffer* argumentBuffer, NS::UInteger startOffset, NS::UInteger arrayElement)
{
    _MTL_msg_v_setArgumentBuffer_startOffset_arrayElement__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, argumentBuffer, startOffset, arrayElement);
}

_MTL_INLINE void MTL::ArgumentEncoder::setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void * MTL::ArgumentEncoder::constantData(NS::UInteger index)
{
    return _MTL_msg_voidp_constantDataAtIndex__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setRenderPipelineState(MTL::RenderPipelineState* pipeline, NS::UInteger index)
{
    _MTL_msg_v_setRenderPipelineState_atIndex__MTL__RenderPipelineStatep_NS__UInteger((const void*)this, nullptr, pipeline, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setRenderPipelineStates(const MTL::RenderPipelineState* const * pipelines, NS::Range range)
{
    _MTL_msg_v_setRenderPipelineStates_withRange__constMTL__RenderPipelineStatepconstp_NS__Range((const void*)this, nullptr, pipelines, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setComputePipelineState(MTL::ComputePipelineState* pipeline, NS::UInteger index)
{
    _MTL_msg_v_setComputePipelineState_atIndex__MTL__ComputePipelineStatep_NS__UInteger((const void*)this, nullptr, pipeline, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setComputePipelineStates(const MTL::ComputePipelineState* const * pipelines, NS::Range range)
{
    _MTL_msg_v_setComputePipelineStates_withRange__constMTL__ComputePipelineStatepconstp_NS__Range((const void*)this, nullptr, pipelines, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIndirectCommandBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::UInteger index)
{
    _MTL_msg_v_setIndirectCommandBuffer_atIndex__MTL__IndirectCommandBufferp_NS__UInteger((const void*)this, nullptr, indirectCommandBuffer, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIndirectCommandBuffers(const MTL::IndirectCommandBuffer* const * buffers, NS::Range range)
{
    _MTL_msg_v_setIndirectCommandBuffers_withRange__constMTL__IndirectCommandBufferpconstp_NS__Range((const void*)this, nullptr, buffers, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger index)
{
    _MTL_msg_v_setAccelerationStructure_atIndex__MTL__AccelerationStructurep_NS__UInteger((const void*)this, nullptr, accelerationStructure, index);
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::ArgumentEncoder::newArgumentEncoder(NS::UInteger index)
{
    return _MTL_msg_MTL__ArgumentEncoderp_newArgumentEncoderForBufferAtIndex__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setVisibleFunctionTable(MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger index)
{
    _MTL_msg_v_setVisibleFunctionTable_atIndex__MTL__VisibleFunctionTablep_NS__UInteger((const void*)this, nullptr, visibleFunctionTable, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setVisibleFunctionTables(const MTL::VisibleFunctionTable* const * visibleFunctionTables, NS::Range range)
{
    _MTL_msg_v_setVisibleFunctionTables_withRange__constMTL__VisibleFunctionTablepconstp_NS__Range((const void*)this, nullptr, visibleFunctionTables, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger index)
{
    _MTL_msg_v_setIntersectionFunctionTable_atIndex__MTL__IntersectionFunctionTablep_NS__UInteger((const void*)this, nullptr, intersectionFunctionTable, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range)
{
    _MTL_msg_v_setIntersectionFunctionTables_withRange__constMTL__IntersectionFunctionTablepconstp_NS__Range((const void*)this, nullptr, intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setDepthStencilState(MTL::DepthStencilState* depthStencilState, NS::UInteger index)
{
    _MTL_msg_v_setDepthStencilState_atIndex__MTL__DepthStencilStatep_NS__UInteger((const void*)this, nullptr, depthStencilState, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setDepthStencilStates(const MTL::DepthStencilState* const * depthStencilStates, NS::Range range)
{
    _MTL_msg_v_setDepthStencilStates_withRange__constMTL__DepthStencilStatepconstp_NS__Range((const void*)this, nullptr, depthStencilStates, range);
}
