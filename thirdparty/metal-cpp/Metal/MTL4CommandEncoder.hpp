#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Fence;
    using Stages = NS::UInteger;
}
namespace MTL4 {
    class CommandBuffer;
}
namespace NS {
    class String;
}

namespace MTL4
{

_MTL4_OPTIONS(NS::UInteger, VisibilityOptions) {
    VisibilityOptionNone = 0,
    VisibilityOptionDevice = 1 << 0,
    VisibilityOptionResourceAlias = 1 << 1,
};


class CommandEncoder : public NS::Referencing<CommandEncoder>
{
public:
    void                 barrierAfterEncoderStages(MTL::Stages afterEncoderStages, MTL::Stages beforeEncoderStages, MTL4::VisibilityOptions visibilityOptions);
    void                 barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages, MTL4::VisibilityOptions visibilityOptions);
    void                 barrierAfterStages(MTL::Stages afterStages, MTL::Stages beforeQueueStages, MTL4::VisibilityOptions visibilityOptions);
    MTL4::CommandBuffer* commandBuffer() const;
    void                 endEncoding();
    void                 insertDebugSignpost(NS::String* string);
    NS::String*          label() const;
    void                 popDebugGroup();
    void                 pushDebugGroup(NS::String* string);
    void                 setLabel(NS::String* label);
    void                 updateFence(MTL::Fence* fence, MTL::Stages afterEncoderStages);
    void                 waitForFence(MTL::Fence* fence, MTL::Stages beforeEncoderStages);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CommandEncoder;

_MTL4_INLINE NS::String* MTL4::CommandEncoder::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandEncoder::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL4::CommandBuffer* MTL4::CommandEncoder::commandBuffer() const
{
    return _MTL4_msg_MTL4__CommandBufferp_commandBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandEncoder::barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages, MTL4::VisibilityOptions visibilityOptions)
{
    _MTL4_msg_v_barrierAfterQueueStages_beforeStages_visibilityOptions__MTL__Stages_MTL__Stages_MTL4__VisibilityOptions((const void*)this, nullptr, afterQueueStages, beforeStages, visibilityOptions);
}

_MTL4_INLINE void MTL4::CommandEncoder::barrierAfterStages(MTL::Stages afterStages, MTL::Stages beforeQueueStages, MTL4::VisibilityOptions visibilityOptions)
{
    _MTL4_msg_v_barrierAfterStages_beforeQueueStages_visibilityOptions__MTL__Stages_MTL__Stages_MTL4__VisibilityOptions((const void*)this, nullptr, afterStages, beforeQueueStages, visibilityOptions);
}

_MTL4_INLINE void MTL4::CommandEncoder::barrierAfterEncoderStages(MTL::Stages afterEncoderStages, MTL::Stages beforeEncoderStages, MTL4::VisibilityOptions visibilityOptions)
{
    _MTL4_msg_v_barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions__MTL__Stages_MTL__Stages_MTL4__VisibilityOptions((const void*)this, nullptr, afterEncoderStages, beforeEncoderStages, visibilityOptions);
}

_MTL4_INLINE void MTL4::CommandEncoder::updateFence(MTL::Fence* fence, MTL::Stages afterEncoderStages)
{
    _MTL4_msg_v_updateFence_afterEncoderStages__MTL__Fencep_MTL__Stages((const void*)this, nullptr, fence, afterEncoderStages);
}

_MTL4_INLINE void MTL4::CommandEncoder::waitForFence(MTL::Fence* fence, MTL::Stages beforeEncoderStages)
{
    _MTL4_msg_v_waitForFence_beforeEncoderStages__MTL__Fencep_MTL__Stages((const void*)this, nullptr, fence, beforeEncoderStages);
}

_MTL4_INLINE void MTL4::CommandEncoder::insertDebugSignpost(NS::String* string)
{
    _MTL4_msg_v_insertDebugSignpost__NS__Stringp((const void*)this, nullptr, string);
}

_MTL4_INLINE void MTL4::CommandEncoder::pushDebugGroup(NS::String* string)
{
    _MTL4_msg_v_pushDebugGroup__NS__Stringp((const void*)this, nullptr, string);
}

_MTL4_INLINE void MTL4::CommandEncoder::popDebugGroup()
{
    _MTL4_msg_v_popDebugGroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandEncoder::endEncoding()
{
    _MTL4_msg_v_endEncoding((const void*)this, nullptr);
}
