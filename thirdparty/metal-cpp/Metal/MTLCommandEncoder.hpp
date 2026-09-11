#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
}
namespace NS {
    class String;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, ResourceUsage) {
    ResourceUsageRead = 1 << 0,
    ResourceUsageWrite = 1 << 1,
    ResourceUsageSample = 1 << 2,
};

_MTL_OPTIONS(NS::UInteger, BarrierScope) {
    BarrierScopeBuffers = 1 << 0,
    BarrierScopeTextures = 1 << 1,
    BarrierScopeRenderTargets = 1 << 2,
};

_MTL_OPTIONS(NS::UInteger, Stages) {
    StageVertex = 1 << 0,
    StageFragment = 1 << 1,
    StageTile = 1 << 2,
    StageObject = 1 << 3,
    StageMesh = 1 << 4,
    StageResourceState = 1 << 26,
    StageDispatch = 1 << 27,
    StageBlit = 1 << 28,
    StageAccelerationStructure = 1 << 29,
    StageMachineLearning = 1 << 30,
    StageAll = 9223372036854775807,
};


class CommandEncoder : public NS::Referencing<CommandEncoder>
{
public:
    void         barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages);
    MTL::Device* device() const;
    void         endEncoding();
    void         insertDebugSignpost(NS::String* string);
    NS::String*  label() const;
    void         popDebugGroup();
    void         pushDebugGroup(NS::String* string);
    void         setLabel(NS::String* label);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLCommandEncoder;

_MTL_INLINE MTL::Device* MTL::CommandEncoder::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::CommandEncoder::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandEncoder::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE void MTL::CommandEncoder::endEncoding()
{
    _MTL_msg_v_endEncoding((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandEncoder::barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages)
{
    _MTL_msg_v_barrierAfterQueueStages_beforeStages__MTL__Stages_MTL__Stages((const void*)this, nullptr, afterQueueStages, beforeStages);
}

_MTL_INLINE void MTL::CommandEncoder::insertDebugSignpost(NS::String* string)
{
    _MTL_msg_v_insertDebugSignpost__NS__Stringp((const void*)this, nullptr, string);
}

_MTL_INLINE void MTL::CommandEncoder::pushDebugGroup(NS::String* string)
{
    _MTL_msg_v_pushDebugGroup__NS__Stringp((const void*)this, nullptr, string);
}

_MTL_INLINE void MTL::CommandEncoder::popDebugGroup()
{
    _MTL_msg_v_popDebugGroup((const void*)this, nullptr);
}
