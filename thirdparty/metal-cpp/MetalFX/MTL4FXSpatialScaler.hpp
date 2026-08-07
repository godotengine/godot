#pragma once

#include "MTL4FXDefines.hpp"
#include "MTL4FXBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLFXSpatialScaler.hpp"

namespace MTL4 {
    class CommandBuffer;
}

namespace MTL4FX
{

class SpatialScaler : public NS::Referencing<SpatialScaler, MTLFX::SpatialScalerBase>
{
public:
    void encodeToCommandBuffer(MTL4::CommandBuffer* commandBuffer);

};

} // namespace MTL4FX

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4FXSpatialScaler;

_MTL4FX_INLINE void MTL4FX::SpatialScaler::encodeToCommandBuffer(MTL4::CommandBuffer* commandBuffer)
{
    _MTL4FX_msg_v_encodeToCommandBuffer__MTL4__CommandBufferp((const void*)this, nullptr, commandBuffer);
}
