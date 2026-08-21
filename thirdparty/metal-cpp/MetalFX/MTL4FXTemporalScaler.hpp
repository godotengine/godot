#pragma once

#include "MTL4FXDefines.hpp"
#include "MTL4FXBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLFXTemporalScaler.hpp"

namespace MTL4 {
    class CommandBuffer;
}

namespace MTL4FX
{

class TemporalScaler : public NS::Referencing<TemporalScaler, MTLFX::TemporalScalerBase>
{
public:
    void encodeToCommandBuffer(MTL4::CommandBuffer* commandBuffer);

};

} // namespace MTL4FX

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4FXTemporalScaler;

_MTL4FX_INLINE void MTL4FX::TemporalScaler::encodeToCommandBuffer(MTL4::CommandBuffer* commandBuffer)
{
    _MTL4FX_msg_v_encodeToCommandBuffer__MTL4__CommandBufferp((const void*)this, nullptr, commandBuffer);
}
