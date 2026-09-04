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
    class RenderCommandEncoder;
    enum StoreAction : NS::UInteger;
    using StoreActionOptions = NS::UInteger;
}

namespace MTL
{

class ParallelRenderCommandEncoder : public NS::Referencing<ParallelRenderCommandEncoder, MTL::CommandEncoder>
{
public:
    MTL::RenderCommandEncoder* renderCommandEncoder();
    void                       setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex);
    void                       setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex);
    void                       setDepthStoreAction(MTL::StoreAction storeAction);
    void                       setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions);
    void                       setStencilStoreAction(MTL::StoreAction storeAction);
    void                       setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLParallelRenderCommandEncoder;

_MTL_INLINE MTL::RenderCommandEncoder* MTL::ParallelRenderCommandEncoder::renderCommandEncoder()
{
    return _MTL_msg_MTL__RenderCommandEncoderp_renderCommandEncoder((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex)
{
    _MTL_msg_v_setColorStoreAction_atIndex__MTL__StoreAction_NS__UInteger((const void*)this, nullptr, storeAction, colorAttachmentIndex);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setDepthStoreAction(MTL::StoreAction storeAction)
{
    _MTL_msg_v_setDepthStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setStencilStoreAction(MTL::StoreAction storeAction)
{
    _MTL_msg_v_setStencilStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex)
{
    _MTL_msg_v_setColorStoreActionOptions_atIndex__MTL__StoreActionOptions_NS__UInteger((const void*)this, nullptr, storeActionOptions, colorAttachmentIndex);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    _MTL_msg_v_setDepthStoreActionOptions__MTL__StoreActionOptions((const void*)this, nullptr, storeActionOptions);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    _MTL_msg_v_setStencilStoreActionOptions__MTL__StoreActionOptions((const void*)this, nullptr, storeActionOptions);
}
