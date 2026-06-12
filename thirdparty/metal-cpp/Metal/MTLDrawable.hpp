#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL
{

class Drawable : public NS::Referencing<Drawable>
{
public:
    void           addPresentedHandler(MTL::DrawablePresentedHandler block);
    void           addPresentedHandler(const MTL::DrawablePresentedHandlerFunction& block);
    NS::UInteger   drawableID() const;
    void           present();
    void           present(CFTimeInterval presentationTime);
    void           presentAfterMinimumDuration(CFTimeInterval duration);
    CFTimeInterval presentedTime() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLDrawable;

_MTL_INLINE CFTimeInterval MTL::Drawable::presentedTime() const
{
    return _MTL_msg_CFTimeInterval_presentedTime((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Drawable::drawableID() const
{
    return _MTL_msg_NS__UInteger_drawableID((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Drawable::present()
{
    _MTL_msg_v_present((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Drawable::present(CFTimeInterval presentationTime)
{
    _MTL_msg_v_presentAtTime__CFTimeInterval((const void*)this, nullptr, presentationTime);
}

_MTL_INLINE void MTL::Drawable::presentAfterMinimumDuration(CFTimeInterval duration)
{
    _MTL_msg_v_presentAfterMinimumDuration__CFTimeInterval((const void*)this, nullptr, duration);
}

_MTL_INLINE void MTL::Drawable::addPresentedHandler(MTL::DrawablePresentedHandler block)
{
    _MTL_msg_v_addPresentedHandler__MTL__DrawablePresentedHandler((const void*)this, nullptr, block);
}

_MTL_INLINE void MTL::Drawable::addPresentedHandler(const MTL::DrawablePresentedHandlerFunction& block)
{
    __block MTL::DrawablePresentedHandlerFunction blockFunction = block;
    addPresentedHandler(^(MTL::Drawable* x0) { blockFunction(x0); });
}
