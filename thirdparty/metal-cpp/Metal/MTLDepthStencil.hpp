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

_MTL_ENUM(NS::UInteger, CompareFunction) {
    CompareFunctionNever = 0,
    CompareFunctionLess = 1,
    CompareFunctionEqual = 2,
    CompareFunctionLessEqual = 3,
    CompareFunctionGreater = 4,
    CompareFunctionNotEqual = 5,
    CompareFunctionGreaterEqual = 6,
    CompareFunctionAlways = 7,
};

_MTL_ENUM(NS::UInteger, StencilOperation) {
    StencilOperationKeep = 0,
    StencilOperationZero = 1,
    StencilOperationReplace = 2,
    StencilOperationIncrementClamp = 3,
    StencilOperationDecrementClamp = 4,
    StencilOperationInvert = 5,
    StencilOperationIncrementWrap = 6,
    StencilOperationDecrementWrap = 7,
};


class StencilDescriptor;
class DepthStencilDescriptor;
class DepthStencilState;

class StencilDescriptor : public NS::Copying<StencilDescriptor>
{
public:
    static StencilDescriptor* alloc();
    StencilDescriptor*        init() const;

    MTL::StencilOperation depthFailureOperation() const;
    MTL::StencilOperation depthStencilPassOperation() const;
    uint32_t              readMask() const;
    void                  setDepthFailureOperation(MTL::StencilOperation depthFailureOperation);
    void                  setDepthStencilPassOperation(MTL::StencilOperation depthStencilPassOperation);
    void                  setReadMask(uint32_t readMask);
    void                  setStencilCompareFunction(MTL::CompareFunction stencilCompareFunction);
    void                  setStencilFailureOperation(MTL::StencilOperation stencilFailureOperation);
    void                  setWriteMask(uint32_t writeMask);
    MTL::CompareFunction  stencilCompareFunction() const;
    MTL::StencilOperation stencilFailureOperation() const;
    uint32_t              writeMask() const;

};

class DepthStencilDescriptor : public NS::Copying<DepthStencilDescriptor>
{
public:
    static DepthStencilDescriptor* alloc();
    DepthStencilDescriptor*        init() const;

    MTL::StencilDescriptor* backFaceStencil() const;
    MTL::CompareFunction    depthCompareFunction() const;
    bool                    depthWriteEnabled() const;
    MTL::StencilDescriptor* frontFaceStencil() const;
    bool                    isDepthWriteEnabled();
    NS::String*             label() const;
    void                    setBackFaceStencil(MTL::StencilDescriptor* backFaceStencil);
    void                    setDepthCompareFunction(MTL::CompareFunction depthCompareFunction);
    void                    setDepthWriteEnabled(bool depthWriteEnabled);
    void                    setFrontFaceStencil(MTL::StencilDescriptor* frontFaceStencil);
    void                    setLabel(NS::String* label);

};

class DepthStencilState : public NS::Referencing<DepthStencilState>
{
public:
    MTL::Device*    device() const;
    MTL::ResourceID gpuResourceID() const;
    NS::String*     label() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLStencilDescriptor;
extern "C" void *OBJC_CLASS_$_MTLDepthStencilDescriptor;
extern "C" void *OBJC_CLASS_$_MTLDepthStencilState;

_MTL_INLINE MTL::StencilDescriptor* MTL::StencilDescriptor::alloc()
{
    return _MTL_msg_MTL__StencilDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLStencilDescriptor, nullptr);
}

_MTL_INLINE MTL::StencilDescriptor* MTL::StencilDescriptor::init() const
{
    return _MTL_msg_MTL__StencilDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CompareFunction MTL::StencilDescriptor::stencilCompareFunction() const
{
    return _MTL_msg_MTL__CompareFunction_stencilCompareFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StencilDescriptor::setStencilCompareFunction(MTL::CompareFunction stencilCompareFunction)
{
    _MTL_msg_v_setStencilCompareFunction__MTL__CompareFunction((const void*)this, nullptr, stencilCompareFunction);
}

_MTL_INLINE MTL::StencilOperation MTL::StencilDescriptor::stencilFailureOperation() const
{
    return _MTL_msg_MTL__StencilOperation_stencilFailureOperation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StencilDescriptor::setStencilFailureOperation(MTL::StencilOperation stencilFailureOperation)
{
    _MTL_msg_v_setStencilFailureOperation__MTL__StencilOperation((const void*)this, nullptr, stencilFailureOperation);
}

_MTL_INLINE MTL::StencilOperation MTL::StencilDescriptor::depthFailureOperation() const
{
    return _MTL_msg_MTL__StencilOperation_depthFailureOperation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StencilDescriptor::setDepthFailureOperation(MTL::StencilOperation depthFailureOperation)
{
    _MTL_msg_v_setDepthFailureOperation__MTL__StencilOperation((const void*)this, nullptr, depthFailureOperation);
}

_MTL_INLINE MTL::StencilOperation MTL::StencilDescriptor::depthStencilPassOperation() const
{
    return _MTL_msg_MTL__StencilOperation_depthStencilPassOperation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StencilDescriptor::setDepthStencilPassOperation(MTL::StencilOperation depthStencilPassOperation)
{
    _MTL_msg_v_setDepthStencilPassOperation__MTL__StencilOperation((const void*)this, nullptr, depthStencilPassOperation);
}

_MTL_INLINE uint32_t MTL::StencilDescriptor::readMask() const
{
    return _MTL_msg_uint32_t_readMask((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StencilDescriptor::setReadMask(uint32_t readMask)
{
    _MTL_msg_v_setReadMask__uint32_t((const void*)this, nullptr, readMask);
}

_MTL_INLINE uint32_t MTL::StencilDescriptor::writeMask() const
{
    return _MTL_msg_uint32_t_writeMask((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StencilDescriptor::setWriteMask(uint32_t writeMask)
{
    _MTL_msg_v_setWriteMask__uint32_t((const void*)this, nullptr, writeMask);
}

_MTL_INLINE MTL::DepthStencilDescriptor* MTL::DepthStencilDescriptor::alloc()
{
    return _MTL_msg_MTL__DepthStencilDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLDepthStencilDescriptor, nullptr);
}

_MTL_INLINE MTL::DepthStencilDescriptor* MTL::DepthStencilDescriptor::init() const
{
    return _MTL_msg_MTL__DepthStencilDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CompareFunction MTL::DepthStencilDescriptor::depthCompareFunction() const
{
    return _MTL_msg_MTL__CompareFunction_depthCompareFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setDepthCompareFunction(MTL::CompareFunction depthCompareFunction)
{
    _MTL_msg_v_setDepthCompareFunction__MTL__CompareFunction((const void*)this, nullptr, depthCompareFunction);
}

_MTL_INLINE bool MTL::DepthStencilDescriptor::depthWriteEnabled() const
{
    return _MTL_msg_bool_depthWriteEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setDepthWriteEnabled(bool depthWriteEnabled)
{
    _MTL_msg_v_setDepthWriteEnabled__bool((const void*)this, nullptr, depthWriteEnabled);
}

_MTL_INLINE MTL::StencilDescriptor* MTL::DepthStencilDescriptor::frontFaceStencil() const
{
    return _MTL_msg_MTL__StencilDescriptorp_frontFaceStencil((const void*)this, nullptr);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setFrontFaceStencil(MTL::StencilDescriptor* frontFaceStencil)
{
    _MTL_msg_v_setFrontFaceStencil__MTL__StencilDescriptorp((const void*)this, nullptr, frontFaceStencil);
}

_MTL_INLINE MTL::StencilDescriptor* MTL::DepthStencilDescriptor::backFaceStencil() const
{
    return _MTL_msg_MTL__StencilDescriptorp_backFaceStencil((const void*)this, nullptr);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setBackFaceStencil(MTL::StencilDescriptor* backFaceStencil)
{
    _MTL_msg_v_setBackFaceStencil__MTL__StencilDescriptorp((const void*)this, nullptr, backFaceStencil);
}

_MTL_INLINE NS::String* MTL::DepthStencilDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE bool MTL::DepthStencilDescriptor::isDepthWriteEnabled()
{
    return _MTL_msg_bool_isDepthWriteEnabled((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::DepthStencilState::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::DepthStencilState::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::DepthStencilState::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}
