//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLDepthStencil.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class DepthStencilDescriptor;
class Device;
class StencilDescriptor;
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

class StencilDescriptor : public NS::Copying<StencilDescriptor>
{
public:
    static StencilDescriptor* alloc();

    StencilOperation          depthFailureOperation() const;

    StencilOperation          depthStencilPassOperation() const;

    StencilDescriptor*        init();

    uint32_t                  readMask() const;

    void                      setDepthFailureOperation(MTL::StencilOperation depthFailureOperation);

    void                      setDepthStencilPassOperation(MTL::StencilOperation depthStencilPassOperation);

    void                      setReadMask(uint32_t readMask);

    void                      setStencilCompareFunction(MTL::CompareFunction stencilCompareFunction);

    void                      setStencilFailureOperation(MTL::StencilOperation stencilFailureOperation);

    void                      setWriteMask(uint32_t writeMask);

    CompareFunction           stencilCompareFunction() const;

    StencilOperation          stencilFailureOperation() const;

    uint32_t                  writeMask() const;
};
class DepthStencilDescriptor : public NS::Copying<DepthStencilDescriptor>
{
public:
    static DepthStencilDescriptor* alloc();

    StencilDescriptor*             backFaceStencil() const;

    CompareFunction                depthCompareFunction() const;

    [[deprecated("please use isDepthWriteEnabled instead")]]
    bool                    depthWriteEnabled() const;

    StencilDescriptor*      frontFaceStencil() const;

    DepthStencilDescriptor* init();

    bool                    isDepthWriteEnabled() const;

    NS::String*             label() const;

    void                    setBackFaceStencil(const MTL::StencilDescriptor* backFaceStencil);

    void                    setDepthCompareFunction(MTL::CompareFunction depthCompareFunction);

    void                    setDepthWriteEnabled(bool depthWriteEnabled);

    void                    setFrontFaceStencil(const MTL::StencilDescriptor* frontFaceStencil);

    void                    setLabel(const NS::String* label);
};
class DepthStencilState : public NS::Referencing<DepthStencilState>
{
public:
    Device*     device() const;

    ResourceID  gpuResourceID() const;

    NS::String* label() const;
};

}
_MTL_INLINE MTL::StencilDescriptor* MTL::StencilDescriptor::alloc()
{
    return NS::Object::alloc<MTL::StencilDescriptor>(_MTL_PRIVATE_CLS(MTLStencilDescriptor));
}

_MTL_INLINE MTL::StencilOperation MTL::StencilDescriptor::depthFailureOperation() const
{
    return Object::sendMessage<MTL::StencilOperation>(this, _MTL_PRIVATE_SEL(depthFailureOperation));
}

_MTL_INLINE MTL::StencilOperation MTL::StencilDescriptor::depthStencilPassOperation() const
{
    return Object::sendMessage<MTL::StencilOperation>(this, _MTL_PRIVATE_SEL(depthStencilPassOperation));
}

_MTL_INLINE MTL::StencilDescriptor* MTL::StencilDescriptor::init()
{
    return NS::Object::init<MTL::StencilDescriptor>();
}

_MTL_INLINE uint32_t MTL::StencilDescriptor::readMask() const
{
    return Object::sendMessage<uint32_t>(this, _MTL_PRIVATE_SEL(readMask));
}

_MTL_INLINE void MTL::StencilDescriptor::setDepthFailureOperation(MTL::StencilOperation depthFailureOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthFailureOperation_), depthFailureOperation);
}

_MTL_INLINE void MTL::StencilDescriptor::setDepthStencilPassOperation(MTL::StencilOperation depthStencilPassOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStencilPassOperation_), depthStencilPassOperation);
}

_MTL_INLINE void MTL::StencilDescriptor::setReadMask(uint32_t readMask)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setReadMask_), readMask);
}

_MTL_INLINE void MTL::StencilDescriptor::setStencilCompareFunction(MTL::CompareFunction stencilCompareFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilCompareFunction_), stencilCompareFunction);
}

_MTL_INLINE void MTL::StencilDescriptor::setStencilFailureOperation(MTL::StencilOperation stencilFailureOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilFailureOperation_), stencilFailureOperation);
}

_MTL_INLINE void MTL::StencilDescriptor::setWriteMask(uint32_t writeMask)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setWriteMask_), writeMask);
}

_MTL_INLINE MTL::CompareFunction MTL::StencilDescriptor::stencilCompareFunction() const
{
    return Object::sendMessage<MTL::CompareFunction>(this, _MTL_PRIVATE_SEL(stencilCompareFunction));
}

_MTL_INLINE MTL::StencilOperation MTL::StencilDescriptor::stencilFailureOperation() const
{
    return Object::sendMessage<MTL::StencilOperation>(this, _MTL_PRIVATE_SEL(stencilFailureOperation));
}

_MTL_INLINE uint32_t MTL::StencilDescriptor::writeMask() const
{
    return Object::sendMessage<uint32_t>(this, _MTL_PRIVATE_SEL(writeMask));
}

_MTL_INLINE MTL::DepthStencilDescriptor* MTL::DepthStencilDescriptor::alloc()
{
    return NS::Object::alloc<MTL::DepthStencilDescriptor>(_MTL_PRIVATE_CLS(MTLDepthStencilDescriptor));
}

_MTL_INLINE MTL::StencilDescriptor* MTL::DepthStencilDescriptor::backFaceStencil() const
{
    return Object::sendMessage<MTL::StencilDescriptor*>(this, _MTL_PRIVATE_SEL(backFaceStencil));
}

_MTL_INLINE MTL::CompareFunction MTL::DepthStencilDescriptor::depthCompareFunction() const
{
    return Object::sendMessage<MTL::CompareFunction>(this, _MTL_PRIVATE_SEL(depthCompareFunction));
}

_MTL_INLINE bool MTL::DepthStencilDescriptor::depthWriteEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isDepthWriteEnabled));
}

_MTL_INLINE MTL::StencilDescriptor* MTL::DepthStencilDescriptor::frontFaceStencil() const
{
    return Object::sendMessage<MTL::StencilDescriptor*>(this, _MTL_PRIVATE_SEL(frontFaceStencil));
}

_MTL_INLINE MTL::DepthStencilDescriptor* MTL::DepthStencilDescriptor::init()
{
    return NS::Object::init<MTL::DepthStencilDescriptor>();
}

_MTL_INLINE bool MTL::DepthStencilDescriptor::isDepthWriteEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isDepthWriteEnabled));
}

_MTL_INLINE NS::String* MTL::DepthStencilDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setBackFaceStencil(const MTL::StencilDescriptor* backFaceStencil)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBackFaceStencil_), backFaceStencil);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setDepthCompareFunction(MTL::CompareFunction depthCompareFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthCompareFunction_), depthCompareFunction);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setDepthWriteEnabled(bool depthWriteEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthWriteEnabled_), depthWriteEnabled);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setFrontFaceStencil(const MTL::StencilDescriptor* frontFaceStencil)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFrontFaceStencil_), frontFaceStencil);
}

_MTL_INLINE void MTL::DepthStencilDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::Device* MTL::DepthStencilState::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::ResourceID MTL::DepthStencilState::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE NS::String* MTL::DepthStencilState::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}
