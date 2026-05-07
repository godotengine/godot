//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLParallelRenderCommandEncoder.hpp
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
#include "MTLCommandEncoder.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLRenderPass.hpp"

namespace MTL
{
class RenderCommandEncoder;

class ParallelRenderCommandEncoder : public NS::Referencing<ParallelRenderCommandEncoder, CommandEncoder>
{
public:
    RenderCommandEncoder* renderCommandEncoder();

    void                  setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex);
    void                  setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex);

    void                  setDepthStoreAction(MTL::StoreAction storeAction);
    void                  setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions);

    void                  setStencilStoreAction(MTL::StoreAction storeAction);
    void                  setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions);
};

}
_MTL_INLINE MTL::RenderCommandEncoder* MTL::ParallelRenderCommandEncoder::renderCommandEncoder()
{
    return Object::sendMessage<MTL::RenderCommandEncoder*>(this, _MTL_PRIVATE_SEL(renderCommandEncoder));
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorStoreAction_atIndex_), storeAction, colorAttachmentIndex);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorStoreActionOptions_atIndex_), storeActionOptions, colorAttachmentIndex);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setDepthStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStoreAction_), storeAction);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStoreActionOptions_), storeActionOptions);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setStencilStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilStoreAction_), storeAction);
}

_MTL_INLINE void MTL::ParallelRenderCommandEncoder::setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilStoreActionOptions_), storeActionOptions);
}
