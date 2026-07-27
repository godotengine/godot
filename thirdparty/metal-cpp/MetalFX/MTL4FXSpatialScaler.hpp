//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTL4FXSpatialScaler.hpp
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "MTLFXDefines.hpp"
#include "MTLFXPrivate.hpp"

#include "MTLFXSpatialScaler.hpp"
#include "../Metal/Metal.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL4FX
{
    class SpatialScaler : public NS::Referencing< SpatialScaler, MTLFX::SpatialScalerBase >
    {
    public:
        void encodeToCommandBuffer( MTL4::CommandBuffer* pCommandBuffer );
    };
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTL4FX::SpatialScaler::encodeToCommandBuffer( MTL4::CommandBuffer* pCommandBuffer )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( encodeToCommandBuffer_ ), pCommandBuffer );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
