//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXSpatialScaler.hpp
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

#include "../Metal/Metal.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL4FX
{
    class SpatialScaler;
}

namespace MTLFX
{
    _MTLFX_ENUM( NS::Integer, SpatialScalerColorProcessingMode )
    {
        SpatialScalerColorProcessingModePerceptual  = 0,
        SpatialScalerColorProcessingModeLinear      = 1,
        SpatialScalerColorProcessingModeHDR         = 2
    };

    class SpatialScalerDescriptor : public NS::Copying< SpatialScalerDescriptor >
    {
    public:
        static class SpatialScalerDescriptor*       alloc();
        class SpatialScalerDescriptor*              init();

        MTL::PixelFormat                            colorTextureFormat() const;
        void                                        setColorTextureFormat( MTL::PixelFormat format );

        MTL::PixelFormat                            outputTextureFormat() const;
        void                                        setOutputTextureFormat( MTL::PixelFormat format );

        NS::UInteger                                inputWidth() const;
        void                                        setInputWidth( NS::UInteger width );

        NS::UInteger                                inputHeight() const;
        void                                        setInputHeight( NS::UInteger height );

        NS::UInteger                                outputWidth() const;
        void                                        setOutputWidth( NS::UInteger width );

        NS::UInteger                                outputHeight() const;
        void                                        setOutputHeight( NS::UInteger height );

        SpatialScalerColorProcessingMode            colorProcessingMode() const;
        void                                        setColorProcessingMode( SpatialScalerColorProcessingMode mode );

        class SpatialScaler*                        newSpatialScaler( const MTL::Device* pDevice ) const;
        MTL4FX::SpatialScaler*                      newSpatialScaler( const MTL::Device* pDevice, const MTL4::Compiler* pCompiler ) const;

        static bool                                 supportsDevice( const MTL::Device* pDevice);
        static bool                                 supportsMetal4FX( const MTL::Device* pDevice );
    };

    class SpatialScalerBase : public NS::Referencing< SpatialScaler >
    {
    public:
        MTL::TextureUsage                           colorTextureUsage() const;
        MTL::TextureUsage                           outputTextureUsage() const;

        NS::UInteger                                inputContentWidth() const;
        void                                        setInputContentWidth( NS::UInteger width );

        NS::UInteger                                inputContentHeight() const;
        void                                        setInputContentHeight( NS::UInteger height );

        MTL::Texture*                               colorTexture() const;
        void                                        setColorTexture( MTL::Texture* pTexture );

        MTL::Texture*                               outputTexture() const;
        void                                        setOutputTexture( MTL::Texture* pTexture );

        MTL::PixelFormat                            colorTextureFormat() const;
        MTL::PixelFormat                            outputTextureFormat() const;
        NS::UInteger                                inputWidth() const;
        NS::UInteger                                inputHeight() const;
        NS::UInteger                                outputWidth() const;
        NS::UInteger                                outputHeight() const;
        SpatialScalerColorProcessingMode            colorProcessingMode() const;

        MTL::Fence*                                 fence() const;
        void                                        setFence( MTL::Fence* pFence );
    };

    class SpatialScaler : public NS::Referencing< SpatialScaler, SpatialScalerBase >
    {
    public:
        void                                        encodeToCommandBuffer( MTL::CommandBuffer* pCommandBuffer );
    };
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::SpatialScalerDescriptor* MTLFX::SpatialScalerDescriptor::alloc()
{
    return NS::Object::alloc< SpatialScalerDescriptor >( _MTLFX_PRIVATE_CLS( MTLFXSpatialScalerDescriptor ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::SpatialScalerDescriptor* MTLFX::SpatialScalerDescriptor::init()
{
    return NS::Object::init< SpatialScalerDescriptor >();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerDescriptor::colorTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setColorTextureFormat( MTL::PixelFormat format )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTextureFormat_ ), format );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerDescriptor::outputTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setOutputTextureFormat( MTL::PixelFormat format )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTextureFormat_ ), format );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::inputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setInputWidth( NS::UInteger width )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputWidth_ ), width );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::inputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setInputHeight( NS::UInteger height )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputHeight_ ), height );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::outputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setOutputWidth( NS::UInteger width )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputWidth_ ), width );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::outputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setOutputHeight( NS::UInteger height )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputHeight_ ), height );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::SpatialScalerColorProcessingMode MTLFX::SpatialScalerDescriptor::colorProcessingMode() const
{
    return Object::sendMessage< SpatialScalerColorProcessingMode >( this, _MTLFX_PRIVATE_SEL( colorProcessingMode ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setColorProcessingMode( SpatialScalerColorProcessingMode mode )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorProcessingMode_ ), mode );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::SpatialScaler* MTLFX::SpatialScalerDescriptor::newSpatialScaler( const MTL::Device* pDevice ) const
{
    return Object::sendMessage< SpatialScaler* >( this, _MTLFX_PRIVATE_SEL( newSpatialScalerWithDevice_ ), pDevice );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL4FX::SpatialScaler* MTLFX::SpatialScalerDescriptor::newSpatialScaler( const MTL::Device* pDevice, const MTL4::Compiler* pCompiler ) const
{
    return Object::sendMessage< MTL4FX::SpatialScaler* >( this, _MTLFX_PRIVATE_SEL( newSpatialScalerWithDevice_compiler_ ), pDevice, pCompiler );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::SpatialScalerDescriptor::supportsDevice( const MTL::Device* pDevice )
{
    return Object::sendMessageSafe< bool >( _NS_PRIVATE_CLS( MTLFXSpatialScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportsDevice_ ), pDevice );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::SpatialScalerDescriptor::supportsMetal4FX( const MTL::Device* pDevice )
{
    return Object::sendMessageSafe< bool >( _NS_PRIVATE_CLS( MTLFXSpatialScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportsMetal4FX_ ), pDevice );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::SpatialScalerBase::colorTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( colorTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::SpatialScalerBase::outputTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( outputTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputContentWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputContentWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setInputContentWidth( NS::UInteger width )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentWidth_ ), width );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputContentHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputContentHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setInputContentHeight( NS::UInteger height )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentHeight_ ), height );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::SpatialScalerBase::colorTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( colorTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setColorTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::SpatialScalerBase::outputTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( outputTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setOutputTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerBase::colorTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerBase::outputTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::outputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::outputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::SpatialScalerColorProcessingMode MTLFX::SpatialScalerBase::colorProcessingMode() const
{
    return Object::sendMessage< SpatialScalerColorProcessingMode >( this, _MTLFX_PRIVATE_SEL( colorProcessingMode ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Fence* MTLFX::SpatialScalerBase::fence() const
{
    return Object::sendMessage< MTL::Fence* >( this, _MTLFX_PRIVATE_SEL( fence ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setFence( MTL::Fence* pFence )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setFence_ ), pFence );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::SpatialScaler::encodeToCommandBuffer( MTL::CommandBuffer* pCommandBuffer )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( encodeToCommandBuffer_ ), pCommandBuffer );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
