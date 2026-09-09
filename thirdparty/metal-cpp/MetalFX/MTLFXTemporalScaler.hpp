//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXTemporalScaler.hpp
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
    class TemporalScaler;
}

namespace MTLFX
{
    class TemporalScalerDescriptor : public NS::Copying< TemporalScalerDescriptor >
    {
    public:
        static class TemporalScalerDescriptor*      alloc();
        class TemporalScalerDescriptor*             init();

        MTL::PixelFormat                            colorTextureFormat() const;
        void                                        setColorTextureFormat( MTL::PixelFormat format );

        MTL::PixelFormat                            depthTextureFormat() const;
        void                                        setDepthTextureFormat( MTL::PixelFormat format );

        MTL::PixelFormat                            motionTextureFormat() const;
        void                                        setMotionTextureFormat( MTL::PixelFormat format );

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

        bool                                        isAutoExposureEnabled() const;
        void                                        setAutoExposureEnabled( bool enabled );

        bool                                        isInputContentPropertiesEnabled() const;
        void                                        setInputContentPropertiesEnabled( bool enabled );

        bool                                        requiresSynchronousInitialization() const;
        void                                        setRequiresSynchronousInitialization(bool requiresSynchronousInitialization);

        bool                                        isReactiveMaskTextureEnabled() const;
        void                                        setReactiveMaskTextureEnabled( bool enabled );

        MTL::PixelFormat                            reactiveMaskTextureFormat() const;
        void                                        setReactiveMaskTextureFormat( MTL::PixelFormat pixelFormat );

        float                                       inputContentMinScale() const;
        void                                        setInputContentMinScale( float scale );

        float                                       inputContentMaxScale() const;
        void                                        setInputContentMaxScale( float scale );

        class TemporalScaler*                       newTemporalScaler( const MTL::Device* pDevice ) const;
        MTL4FX::TemporalScaler*                     newTemporalScaler( const MTL::Device* pDevice, const MTL4::Compiler* pCompiler) const;

        static float                                supportedInputContentMinScale( const MTL::Device* pDevice );
        static float                                supportedInputContentMaxScale( const MTL::Device* pDevice );

        static bool                                 supportsDevice( const MTL::Device* pDevice );
        static bool                                 supportsMetal4FX( const MTL::Device* pDevice );
    };

    class FrameInterpolatableScaler : public NS::Copying< FrameInterpolatableScaler >
    {
	};

    class TemporalScalerBase : public NS::Referencing< TemporalScaler, FrameInterpolatableScaler >
    {
    public:
        MTL::TextureUsage                           colorTextureUsage() const;
        MTL::TextureUsage                           depthTextureUsage() const;
        MTL::TextureUsage                           motionTextureUsage() const;
        MTL::TextureUsage                           outputTextureUsage() const;

        NS::UInteger                                inputContentWidth() const;
        void                                        setInputContentWidth( NS::UInteger width );

        NS::UInteger                                inputContentHeight() const;
        void                                        setInputContentHeight( NS::UInteger height );

        MTL::Texture*                               colorTexture() const;
        void                                        setColorTexture( MTL::Texture* pTexture );

        MTL::Texture*                               depthTexture() const;
        void                                        setDepthTexture( MTL::Texture* pTexture );

        MTL::Texture*                               motionTexture() const;
        void                                        setMotionTexture( MTL::Texture* pTexture );

        MTL::Texture*                               outputTexture() const;
        void                                        setOutputTexture( MTL::Texture* pTexture );

        MTL::Texture*                               exposureTexture() const;
        void                                        setExposureTexture( MTL::Texture* pTexture );

        float                                       preExposure() const;
        void                                        setPreExposure( float preExposure );
        
        float                                       jitterOffsetX() const;
        void                                        setJitterOffsetX( float offset );

        float                                       jitterOffsetY() const;
        void                                        setJitterOffsetY( float offset );

        float                                       motionVectorScaleX() const;
        void                                        setMotionVectorScaleX( float scale );

        float                                       motionVectorScaleY() const;
        void                                        setMotionVectorScaleY( float scale );

        MTL::Texture*                               reactiveMaskTexture() const;
        void                                        setReactiveMaskTexture( MTL::Texture* reactiveMaskTexture );

        MTL::TextureUsage                           reactiveTextureUsage() const;

        bool                                        reset() const;
        void                                        setReset( bool reset );

        bool                                        isDepthReversed() const;
        void                                        setDepthReversed( bool depthReversed );

        MTL::PixelFormat                            colorTextureFormat() const;
        MTL::PixelFormat                            depthTextureFormat() const;
        MTL::PixelFormat                            motionTextureFormat() const;
        MTL::PixelFormat                            reactiveTextureFormat() const;
        MTL::PixelFormat                            outputTextureFormat() const;
        NS::UInteger                                inputWidth() const;
        NS::UInteger                                inputHeight() const;
        NS::UInteger                                outputWidth() const;
        NS::UInteger                                outputHeight() const;
        float                                       inputContentMinScale() const;
        float                                       inputContentMaxScale() const;

        MTL::Fence*                                 fence() const;
        void                                        setFence( MTL::Fence* pFence );
    };

    class TemporalScaler : public NS::Referencing< TemporalScaler, TemporalScalerBase >
    {
    public:
        void                                        encodeToCommandBuffer( MTL::CommandBuffer* pCommandBuffer );
    };
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::TemporalScalerDescriptor* MTLFX::TemporalScalerDescriptor::alloc()
{
    return NS::Object::alloc< TemporalScalerDescriptor >( _MTLFX_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::TemporalScalerDescriptor* MTLFX::TemporalScalerDescriptor::init()
{
    return NS::Object::init< TemporalScalerDescriptor >();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::colorTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setColorTextureFormat( MTL::PixelFormat format )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTextureFormat_ ), format );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::depthTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( depthTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setDepthTextureFormat( MTL::PixelFormat format )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthTextureFormat_ ), format );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::motionTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( motionTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setMotionTextureFormat( MTL::PixelFormat format )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionTextureFormat_ ), format );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::outputTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setOutputTextureFormat( MTL::PixelFormat format )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTextureFormat_ ), format );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::inputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputWidth( NS::UInteger width )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputWidth_ ), width );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::inputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputHeight( NS::UInteger height )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputHeight_ ), height );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::outputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setOutputWidth( NS::UInteger width )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputWidth_ ), width );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::outputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setOutputHeight( NS::UInteger height )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputHeight_ ), height );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::isAutoExposureEnabled() const
{
    return Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isAutoExposureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setAutoExposureEnabled( bool enabled )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setAutoExposureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::isInputContentPropertiesEnabled() const
{
    return Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isInputContentPropertiesEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputContentPropertiesEnabled( bool enabled )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentPropertiesEnabled_ ), enabled );
}


//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::requiresSynchronousInitialization() const
{
    return Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( requiresSynchronousInitialization ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setRequiresSynchronousInitialization(bool requiresSynchronousInitialization)
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setRequiresSynchronousInitialization_ ), requiresSynchronousInitialization );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::isReactiveMaskTextureEnabled() const
{
    return Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isReactiveMaskTextureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setReactiveMaskTextureEnabled( bool enabled )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReactiveMaskTextureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::reactiveMaskTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( reactiveMaskTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setReactiveMaskTextureFormat( MTL::PixelFormat pixelFormat )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReactiveMaskTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::inputContentMinScale() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMinScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputContentMinScale( float scale )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentMinScale_ ), scale );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::inputContentMaxScale() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMaxScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputContentMaxScale( float scale )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentMaxScale_ ), scale );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::TemporalScaler* MTLFX::TemporalScalerDescriptor::newTemporalScaler( const MTL::Device* pDevice ) const
{
    return Object::sendMessage< TemporalScaler* >( this, _MTLFX_PRIVATE_SEL( newTemporalScalerWithDevice_ ), pDevice );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL4FX::TemporalScaler* MTLFX::TemporalScalerDescriptor::newTemporalScaler( const MTL::Device* pDevice, const MTL4::Compiler* pCompiler ) const
{
    return Object::sendMessage< MTL4FX::TemporalScaler* >( this, _MTLFX_PRIVATE_SEL( newTemporalScalerWithDevice_compiler_ ), pDevice, pCompiler );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::supportedInputContentMinScale( const MTL::Device* pDevice )
{
    float scale = 1.0f;

    if ( nullptr != methodSignatureForSelector( _NS_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMinScaleForDevice_ ) ) )
    {
        scale = sendMessage< float >( _NS_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMinScaleForDevice_ ), pDevice );
    }

    return scale;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::supportedInputContentMaxScale( const MTL::Device* pDevice )
{
    float scale = 1.0f;

    if ( nullptr != methodSignatureForSelector( _NS_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMaxScaleForDevice_ ) ) )
    {
        scale = sendMessage< float >( _NS_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMaxScaleForDevice_ ), pDevice );
    }
    else if ( supportsDevice( pDevice ) )
    {
        scale = 2.0f;
    }

    return scale;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::supportsDevice( const MTL::Device* pDevice )
{
    return Object::sendMessageSafe< bool >( _NS_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportsDevice_ ), pDevice );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::supportsMetal4FX( const MTL::Device* pDevice )
{
    return Object::sendMessageSafe< bool >( _NS_PRIVATE_CLS( MTLFXTemporalScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportsMetal4FX_ ), pDevice );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::colorTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( colorTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::depthTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( depthTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::motionTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( motionTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::outputTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( outputTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputContentWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputContentWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setInputContentWidth( NS::UInteger width )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentWidth_ ), width );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputContentHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputContentHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setInputContentHeight( NS::UInteger height )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentHeight_ ), height );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::colorTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( colorTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setColorTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::depthTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( depthTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setDepthTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::motionTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( motionTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setMotionTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::outputTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( outputTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setOutputTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::exposureTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( exposureTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setExposureTexture( MTL::Texture* pTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setExposureTexture_ ), pTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::preExposure() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( preExposure ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setPreExposure( float preExposure )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setPreExposure_ ), preExposure );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::jitterOffsetX() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( jitterOffsetX ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setJitterOffsetX( float offset )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setJitterOffsetX_ ), offset );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::jitterOffsetY() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( jitterOffsetY ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setJitterOffsetY( float offset )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setJitterOffsetY_ ), offset );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::motionVectorScaleX() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( motionVectorScaleX ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setMotionVectorScaleX( float scale )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionVectorScaleX_ ), scale );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::motionVectorScaleY() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( motionVectorScaleY ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setMotionVectorScaleY( float scale )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionVectorScaleY_ ), scale );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::reactiveMaskTexture() const
{
    return Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( reactiveMaskTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setReactiveMaskTexture( MTL::Texture* reactiveMaskTexture )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReactiveMaskTexture_ ), reactiveMaskTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::reactiveTextureUsage() const
{
    return Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( reactiveTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerBase::reset() const
{
    return Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( reset ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setReset( bool reset )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReset_ ), reset );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalScalerBase::isDepthReversed() const
{
    return Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isDepthReversed ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setDepthReversed( bool depthReversed )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthReversed_ ), depthReversed );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::colorTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::depthTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( depthTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::motionTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( motionTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::outputTextureFormat() const
{
    return Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::outputWidth() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::outputHeight() const
{
    return Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::inputContentMinScale() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMinScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalScalerBase::inputContentMaxScale() const
{
    return Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMaxScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Fence* MTLFX::TemporalScalerBase::fence() const
{
    return Object::sendMessage< MTL::Fence* >( this, _MTLFX_PRIVATE_SEL( fence ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setFence( MTL::Fence* pFence )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setFence_ ), pFence );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalScaler::encodeToCommandBuffer( MTL::CommandBuffer* pCommandBuffer )
{
    Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( encodeToCommandBuffer_ ), pCommandBuffer );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
