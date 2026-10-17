//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXTemporalDenoisedScaler.hpp
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
#include "MTLFXTemporalScaler.hpp"

#include "../Metal/Metal.hpp"

#include <simd/simd.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL4FX
{
    class TemporalDenoisedScaler;
}

namespace MTLFX
{
    class TemporalDenoisedScalerDescriptor : public NS::Copying< TemporalDenoisedScalerDescriptor >
    {
        public:
            static class TemporalDenoisedScalerDescriptor*       alloc();
            class TemporalDenoisedScalerDescriptor*              init();

            MTL::PixelFormat                                     colorTextureFormat() const;
            void                                                 setColorTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     depthTextureFormat() const;
            void                                                 setDepthTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     motionTextureFormat() const;
            void                                                 setMotionTextureFormat( MTL::PixelFormat pixelFormal );

            MTL::PixelFormat                                     diffuseAlbedoTextureFormat() const;
            void                                                 setDiffuseAlbedoTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     specularAlbedoTextureFormat() const;
            void                                                 setSpecularAlbedoTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     normalTextureFormat() const;
            void                                                 setNormalTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     roughnessTextureFormat() const;
            void                                                 setRoughnessTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     specularHitDistanceTextureFormat() const;
            void                                                 setSpecularHitDistanceTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     denoiseStrengthMaskTextureFormat() const;
            void                                                 setDenoiseStrengthMaskTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     transparencyOverlayTextureFormat() const;
            void                                                 setTransparencyOverlayTextureFormat( MTL::PixelFormat pixelFormat );

            MTL::PixelFormat                                     outputTextureFormat() const;
            void                                                 setOutputTextureFormat( MTL::PixelFormat pixelFormat );

            NS::UInteger                                         inputWidth() const;
            void                                                 setInputWidth( NS::UInteger inputWidth );

            NS::UInteger                                         inputHeight() const;
            void                                                 setInputHeight( NS::UInteger inputHeight );

            NS::UInteger                                         outputWidth() const;
            void                                                 setOutputWidth( NS::UInteger outputWidth );

            NS::UInteger                                         outputHeight() const;
            void                                                 setOutputHeight( NS::UInteger outputHeight );

            bool                                                 requiresSynchronousInitialization() const;
            void                                                 setRequiresSynchronousInitialization( bool requiresSynchronousInitialization );

            bool                                                 isAutoExposureEnabled() const;
            void                                                 setAutoExposureEnabled( bool enabled );

            bool                                                 isInputContentPropertiesEnabled() const;
            void                                                 setInputContentPropertiesEnabled( bool enabled );

            float                                                inputContentMinScale() const;
            void                                                 setInputContentMinScale( float inputContentMinScale );

            float                                                inputContentMaxScale() const;
            void                                                 setInputContentMaxScale( float inputContentMaxScale );

            bool                                                 isReactiveMaskTextureEnabled() const;
            void                                                 setReactiveMaskTextureEnabled( bool enabled );

            MTL::PixelFormat                                     reactiveMaskTextureFormat() const;
            void                                                 setReactiveMaskTextureFormat( MTL::PixelFormat pixelFormat );

            bool                                                 isSpecularHitDistanceTextureEnabled() const;
            void                                                 setSpecularHitDistanceTextureEnabled( bool enabled );

            bool                                                 isDenoiseStrengthMaskTextureEnabled() const;
            void                                                 setDenoiseStrengthMaskTextureEnabled( bool enabled );

            bool                                                 isTransparencyOverlayTextureEnabled() const;
            void                                                 setTransparencyOverlayTextureEnabled( bool enabled );

            class TemporalDenoisedScaler*                        newTemporalDenoisedScaler( const MTL::Device* device ) const;
            MTL4FX::TemporalDenoisedScaler*                      newTemporalDenoisedScaler( const MTL::Device* device, const MTL4::Compiler* compiler) const;

            static float                                         supportedInputContentMinScale(MTL::Device* device);
            static float                                         supportedInputContentMaxScale(MTL::Device* device);

            static bool                                          supportsMetal4FX( MTL::Device* device);
            static bool                                          supportsDevice( MTL::Device* device);
    };

    class TemporalDenoisedScalerBase : public NS::Referencing< TemporalDenoisedScalerBase, FrameInterpolatableScaler >
    {
    public:
        MTL::TextureUsage                                       colorTextureUsage() const;
        MTL::TextureUsage                                       depthTextureUsage() const;
        MTL::TextureUsage                                       motionTextureUsage() const;
        MTL::TextureUsage                                       reactiveTextureUsage() const;
        MTL::TextureUsage                                       diffuseAlbedoTextureUsage() const;
        MTL::TextureUsage                                       specularAlbedoTextureUsage() const;
        MTL::TextureUsage                                       normalTextureUsage() const;
        MTL::TextureUsage                                       roughnessTextureUsage() const;
        MTL::TextureUsage                                       specularHitDistanceTextureUsage() const;
        MTL::TextureUsage                                       denoiseStrengthMaskTextureUsage() const;
        MTL::TextureUsage                                       transparencyOverlayTextureUsage() const;
        MTL::TextureUsage                                       outputTextureUsage() const;

        MTL::Texture*                                           colorTexture() const;
        void                                                    setColorTexture( MTL::Texture* colorTexture );

        MTL::Texture*                                           depthTexture() const;
        void                                                    setDepthTexture( MTL::Texture* depthTexture );

        MTL::Texture*                                           motionTexture() const;
        void                                                    setMotionTexture( MTL::Texture* motionTexture );

        MTL::Texture*                                           diffuseAlbedoTexture() const;
        void                                                    setDiffuseAlbedoTexture( MTL::Texture* diffuseAlbedoTexture );

        MTL::Texture*                                           specularAlbedoTexture() const;
        void                                                    setSpecularAlbedoTexture( MTL::Texture* specularAlbedoTexture );

        MTL::Texture*                                           normalTexture() const;
        void                                                    setNormalTexture( MTL::Texture* normalTexture );

        MTL::Texture*                                           roughnessTexture() const;
        void                                                    setRoughnessTexture( MTL::Texture* roughnessTexture );

        MTL::Texture*                                           specularHitDistanceTexture() const;
        void                                                    setSpecularHitDistanceTexture( MTL::Texture* specularHitDistanceTexture );

        MTL::Texture*                                           denoiseStrengthMaskTexture() const;
        void                                                    setDenoiseStrengthMaskTexture( MTL::Texture* denoiseStrengthMaskTexture );

        MTL::Texture*                                           transparencyOverlayTexture() const;
        void                                                    setTransparencyOverlayTexture( MTL::Texture* transparencyOverlayTexture );

        MTL::Texture*                                           outputTexture() const;
        void                                                    setOutputTexture( MTL::Texture* outputTexture );

        MTL::Texture*                                           exposureTexture() const;
        void                                                    setExposureTexture( MTL::Texture* exposureTexture );

        float                                                   preExposure() const;
        void                                                    setPreExposure( float preExposure );

        MTL::Texture*                                           reactiveMaskTexture() const;
        void                                                    setReactiveMaskTexture( MTL::Texture* reactiveMaskTexture );

        float                                                   jitterOffsetX() const;
        void                                                    setJitterOffsetX( float jitterOffsetX );

        float                                                   jitterOffsetY() const;
        void                                                    setJitterOffsetY( float jitterOffsetY );

        float                                                   motionVectorScaleX() const;
        void                                                    setMotionVectorScaleX( float motionVectorScaleX );

        float                                                   motionVectorScaleY() const;
        void                                                    setMotionVectorScaleY( float motionVectorScaleY );

        bool                                                    shouldResetHistory() const;
        void                                                    setShouldResetHistory( bool shouldResetHistory );

        bool                                                    isDepthReversed() const;
        void                                                    setDepthReversed( bool depthReversed );

        MTL::PixelFormat                                        colorTextureFormat() const;
        MTL::PixelFormat                                        depthTextureFormat() const;
        MTL::PixelFormat                                        motionTextureFormat() const;
        MTL::PixelFormat                                        diffuseAlbedoTextureFormat() const;
        MTL::PixelFormat                                        specularAlbedoTextureFormat() const;
        MTL::PixelFormat                                        normalTextureFormat() const;
        MTL::PixelFormat                                        roughnessTextureFormat() const;
        MTL::PixelFormat                                        specularHitDistanceTextureFormat() const;
        MTL::PixelFormat                                        denoiseStrengthMaskTextureFormat() const;
        MTL::PixelFormat                                        transparencyOverlayTextureFormat() const;
        MTL::PixelFormat                                        reactiveMaskTextureFormat() const;
        MTL::PixelFormat                                        outputTextureFormat() const;

        NS::UInteger                                            inputWidth() const;
        NS::UInteger                                            inputHeight() const;
        NS::UInteger                                            outputWidth() const;
        NS::UInteger                                            outputHeight() const;
        float                                                   inputContentMinScale() const;
        float                                                   inputContentMaxScale() const;

        simd::float4x4                                          worldToViewMatrix() const;
        void                                                    setWorldToViewMatrix( simd::float4x4 worldToViewMatrix );

        simd::float4x4                                          viewToClipMatrix() const;
        void                                                    setViewToClipMatrix( simd::float4x4 viewToClipMatrix );

        MTL::Fence*                                             fence() const;
        void                                                    setFence( MTL::Fence* fence );
    };

    class TemporalDenoisedScaler : public NS::Referencing< TemporalDenoisedScaler, TemporalDenoisedScalerBase >
    {
    public:

        void                                                    encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer);
    };
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::TemporalDenoisedScalerDescriptor* MTLFX::TemporalDenoisedScalerDescriptor::alloc()
{
    return NS::Object::alloc< TemporalDenoisedScalerDescriptor >( _MTLFX_PRIVATE_CLS( MTLFXTemporalDenoisedScalerDescriptor ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::TemporalDenoisedScalerDescriptor* MTLFX::TemporalDenoisedScalerDescriptor::init()
{
    return NS::Object::init< TemporalDenoisedScalerDescriptor >();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::colorTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setColorTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::depthTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( depthTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDepthTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::motionTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( motionTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setMotionTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::diffuseAlbedoTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( diffuseAlbedoTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDiffuseAlbedoTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDiffuseAlbedoTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::specularAlbedoTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( specularAlbedoTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setSpecularAlbedoTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setSpecularAlbedoTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::normalTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( normalTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setNormalTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setNormalTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::roughnessTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( roughnessTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setRoughnessTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setRoughnessTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::specularHitDistanceTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( specularHitDistanceTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setSpecularHitDistanceTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setSpecularHitDistanceTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::denoiseStrengthMaskTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( denoiseStrengthMaskTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDenoiseStrengthMaskTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDenoiseStrengthMaskTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::transparencyOverlayTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( transparencyOverlayTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setTransparencyOverlayTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setTransparencyOverlayTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::outputTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setOutputTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::inputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputWidth( NS::UInteger inputWidth )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputWidth_ ), inputWidth );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::inputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputHeight( NS::UInteger inputHeight )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputHeight_ ), inputHeight );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::outputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setOutputWidth( NS::UInteger outputWidth )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputWidth_ ), outputWidth );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::outputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setOutputHeight( NS::UInteger outputHeight )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputHeight_ ), outputHeight );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::requiresSynchronousInitialization() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( requiresSynchronousInitialization ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setRequiresSynchronousInitialization( bool requiresSynchronousInitialization )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setRequiresSynchronousInitialization_ ), requiresSynchronousInitialization );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isAutoExposureEnabled() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isAutoExposureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setAutoExposureEnabled( bool enabled )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setAutoExposureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isInputContentPropertiesEnabled() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isInputContentPropertiesEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputContentPropertiesEnabled( bool enabled )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentPropertiesEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerDescriptor::inputContentMinScale() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMinScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputContentMinScale( float inputContentMinScale )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentMinScale_ ), inputContentMinScale );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerDescriptor::inputContentMaxScale() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMaxScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputContentMaxScale( float inputContentMaxScale )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputContentMaxScale_ ), inputContentMaxScale );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isReactiveMaskTextureEnabled() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isReactiveMaskTextureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setReactiveMaskTextureEnabled( bool enabled )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReactiveMaskTextureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::reactiveMaskTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( reactiveMaskTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setReactiveMaskTextureFormat( MTL::PixelFormat pixelFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReactiveMaskTextureFormat_ ), pixelFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isSpecularHitDistanceTextureEnabled() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isSpecularHitDistanceTextureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setSpecularHitDistanceTextureEnabled( bool enabled )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setSpecularHitDistanceTextureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isDenoiseStrengthMaskTextureEnabled() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isDenoiseStrengthMaskTextureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDenoiseStrengthMaskTextureEnabled( bool enabled )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDenoiseStrengthMaskTextureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isTransparencyOverlayTextureEnabled() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isTransparencyOverlayTextureEnabled ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setTransparencyOverlayTextureEnabled( bool enabled )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setTransparencyOverlayTextureEnabled_ ), enabled );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::TemporalDenoisedScaler* MTLFX::TemporalDenoisedScalerDescriptor::newTemporalDenoisedScaler( const MTL::Device* device ) const
{
    return NS::Object::sendMessage< TemporalDenoisedScaler* >( this, _MTLFX_PRIVATE_SEL( newTemporalDenoisedScalerWithDevice_ ), device );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL4FX::TemporalDenoisedScaler* MTLFX::TemporalDenoisedScalerDescriptor::newTemporalDenoisedScaler( const MTL::Device* device, const MTL4::Compiler* compiler ) const
{
    return NS::Object::sendMessage< MTL4FX::TemporalDenoisedScaler* >( this, _MTLFX_PRIVATE_SEL( newTemporalDenoisedScalerWithDevice_compiler_ ), device, compiler );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerDescriptor::supportedInputContentMinScale( MTL::Device* pDevice )
{
    float scale = 1.0f;

    if ( nullptr != methodSignatureForSelector( _MTLFX_PRIVATE_CLS( MTLFXTemporalDenoisedScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMinScaleForDevice_ ) ) )
    {
        scale = sendMessage< float >( _NS_PRIVATE_CLS( MTLFXTemporalDenoisedScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMinScaleForDevice_ ), pDevice );
    }

    return scale;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerDescriptor::supportedInputContentMaxScale( MTL::Device* pDevice )
{
    float scale = 1.0f;

    if ( nullptr != methodSignatureForSelector( _MTLFX_PRIVATE_CLS( MTLFXTemporalDenoisedScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMaxScaleForDevice_ ) ) )
    {
        scale = sendMessage< float >( _MTLFX_PRIVATE_CLS( MTLFXTemporalDenoisedScalerDescriptor ), _MTLFX_PRIVATE_SEL( supportedInputContentMaxScaleForDevice_ ), pDevice );
    }
    else if ( supportsDevice( pDevice ) )
    {
        scale = 2.0f;
    }

    return scale;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::supportsMetal4FX( MTL::Device* device )
{
    return NS::Object::sendMessageSafe< bool >( _MTLFX_PRIVATE_CLS(MTLFXTemporalDenoisedScalerDescriptor), _MTLFX_PRIVATE_SEL( supportsMetal4FX_ ), device );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::supportsDevice( MTL::Device* device )
{
    return NS::Object::sendMessageSafe< bool >( _MTLFX_PRIVATE_CLS(MTLFXTemporalDenoisedScalerDescriptor), _MTLFX_PRIVATE_SEL( supportsDevice_ ), device );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::colorTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( colorTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::depthTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( depthTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::motionTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( motionTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::reactiveTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( reactiveTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::diffuseAlbedoTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( diffuseAlbedoTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::specularAlbedoTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( specularAlbedoTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::normalTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( normalTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::roughnessTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( roughnessTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::specularHitDistanceTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( specularHitDistanceTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::denoiseStrengthMaskTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( denoiseStrengthMaskTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::transparencyOverlayTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( transparencyOverlayTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::outputTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( outputTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::colorTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( colorTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setColorTexture( MTL::Texture* colorTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTexture_ ), colorTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::depthTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( depthTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDepthTexture( MTL::Texture* depthTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthTexture_ ), depthTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::motionTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( motionTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setMotionTexture( MTL::Texture* motionTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionTexture_ ), motionTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::diffuseAlbedoTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( diffuseAlbedoTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDiffuseAlbedoTexture( MTL::Texture* diffuseAlbedoTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDiffuseAlbedoTexture_ ), diffuseAlbedoTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::specularAlbedoTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( specularAlbedoTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setSpecularAlbedoTexture( MTL::Texture* specularAlbedoTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setSpecularAlbedoTexture_ ), specularAlbedoTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::normalTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( normalTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setNormalTexture( MTL::Texture* normalTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setNormalTexture_ ), normalTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::roughnessTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( roughnessTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setRoughnessTexture( MTL::Texture* roughnessTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setRoughnessTexture_ ), roughnessTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::specularHitDistanceTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( specularHitDistanceTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setSpecularHitDistanceTexture( MTL::Texture* specularHitDistanceTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setSpecularHitDistanceTexture_ ), specularHitDistanceTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::denoiseStrengthMaskTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( denoiseStrengthMaskTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDenoiseStrengthMaskTexture( MTL::Texture* denoiseStrengthMaskTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDenoiseStrengthMaskTexture_ ), denoiseStrengthMaskTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::transparencyOverlayTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( transparencyOverlayTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setTransparencyOverlayTexture( MTL::Texture* transparencyOverlayTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setTransparencyOverlayTexture_ ), transparencyOverlayTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::outputTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( outputTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setOutputTexture( MTL::Texture* outputTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTexture_ ), outputTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::exposureTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( exposureTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setExposureTexture( MTL::Texture* exposureTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setExposureTexture_ ), exposureTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::preExposure() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( preExposure ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setPreExposure( float preExposure )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setPreExposure_ ), preExposure );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::reactiveMaskTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( reactiveMaskTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setReactiveMaskTexture( MTL::Texture* reactiveMaskTexture )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setReactiveMaskTexture_ ), reactiveMaskTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::jitterOffsetX() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( jitterOffsetX ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setJitterOffsetX( float jitterOffsetX )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setJitterOffsetX_ ), jitterOffsetX );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::jitterOffsetY() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( jitterOffsetY ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setJitterOffsetY( float jitterOffsetY )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setJitterOffsetY_ ), jitterOffsetY );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::motionVectorScaleX() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( motionVectorScaleX ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setMotionVectorScaleX( float motionVectorScaleX )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionVectorScaleX_ ), motionVectorScaleX );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::motionVectorScaleY() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( motionVectorScaleY ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setMotionVectorScaleY( float motionVectorScaleY )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionVectorScaleY_ ), motionVectorScaleY );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerBase::shouldResetHistory() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( shouldResetHistory ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setShouldResetHistory( bool shouldResetHistory )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setShouldResetHistory_ ), shouldResetHistory );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerBase::isDepthReversed() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isDepthReversed ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDepthReversed( bool depthReversed )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthReversed_ ), depthReversed );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::colorTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::depthTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( depthTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::motionTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( motionTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::diffuseAlbedoTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( diffuseAlbedoTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::specularAlbedoTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( specularAlbedoTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::normalTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( normalTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::roughnessTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( roughnessTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::specularHitDistanceTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( specularHitDistanceTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::denoiseStrengthMaskTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( denoiseStrengthMaskTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::transparencyOverlayTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( transparencyOverlayTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::reactiveMaskTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( reactiveMaskTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::outputTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::inputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::inputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::outputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::outputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::inputContentMinScale() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMinScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::inputContentMaxScale() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( inputContentMaxScale ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE simd::float4x4 MTLFX::TemporalDenoisedScalerBase::worldToViewMatrix() const
{
    return NS::Object::sendMessage< simd::float4x4 >( this, _MTLFX_PRIVATE_SEL( worldToViewMatrix ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setWorldToViewMatrix( simd::float4x4 worldToViewMatrix )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setWorldToViewMatrix_ ), worldToViewMatrix );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE simd::float4x4 MTLFX::TemporalDenoisedScalerBase::viewToClipMatrix() const
{
    return NS::Object::sendMessage< simd::float4x4 >( this, _MTLFX_PRIVATE_SEL( viewToClipMatrix ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setViewToClipMatrix( simd::float4x4 viewToClipMatrix )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setViewToClipMatrix_ ), viewToClipMatrix );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Fence* MTLFX::TemporalDenoisedScalerBase::fence() const
{
    return NS::Object::sendMessage< MTL::Fence* >( this, _MTLFX_PRIVATE_SEL( fence ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setFence( MTL::Fence* fence )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setFence_ ), fence );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::TemporalDenoisedScaler::encodeToCommandBuffer( MTL::CommandBuffer* commandBuffer )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( encodeToCommandBuffer_ ), commandBuffer );
}
