//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXFrameInterpolator.hpp
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTL4FX
{
    class TemporalScaler;
    class TemporalDenoisedScaler;
    class FrameInterpolator;
}

namespace MTLFX
{
    class FrameInterpolatorDescriptor : public NS::Copying< FrameInterpolatorDescriptor >
    {
    public:
        static FrameInterpolatorDescriptor* alloc();
        FrameInterpolatorDescriptor*        init();

        MTL::PixelFormat                    colorTextureFormat() const;
        void                                setColorTextureFormat(MTL::PixelFormat colorTextureFormat);

        MTL::PixelFormat                    outputTextureFormat() const;
        void                                setOutputTextureFormat(MTL::PixelFormat outputTextureFormat);

        MTL::PixelFormat                    depthTextureFormat() const;
        void                                setDepthTextureFormat(MTL::PixelFormat depthTextureFormat);

        MTL::PixelFormat                    motionTextureFormat() const;
        void                                setMotionTextureFormat(MTL::PixelFormat motionTextureFormat);

        MTL::PixelFormat                    uiTextureFormat() const;
        void                                setUITextureFormat(MTL::PixelFormat uiTextureFormat);

        MTLFX::FrameInterpolatableScaler*   scaler() const;
        void                                setScaler(MTLFX::FrameInterpolatableScaler* scaler);

        NS::UInteger                        inputWidth() const;
        void                                setInputWidth( NS::UInteger inputWidth );

        NS::UInteger                        inputHeight() const;
        void                                setInputHeight( NS::UInteger inputHeight );

        NS::UInteger                        outputWidth() const;
        void                                setOutputWidth( NS::UInteger outputWidth );

        NS::UInteger                        outputHeight() const;
        void                                setOutputHeight( NS::UInteger outputHeight );

        class FrameInterpolator*            newFrameInterpolator( const MTL::Device* pDevice) const;
        MTL4FX::FrameInterpolator*          newFrameInterpolator( const MTL::Device* pDevice, const MTL4::Compiler* pCompiler) const;

        static bool                         supportsMetal4FX(MTL::Device* device);
        static bool                         supportsDevice(MTL::Device* device);
    };

    class FrameInterpolatorBase : public NS::Referencing<FrameInterpolatorBase>
    {
    public:
        MTL::TextureUsage colorTextureUsage() const;
        MTL::TextureUsage outputTextureUsage() const;
        MTL::TextureUsage depthTextureUsage() const;
        MTL::TextureUsage motionTextureUsage() const;
        MTL::TextureUsage uiTextureUsage() const;

        MTL::PixelFormat  colorTextureFormat() const;
        MTL::PixelFormat  depthTextureFormat() const;
        MTL::PixelFormat  motionTextureFormat() const;
        MTL::PixelFormat  outputTextureFormat() const;

        NS::UInteger      inputWidth() const;
        NS::UInteger      inputHeight() const;
        NS::UInteger      outputWidth() const;
        NS::UInteger      outputHeight() const;
        MTL::PixelFormat  uiTextureFormat() const;

        MTL::Texture*     colorTexture() const;
        void              setColorTexture(MTL::Texture* colorTexture);

        MTL::Texture*     prevColorTexture() const;
        void              setPrevColorTexture(MTL::Texture* prevColorTexture);

        MTL::Texture*     depthTexture() const;
        void              setDepthTexture(MTL::Texture* depthTexture);

        MTL::Texture*     motionTexture() const;
        void              setMotionTexture(MTL::Texture* motionTexture);

        float             motionVectorScaleX() const;
        void              setMotionVectorScaleX(float scaleX);

        float             motionVectorScaleY() const;
        void              setMotionVectorScaleY(float scaleY);

        float             deltaTime() const;
        void              setDeltaTime( float deltaTime );

        float             nearPlane() const;
        void              setNearPlane( float nearPlane );

        float             farPlane() const;
        void              setFarPlane( float farPlane );

        float             fieldOfView() const;
        void              setFieldOfView( float fieldOfView );

        float             aspectRatio() const;
        void              setAspectRatio( float aspectRatio );

        MTL::Texture*     uiTexture() const;
        void              setUITexture(MTL::Texture* uiTexture);

        float             jitterOffsetX() const;
        void              setJitterOffsetX( float jitterOffsetX );

        float             jitterOffsetY() const;
        void              setJitterOffsetY( float jitterOffsetY );

        bool              isUITextureComposited() const;
        void              setIsUITextureComposited( bool uiTextureComposited );

        bool              shouldResetHistory() const;
        void              setShouldResetHistory( bool shouldResetHistory );

        MTL::Texture*     outputTexture() const;
        void              setOutputTexture( MTL::Texture* outputTexture );

        MTL::Fence*       fence() const;
        void              setFence( MTL::Fence* fence );

        bool              isDepthReversed() const;
        void              setDepthReversed( bool depthReversed );
    };

    class FrameInterpolator : public NS::Referencing<FrameInterpolator, FrameInterpolatorBase>
    {
    public:
        void              encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer);
    };

}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::FrameInterpolatorDescriptor* MTLFX::FrameInterpolatorDescriptor::alloc()
{
    return NS::Object::alloc< FrameInterpolatorDescriptor >( _MTLFX_PRIVATE_CLS( MTLFXFrameInterpolatorDescriptor ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::FrameInterpolatorDescriptor* MTLFX::FrameInterpolatorDescriptor::init()
{
    return NS::Object::init< FrameInterpolatorDescriptor >();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::colorTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setColorTextureFormat( MTL::PixelFormat colorTextureFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTextureFormat_ ), colorTextureFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::outputTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setOutputTextureFormat( MTL::PixelFormat outputTextureFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTextureFormat_ ), outputTextureFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::depthTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( depthTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setDepthTextureFormat( MTL::PixelFormat depthTextureFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthTextureFormat_ ), depthTextureFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::motionTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( motionTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setMotionTextureFormat( MTL::PixelFormat motionTextureFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionTextureFormat_ ), motionTextureFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::uiTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( uiTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setUITextureFormat( MTL::PixelFormat uiTextureFormat )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setUITextureFormat_ ), uiTextureFormat );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::FrameInterpolatableScaler* MTLFX::FrameInterpolatorDescriptor::scaler() const
{
    return NS::Object::sendMessage< MTLFX::FrameInterpolatableScaler* >( this, _MTLFX_PRIVATE_SEL( scaler ) );
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setScaler(MTLFX::FrameInterpolatableScaler* scaler)
{
    NS::Object::sendMessage< void >(this, _MTLFX_PRIVATE_SEL( setScaler_ ), scaler );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::inputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setInputWidth( NS::UInteger inputWidth )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputWidth_ ), inputWidth );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::inputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setInputHeight( NS::UInteger inputHeight )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setInputHeight_ ), inputHeight );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::outputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setOutputWidth( NS::UInteger outputWidth )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputWidth_ ), outputWidth );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::outputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setOutputHeight( NS::UInteger outputHeight )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputHeight_ ), outputHeight );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTLFX::FrameInterpolator* MTLFX::FrameInterpolatorDescriptor::newFrameInterpolator( const MTL::Device* device ) const
{
    return NS::Object::sendMessage< MTLFX::FrameInterpolator* >( this, _MTLFX_PRIVATE_SEL( newFrameInterpolatorWithDevice_ ), device );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL4FX::FrameInterpolator* MTLFX::FrameInterpolatorDescriptor::newFrameInterpolator( const MTL::Device* device, const MTL4::Compiler* compiler ) const
{
    return NS::Object::sendMessage< MTL4FX::FrameInterpolator* >( this, _MTLFX_PRIVATE_SEL( newFrameInterpolatorWithDevice_compiler_ ), device, compiler );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::FrameInterpolatorDescriptor::supportsMetal4FX(MTL::Device* device)
{
    return NS::Object::sendMessageSafe< bool >( _MTLFX_PRIVATE_CLS(MTLFXFrameInterpolatorDescriptor), _MTLFX_PRIVATE_SEL( supportsMetal4FX_ ), device );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::FrameInterpolatorDescriptor::supportsDevice(MTL::Device* device)
{
    return NS::Object::sendMessageSafe< bool >( _MTLFX_PRIVATE_CLS(MTLFXFrameInterpolatorDescriptor), _MTLFX_PRIVATE_SEL( supportsDevice_ ), device );
}


//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::colorTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( colorTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::outputTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( outputTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::depthTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( depthTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::motionTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( motionTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::uiTextureUsage() const
{
    return NS::Object::sendMessage< MTL::TextureUsage >( this, _MTLFX_PRIVATE_SEL( uiTextureUsage ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::colorTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( colorTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::depthTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( depthTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::motionTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( motionTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::outputTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( outputTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::inputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::inputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( inputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::outputWidth() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputWidth ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::outputHeight() const
{
    return NS::Object::sendMessage< NS::UInteger >( this, _MTLFX_PRIVATE_SEL( outputHeight ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::uiTextureFormat() const
{
    return NS::Object::sendMessage< MTL::PixelFormat >( this, _MTLFX_PRIVATE_SEL( uiTextureFormat ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::colorTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( colorTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setColorTexture(MTL::Texture* colorTexture)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setColorTexture_ ), colorTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::prevColorTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( prevColorTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setPrevColorTexture(MTL::Texture* prevColorTexture)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setPrevColorTexture_ ), prevColorTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::depthTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( depthTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setDepthTexture(MTL::Texture* depthTexture)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthTexture_ ), depthTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::motionTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( motionTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setMotionTexture(MTL::Texture* motionTexture)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionTexture_ ), motionTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::motionVectorScaleX() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( motionVectorScaleX ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setMotionVectorScaleX(float scaleX)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionVectorScaleX_ ), scaleX );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::motionVectorScaleY() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( motionVectorScaleY ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setMotionVectorScaleY(float scaleY)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setMotionVectorScaleY_ ), scaleY );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float  MTLFX::FrameInterpolatorBase::deltaTime() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( deltaTime ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void   MTLFX::FrameInterpolatorBase::setDeltaTime( float deltaTime )
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDeltaTime_ ), deltaTime );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float  MTLFX::FrameInterpolatorBase::nearPlane() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( nearPlane ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void  MTLFX::FrameInterpolatorBase::setNearPlane( float nearPlane )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setNearPlane_ ), nearPlane );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float  MTLFX::FrameInterpolatorBase::farPlane() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( farPlane ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setFarPlane( float farPlane )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setFarPlane_ ), farPlane );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float  MTLFX::FrameInterpolatorBase::fieldOfView() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( fieldOfView ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void   MTLFX::FrameInterpolatorBase::setFieldOfView( float fieldOfView )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setFieldOfView_ ), fieldOfView );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float  MTLFX::FrameInterpolatorBase::aspectRatio() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( aspectRatio ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void   MTLFX::FrameInterpolatorBase::setAspectRatio( float aspectRatio )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setAspectRatio_ ), aspectRatio );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::uiTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( uiTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setUITexture(MTL::Texture* uiTexture)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setUITexture_ ), uiTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::jitterOffsetX() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( jitterOffsetX ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setJitterOffsetX( float jitterOffsetX )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setJitterOffsetX_ ), jitterOffsetX );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::jitterOffsetY() const
{
    return NS::Object::sendMessage< float >( this, _MTLFX_PRIVATE_SEL( jitterOffsetY ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setJitterOffsetY( float jitterOffsetY )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setJitterOffsetY_ ), jitterOffsetY );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::isUITextureComposited() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isUITextureComposited ) );
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setIsUITextureComposited( bool uiTextureComposited )
{
    NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setIsUITextureComposited_ ), uiTextureComposited );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::shouldResetHistory() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( shouldResetHistory ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setShouldResetHistory(bool shouldResetHistory)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setShouldResetHistory_ ), shouldResetHistory );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::outputTexture() const
{
    return NS::Object::sendMessage< MTL::Texture* >( this, _MTLFX_PRIVATE_SEL( outputTexture ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setOutputTexture(MTL::Texture* outputTexture)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setOutputTexture_ ), outputTexture );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE MTL::Fence* MTLFX::FrameInterpolatorBase::fence() const
{
    return NS::Object::sendMessage< MTL::Fence* >( this, _MTLFX_PRIVATE_SEL( fence ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setFence(MTL::Fence* fence)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setFence_ ), fence );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::isDepthReversed() const
{
    return NS::Object::sendMessage< bool >( this, _MTLFX_PRIVATE_SEL( isDepthReversed ) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setDepthReversed(bool depthReversed)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( setDepthReversed_ ), depthReversed );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_MTLFX_INLINE void MTLFX::FrameInterpolator::encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer)
{
    return NS::Object::sendMessage< void >( this, _MTLFX_PRIVATE_SEL( encodeToCommandBuffer_ ), commandBuffer );
}
