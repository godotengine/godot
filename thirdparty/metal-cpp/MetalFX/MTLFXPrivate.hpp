//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXPrivate.hpp
//
// Copyright 2020-2024 Apple Inc.
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

#include <objc/runtime.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define _MTLFX_PRIVATE_CLS( symbol )                    ( MTLFX::Private::Class::s_k##symbol )
#define _MTLFX_PRIVATE_SEL( accessor )                  ( MTLFX::Private::Selector::s_k##accessor )

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#if defined( MTLFX_PRIVATE_IMPLEMENTATION )

#if defined( METALCPP_SYMBOL_VISIBILITY_HIDDEN )
#define _MTLFX_PRIVATE_VISIBILITY                       __attribute__( ( visibility("hidden" ) ) )
#else
#define _MTLFX_PRIVATE_VISIBILITY                       __attribute__( ( visibility("default" ) ) )
#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN

#define _MTLFX_PRIVATE_IMPORT                           __attribute__( ( weak_import ) )

#ifdef __OBJC__
#define _MTLFX_PRIVATE_OBJC_LOOKUP_CLASS( symbol )      ( ( __bridge void* ) objc_lookUpClass( #symbol ) )
#define _MTLFX_PRIVATE_OBJC_GET_PROTOCOL( symbol )      ( ( __bridge void* ) objc_getProtocol( #symbol ) )
#else
#define _MTLFX_PRIVATE_OBJC_LOOKUP_CLASS( symbol )      objc_lookUpClass(#symbol)
#define _MTLFX_PRIVATE_OBJC_GET_PROTOCOL( symbol )      objc_getProtocol(#symbol)
#endif // __OBJC__

#define _MTLFX_PRIVATE_DEF_CLS( symbol )                void* s_k##symbol _MTLFX_PRIVATE_VISIBILITY = _MTLFX_PRIVATE_OBJC_LOOKUP_CLASS( symbol )
#define _MTLFX_PRIVATE_DEF_PRO( symbol )                void* s_k##symbol _MTLFX_PRIVATE_VISIBILITY = _MTLFX_PRIVATE_OBJC_GET_PROTOCOL( symbol )
#define _MTLFX_PRIVATE_DEF_SEL( accessor, symbol )       SEL s_k##accessor _MTLFX_PRIVATE_VISIBILITY = sel_registerName( symbol )

#include <dlfcn.h>
#define MTLFX_DEF_FUNC( name, signature )               using Fn##name = signature; \
                                                        Fn##name name = reinterpret_cast< Fn##name >( dlsym( RTLD_DEFAULT, #name ) )

namespace MTLFX::Private
{
    template <typename _Type>

    inline _Type const LoadSymbol(const char* pSymbol)
    {
        const _Type* pAddress = static_cast<_Type*>(dlsym(RTLD_DEFAULT, pSymbol));

        return pAddress ? *pAddress : nullptr;
    }
} // MTLFX::Private

#if defined(__MAC_26_0) || defined(__IPHONE_26_0) || defined(__TVOS_26_0)

#define _MTLFX_PRIVATE_DEF_STR( type, symbol )                                                                                  \
    _MTLFX_EXTERN type const                            MTLFX##symbol _MTLFX_PRIVATE_IMPORT;                                    \
    type const                                          MTLFX::symbol = ( nullptr != &MTLFX##symbol ) ? MTLFX##ssymbol : nullptr

#define _MTLFX_PRIVATE_DEF_CONST( type, symbol )                                                                                \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol _MTLFX_PRIVATE_IMPORT;                                   \
    type const                                          MTLFX::symbol = (nullptr != &MTLFX##ssymbol) ? MTLFX##ssymbol : nullptr

#define _MTLFX_PRIVATE_DEF_WEAK_CONST( type, symbol )                                                                           \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol;                                                         \
    type const                                          MTLFX::symbol = Private::LoadSymbol< type >( "MTLFX" #symbol )

#else

#define _MTLFX_PRIVATE_DEF_STR( type, symbol )                                                                                  \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol;                                                         \
    type const                                          MTLFX::symbol = Private::LoadSymbol< type >( "MTLFX" #symbol )

#define _MTLFX_PRIVATE_DEF_CONST( type, symbol )                                                                                \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol;                                                         \
    type const                                          MTLFX::symbol = Private::LoadSymbol< type >( "MTLFX" #symbol )

#define _MTLFX_PRIVATE_DEF_WEAK_CONST( type, symbol )   _MTLFX_PRIVATE_DEF_CONST( type, symbol )

#endif

#else

#define _MTLFX_PRIVATE_DEF_CLS( symbol )                extern void* s_k##symbol
#define _MTLFX_PRIVATE_DEF_PRO( symbol )                extern void* s_k##symbol
#define _MTLFX_PRIVATE_DEF_SEL( accessor, symbol )      extern SEL s_k##accessor
#define _MTLFX_PRIVATE_DEF_STR( type, symbol )          extern type const MTLFX::symbol
#define _MTLFX_PRIVATE_DEF_CONST( type, symbol )        extern type const MTLFX::symbol
#define _MTLFX_PRIVATE_DEF_WEAK_CONST( type, symbol )   extern type const MTLFX::symbol

#endif // MTLFX_PRIVATE_IMPLEMENTATION

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTLFX
{
    namespace Private
    {
        namespace Class
        {
            _MTLFX_PRIVATE_DEF_CLS( MTLFXSpatialScalerDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTLFXTemporalScalerDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTLFXFrameInterpolatorDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTLFXTemporalDenoisedScalerDescriptor );

            _MTLFX_PRIVATE_DEF_CLS( MTL4FXSpatialScalerDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTL4FXTemporalScalerDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTL4FXFrameInterpolatorDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTL4FXTemporalDenoisedScalerDescriptor );
        } // Class
    } // Private
} // MTLFX

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTLFX
{
    namespace Private
    {
        namespace Protocol
        {
            _MTLFX_PRIVATE_DEF_PRO( MTLFXSpatialScaler );
            _MTLFX_PRIVATE_DEF_PRO( MTLFXTemporalScaler );
        } // Protocol
    } // Private
} // MTLFX

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTLFX
{
    namespace Private
    {
        namespace Selector
        {
            _MTLFX_PRIVATE_DEF_SEL( aspectRatio,
                                    "aspectRatio" );
            _MTLFX_PRIVATE_DEF_SEL( colorProcessingMode,
                                    "colorProcessingMode" );
            _MTLFX_PRIVATE_DEF_SEL( colorTexture,
                                    "colorTexture" );
            _MTLFX_PRIVATE_DEF_SEL( colorTextureFormat,
                                    "colorTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( colorTextureUsage,
                                    "colorTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( deltaTime,
                                    "deltaTime" );
            _MTLFX_PRIVATE_DEF_SEL( denoiseStrengthMaskTexture,
                                    "denoiseStrengthMaskTexture" );
            _MTLFX_PRIVATE_DEF_SEL( denoiseStrengthMaskTextureFormat,
                                    "denoiseStrengthMaskTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( denoiseStrengthMaskTextureUsage,
                                    "denoiseStrengthMaskTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( depthTexture,
                                    "depthTexture" );
            _MTLFX_PRIVATE_DEF_SEL( depthTextureFormat,
                                    "depthTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( depthTextureUsage,
                                    "depthTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( diffuseAlbedoTexture,
                                    "diffuseAlbedoTexture" );
            _MTLFX_PRIVATE_DEF_SEL( diffuseAlbedoTextureFormat,
                                    "diffuseAlbedoTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( diffuseAlbedoTextureUsage,
                                    "diffuseAlbedoTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( encodeToCommandBuffer_,
                                    "encodeToCommandBuffer:" );
            _MTLFX_PRIVATE_DEF_SEL( exposureTexture,
                                    "exposureTexture" );
            _MTLFX_PRIVATE_DEF_SEL( farPlane,
                                    "farPlane" );
            _MTLFX_PRIVATE_DEF_SEL( fence,
                                    "fence" );
            _MTLFX_PRIVATE_DEF_SEL( fieldOfView,
                                    "fieldOfView" );
            _MTLFX_PRIVATE_DEF_SEL( height,
                                    "height" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentHeight,
                                    "inputContentHeight" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentMaxScale,
                                    "inputContentMaxScale" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentMinScale,
                                    "inputContentMinScale" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentWidth,
                                    "inputContentWidth" );
            _MTLFX_PRIVATE_DEF_SEL( inputHeight,
                                    "inputHeight" );
            _MTLFX_PRIVATE_DEF_SEL( inputWidth,
                                    "inputWidth" );
            _MTLFX_PRIVATE_DEF_SEL( isAutoExposureEnabled,
                                    "isAutoExposureEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isDenoiseStrengthMaskTextureEnabled,
                                    "isDenoiseStrengthMaskTextureEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isDepthReversed,
                                    "isDepthReversed" );
            _MTLFX_PRIVATE_DEF_SEL( isInputContentPropertiesEnabled,
                                    "isInputContentPropertiesEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isTransparencyOverlayTextureEnabled,
                                    "isTransparencyOverlayTextureEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isReactiveMaskTextureEnabled,
                                    "isReactiveMaskTextureEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isSpecularHitDistanceTextureEnabled,
                                    "isSpecularHitDistanceTextureEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isUITextureComposited,
                                    "isUITextureComposited" );
            _MTLFX_PRIVATE_DEF_SEL( jitterOffsetX,
                                    "jitterOffsetX" );
            _MTLFX_PRIVATE_DEF_SEL( jitterOffsetY,
                                    "jitterOffsetY" );
            _MTLFX_PRIVATE_DEF_SEL( maskTexture,
                                    "maskTexture" );
            _MTLFX_PRIVATE_DEF_SEL( maskTextureFormat,
                                    "maskTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( maskTextureUsage,
                                    "maskTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( motionTexture,
                                    "motionTexture" );
            _MTLFX_PRIVATE_DEF_SEL( motionTextureFormat,
                                    "motionTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( motionTextureUsage,
                                    "motionTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( motionVectorScaleX,
                                    "motionVectorScaleX" );
            _MTLFX_PRIVATE_DEF_SEL( motionVectorScaleY,
                                    "motionVectorScaleY" );
            _MTLFX_PRIVATE_DEF_SEL( nearPlane,
                                    "nearPlane" );
            _MTLFX_PRIVATE_DEF_SEL( newFrameInterpolatorWithDevice_,
                                    "newFrameInterpolatorWithDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( newFrameInterpolatorWithDevice_compiler_,
                                    "newFrameInterpolatorWithDevice:compiler:" );
            _MTLFX_PRIVATE_DEF_SEL( newTemporalDenoisedScalerWithDevice_,
                                    "newTemporalDenoisedScalerWithDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( newTemporalDenoisedScalerWithDevice_compiler_,
                                    "newTemporalDenoisedScalerWithDevice:compiler:" );
            _MTLFX_PRIVATE_DEF_SEL( newSpatialScalerWithDevice_,
                                    "newSpatialScalerWithDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( newSpatialScalerWithDevice_compiler_,
                                    "newSpatialScalerWithDevice:compiler:" );
            _MTLFX_PRIVATE_DEF_SEL( newTemporalScalerWithDevice_,
                                    "newTemporalScalerWithDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( newTemporalScalerWithDevice_compiler_,
                                    "newTemporalScalerWithDevice:compiler:" );
            _MTLFX_PRIVATE_DEF_SEL( normalTexture,
                                    "normalTexture" );
            _MTLFX_PRIVATE_DEF_SEL( normalTextureFormat,
                                    "normalTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( normalTextureUsage,
                                    "normalTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( outputHeight,
                                    "outputHeight" );
            _MTLFX_PRIVATE_DEF_SEL( outputTexture,
                                    "outputTexture" );
            _MTLFX_PRIVATE_DEF_SEL( outputTextureFormat,
                                    "outputTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( outputTextureUsage,
                                    "outputTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( outputWidth,
                                    "outputWidth" );
            _MTLFX_PRIVATE_DEF_SEL( preExposure,
                                    "preExposure" );
            _MTLFX_PRIVATE_DEF_SEL( transparencyOverlayTextureFormat,
                                    "transparencyOverlayTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( transparencyOverlayTextureUsage,
                                    "transparencyOverlayTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( prevColorTexture,
                                    "prevColorTexture" );
            _MTLFX_PRIVATE_DEF_SEL( reactiveMaskTextureFormat,
                                    "reactiveMaskTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( reactiveTextureUsage,
                                    "reactiveTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( reactiveMaskTexture,
                                    "reactiveMaskTexture" );
            _MTLFX_PRIVATE_DEF_SEL( reset,
                                    "reset" );
            _MTLFX_PRIVATE_DEF_SEL( requiresSynchronousInitialization,
                                    "requiresSynchronousInitialization" );
            _MTLFX_PRIVATE_DEF_SEL( roughnessTextureFormat,
                                    "roughnessTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( roughnessTextureUsage,
                                    "roughnessTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( scaler,
                                    "scaler" );
            _MTLFX_PRIVATE_DEF_SEL( scaler4,
                                    "scaler4" );
            _MTLFX_PRIVATE_DEF_SEL( setAspectRatio_,
                                    "setAspectRatio:" );
            _MTLFX_PRIVATE_DEF_SEL( setAutoExposureEnabled_,
                                    "setAutoExposureEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setColorProcessingMode_,
                                    "setColorProcessingMode:" );
            _MTLFX_PRIVATE_DEF_SEL( setColorTexture_,
                                    "setColorTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setColorTextureFormat_,
                                    "setColorTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setDeltaTime_,
                                    "setDeltaTime:" );
            _MTLFX_PRIVATE_DEF_SEL( setDenoiseStrengthMaskTexture_,
                                    "setDenoiseStrengthMaskTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setDenoiseStrengthMaskTextureEnabled_,
                                    "setDenoiseStrengthMaskTextureEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setDenoiseStrengthMaskTextureFormat_,
                                    "setDenoiseStrengthMaskTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthInverted_,
                                    "setDepthInverted:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthReversed_,
                                    "setDepthReversed:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthTexture_,
                                    "setDepthTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthTextureFormat_,
                                    "setDepthTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setDiffuseAlbedoTexture_,
                                    "setDiffuseAlbedoTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setDiffuseAlbedoTextureFormat_,
                                    "setDiffuseAlbedoTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setExposureTexture_,
                                    "setExposureTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setFarPlane_,
                                    "setFarPlane:" );
            _MTLFX_PRIVATE_DEF_SEL( setFence_,
                                    "setFence:" );
            _MTLFX_PRIVATE_DEF_SEL( setFieldOfView_,
                                    "setFieldOfView:" );
            _MTLFX_PRIVATE_DEF_SEL( setHeight_,
                                    "setHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentHeight_,
                                    "setInputContentHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentMaxScale_,
                                    "setInputContentMaxScale:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentMinScale_,
                                    "setInputContentMinScale:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentPropertiesEnabled_,
                                    "setInputContentPropertiesEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentWidth_,
                                    "setInputContentWidth:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputHeight_,
                                    "setInputHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputWidth_,
                                    "setInputWidth:" );
            _MTLFX_PRIVATE_DEF_SEL( setIsUITextureComposited_,
                                    "setIsUITextureComposited:" );
            _MTLFX_PRIVATE_DEF_SEL( setJitterOffsetX_,
                                    "setJitterOffsetX:" );
            _MTLFX_PRIVATE_DEF_SEL( setJitterOffsetY_,
                                    "setJitterOffsetY:" );
            _MTLFX_PRIVATE_DEF_SEL( setNearPlane_,
                                    "setNearPlane:" );
            _MTLFX_PRIVATE_DEF_SEL( setMaskTexture_,
                                    "setMaskTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setMaskTextureFormat_,
                                    "setMaskTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionTexture_,
                                    "setMotionTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionTextureFormat_,
                                    "setMotionTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionVectorScaleX_,
                                    "setMotionVectorScaleX:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionVectorScaleY_,
                                    "setMotionVectorScaleY:" );
            _MTLFX_PRIVATE_DEF_SEL( setNormalTexture_,
                                    "setNormalTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setNormalTextureFormat_,
                                    "setNormalTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputHeight_,
                                    "setOutputHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputTexture_,
                                    "setOutputTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputTextureFormat_,
                                    "setOutputTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputWidth_,
                                    "setOutputWidth:" );
            _MTLFX_PRIVATE_DEF_SEL( transparencyOverlayTexture,
                                    "transparencyOverlayTexture" );
            _MTLFX_PRIVATE_DEF_SEL( setTransparencyOverlayTexture_,
                                    "setTransparencyOverlayTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setTransparencyOverlayTextureEnabled_,
                                    "setTransparencyOverlayTextureEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setPreExposure_,
                                    "setPreExposure:" );
            _MTLFX_PRIVATE_DEF_SEL( setTransparencyOverlayTextureFormat_,
                                    "setTransparencyOverlayTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setPrevColorTexture_,
                                    "setPrevColorTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setReactiveMaskTexture_,
                                    "setReactiveMaskTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setReactiveMaskTextureEnabled_,
                                    "setReactiveMaskTextureEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setReactiveMaskTextureFormat_,
                                    "setReactiveMaskTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setRequiresSynchronousInitialization_,
                                    "setRequiresSynchronousInitialization:" );
            _MTLFX_PRIVATE_DEF_SEL( setReset_,
                                    "setReset:" );
            _MTLFX_PRIVATE_DEF_SEL( roughnessTexture,
                                    "roughnessTexture" );
            _MTLFX_PRIVATE_DEF_SEL( setRoughnessTexture_,
                                    "setRoughnessTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setRoughnessTextureFormat_,
                                    "setRoughnessTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setScaler_,
                                    "setScaler:" );
            _MTLFX_PRIVATE_DEF_SEL( setShouldResetHistory_,
                                    "setShouldResetHistory:" );
            _MTLFX_PRIVATE_DEF_SEL( specularHitDistanceTexture,
                                    "specularHitDistanceTexture" );
            _MTLFX_PRIVATE_DEF_SEL( setSpecularHitDistanceTexture_,
                                    "setSpecularHitDistanceTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setSpecularHitDistanceTextureEnabled_,
                                    "setSpecularHitDistanceTextureEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setSpecularAlbedoTexture_,
                                    "setSpecularAlbedoTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setSpecularAlbedoTextureFormat_,
                                    "setSpecularAlbedoTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setSpecularHitDistanceTextureFormat_,
                                    "setSpecularHitDistanceTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setUITexture_,
                                    "setUITexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setUITextureFormat_,
                                    "setUITextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setViewToClipMatrix_,
                                    "setViewToClipMatrix:" );
            _MTLFX_PRIVATE_DEF_SEL( setWidth_,
                                    "setWidth:" );
            _MTLFX_PRIVATE_DEF_SEL( setWorldToViewMatrix_,
                                    "setWorldToViewMatrix:" );
            _MTLFX_PRIVATE_DEF_SEL( shouldResetHistory,
                                    "shouldResetHistory" );
            _MTLFX_PRIVATE_DEF_SEL( specularAlbedoTexture,
                                    "specularAlbedoTexture" );
            _MTLFX_PRIVATE_DEF_SEL( specularAlbedoTextureFormat,
                                    "specularAlbedoTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( specularAlbedoTextureUsage,
                                    "specularAlbedoTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( specularHitDistanceTextureFormat,
                                    "specularHitDistanceTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( specularHitDistanceTextureUsage,
                                    "specularHitDistanceTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( supportedInputContentMaxScaleForDevice_,
                                    "supportedInputContentMaxScaleForDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( supportedInputContentMinScaleForDevice_,
                                    "supportedInputContentMinScaleForDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( supportsDevice_,
                                    "supportsDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( supportsMetal4FX_,
                                    "supportsMetal4FX:" );
            _MTLFX_PRIVATE_DEF_SEL( uiTexture,
                                    "uiTexture" );
            _MTLFX_PRIVATE_DEF_SEL( uiTextureFormat,
                                    "uiTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( uiTextureUsage,
                                    "uiTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( viewToClipMatrix,
                                    "viewToClipMatrix" );
            _MTLFX_PRIVATE_DEF_SEL( width,
                                    "width" );
            _MTLFX_PRIVATE_DEF_SEL( worldToViewMatrix,
                                    "worldToViewMatrix" );
        } // Selector
    } // Private
} // MTLFX

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
