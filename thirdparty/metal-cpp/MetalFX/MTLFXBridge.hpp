#pragma once

// Consolidated extern "C" trampoline decls for this framework.
// One entry per (return, args, selector) — identical C++ signatures
// across multiple classes collapse to a single linker alias of
// `_objc_msgSend$<selector>`. Per-class headers include this file
// instead of declaring their own externs.

#include "MTLFXDefines.hpp"
#include <objc/objc.h>
#include "../Foundation/NSTypes.hpp"

namespace MTL {
    class CommandBuffer;
    class Device;
    class Fence;
    class Texture;
    enum PixelFormat : NS::UInteger;
    using TextureUsage = NS::UInteger;
}
namespace MTL4 {
    class Compiler;
}
namespace MTL4FX {
    class FrameInterpolator;
    class SpatialScaler;
    class TemporalDenoisedScaler;
    class TemporalScaler;
}
namespace MTLFX {
    class FrameInterpolator;
    class FrameInterpolatorDescriptor;
    class SpatialScaler;
    class SpatialScalerDescriptor;
    class TemporalDenoisedScaler;
    class TemporalDenoisedScalerDescriptor;
    class TemporalScaler;
    class TemporalScalerDescriptor;
    enum SpatialScalerColorProcessingMode : NS::Integer;
}
namespace NS {
    class Object;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

extern "C" {
MTLFX::FrameInterpolatorDescriptor* _MTLFX_msg_MTLFX__FrameInterpolatorDescriptorp_alloc(const void*, SEL) __asm__("_objc_msgSend$" "alloc");
MTLFX::SpatialScalerDescriptor* _MTLFX_msg_MTLFX__SpatialScalerDescriptorp_alloc(const void*, SEL) __asm__("_objc_msgSend$" "alloc");
MTLFX::TemporalDenoisedScalerDescriptor* _MTLFX_msg_MTLFX__TemporalDenoisedScalerDescriptorp_alloc(const void*, SEL) __asm__("_objc_msgSend$" "alloc");
MTLFX::TemporalScalerDescriptor* _MTLFX_msg_MTLFX__TemporalScalerDescriptorp_alloc(const void*, SEL) __asm__("_objc_msgSend$" "alloc");
float _MTLFX_msg_float_aspectRatio(const void*, SEL) __asm__("_objc_msgSend$" "aspectRatio");
bool _MTLFX_msg_bool_autoExposureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "autoExposureEnabled");
MTLFX::SpatialScalerColorProcessingMode _MTLFX_msg_MTLFX__SpatialScalerColorProcessingMode_colorProcessingMode(const void*, SEL) __asm__("_objc_msgSend$" "colorProcessingMode");
MTL::Texture* _MTLFX_msg_MTL__Texturep_colorTexture(const void*, SEL) __asm__("_objc_msgSend$" "colorTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_colorTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "colorTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_colorTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "colorTextureUsage");
float _MTLFX_msg_float_deltaTime(const void*, SEL) __asm__("_objc_msgSend$" "deltaTime");
MTL::Texture* _MTLFX_msg_MTL__Texturep_denoiseStrengthMaskTexture(const void*, SEL) __asm__("_objc_msgSend$" "denoiseStrengthMaskTexture");
bool _MTLFX_msg_bool_denoiseStrengthMaskTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "denoiseStrengthMaskTextureEnabled");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_denoiseStrengthMaskTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "denoiseStrengthMaskTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_denoiseStrengthMaskTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "denoiseStrengthMaskTextureUsage");
bool _MTLFX_msg_bool_depthReversed(const void*, SEL) __asm__("_objc_msgSend$" "depthReversed");
MTL::Texture* _MTLFX_msg_MTL__Texturep_depthTexture(const void*, SEL) __asm__("_objc_msgSend$" "depthTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_depthTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "depthTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_depthTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "depthTextureUsage");
MTL::Texture* _MTLFX_msg_MTL__Texturep_diffuseAlbedoTexture(const void*, SEL) __asm__("_objc_msgSend$" "diffuseAlbedoTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_diffuseAlbedoTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "diffuseAlbedoTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_diffuseAlbedoTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "diffuseAlbedoTextureUsage");
void _MTLFX_msg_v_encodeToCommandBuffer__MTL__CommandBufferp(const void*, SEL, MTL::CommandBuffer*) __asm__("_objc_msgSend$" "encodeToCommandBuffer:");
MTL::Texture* _MTLFX_msg_MTL__Texturep_exposureTexture(const void*, SEL) __asm__("_objc_msgSend$" "exposureTexture");
float _MTLFX_msg_float_farPlane(const void*, SEL) __asm__("_objc_msgSend$" "farPlane");
MTL::Fence* _MTLFX_msg_MTL__Fencep_fence(const void*, SEL) __asm__("_objc_msgSend$" "fence");
float _MTLFX_msg_float_fieldOfView(const void*, SEL) __asm__("_objc_msgSend$" "fieldOfView");
MTLFX::FrameInterpolatorDescriptor* _MTLFX_msg_MTLFX__FrameInterpolatorDescriptorp_init(const void*, SEL) __asm__("_objc_msgSend$" "init");
MTLFX::SpatialScalerDescriptor* _MTLFX_msg_MTLFX__SpatialScalerDescriptorp_init(const void*, SEL) __asm__("_objc_msgSend$" "init");
MTLFX::TemporalDenoisedScalerDescriptor* _MTLFX_msg_MTLFX__TemporalDenoisedScalerDescriptorp_init(const void*, SEL) __asm__("_objc_msgSend$" "init");
MTLFX::TemporalScalerDescriptor* _MTLFX_msg_MTLFX__TemporalScalerDescriptorp_init(const void*, SEL) __asm__("_objc_msgSend$" "init");
NS::UInteger _MTLFX_msg_NS__UInteger_inputContentHeight(const void*, SEL) __asm__("_objc_msgSend$" "inputContentHeight");
float _MTLFX_msg_float_inputContentMaxScale(const void*, SEL) __asm__("_objc_msgSend$" "inputContentMaxScale");
float _MTLFX_msg_float_inputContentMinScale(const void*, SEL) __asm__("_objc_msgSend$" "inputContentMinScale");
bool _MTLFX_msg_bool_inputContentPropertiesEnabled(const void*, SEL) __asm__("_objc_msgSend$" "inputContentPropertiesEnabled");
NS::UInteger _MTLFX_msg_NS__UInteger_inputContentWidth(const void*, SEL) __asm__("_objc_msgSend$" "inputContentWidth");
NS::UInteger _MTLFX_msg_NS__UInteger_inputHeight(const void*, SEL) __asm__("_objc_msgSend$" "inputHeight");
NS::UInteger _MTLFX_msg_NS__UInteger_inputWidth(const void*, SEL) __asm__("_objc_msgSend$" "inputWidth");
bool _MTLFX_msg_bool_isAutoExposureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "isAutoExposureEnabled");
bool _MTLFX_msg_bool_isDenoiseStrengthMaskTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "isDenoiseStrengthMaskTextureEnabled");
bool _MTLFX_msg_bool_isDepthReversed(const void*, SEL) __asm__("_objc_msgSend$" "isDepthReversed");
bool _MTLFX_msg_bool_isInputContentPropertiesEnabled(const void*, SEL) __asm__("_objc_msgSend$" "isInputContentPropertiesEnabled");
bool _MTLFX_msg_bool_isReactiveMaskTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "isReactiveMaskTextureEnabled");
bool _MTLFX_msg_bool_isSpecularHitDistanceTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "isSpecularHitDistanceTextureEnabled");
bool _MTLFX_msg_bool_isTransparencyOverlayTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "isTransparencyOverlayTextureEnabled");
bool _MTLFX_msg_bool_isUITextureComposited(const void*, SEL) __asm__("_objc_msgSend$" "isUITextureComposited");
float _MTLFX_msg_float_jitterOffsetX(const void*, SEL) __asm__("_objc_msgSend$" "jitterOffsetX");
float _MTLFX_msg_float_jitterOffsetY(const void*, SEL) __asm__("_objc_msgSend$" "jitterOffsetY");
MTL::Texture* _MTLFX_msg_MTL__Texturep_motionTexture(const void*, SEL) __asm__("_objc_msgSend$" "motionTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_motionTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "motionTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_motionTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "motionTextureUsage");
float _MTLFX_msg_float_motionVectorScaleX(const void*, SEL) __asm__("_objc_msgSend$" "motionVectorScaleX");
float _MTLFX_msg_float_motionVectorScaleY(const void*, SEL) __asm__("_objc_msgSend$" "motionVectorScaleY");
float _MTLFX_msg_float_nearPlane(const void*, SEL) __asm__("_objc_msgSend$" "nearPlane");
MTLFX::FrameInterpolator* _MTLFX_msg_MTLFX__FrameInterpolatorp_newFrameInterpolatorWithDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "newFrameInterpolatorWithDevice:");
MTL4FX::FrameInterpolator* _MTLFX_msg_MTL4FX__FrameInterpolatorp_newFrameInterpolatorWithDevice_compiler__MTL__Devicep_MTL4__Compilerp(const void*, SEL, MTL::Device*, MTL4::Compiler*) __asm__("_objc_msgSend$" "newFrameInterpolatorWithDevice:compiler:");
MTLFX::SpatialScaler* _MTLFX_msg_MTLFX__SpatialScalerp_newSpatialScalerWithDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "newSpatialScalerWithDevice:");
MTL4FX::SpatialScaler* _MTLFX_msg_MTL4FX__SpatialScalerp_newSpatialScalerWithDevice_compiler__MTL__Devicep_MTL4__Compilerp(const void*, SEL, MTL::Device*, MTL4::Compiler*) __asm__("_objc_msgSend$" "newSpatialScalerWithDevice:compiler:");
MTLFX::TemporalDenoisedScaler* _MTLFX_msg_MTLFX__TemporalDenoisedScalerp_newTemporalDenoisedScalerWithDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "newTemporalDenoisedScalerWithDevice:");
MTL4FX::TemporalDenoisedScaler* _MTLFX_msg_MTL4FX__TemporalDenoisedScalerp_newTemporalDenoisedScalerWithDevice_compiler__MTL__Devicep_MTL4__Compilerp(const void*, SEL, MTL::Device*, MTL4::Compiler*) __asm__("_objc_msgSend$" "newTemporalDenoisedScalerWithDevice:compiler:");
MTLFX::TemporalScaler* _MTLFX_msg_MTLFX__TemporalScalerp_newTemporalScalerWithDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "newTemporalScalerWithDevice:");
MTL4FX::TemporalScaler* _MTLFX_msg_MTL4FX__TemporalScalerp_newTemporalScalerWithDevice_compiler__MTL__Devicep_MTL4__Compilerp(const void*, SEL, MTL::Device*, MTL4::Compiler*) __asm__("_objc_msgSend$" "newTemporalScalerWithDevice:compiler:");
MTL::Texture* _MTLFX_msg_MTL__Texturep_normalTexture(const void*, SEL) __asm__("_objc_msgSend$" "normalTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_normalTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "normalTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_normalTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "normalTextureUsage");
NS::UInteger _MTLFX_msg_NS__UInteger_outputHeight(const void*, SEL) __asm__("_objc_msgSend$" "outputHeight");
MTL::Texture* _MTLFX_msg_MTL__Texturep_outputTexture(const void*, SEL) __asm__("_objc_msgSend$" "outputTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_outputTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "outputTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_outputTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "outputTextureUsage");
NS::UInteger _MTLFX_msg_NS__UInteger_outputWidth(const void*, SEL) __asm__("_objc_msgSend$" "outputWidth");
float _MTLFX_msg_float_preExposure(const void*, SEL) __asm__("_objc_msgSend$" "preExposure");
MTL::Texture* _MTLFX_msg_MTL__Texturep_prevColorTexture(const void*, SEL) __asm__("_objc_msgSend$" "prevColorTexture");
MTL::Texture* _MTLFX_msg_MTL__Texturep_reactiveMaskTexture(const void*, SEL) __asm__("_objc_msgSend$" "reactiveMaskTexture");
bool _MTLFX_msg_bool_reactiveMaskTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "reactiveMaskTextureEnabled");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_reactiveMaskTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "reactiveMaskTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_reactiveTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "reactiveTextureUsage");
bool _MTLFX_msg_bool_requiresSynchronousInitialization(const void*, SEL) __asm__("_objc_msgSend$" "requiresSynchronousInitialization");
bool _MTLFX_msg_bool_reset(const void*, SEL) __asm__("_objc_msgSend$" "reset");
MTL::Texture* _MTLFX_msg_MTL__Texturep_roughnessTexture(const void*, SEL) __asm__("_objc_msgSend$" "roughnessTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_roughnessTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "roughnessTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_roughnessTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "roughnessTextureUsage");
NS::Object* _MTLFX_msg_NS__Objectp_scaler(const void*, SEL) __asm__("_objc_msgSend$" "scaler");
void _MTLFX_msg_v_setAspectRatio__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setAspectRatio:");
void _MTLFX_msg_v_setAutoExposureEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setAutoExposureEnabled:");
void _MTLFX_msg_v_setColorProcessingMode__MTLFX__SpatialScalerColorProcessingMode(const void*, SEL, MTLFX::SpatialScalerColorProcessingMode) __asm__("_objc_msgSend$" "setColorProcessingMode:");
void _MTLFX_msg_v_setColorTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setColorTexture:");
void _MTLFX_msg_v_setColorTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setColorTextureFormat:");
void _MTLFX_msg_v_setDeltaTime__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setDeltaTime:");
void _MTLFX_msg_v_setDenoiseStrengthMaskTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setDenoiseStrengthMaskTexture:");
void _MTLFX_msg_v_setDenoiseStrengthMaskTextureEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setDenoiseStrengthMaskTextureEnabled:");
void _MTLFX_msg_v_setDenoiseStrengthMaskTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setDenoiseStrengthMaskTextureFormat:");
void _MTLFX_msg_v_setDepthReversed__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setDepthReversed:");
void _MTLFX_msg_v_setDepthTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setDepthTexture:");
void _MTLFX_msg_v_setDepthTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setDepthTextureFormat:");
void _MTLFX_msg_v_setDiffuseAlbedoTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setDiffuseAlbedoTexture:");
void _MTLFX_msg_v_setDiffuseAlbedoTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setDiffuseAlbedoTextureFormat:");
void _MTLFX_msg_v_setExposureTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setExposureTexture:");
void _MTLFX_msg_v_setFarPlane__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setFarPlane:");
void _MTLFX_msg_v_setFence__MTL__Fencep(const void*, SEL, MTL::Fence*) __asm__("_objc_msgSend$" "setFence:");
void _MTLFX_msg_v_setFieldOfView__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setFieldOfView:");
void _MTLFX_msg_v_setInputContentHeight__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setInputContentHeight:");
void _MTLFX_msg_v_setInputContentMaxScale__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setInputContentMaxScale:");
void _MTLFX_msg_v_setInputContentMinScale__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setInputContentMinScale:");
void _MTLFX_msg_v_setInputContentPropertiesEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setInputContentPropertiesEnabled:");
void _MTLFX_msg_v_setInputContentWidth__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setInputContentWidth:");
void _MTLFX_msg_v_setInputHeight__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setInputHeight:");
void _MTLFX_msg_v_setInputWidth__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setInputWidth:");
void _MTLFX_msg_v_setIsUITextureComposited__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setIsUITextureComposited:");
void _MTLFX_msg_v_setJitterOffsetX__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setJitterOffsetX:");
void _MTLFX_msg_v_setJitterOffsetY__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setJitterOffsetY:");
void _MTLFX_msg_v_setMotionTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setMotionTexture:");
void _MTLFX_msg_v_setMotionTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setMotionTextureFormat:");
void _MTLFX_msg_v_setMotionVectorScaleX__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setMotionVectorScaleX:");
void _MTLFX_msg_v_setMotionVectorScaleY__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setMotionVectorScaleY:");
void _MTLFX_msg_v_setNearPlane__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setNearPlane:");
void _MTLFX_msg_v_setNormalTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setNormalTexture:");
void _MTLFX_msg_v_setNormalTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setNormalTextureFormat:");
void _MTLFX_msg_v_setOutputHeight__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setOutputHeight:");
void _MTLFX_msg_v_setOutputTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setOutputTexture:");
void _MTLFX_msg_v_setOutputTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setOutputTextureFormat:");
void _MTLFX_msg_v_setOutputWidth__NS__UInteger(const void*, SEL, NS::UInteger) __asm__("_objc_msgSend$" "setOutputWidth:");
void _MTLFX_msg_v_setPreExposure__float(const void*, SEL, float) __asm__("_objc_msgSend$" "setPreExposure:");
void _MTLFX_msg_v_setPrevColorTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setPrevColorTexture:");
void _MTLFX_msg_v_setReactiveMaskTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setReactiveMaskTexture:");
void _MTLFX_msg_v_setReactiveMaskTextureEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setReactiveMaskTextureEnabled:");
void _MTLFX_msg_v_setReactiveMaskTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setReactiveMaskTextureFormat:");
void _MTLFX_msg_v_setRequiresSynchronousInitialization__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setRequiresSynchronousInitialization:");
void _MTLFX_msg_v_setReset__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setReset:");
void _MTLFX_msg_v_setRoughnessTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setRoughnessTexture:");
void _MTLFX_msg_v_setRoughnessTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setRoughnessTextureFormat:");
void _MTLFX_msg_v_setScaler__NS__Objectp(const void*, SEL, NS::Object*) __asm__("_objc_msgSend$" "setScaler:");
void _MTLFX_msg_v_setShouldResetHistory__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setShouldResetHistory:");
void _MTLFX_msg_v_setSpecularAlbedoTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setSpecularAlbedoTexture:");
void _MTLFX_msg_v_setSpecularAlbedoTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setSpecularAlbedoTextureFormat:");
void _MTLFX_msg_v_setSpecularHitDistanceTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setSpecularHitDistanceTexture:");
void _MTLFX_msg_v_setSpecularHitDistanceTextureEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setSpecularHitDistanceTextureEnabled:");
void _MTLFX_msg_v_setSpecularHitDistanceTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setSpecularHitDistanceTextureFormat:");
void _MTLFX_msg_v_setTransparencyOverlayTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setTransparencyOverlayTexture:");
void _MTLFX_msg_v_setTransparencyOverlayTextureEnabled__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setTransparencyOverlayTextureEnabled:");
void _MTLFX_msg_v_setTransparencyOverlayTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setTransparencyOverlayTextureFormat:");
void _MTLFX_msg_v_setUITexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setUITexture:");
void _MTLFX_msg_v_setUITextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setUITextureFormat:");
void _MTLFX_msg_v_setUiTexture__MTL__Texturep(const void*, SEL, MTL::Texture*) __asm__("_objc_msgSend$" "setUiTexture:");
void _MTLFX_msg_v_setUiTextureComposited__bool(const void*, SEL, bool) __asm__("_objc_msgSend$" "setUiTextureComposited:");
void _MTLFX_msg_v_setUiTextureFormat__MTL__PixelFormat(const void*, SEL, MTL::PixelFormat) __asm__("_objc_msgSend$" "setUiTextureFormat:");
void _MTLFX_msg_v_setViewToClipMatrix__voidp(const void*, SEL, void*) __asm__("_objc_msgSend$" "setViewToClipMatrix:");
void _MTLFX_msg_v_setWorldToViewMatrix__voidp(const void*, SEL, void*) __asm__("_objc_msgSend$" "setWorldToViewMatrix:");
bool _MTLFX_msg_bool_shouldResetHistory(const void*, SEL) __asm__("_objc_msgSend$" "shouldResetHistory");
MTL::Texture* _MTLFX_msg_MTL__Texturep_specularAlbedoTexture(const void*, SEL) __asm__("_objc_msgSend$" "specularAlbedoTexture");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_specularAlbedoTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "specularAlbedoTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_specularAlbedoTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "specularAlbedoTextureUsage");
MTL::Texture* _MTLFX_msg_MTL__Texturep_specularHitDistanceTexture(const void*, SEL) __asm__("_objc_msgSend$" "specularHitDistanceTexture");
bool _MTLFX_msg_bool_specularHitDistanceTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "specularHitDistanceTextureEnabled");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_specularHitDistanceTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "specularHitDistanceTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_specularHitDistanceTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "specularHitDistanceTextureUsage");
float _MTLFX_msg_float_supportedInputContentMaxScaleForDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "supportedInputContentMaxScaleForDevice:");
float _MTLFX_msg_float_supportedInputContentMinScaleForDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "supportedInputContentMinScaleForDevice:");
bool _MTLFX_msg_bool_supportsDevice__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "supportsDevice:");
bool _MTLFX_msg_bool_supportsMetal4FX__MTL__Devicep(const void*, SEL, MTL::Device*) __asm__("_objc_msgSend$" "supportsMetal4FX:");
MTL::Texture* _MTLFX_msg_MTL__Texturep_transparencyOverlayTexture(const void*, SEL) __asm__("_objc_msgSend$" "transparencyOverlayTexture");
bool _MTLFX_msg_bool_transparencyOverlayTextureEnabled(const void*, SEL) __asm__("_objc_msgSend$" "transparencyOverlayTextureEnabled");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_transparencyOverlayTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "transparencyOverlayTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_transparencyOverlayTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "transparencyOverlayTextureUsage");
MTL::Texture* _MTLFX_msg_MTL__Texturep_uiTexture(const void*, SEL) __asm__("_objc_msgSend$" "uiTexture");
bool _MTLFX_msg_bool_uiTextureComposited(const void*, SEL) __asm__("_objc_msgSend$" "uiTextureComposited");
MTL::PixelFormat _MTLFX_msg_MTL__PixelFormat_uiTextureFormat(const void*, SEL) __asm__("_objc_msgSend$" "uiTextureFormat");
MTL::TextureUsage _MTLFX_msg_MTL__TextureUsage_uiTextureUsage(const void*, SEL) __asm__("_objc_msgSend$" "uiTextureUsage");
void* _MTLFX_msg_voidp_viewToClipMatrix(const void*, SEL) __asm__("_objc_msgSend$" "viewToClipMatrix");
void* _MTLFX_msg_voidp_worldToViewMatrix(const void*, SEL) __asm__("_objc_msgSend$" "worldToViewMatrix");
} // extern "C"

#pragma clang diagnostic pop
