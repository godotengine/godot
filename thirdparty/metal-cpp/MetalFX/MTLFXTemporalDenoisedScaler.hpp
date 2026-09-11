#pragma once

#include "MTLFXDefines.hpp"
#include "MTLFXBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

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
    class TemporalDenoisedScaler;
}

namespace MTLFX
{

class TemporalDenoisedScalerDescriptor;
class TemporalDenoisedScalerBase;
class TemporalDenoisedScaler;

class TemporalDenoisedScalerDescriptor : public NS::Copying<TemporalDenoisedScalerDescriptor>
{
public:
    static TemporalDenoisedScalerDescriptor* alloc();
    TemporalDenoisedScalerDescriptor*        init() const;

    static float supportedInputContentMaxScale(MTL::Device* device);
    static float supportedInputContentMinScale(MTL::Device* device);
    static bool  supportsDevice(MTL::Device* device);
    static bool  supportsMetal4FX(MTL::Device* device);

    bool                            autoExposureEnabled() const;
    MTL::PixelFormat                colorTextureFormat() const;
    bool                            denoiseStrengthMaskTextureEnabled() const;
    MTL::PixelFormat                denoiseStrengthMaskTextureFormat() const;
    MTL::PixelFormat                depthTextureFormat() const;
    MTL::PixelFormat                diffuseAlbedoTextureFormat() const;
    NS::UInteger                    inputHeight() const;
    NS::UInteger                    inputWidth() const;
    bool                            isAutoExposureEnabled();
    bool                            isDenoiseStrengthMaskTextureEnabled();
    bool                            isReactiveMaskTextureEnabled();
    bool                            isSpecularHitDistanceTextureEnabled();
    bool                            isTransparencyOverlayTextureEnabled();
    MTL::PixelFormat                motionTextureFormat() const;
    MTLFX::TemporalDenoisedScaler*  newTemporalDenoisedScaler(MTL::Device* device);
    MTL4FX::TemporalDenoisedScaler* newTemporalDenoisedScaler(MTL::Device* device, MTL4::Compiler* compiler);
    MTL::PixelFormat                normalTextureFormat() const;
    NS::UInteger                    outputHeight() const;
    MTL::PixelFormat                outputTextureFormat() const;
    NS::UInteger                    outputWidth() const;
    bool                            reactiveMaskTextureEnabled() const;
    MTL::PixelFormat                reactiveMaskTextureFormat() const;
    bool                            requiresSynchronousInitialization() const;
    MTL::PixelFormat                roughnessTextureFormat() const;
    void                            setAutoExposureEnabled(bool autoExposureEnabled);
    void                            setColorTextureFormat(MTL::PixelFormat colorTextureFormat);
    void                            setDenoiseStrengthMaskTextureEnabled(bool denoiseStrengthMaskTextureEnabled);
    void                            setDenoiseStrengthMaskTextureFormat(MTL::PixelFormat denoiseStrengthMaskTextureFormat);
    void                            setDepthTextureFormat(MTL::PixelFormat depthTextureFormat);
    void                            setDiffuseAlbedoTextureFormat(MTL::PixelFormat diffuseAlbedoTextureFormat);
    void                            setInputHeight(NS::UInteger inputHeight);
    void                            setInputWidth(NS::UInteger inputWidth);
    void                            setMotionTextureFormat(MTL::PixelFormat motionTextureFormat);
    void                            setNormalTextureFormat(MTL::PixelFormat normalTextureFormat);
    void                            setOutputHeight(NS::UInteger outputHeight);
    void                            setOutputTextureFormat(MTL::PixelFormat outputTextureFormat);
    void                            setOutputWidth(NS::UInteger outputWidth);
    void                            setReactiveMaskTextureEnabled(bool reactiveMaskTextureEnabled);
    void                            setReactiveMaskTextureFormat(MTL::PixelFormat reactiveMaskTextureFormat);
    void                            setRequiresSynchronousInitialization(bool requiresSynchronousInitialization);
    void                            setRoughnessTextureFormat(MTL::PixelFormat roughnessTextureFormat);
    void                            setSpecularAlbedoTextureFormat(MTL::PixelFormat specularAlbedoTextureFormat);
    void                            setSpecularHitDistanceTextureEnabled(bool specularHitDistanceTextureEnabled);
    void                            setSpecularHitDistanceTextureFormat(MTL::PixelFormat specularHitDistanceTextureFormat);
    void                            setTransparencyOverlayTextureEnabled(bool transparencyOverlayTextureEnabled);
    void                            setTransparencyOverlayTextureFormat(MTL::PixelFormat transparencyOverlayTextureFormat);
    MTL::PixelFormat                specularAlbedoTextureFormat() const;
    bool                            specularHitDistanceTextureEnabled() const;
    MTL::PixelFormat                specularHitDistanceTextureFormat() const;
    bool                            transparencyOverlayTextureEnabled() const;
    MTL::PixelFormat                transparencyOverlayTextureFormat() const;

};

class TemporalDenoisedScalerBase : public NS::Referencing<TemporalDenoisedScalerBase>
{
public:
    MTL::Texture*     colorTexture() const;
    MTL::PixelFormat  colorTextureFormat() const;
    MTL::TextureUsage colorTextureUsage() const;
    MTL::Texture*     denoiseStrengthMaskTexture() const;
    MTL::PixelFormat  denoiseStrengthMaskTextureFormat() const;
    MTL::TextureUsage denoiseStrengthMaskTextureUsage() const;
    bool              depthReversed() const;
    MTL::Texture*     depthTexture() const;
    MTL::PixelFormat  depthTextureFormat() const;
    MTL::TextureUsage depthTextureUsage() const;
    MTL::Texture*     diffuseAlbedoTexture() const;
    MTL::PixelFormat  diffuseAlbedoTextureFormat() const;
    MTL::TextureUsage diffuseAlbedoTextureUsage() const;
    MTL::Texture*     exposureTexture() const;
    MTL::Fence*       fence() const;
    float             inputContentMaxScale() const;
    float             inputContentMinScale() const;
    NS::UInteger      inputHeight() const;
    NS::UInteger      inputWidth() const;
    bool              isDepthReversed();
    float             jitterOffsetX() const;
    float             jitterOffsetY() const;
    MTL::Texture*     motionTexture() const;
    MTL::PixelFormat  motionTextureFormat() const;
    MTL::TextureUsage motionTextureUsage() const;
    float             motionVectorScaleX() const;
    float             motionVectorScaleY() const;
    MTL::Texture*     normalTexture() const;
    MTL::PixelFormat  normalTextureFormat() const;
    MTL::TextureUsage normalTextureUsage() const;
    NS::UInteger      outputHeight() const;
    MTL::Texture*     outputTexture() const;
    MTL::PixelFormat  outputTextureFormat() const;
    MTL::TextureUsage outputTextureUsage() const;
    NS::UInteger      outputWidth() const;
    float             preExposure() const;
    MTL::Texture*     reactiveMaskTexture() const;
    MTL::PixelFormat  reactiveMaskTextureFormat() const;
    MTL::TextureUsage reactiveTextureUsage() const;
    MTL::Texture*     roughnessTexture() const;
    MTL::PixelFormat  roughnessTextureFormat() const;
    MTL::TextureUsage roughnessTextureUsage() const;
    void              setColorTexture(MTL::Texture* colorTexture);
    void              setDenoiseStrengthMaskTexture(MTL::Texture* denoiseStrengthMaskTexture);
    void              setDepthReversed(bool depthReversed);
    void              setDepthTexture(MTL::Texture* depthTexture);
    void              setDiffuseAlbedoTexture(MTL::Texture* diffuseAlbedoTexture);
    void              setExposureTexture(MTL::Texture* exposureTexture);
    void              setFence(MTL::Fence* fence);
    void              setJitterOffsetX(float jitterOffsetX);
    void              setJitterOffsetY(float jitterOffsetY);
    void              setMotionTexture(MTL::Texture* motionTexture);
    void              setMotionVectorScaleX(float motionVectorScaleX);
    void              setMotionVectorScaleY(float motionVectorScaleY);
    void              setNormalTexture(MTL::Texture* normalTexture);
    void              setOutputTexture(MTL::Texture* outputTexture);
    void              setPreExposure(float preExposure);
    void              setReactiveMaskTexture(MTL::Texture* reactiveMaskTexture);
    void              setRoughnessTexture(MTL::Texture* roughnessTexture);
    void              setShouldResetHistory(bool shouldResetHistory);
    void              setSpecularAlbedoTexture(MTL::Texture* specularAlbedoTexture);
    void              setSpecularHitDistanceTexture(MTL::Texture* specularHitDistanceTexture);
    void              setTransparencyOverlayTexture(MTL::Texture* transparencyOverlayTexture);
    void              setViewToClipMatrix(void* viewToClipMatrix);
    void              setWorldToViewMatrix(void* worldToViewMatrix);
    bool              shouldResetHistory() const;
    MTL::Texture*     specularAlbedoTexture() const;
    MTL::PixelFormat  specularAlbedoTextureFormat() const;
    MTL::TextureUsage specularAlbedoTextureUsage() const;
    MTL::Texture*     specularHitDistanceTexture() const;
    MTL::PixelFormat  specularHitDistanceTextureFormat() const;
    MTL::TextureUsage specularHitDistanceTextureUsage() const;
    MTL::Texture*     transparencyOverlayTexture() const;
    MTL::PixelFormat  transparencyOverlayTextureFormat() const;
    MTL::TextureUsage transparencyOverlayTextureUsage() const;
    void*             viewToClipMatrix() const;
    void*             worldToViewMatrix() const;

};

class TemporalDenoisedScaler : public NS::Referencing<TemporalDenoisedScaler, MTLFX::TemporalDenoisedScalerBase>
{
public:
    void encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer);

};

} // namespace MTLFX

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFXTemporalDenoisedScalerDescriptor;
extern "C" void *OBJC_CLASS_$_MTLFXTemporalDenoisedScalerBase;
extern "C" void *OBJC_CLASS_$_MTLFXTemporalDenoisedScaler;

_MTLFX_INLINE MTLFX::TemporalDenoisedScalerDescriptor* MTLFX::TemporalDenoisedScalerDescriptor::alloc()
{
    return _MTLFX_msg_MTLFX__TemporalDenoisedScalerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLFXTemporalDenoisedScalerDescriptor, nullptr);
}

_MTLFX_INLINE MTLFX::TemporalDenoisedScalerDescriptor* MTLFX::TemporalDenoisedScalerDescriptor::init() const
{
    return _MTLFX_msg_MTLFX__TemporalDenoisedScalerDescriptorp_init((const void*)this, nullptr);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerDescriptor::supportedInputContentMinScale(MTL::Device* device)
{
    return _MTLFX_msg_float_supportedInputContentMinScaleForDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalDenoisedScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerDescriptor::supportedInputContentMaxScale(MTL::Device* device)
{
    return _MTLFX_msg_float_supportedInputContentMaxScaleForDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalDenoisedScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::supportsMetal4FX(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsMetal4FX__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalDenoisedScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::supportsDevice(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalDenoisedScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setColorTextureFormat(MTL::PixelFormat colorTextureFormat)
{
    _MTLFX_msg_v_setColorTextureFormat__MTL__PixelFormat((const void*)this, nullptr, colorTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::depthTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_depthTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDepthTextureFormat(MTL::PixelFormat depthTextureFormat)
{
    _MTLFX_msg_v_setDepthTextureFormat__MTL__PixelFormat((const void*)this, nullptr, depthTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::motionTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_motionTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setMotionTextureFormat(MTL::PixelFormat motionTextureFormat)
{
    _MTLFX_msg_v_setMotionTextureFormat__MTL__PixelFormat((const void*)this, nullptr, motionTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::diffuseAlbedoTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_diffuseAlbedoTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDiffuseAlbedoTextureFormat(MTL::PixelFormat diffuseAlbedoTextureFormat)
{
    _MTLFX_msg_v_setDiffuseAlbedoTextureFormat__MTL__PixelFormat((const void*)this, nullptr, diffuseAlbedoTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::specularAlbedoTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_specularAlbedoTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setSpecularAlbedoTextureFormat(MTL::PixelFormat specularAlbedoTextureFormat)
{
    _MTLFX_msg_v_setSpecularAlbedoTextureFormat__MTL__PixelFormat((const void*)this, nullptr, specularAlbedoTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::normalTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_normalTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setNormalTextureFormat(MTL::PixelFormat normalTextureFormat)
{
    _MTLFX_msg_v_setNormalTextureFormat__MTL__PixelFormat((const void*)this, nullptr, normalTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::roughnessTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_roughnessTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setRoughnessTextureFormat(MTL::PixelFormat roughnessTextureFormat)
{
    _MTLFX_msg_v_setRoughnessTextureFormat__MTL__PixelFormat((const void*)this, nullptr, roughnessTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::specularHitDistanceTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_specularHitDistanceTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setSpecularHitDistanceTextureFormat(MTL::PixelFormat specularHitDistanceTextureFormat)
{
    _MTLFX_msg_v_setSpecularHitDistanceTextureFormat__MTL__PixelFormat((const void*)this, nullptr, specularHitDistanceTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::denoiseStrengthMaskTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_denoiseStrengthMaskTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDenoiseStrengthMaskTextureFormat(MTL::PixelFormat denoiseStrengthMaskTextureFormat)
{
    _MTLFX_msg_v_setDenoiseStrengthMaskTextureFormat__MTL__PixelFormat((const void*)this, nullptr, denoiseStrengthMaskTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::transparencyOverlayTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_transparencyOverlayTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setTransparencyOverlayTextureFormat(MTL::PixelFormat transparencyOverlayTextureFormat)
{
    _MTLFX_msg_v_setTransparencyOverlayTextureFormat__MTL__PixelFormat((const void*)this, nullptr, transparencyOverlayTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setOutputTextureFormat(MTL::PixelFormat outputTextureFormat)
{
    _MTLFX_msg_v_setOutputTextureFormat__MTL__PixelFormat((const void*)this, nullptr, outputTextureFormat);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputWidth(NS::UInteger inputWidth)
{
    _MTLFX_msg_v_setInputWidth__NS__UInteger((const void*)this, nullptr, inputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setInputHeight(NS::UInteger inputHeight)
{
    _MTLFX_msg_v_setInputHeight__NS__UInteger((const void*)this, nullptr, inputHeight);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setOutputWidth(NS::UInteger outputWidth)
{
    _MTLFX_msg_v_setOutputWidth__NS__UInteger((const void*)this, nullptr, outputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerDescriptor::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setOutputHeight(NS::UInteger outputHeight)
{
    _MTLFX_msg_v_setOutputHeight__NS__UInteger((const void*)this, nullptr, outputHeight);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::requiresSynchronousInitialization() const
{
    return _MTLFX_msg_bool_requiresSynchronousInitialization((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setRequiresSynchronousInitialization(bool requiresSynchronousInitialization)
{
    _MTLFX_msg_v_setRequiresSynchronousInitialization__bool((const void*)this, nullptr, requiresSynchronousInitialization);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::autoExposureEnabled() const
{
    return _MTLFX_msg_bool_autoExposureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setAutoExposureEnabled(bool autoExposureEnabled)
{
    _MTLFX_msg_v_setAutoExposureEnabled__bool((const void*)this, nullptr, autoExposureEnabled);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::reactiveMaskTextureEnabled() const
{
    return _MTLFX_msg_bool_reactiveMaskTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setReactiveMaskTextureEnabled(bool reactiveMaskTextureEnabled)
{
    _MTLFX_msg_v_setReactiveMaskTextureEnabled__bool((const void*)this, nullptr, reactiveMaskTextureEnabled);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerDescriptor::reactiveMaskTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_reactiveMaskTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setReactiveMaskTextureFormat(MTL::PixelFormat reactiveMaskTextureFormat)
{
    _MTLFX_msg_v_setReactiveMaskTextureFormat__MTL__PixelFormat((const void*)this, nullptr, reactiveMaskTextureFormat);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::specularHitDistanceTextureEnabled() const
{
    return _MTLFX_msg_bool_specularHitDistanceTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setSpecularHitDistanceTextureEnabled(bool specularHitDistanceTextureEnabled)
{
    _MTLFX_msg_v_setSpecularHitDistanceTextureEnabled__bool((const void*)this, nullptr, specularHitDistanceTextureEnabled);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::denoiseStrengthMaskTextureEnabled() const
{
    return _MTLFX_msg_bool_denoiseStrengthMaskTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setDenoiseStrengthMaskTextureEnabled(bool denoiseStrengthMaskTextureEnabled)
{
    _MTLFX_msg_v_setDenoiseStrengthMaskTextureEnabled__bool((const void*)this, nullptr, denoiseStrengthMaskTextureEnabled);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::transparencyOverlayTextureEnabled() const
{
    return _MTLFX_msg_bool_transparencyOverlayTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerDescriptor::setTransparencyOverlayTextureEnabled(bool transparencyOverlayTextureEnabled)
{
    _MTLFX_msg_v_setTransparencyOverlayTextureEnabled__bool((const void*)this, nullptr, transparencyOverlayTextureEnabled);
}

_MTLFX_INLINE MTLFX::TemporalDenoisedScaler* MTLFX::TemporalDenoisedScalerDescriptor::newTemporalDenoisedScaler(MTL::Device* device)
{
    return _MTLFX_msg_MTLFX__TemporalDenoisedScalerp_newTemporalDenoisedScalerWithDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTLFX_INLINE MTL4FX::TemporalDenoisedScaler* MTLFX::TemporalDenoisedScalerDescriptor::newTemporalDenoisedScaler(MTL::Device* device, MTL4::Compiler* compiler)
{
    return _MTLFX_msg_MTL4FX__TemporalDenoisedScalerp_newTemporalDenoisedScalerWithDevice_compiler__MTL__Devicep_MTL4__Compilerp((const void*)this, nullptr, device, compiler);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isAutoExposureEnabled()
{
    return _MTLFX_msg_bool_isAutoExposureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isReactiveMaskTextureEnabled()
{
    return _MTLFX_msg_bool_isReactiveMaskTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isSpecularHitDistanceTextureEnabled()
{
    return _MTLFX_msg_bool_isSpecularHitDistanceTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isDenoiseStrengthMaskTextureEnabled()
{
    return _MTLFX_msg_bool_isDenoiseStrengthMaskTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerDescriptor::isTransparencyOverlayTextureEnabled()
{
    return _MTLFX_msg_bool_isTransparencyOverlayTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::colorTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_colorTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::depthTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_depthTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::motionTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_motionTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::reactiveTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_reactiveTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::diffuseAlbedoTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_diffuseAlbedoTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::specularAlbedoTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_specularAlbedoTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::normalTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_normalTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::roughnessTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_roughnessTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::specularHitDistanceTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_specularHitDistanceTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::denoiseStrengthMaskTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_denoiseStrengthMaskTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::transparencyOverlayTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_transparencyOverlayTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalDenoisedScalerBase::outputTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_outputTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::colorTexture() const
{
    return _MTLFX_msg_MTL__Texturep_colorTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setColorTexture(MTL::Texture* colorTexture)
{
    _MTLFX_msg_v_setColorTexture__MTL__Texturep((const void*)this, nullptr, colorTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::depthTexture() const
{
    return _MTLFX_msg_MTL__Texturep_depthTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDepthTexture(MTL::Texture* depthTexture)
{
    _MTLFX_msg_v_setDepthTexture__MTL__Texturep((const void*)this, nullptr, depthTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::motionTexture() const
{
    return _MTLFX_msg_MTL__Texturep_motionTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setMotionTexture(MTL::Texture* motionTexture)
{
    _MTLFX_msg_v_setMotionTexture__MTL__Texturep((const void*)this, nullptr, motionTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::diffuseAlbedoTexture() const
{
    return _MTLFX_msg_MTL__Texturep_diffuseAlbedoTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDiffuseAlbedoTexture(MTL::Texture* diffuseAlbedoTexture)
{
    _MTLFX_msg_v_setDiffuseAlbedoTexture__MTL__Texturep((const void*)this, nullptr, diffuseAlbedoTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::specularAlbedoTexture() const
{
    return _MTLFX_msg_MTL__Texturep_specularAlbedoTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setSpecularAlbedoTexture(MTL::Texture* specularAlbedoTexture)
{
    _MTLFX_msg_v_setSpecularAlbedoTexture__MTL__Texturep((const void*)this, nullptr, specularAlbedoTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::normalTexture() const
{
    return _MTLFX_msg_MTL__Texturep_normalTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setNormalTexture(MTL::Texture* normalTexture)
{
    _MTLFX_msg_v_setNormalTexture__MTL__Texturep((const void*)this, nullptr, normalTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::roughnessTexture() const
{
    return _MTLFX_msg_MTL__Texturep_roughnessTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setRoughnessTexture(MTL::Texture* roughnessTexture)
{
    _MTLFX_msg_v_setRoughnessTexture__MTL__Texturep((const void*)this, nullptr, roughnessTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::specularHitDistanceTexture() const
{
    return _MTLFX_msg_MTL__Texturep_specularHitDistanceTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setSpecularHitDistanceTexture(MTL::Texture* specularHitDistanceTexture)
{
    _MTLFX_msg_v_setSpecularHitDistanceTexture__MTL__Texturep((const void*)this, nullptr, specularHitDistanceTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::denoiseStrengthMaskTexture() const
{
    return _MTLFX_msg_MTL__Texturep_denoiseStrengthMaskTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDenoiseStrengthMaskTexture(MTL::Texture* denoiseStrengthMaskTexture)
{
    _MTLFX_msg_v_setDenoiseStrengthMaskTexture__MTL__Texturep((const void*)this, nullptr, denoiseStrengthMaskTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::transparencyOverlayTexture() const
{
    return _MTLFX_msg_MTL__Texturep_transparencyOverlayTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setTransparencyOverlayTexture(MTL::Texture* transparencyOverlayTexture)
{
    _MTLFX_msg_v_setTransparencyOverlayTexture__MTL__Texturep((const void*)this, nullptr, transparencyOverlayTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::outputTexture() const
{
    return _MTLFX_msg_MTL__Texturep_outputTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setOutputTexture(MTL::Texture* outputTexture)
{
    _MTLFX_msg_v_setOutputTexture__MTL__Texturep((const void*)this, nullptr, outputTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::exposureTexture() const
{
    return _MTLFX_msg_MTL__Texturep_exposureTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setExposureTexture(MTL::Texture* exposureTexture)
{
    _MTLFX_msg_v_setExposureTexture__MTL__Texturep((const void*)this, nullptr, exposureTexture);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::preExposure() const
{
    return _MTLFX_msg_float_preExposure((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setPreExposure(float preExposure)
{
    _MTLFX_msg_v_setPreExposure__float((const void*)this, nullptr, preExposure);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalDenoisedScalerBase::reactiveMaskTexture() const
{
    return _MTLFX_msg_MTL__Texturep_reactiveMaskTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setReactiveMaskTexture(MTL::Texture* reactiveMaskTexture)
{
    _MTLFX_msg_v_setReactiveMaskTexture__MTL__Texturep((const void*)this, nullptr, reactiveMaskTexture);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::jitterOffsetX() const
{
    return _MTLFX_msg_float_jitterOffsetX((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setJitterOffsetX(float jitterOffsetX)
{
    _MTLFX_msg_v_setJitterOffsetX__float((const void*)this, nullptr, jitterOffsetX);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::jitterOffsetY() const
{
    return _MTLFX_msg_float_jitterOffsetY((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setJitterOffsetY(float jitterOffsetY)
{
    _MTLFX_msg_v_setJitterOffsetY__float((const void*)this, nullptr, jitterOffsetY);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::motionVectorScaleX() const
{
    return _MTLFX_msg_float_motionVectorScaleX((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setMotionVectorScaleX(float motionVectorScaleX)
{
    _MTLFX_msg_v_setMotionVectorScaleX__float((const void*)this, nullptr, motionVectorScaleX);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::motionVectorScaleY() const
{
    return _MTLFX_msg_float_motionVectorScaleY((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setMotionVectorScaleY(float motionVectorScaleY)
{
    _MTLFX_msg_v_setMotionVectorScaleY__float((const void*)this, nullptr, motionVectorScaleY);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerBase::shouldResetHistory() const
{
    return _MTLFX_msg_bool_shouldResetHistory((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setShouldResetHistory(bool shouldResetHistory)
{
    _MTLFX_msg_v_setShouldResetHistory__bool((const void*)this, nullptr, shouldResetHistory);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerBase::depthReversed() const
{
    return _MTLFX_msg_bool_depthReversed((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setDepthReversed(bool depthReversed)
{
    _MTLFX_msg_v_setDepthReversed__bool((const void*)this, nullptr, depthReversed);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::depthTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_depthTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::motionTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_motionTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::diffuseAlbedoTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_diffuseAlbedoTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::specularAlbedoTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_specularAlbedoTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::normalTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_normalTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::roughnessTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_roughnessTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::specularHitDistanceTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_specularHitDistanceTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::denoiseStrengthMaskTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_denoiseStrengthMaskTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::transparencyOverlayTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_transparencyOverlayTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::reactiveMaskTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_reactiveMaskTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalDenoisedScalerBase::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalDenoisedScalerBase::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::inputContentMinScale() const
{
    return _MTLFX_msg_float_inputContentMinScale((const void*)this, nullptr);
}

_MTLFX_INLINE float MTLFX::TemporalDenoisedScalerBase::inputContentMaxScale() const
{
    return _MTLFX_msg_float_inputContentMaxScale((const void*)this, nullptr);
}

_MTLFX_INLINE void* MTLFX::TemporalDenoisedScalerBase::worldToViewMatrix() const
{
    return _MTLFX_msg_voidp_worldToViewMatrix((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setWorldToViewMatrix(void* worldToViewMatrix)
{
    _MTLFX_msg_v_setWorldToViewMatrix__voidp((const void*)this, nullptr, worldToViewMatrix);
}

_MTLFX_INLINE void* MTLFX::TemporalDenoisedScalerBase::viewToClipMatrix() const
{
    return _MTLFX_msg_voidp_viewToClipMatrix((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setViewToClipMatrix(void* viewToClipMatrix)
{
    _MTLFX_msg_v_setViewToClipMatrix__voidp((const void*)this, nullptr, viewToClipMatrix);
}

_MTLFX_INLINE MTL::Fence* MTLFX::TemporalDenoisedScalerBase::fence() const
{
    return _MTLFX_msg_MTL__Fencep_fence((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScalerBase::setFence(MTL::Fence* fence)
{
    _MTLFX_msg_v_setFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTLFX_INLINE bool MTLFX::TemporalDenoisedScalerBase::isDepthReversed()
{
    return _MTLFX_msg_bool_isDepthReversed((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalDenoisedScaler::encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer)
{
    _MTLFX_msg_v_encodeToCommandBuffer__MTL__CommandBufferp((const void*)this, nullptr, commandBuffer);
}
