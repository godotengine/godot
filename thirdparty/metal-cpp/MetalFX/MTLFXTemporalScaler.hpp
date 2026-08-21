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
    class TemporalScaler;
}

namespace MTLFX
{

class TemporalScalerDescriptor;
class FrameInterpolatableScaler;
class TemporalScalerBase;
class TemporalScaler;

class TemporalScalerDescriptor : public NS::Copying<TemporalScalerDescriptor>
{
public:
    static TemporalScalerDescriptor* alloc();
    TemporalScalerDescriptor*        init() const;

    static float supportedInputContentMaxScale(MTL::Device* device);
    static float supportedInputContentMinScale(MTL::Device* device);
    static bool  supportsDevice(MTL::Device* device);
    static bool  supportsMetal4FX(MTL::Device* device);

    bool                    autoExposureEnabled() const;
    MTL::PixelFormat        colorTextureFormat() const;
    MTL::PixelFormat        depthTextureFormat() const;
    float                   inputContentMaxScale() const;
    float                   inputContentMinScale() const;
    bool                    inputContentPropertiesEnabled() const;
    NS::UInteger            inputHeight() const;
    NS::UInteger            inputWidth() const;
    bool                    isAutoExposureEnabled();
    bool                    isInputContentPropertiesEnabled();
    bool                    isReactiveMaskTextureEnabled();
    MTL::PixelFormat        motionTextureFormat() const;
    MTLFX::TemporalScaler*  newTemporalScaler(MTL::Device* device);
    MTL4FX::TemporalScaler* newTemporalScaler(MTL::Device* device, MTL4::Compiler* compiler);
    NS::UInteger            outputHeight() const;
    MTL::PixelFormat        outputTextureFormat() const;
    NS::UInteger            outputWidth() const;
    bool                    reactiveMaskTextureEnabled() const;
    MTL::PixelFormat        reactiveMaskTextureFormat() const;
    bool                    requiresSynchronousInitialization() const;
    void                    setAutoExposureEnabled(bool autoExposureEnabled);
    void                    setColorTextureFormat(MTL::PixelFormat colorTextureFormat);
    void                    setDepthTextureFormat(MTL::PixelFormat depthTextureFormat);
    void                    setInputContentMaxScale(float inputContentMaxScale);
    void                    setInputContentMinScale(float inputContentMinScale);
    void                    setInputContentPropertiesEnabled(bool inputContentPropertiesEnabled);
    void                    setInputHeight(NS::UInteger inputHeight);
    void                    setInputWidth(NS::UInteger inputWidth);
    void                    setMotionTextureFormat(MTL::PixelFormat motionTextureFormat);
    void                    setOutputHeight(NS::UInteger outputHeight);
    void                    setOutputTextureFormat(MTL::PixelFormat outputTextureFormat);
    void                    setOutputWidth(NS::UInteger outputWidth);
    void                    setReactiveMaskTextureEnabled(bool reactiveMaskTextureEnabled);
    void                    setReactiveMaskTextureFormat(MTL::PixelFormat reactiveMaskTextureFormat);
    void                    setRequiresSynchronousInitialization(bool requiresSynchronousInitialization);

};

class FrameInterpolatableScaler : public NS::Referencing<FrameInterpolatableScaler>
{
public:
};

class TemporalScalerBase : public NS::Referencing<TemporalScalerBase, MTLFX::FrameInterpolatableScaler>
{
public:
    MTL::Texture*     colorTexture() const;
    MTL::PixelFormat  colorTextureFormat() const;
    MTL::TextureUsage colorTextureUsage() const;
    bool              depthReversed() const;
    MTL::Texture*     depthTexture() const;
    MTL::PixelFormat  depthTextureFormat() const;
    MTL::TextureUsage depthTextureUsage() const;
    MTL::Texture*     exposureTexture() const;
    MTL::Fence*       fence() const;
    NS::UInteger      inputContentHeight() const;
    float             inputContentMaxScale() const;
    float             inputContentMinScale() const;
    NS::UInteger      inputContentWidth() const;
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
    NS::UInteger      outputHeight() const;
    MTL::Texture*     outputTexture() const;
    MTL::PixelFormat  outputTextureFormat() const;
    MTL::TextureUsage outputTextureUsage() const;
    NS::UInteger      outputWidth() const;
    float             preExposure() const;
    MTL::Texture*     reactiveMaskTexture() const;
    MTL::PixelFormat  reactiveMaskTextureFormat() const;
    MTL::TextureUsage reactiveTextureUsage() const;
    bool              reset() const;
    void              setColorTexture(MTL::Texture* colorTexture);
    void              setDepthReversed(bool depthReversed);
    void              setDepthTexture(MTL::Texture* depthTexture);
    void              setExposureTexture(MTL::Texture* exposureTexture);
    void              setFence(MTL::Fence* fence);
    void              setInputContentHeight(NS::UInteger inputContentHeight);
    void              setInputContentWidth(NS::UInteger inputContentWidth);
    void              setJitterOffsetX(float jitterOffsetX);
    void              setJitterOffsetY(float jitterOffsetY);
    void              setMotionTexture(MTL::Texture* motionTexture);
    void              setMotionVectorScaleX(float motionVectorScaleX);
    void              setMotionVectorScaleY(float motionVectorScaleY);
    void              setOutputTexture(MTL::Texture* outputTexture);
    void              setPreExposure(float preExposure);
    void              setReactiveMaskTexture(MTL::Texture* reactiveMaskTexture);
    void              setReset(bool reset);

};

class TemporalScaler : public NS::Referencing<TemporalScaler, MTLFX::TemporalScalerBase>
{
public:
    void encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer);

};

} // namespace MTLFX

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFXTemporalScalerDescriptor;
extern "C" void *OBJC_CLASS_$_MTLFXFrameInterpolatableScaler;
extern "C" void *OBJC_CLASS_$_MTLFXTemporalScalerBase;
extern "C" void *OBJC_CLASS_$_MTLFXTemporalScaler;

_MTLFX_INLINE MTLFX::TemporalScalerDescriptor* MTLFX::TemporalScalerDescriptor::alloc()
{
    return _MTLFX_msg_MTLFX__TemporalScalerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLFXTemporalScalerDescriptor, nullptr);
}

_MTLFX_INLINE MTLFX::TemporalScalerDescriptor* MTLFX::TemporalScalerDescriptor::init() const
{
    return _MTLFX_msg_MTLFX__TemporalScalerDescriptorp_init((const void*)this, nullptr);
}

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::supportedInputContentMinScale(MTL::Device* device)
{
    return _MTLFX_msg_float_supportedInputContentMinScaleForDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::supportedInputContentMaxScale(MTL::Device* device)
{
    return _MTLFX_msg_float_supportedInputContentMaxScaleForDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::supportsDevice(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::supportsMetal4FX(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsMetal4FX__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXTemporalScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setColorTextureFormat(MTL::PixelFormat colorTextureFormat)
{
    _MTLFX_msg_v_setColorTextureFormat__MTL__PixelFormat((const void*)this, nullptr, colorTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::depthTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_depthTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setDepthTextureFormat(MTL::PixelFormat depthTextureFormat)
{
    _MTLFX_msg_v_setDepthTextureFormat__MTL__PixelFormat((const void*)this, nullptr, depthTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::motionTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_motionTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setMotionTextureFormat(MTL::PixelFormat motionTextureFormat)
{
    _MTLFX_msg_v_setMotionTextureFormat__MTL__PixelFormat((const void*)this, nullptr, motionTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setOutputTextureFormat(MTL::PixelFormat outputTextureFormat)
{
    _MTLFX_msg_v_setOutputTextureFormat__MTL__PixelFormat((const void*)this, nullptr, outputTextureFormat);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputWidth(NS::UInteger inputWidth)
{
    _MTLFX_msg_v_setInputWidth__NS__UInteger((const void*)this, nullptr, inputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputHeight(NS::UInteger inputHeight)
{
    _MTLFX_msg_v_setInputHeight__NS__UInteger((const void*)this, nullptr, inputHeight);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setOutputWidth(NS::UInteger outputWidth)
{
    _MTLFX_msg_v_setOutputWidth__NS__UInteger((const void*)this, nullptr, outputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerDescriptor::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setOutputHeight(NS::UInteger outputHeight)
{
    _MTLFX_msg_v_setOutputHeight__NS__UInteger((const void*)this, nullptr, outputHeight);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::autoExposureEnabled() const
{
    return _MTLFX_msg_bool_autoExposureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setAutoExposureEnabled(bool autoExposureEnabled)
{
    _MTLFX_msg_v_setAutoExposureEnabled__bool((const void*)this, nullptr, autoExposureEnabled);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::requiresSynchronousInitialization() const
{
    return _MTLFX_msg_bool_requiresSynchronousInitialization((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setRequiresSynchronousInitialization(bool requiresSynchronousInitialization)
{
    _MTLFX_msg_v_setRequiresSynchronousInitialization__bool((const void*)this, nullptr, requiresSynchronousInitialization);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::inputContentPropertiesEnabled() const
{
    return _MTLFX_msg_bool_inputContentPropertiesEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputContentPropertiesEnabled(bool inputContentPropertiesEnabled)
{
    _MTLFX_msg_v_setInputContentPropertiesEnabled__bool((const void*)this, nullptr, inputContentPropertiesEnabled);
}

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::inputContentMinScale() const
{
    return _MTLFX_msg_float_inputContentMinScale((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputContentMinScale(float inputContentMinScale)
{
    _MTLFX_msg_v_setInputContentMinScale__float((const void*)this, nullptr, inputContentMinScale);
}

_MTLFX_INLINE float MTLFX::TemporalScalerDescriptor::inputContentMaxScale() const
{
    return _MTLFX_msg_float_inputContentMaxScale((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setInputContentMaxScale(float inputContentMaxScale)
{
    _MTLFX_msg_v_setInputContentMaxScale__float((const void*)this, nullptr, inputContentMaxScale);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::reactiveMaskTextureEnabled() const
{
    return _MTLFX_msg_bool_reactiveMaskTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setReactiveMaskTextureEnabled(bool reactiveMaskTextureEnabled)
{
    _MTLFX_msg_v_setReactiveMaskTextureEnabled__bool((const void*)this, nullptr, reactiveMaskTextureEnabled);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerDescriptor::reactiveMaskTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_reactiveMaskTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerDescriptor::setReactiveMaskTextureFormat(MTL::PixelFormat reactiveMaskTextureFormat)
{
    _MTLFX_msg_v_setReactiveMaskTextureFormat__MTL__PixelFormat((const void*)this, nullptr, reactiveMaskTextureFormat);
}

_MTLFX_INLINE MTLFX::TemporalScaler* MTLFX::TemporalScalerDescriptor::newTemporalScaler(MTL::Device* device)
{
    return _MTLFX_msg_MTLFX__TemporalScalerp_newTemporalScalerWithDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTLFX_INLINE MTL4FX::TemporalScaler* MTLFX::TemporalScalerDescriptor::newTemporalScaler(MTL::Device* device, MTL4::Compiler* compiler)
{
    return _MTLFX_msg_MTL4FX__TemporalScalerp_newTemporalScalerWithDevice_compiler__MTL__Devicep_MTL4__Compilerp((const void*)this, nullptr, device, compiler);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::isAutoExposureEnabled()
{
    return _MTLFX_msg_bool_isAutoExposureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::isInputContentPropertiesEnabled()
{
    return _MTLFX_msg_bool_isInputContentPropertiesEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerDescriptor::isReactiveMaskTextureEnabled()
{
    return _MTLFX_msg_bool_isReactiveMaskTextureEnabled((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::colorTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_colorTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::depthTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_depthTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::motionTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_motionTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::reactiveTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_reactiveTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::TemporalScalerBase::outputTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_outputTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputContentWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputContentWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setInputContentWidth(NS::UInteger inputContentWidth)
{
    _MTLFX_msg_v_setInputContentWidth__NS__UInteger((const void*)this, nullptr, inputContentWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputContentHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputContentHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setInputContentHeight(NS::UInteger inputContentHeight)
{
    _MTLFX_msg_v_setInputContentHeight__NS__UInteger((const void*)this, nullptr, inputContentHeight);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::colorTexture() const
{
    return _MTLFX_msg_MTL__Texturep_colorTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setColorTexture(MTL::Texture* colorTexture)
{
    _MTLFX_msg_v_setColorTexture__MTL__Texturep((const void*)this, nullptr, colorTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::depthTexture() const
{
    return _MTLFX_msg_MTL__Texturep_depthTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setDepthTexture(MTL::Texture* depthTexture)
{
    _MTLFX_msg_v_setDepthTexture__MTL__Texturep((const void*)this, nullptr, depthTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::motionTexture() const
{
    return _MTLFX_msg_MTL__Texturep_motionTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setMotionTexture(MTL::Texture* motionTexture)
{
    _MTLFX_msg_v_setMotionTexture__MTL__Texturep((const void*)this, nullptr, motionTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::outputTexture() const
{
    return _MTLFX_msg_MTL__Texturep_outputTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setOutputTexture(MTL::Texture* outputTexture)
{
    _MTLFX_msg_v_setOutputTexture__MTL__Texturep((const void*)this, nullptr, outputTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::exposureTexture() const
{
    return _MTLFX_msg_MTL__Texturep_exposureTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setExposureTexture(MTL::Texture* exposureTexture)
{
    _MTLFX_msg_v_setExposureTexture__MTL__Texturep((const void*)this, nullptr, exposureTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::TemporalScalerBase::reactiveMaskTexture() const
{
    return _MTLFX_msg_MTL__Texturep_reactiveMaskTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setReactiveMaskTexture(MTL::Texture* reactiveMaskTexture)
{
    _MTLFX_msg_v_setReactiveMaskTexture__MTL__Texturep((const void*)this, nullptr, reactiveMaskTexture);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::preExposure() const
{
    return _MTLFX_msg_float_preExposure((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setPreExposure(float preExposure)
{
    _MTLFX_msg_v_setPreExposure__float((const void*)this, nullptr, preExposure);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::jitterOffsetX() const
{
    return _MTLFX_msg_float_jitterOffsetX((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setJitterOffsetX(float jitterOffsetX)
{
    _MTLFX_msg_v_setJitterOffsetX__float((const void*)this, nullptr, jitterOffsetX);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::jitterOffsetY() const
{
    return _MTLFX_msg_float_jitterOffsetY((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setJitterOffsetY(float jitterOffsetY)
{
    _MTLFX_msg_v_setJitterOffsetY__float((const void*)this, nullptr, jitterOffsetY);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::motionVectorScaleX() const
{
    return _MTLFX_msg_float_motionVectorScaleX((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setMotionVectorScaleX(float motionVectorScaleX)
{
    _MTLFX_msg_v_setMotionVectorScaleX__float((const void*)this, nullptr, motionVectorScaleX);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::motionVectorScaleY() const
{
    return _MTLFX_msg_float_motionVectorScaleY((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setMotionVectorScaleY(float motionVectorScaleY)
{
    _MTLFX_msg_v_setMotionVectorScaleY__float((const void*)this, nullptr, motionVectorScaleY);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerBase::reset() const
{
    return _MTLFX_msg_bool_reset((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setReset(bool reset)
{
    _MTLFX_msg_v_setReset__bool((const void*)this, nullptr, reset);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerBase::depthReversed() const
{
    return _MTLFX_msg_bool_depthReversed((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setDepthReversed(bool depthReversed)
{
    _MTLFX_msg_v_setDepthReversed__bool((const void*)this, nullptr, depthReversed);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::depthTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_depthTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::motionTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_motionTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::reactiveMaskTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_reactiveMaskTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::TemporalScalerBase::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::TemporalScalerBase::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::inputContentMinScale() const
{
    return _MTLFX_msg_float_inputContentMinScale((const void*)this, nullptr);
}

_MTLFX_INLINE float MTLFX::TemporalScalerBase::inputContentMaxScale() const
{
    return _MTLFX_msg_float_inputContentMaxScale((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::Fence* MTLFX::TemporalScalerBase::fence() const
{
    return _MTLFX_msg_MTL__Fencep_fence((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScalerBase::setFence(MTL::Fence* fence)
{
    _MTLFX_msg_v_setFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTLFX_INLINE bool MTLFX::TemporalScalerBase::isDepthReversed()
{
    return _MTLFX_msg_bool_isDepthReversed((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::TemporalScaler::encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer)
{
    _MTLFX_msg_v_encodeToCommandBuffer__MTL__CommandBufferp((const void*)this, nullptr, commandBuffer);
}
