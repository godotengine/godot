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
    class FrameInterpolator;
}
namespace NS {
    class Object;
}

namespace MTLFX
{

class FrameInterpolatorDescriptor;
class FrameInterpolatorBase;
class FrameInterpolator;

class FrameInterpolatorDescriptor : public NS::Copying<FrameInterpolatorDescriptor>
{
public:
    static FrameInterpolatorDescriptor* alloc();
    FrameInterpolatorDescriptor*        init() const;

    static bool supportsDevice(MTL::Device* device);
    static bool supportsMetal4FX(MTL::Device* device);

    MTL::PixelFormat           colorTextureFormat() const;
    MTL::PixelFormat           depthTextureFormat() const;
    NS::UInteger               inputHeight() const;
    NS::UInteger               inputWidth() const;
    MTL::PixelFormat           motionTextureFormat() const;
    MTLFX::FrameInterpolator*  newFrameInterpolator(MTL::Device* device);
    MTL4FX::FrameInterpolator* newFrameInterpolator(MTL::Device* device, MTL4::Compiler* compiler);
    NS::UInteger               outputHeight() const;
    MTL::PixelFormat           outputTextureFormat() const;
    NS::UInteger               outputWidth() const;
    NS::Object*                scaler() const;
    void                       setColorTextureFormat(MTL::PixelFormat colorTextureFormat);
    void                       setDepthTextureFormat(MTL::PixelFormat depthTextureFormat);
    void                       setInputHeight(NS::UInteger inputHeight);
    void                       setInputWidth(NS::UInteger inputWidth);
    void                       setMotionTextureFormat(MTL::PixelFormat motionTextureFormat);
    void                       setOutputHeight(NS::UInteger outputHeight);
    void                       setOutputTextureFormat(MTL::PixelFormat outputTextureFormat);
    void                       setOutputWidth(NS::UInteger outputWidth);
    void                       setScaler(NS::Object* scaler);
    void                       setUITextureFormat(MTL::PixelFormat uiTextureFormat);
    void                       setUiTextureFormat(MTL::PixelFormat uiTextureFormat);
    MTL::PixelFormat           uiTextureFormat() const;

};

class FrameInterpolatorBase : public NS::Referencing<FrameInterpolatorBase>
{
public:
    float             aspectRatio() const;
    MTL::Texture*     colorTexture() const;
    MTL::PixelFormat  colorTextureFormat() const;
    MTL::TextureUsage colorTextureUsage() const;
    float             deltaTime() const;
    bool              depthReversed() const;
    MTL::Texture*     depthTexture() const;
    MTL::PixelFormat  depthTextureFormat() const;
    MTL::TextureUsage depthTextureUsage() const;
    float             farPlane() const;
    MTL::Fence*       fence() const;
    float             fieldOfView() const;
    NS::UInteger      inputHeight() const;
    NS::UInteger      inputWidth() const;
    bool              isDepthReversed();
    bool              isUITextureComposited();
    float             jitterOffsetX() const;
    float             jitterOffsetY() const;
    MTL::Texture*     motionTexture() const;
    MTL::PixelFormat  motionTextureFormat() const;
    MTL::TextureUsage motionTextureUsage() const;
    float             motionVectorScaleX() const;
    float             motionVectorScaleY() const;
    float             nearPlane() const;
    NS::UInteger      outputHeight() const;
    MTL::Texture*     outputTexture() const;
    MTL::PixelFormat  outputTextureFormat() const;
    MTL::TextureUsage outputTextureUsage() const;
    NS::UInteger      outputWidth() const;
    MTL::Texture*     prevColorTexture() const;
    void              setAspectRatio(float aspectRatio);
    void              setColorTexture(MTL::Texture* colorTexture);
    void              setDeltaTime(float deltaTime);
    void              setDepthReversed(bool depthReversed);
    void              setDepthTexture(MTL::Texture* depthTexture);
    void              setFarPlane(float farPlane);
    void              setFence(MTL::Fence* fence);
    void              setFieldOfView(float fieldOfView);
    void              setIsUITextureComposited(bool uiTextureComposited);
    void              setJitterOffsetX(float jitterOffsetX);
    void              setJitterOffsetY(float jitterOffsetY);
    void              setMotionTexture(MTL::Texture* motionTexture);
    void              setMotionVectorScaleX(float motionVectorScaleX);
    void              setMotionVectorScaleY(float motionVectorScaleY);
    void              setNearPlane(float nearPlane);
    void              setOutputTexture(MTL::Texture* outputTexture);
    void              setPrevColorTexture(MTL::Texture* prevColorTexture);
    void              setShouldResetHistory(bool shouldResetHistory);
    void              setUITexture(MTL::Texture* uiTexture);
    void              setUiTexture(MTL::Texture* uiTexture);
    void              setUiTextureComposited(bool uiTextureComposited);
    bool              shouldResetHistory() const;
    MTL::Texture*     uiTexture() const;
    bool              uiTextureComposited() const;
    MTL::PixelFormat  uiTextureFormat() const;
    MTL::TextureUsage uiTextureUsage() const;

};

class FrameInterpolator : public NS::Referencing<FrameInterpolator, MTLFX::FrameInterpolatorBase>
{
public:
    void encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer);

};

} // namespace MTLFX

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFXFrameInterpolatorDescriptor;
extern "C" void *OBJC_CLASS_$_MTLFXFrameInterpolatorBase;
extern "C" void *OBJC_CLASS_$_MTLFXFrameInterpolator;

_MTLFX_INLINE MTLFX::FrameInterpolatorDescriptor* MTLFX::FrameInterpolatorDescriptor::alloc()
{
    return _MTLFX_msg_MTLFX__FrameInterpolatorDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLFXFrameInterpolatorDescriptor, nullptr);
}

_MTLFX_INLINE MTLFX::FrameInterpolatorDescriptor* MTLFX::FrameInterpolatorDescriptor::init() const
{
    return _MTLFX_msg_MTLFX__FrameInterpolatorDescriptorp_init((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorDescriptor::supportsMetal4FX(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsMetal4FX__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXFrameInterpolatorDescriptor, nullptr, device);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorDescriptor::supportsDevice(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXFrameInterpolatorDescriptor, nullptr, device);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setColorTextureFormat(MTL::PixelFormat colorTextureFormat)
{
    _MTLFX_msg_v_setColorTextureFormat__MTL__PixelFormat((const void*)this, nullptr, colorTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setOutputTextureFormat(MTL::PixelFormat outputTextureFormat)
{
    _MTLFX_msg_v_setOutputTextureFormat__MTL__PixelFormat((const void*)this, nullptr, outputTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::depthTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_depthTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setDepthTextureFormat(MTL::PixelFormat depthTextureFormat)
{
    _MTLFX_msg_v_setDepthTextureFormat__MTL__PixelFormat((const void*)this, nullptr, depthTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::motionTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_motionTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setMotionTextureFormat(MTL::PixelFormat motionTextureFormat)
{
    _MTLFX_msg_v_setMotionTextureFormat__MTL__PixelFormat((const void*)this, nullptr, motionTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorDescriptor::uiTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_uiTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setUiTextureFormat(MTL::PixelFormat uiTextureFormat)
{
    _MTLFX_msg_v_setUiTextureFormat__MTL__PixelFormat((const void*)this, nullptr, uiTextureFormat);
}

_MTLFX_INLINE NS::Object* MTLFX::FrameInterpolatorDescriptor::scaler() const
{
    return _MTLFX_msg_NS__Objectp_scaler((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setScaler(NS::Object* scaler)
{
    _MTLFX_msg_v_setScaler__NS__Objectp((const void*)this, nullptr, scaler);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setInputWidth(NS::UInteger inputWidth)
{
    _MTLFX_msg_v_setInputWidth__NS__UInteger((const void*)this, nullptr, inputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setInputHeight(NS::UInteger inputHeight)
{
    _MTLFX_msg_v_setInputHeight__NS__UInteger((const void*)this, nullptr, inputHeight);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setOutputWidth(NS::UInteger outputWidth)
{
    _MTLFX_msg_v_setOutputWidth__NS__UInteger((const void*)this, nullptr, outputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorDescriptor::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setOutputHeight(NS::UInteger outputHeight)
{
    _MTLFX_msg_v_setOutputHeight__NS__UInteger((const void*)this, nullptr, outputHeight);
}

_MTLFX_INLINE MTLFX::FrameInterpolator* MTLFX::FrameInterpolatorDescriptor::newFrameInterpolator(MTL::Device* device)
{
    return _MTLFX_msg_MTLFX__FrameInterpolatorp_newFrameInterpolatorWithDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTLFX_INLINE MTL4FX::FrameInterpolator* MTLFX::FrameInterpolatorDescriptor::newFrameInterpolator(MTL::Device* device, MTL4::Compiler* compiler)
{
    return _MTLFX_msg_MTL4FX__FrameInterpolatorp_newFrameInterpolatorWithDevice_compiler__MTL__Devicep_MTL4__Compilerp((const void*)this, nullptr, device, compiler);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorDescriptor::setUITextureFormat(MTL::PixelFormat uiTextureFormat)
{
    _MTLFX_msg_v_setUITextureFormat__MTL__PixelFormat((const void*)this, nullptr, uiTextureFormat);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::colorTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_colorTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::outputTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_outputTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::depthTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_depthTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::motionTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_motionTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::FrameInterpolatorBase::uiTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_uiTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::depthTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_depthTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::motionTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_motionTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::FrameInterpolatorBase::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::FrameInterpolatorBase::uiTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_uiTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::colorTexture() const
{
    return _MTLFX_msg_MTL__Texturep_colorTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setColorTexture(MTL::Texture* colorTexture)
{
    _MTLFX_msg_v_setColorTexture__MTL__Texturep((const void*)this, nullptr, colorTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::prevColorTexture() const
{
    return _MTLFX_msg_MTL__Texturep_prevColorTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setPrevColorTexture(MTL::Texture* prevColorTexture)
{
    _MTLFX_msg_v_setPrevColorTexture__MTL__Texturep((const void*)this, nullptr, prevColorTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::depthTexture() const
{
    return _MTLFX_msg_MTL__Texturep_depthTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setDepthTexture(MTL::Texture* depthTexture)
{
    _MTLFX_msg_v_setDepthTexture__MTL__Texturep((const void*)this, nullptr, depthTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::motionTexture() const
{
    return _MTLFX_msg_MTL__Texturep_motionTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setMotionTexture(MTL::Texture* motionTexture)
{
    _MTLFX_msg_v_setMotionTexture__MTL__Texturep((const void*)this, nullptr, motionTexture);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::motionVectorScaleX() const
{
    return _MTLFX_msg_float_motionVectorScaleX((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setMotionVectorScaleX(float motionVectorScaleX)
{
    _MTLFX_msg_v_setMotionVectorScaleX__float((const void*)this, nullptr, motionVectorScaleX);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::motionVectorScaleY() const
{
    return _MTLFX_msg_float_motionVectorScaleY((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setMotionVectorScaleY(float motionVectorScaleY)
{
    _MTLFX_msg_v_setMotionVectorScaleY__float((const void*)this, nullptr, motionVectorScaleY);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::deltaTime() const
{
    return _MTLFX_msg_float_deltaTime((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setDeltaTime(float deltaTime)
{
    _MTLFX_msg_v_setDeltaTime__float((const void*)this, nullptr, deltaTime);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::nearPlane() const
{
    return _MTLFX_msg_float_nearPlane((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setNearPlane(float nearPlane)
{
    _MTLFX_msg_v_setNearPlane__float((const void*)this, nullptr, nearPlane);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::farPlane() const
{
    return _MTLFX_msg_float_farPlane((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setFarPlane(float farPlane)
{
    _MTLFX_msg_v_setFarPlane__float((const void*)this, nullptr, farPlane);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::fieldOfView() const
{
    return _MTLFX_msg_float_fieldOfView((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setFieldOfView(float fieldOfView)
{
    _MTLFX_msg_v_setFieldOfView__float((const void*)this, nullptr, fieldOfView);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::aspectRatio() const
{
    return _MTLFX_msg_float_aspectRatio((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setAspectRatio(float aspectRatio)
{
    _MTLFX_msg_v_setAspectRatio__float((const void*)this, nullptr, aspectRatio);
}

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::uiTexture() const
{
    return _MTLFX_msg_MTL__Texturep_uiTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setUiTexture(MTL::Texture* uiTexture)
{
    _MTLFX_msg_v_setUiTexture__MTL__Texturep((const void*)this, nullptr, uiTexture);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::jitterOffsetX() const
{
    return _MTLFX_msg_float_jitterOffsetX((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setJitterOffsetX(float jitterOffsetX)
{
    _MTLFX_msg_v_setJitterOffsetX__float((const void*)this, nullptr, jitterOffsetX);
}

_MTLFX_INLINE float MTLFX::FrameInterpolatorBase::jitterOffsetY() const
{
    return _MTLFX_msg_float_jitterOffsetY((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setJitterOffsetY(float jitterOffsetY)
{
    _MTLFX_msg_v_setJitterOffsetY__float((const void*)this, nullptr, jitterOffsetY);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::uiTextureComposited() const
{
    return _MTLFX_msg_bool_uiTextureComposited((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setUiTextureComposited(bool uiTextureComposited)
{
    _MTLFX_msg_v_setUiTextureComposited__bool((const void*)this, nullptr, uiTextureComposited);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::shouldResetHistory() const
{
    return _MTLFX_msg_bool_shouldResetHistory((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setShouldResetHistory(bool shouldResetHistory)
{
    _MTLFX_msg_v_setShouldResetHistory__bool((const void*)this, nullptr, shouldResetHistory);
}

_MTLFX_INLINE MTL::Texture* MTLFX::FrameInterpolatorBase::outputTexture() const
{
    return _MTLFX_msg_MTL__Texturep_outputTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setOutputTexture(MTL::Texture* outputTexture)
{
    _MTLFX_msg_v_setOutputTexture__MTL__Texturep((const void*)this, nullptr, outputTexture);
}

_MTLFX_INLINE MTL::Fence* MTLFX::FrameInterpolatorBase::fence() const
{
    return _MTLFX_msg_MTL__Fencep_fence((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setFence(MTL::Fence* fence)
{
    _MTLFX_msg_v_setFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::depthReversed() const
{
    return _MTLFX_msg_bool_depthReversed((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setDepthReversed(bool depthReversed)
{
    _MTLFX_msg_v_setDepthReversed__bool((const void*)this, nullptr, depthReversed);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setUITexture(MTL::Texture* uiTexture)
{
    _MTLFX_msg_v_setUITexture__MTL__Texturep((const void*)this, nullptr, uiTexture);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::isUITextureComposited()
{
    return _MTLFX_msg_bool_isUITextureComposited((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolatorBase::setIsUITextureComposited(bool uiTextureComposited)
{
    _MTLFX_msg_v_setIsUITextureComposited__bool((const void*)this, nullptr, uiTextureComposited);
}

_MTLFX_INLINE bool MTLFX::FrameInterpolatorBase::isDepthReversed()
{
    return _MTLFX_msg_bool_isDepthReversed((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::FrameInterpolator::encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer)
{
    _MTLFX_msg_v_encodeToCommandBuffer__MTL__CommandBufferp((const void*)this, nullptr, commandBuffer);
}
