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
    class SpatialScaler;
}

namespace MTLFX
{

_MTLFX_ENUM(NS::Integer, SpatialScalerColorProcessingMode) {
    SpatialScalerColorProcessingModePerceptual = 0,
    SpatialScalerColorProcessingModeLinear = 1,
    SpatialScalerColorProcessingModeHDR = 2,
};


class SpatialScalerDescriptor;
class SpatialScalerBase;
class SpatialScaler;

class SpatialScalerDescriptor : public NS::Copying<SpatialScalerDescriptor>
{
public:
    static SpatialScalerDescriptor* alloc();
    SpatialScalerDescriptor*        init() const;

    static bool supportsDevice(MTL::Device* device);
    static bool supportsMetal4FX(MTL::Device* device);

    MTLFX::SpatialScalerColorProcessingMode colorProcessingMode() const;
    MTL::PixelFormat                        colorTextureFormat() const;
    NS::UInteger                            inputHeight() const;
    NS::UInteger                            inputWidth() const;
    MTLFX::SpatialScaler*                   newSpatialScaler(MTL::Device* device);
    MTL4FX::SpatialScaler*                  newSpatialScaler(MTL::Device* device, MTL4::Compiler* compiler);
    NS::UInteger                            outputHeight() const;
    MTL::PixelFormat                        outputTextureFormat() const;
    NS::UInteger                            outputWidth() const;
    void                                    setColorProcessingMode(MTLFX::SpatialScalerColorProcessingMode colorProcessingMode);
    void                                    setColorTextureFormat(MTL::PixelFormat colorTextureFormat);
    void                                    setInputHeight(NS::UInteger inputHeight);
    void                                    setInputWidth(NS::UInteger inputWidth);
    void                                    setOutputHeight(NS::UInteger outputHeight);
    void                                    setOutputTextureFormat(MTL::PixelFormat outputTextureFormat);
    void                                    setOutputWidth(NS::UInteger outputWidth);

};

class SpatialScalerBase : public NS::Referencing<SpatialScalerBase>
{
public:
    MTLFX::SpatialScalerColorProcessingMode colorProcessingMode() const;
    MTL::Texture*                           colorTexture() const;
    MTL::PixelFormat                        colorTextureFormat() const;
    MTL::TextureUsage                       colorTextureUsage() const;
    MTL::Fence*                             fence() const;
    NS::UInteger                            inputContentHeight() const;
    NS::UInteger                            inputContentWidth() const;
    NS::UInteger                            inputHeight() const;
    NS::UInteger                            inputWidth() const;
    NS::UInteger                            outputHeight() const;
    MTL::Texture*                           outputTexture() const;
    MTL::PixelFormat                        outputTextureFormat() const;
    MTL::TextureUsage                       outputTextureUsage() const;
    NS::UInteger                            outputWidth() const;
    void                                    setColorTexture(MTL::Texture* colorTexture);
    void                                    setFence(MTL::Fence* fence);
    void                                    setInputContentHeight(NS::UInteger inputContentHeight);
    void                                    setInputContentWidth(NS::UInteger inputContentWidth);
    void                                    setOutputTexture(MTL::Texture* outputTexture);

};

class SpatialScaler : public NS::Referencing<SpatialScaler, MTLFX::SpatialScalerBase>
{
public:
    void encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer);

};

} // namespace MTLFX

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFXSpatialScalerDescriptor;
extern "C" void *OBJC_CLASS_$_MTLFXSpatialScalerBase;
extern "C" void *OBJC_CLASS_$_MTLFXSpatialScaler;

_MTLFX_INLINE MTLFX::SpatialScalerDescriptor* MTLFX::SpatialScalerDescriptor::alloc()
{
    return _MTLFX_msg_MTLFX__SpatialScalerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLFXSpatialScalerDescriptor, nullptr);
}

_MTLFX_INLINE MTLFX::SpatialScalerDescriptor* MTLFX::SpatialScalerDescriptor::init() const
{
    return _MTLFX_msg_MTLFX__SpatialScalerDescriptorp_init((const void*)this, nullptr);
}

_MTLFX_INLINE bool MTLFX::SpatialScalerDescriptor::supportsMetal4FX(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsMetal4FX__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXSpatialScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE bool MTLFX::SpatialScalerDescriptor::supportsDevice(MTL::Device* device)
{
    return _MTLFX_msg_bool_supportsDevice__MTL__Devicep((const void*)&OBJC_CLASS_$_MTLFXSpatialScalerDescriptor, nullptr, device);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerDescriptor::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setColorTextureFormat(MTL::PixelFormat colorTextureFormat)
{
    _MTLFX_msg_v_setColorTextureFormat__MTL__PixelFormat((const void*)this, nullptr, colorTextureFormat);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerDescriptor::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setOutputTextureFormat(MTL::PixelFormat outputTextureFormat)
{
    _MTLFX_msg_v_setOutputTextureFormat__MTL__PixelFormat((const void*)this, nullptr, outputTextureFormat);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setInputWidth(NS::UInteger inputWidth)
{
    _MTLFX_msg_v_setInputWidth__NS__UInteger((const void*)this, nullptr, inputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setInputHeight(NS::UInteger inputHeight)
{
    _MTLFX_msg_v_setInputHeight__NS__UInteger((const void*)this, nullptr, inputHeight);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setOutputWidth(NS::UInteger outputWidth)
{
    _MTLFX_msg_v_setOutputWidth__NS__UInteger((const void*)this, nullptr, outputWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerDescriptor::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setOutputHeight(NS::UInteger outputHeight)
{
    _MTLFX_msg_v_setOutputHeight__NS__UInteger((const void*)this, nullptr, outputHeight);
}

_MTLFX_INLINE MTLFX::SpatialScalerColorProcessingMode MTLFX::SpatialScalerDescriptor::colorProcessingMode() const
{
    return _MTLFX_msg_MTLFX__SpatialScalerColorProcessingMode_colorProcessingMode((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerDescriptor::setColorProcessingMode(MTLFX::SpatialScalerColorProcessingMode colorProcessingMode)
{
    _MTLFX_msg_v_setColorProcessingMode__MTLFX__SpatialScalerColorProcessingMode((const void*)this, nullptr, colorProcessingMode);
}

_MTLFX_INLINE MTLFX::SpatialScaler* MTLFX::SpatialScalerDescriptor::newSpatialScaler(MTL::Device* device)
{
    return _MTLFX_msg_MTLFX__SpatialScalerp_newSpatialScalerWithDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTLFX_INLINE MTL4FX::SpatialScaler* MTLFX::SpatialScalerDescriptor::newSpatialScaler(MTL::Device* device, MTL4::Compiler* compiler)
{
    return _MTLFX_msg_MTL4FX__SpatialScalerp_newSpatialScalerWithDevice_compiler__MTL__Devicep_MTL4__Compilerp((const void*)this, nullptr, device, compiler);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::SpatialScalerBase::colorTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_colorTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::TextureUsage MTLFX::SpatialScalerBase::outputTextureUsage() const
{
    return _MTLFX_msg_MTL__TextureUsage_outputTextureUsage((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputContentWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputContentWidth((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setInputContentWidth(NS::UInteger inputContentWidth)
{
    _MTLFX_msg_v_setInputContentWidth__NS__UInteger((const void*)this, nullptr, inputContentWidth);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputContentHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputContentHeight((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setInputContentHeight(NS::UInteger inputContentHeight)
{
    _MTLFX_msg_v_setInputContentHeight__NS__UInteger((const void*)this, nullptr, inputContentHeight);
}

_MTLFX_INLINE MTL::Texture* MTLFX::SpatialScalerBase::colorTexture() const
{
    return _MTLFX_msg_MTL__Texturep_colorTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setColorTexture(MTL::Texture* colorTexture)
{
    _MTLFX_msg_v_setColorTexture__MTL__Texturep((const void*)this, nullptr, colorTexture);
}

_MTLFX_INLINE MTL::Texture* MTLFX::SpatialScalerBase::outputTexture() const
{
    return _MTLFX_msg_MTL__Texturep_outputTexture((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setOutputTexture(MTL::Texture* outputTexture)
{
    _MTLFX_msg_v_setOutputTexture__MTL__Texturep((const void*)this, nullptr, outputTexture);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerBase::colorTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_colorTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::PixelFormat MTLFX::SpatialScalerBase::outputTextureFormat() const
{
    return _MTLFX_msg_MTL__PixelFormat_outputTextureFormat((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputWidth() const
{
    return _MTLFX_msg_NS__UInteger_inputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::inputHeight() const
{
    return _MTLFX_msg_NS__UInteger_inputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::outputWidth() const
{
    return _MTLFX_msg_NS__UInteger_outputWidth((const void*)this, nullptr);
}

_MTLFX_INLINE NS::UInteger MTLFX::SpatialScalerBase::outputHeight() const
{
    return _MTLFX_msg_NS__UInteger_outputHeight((const void*)this, nullptr);
}

_MTLFX_INLINE MTLFX::SpatialScalerColorProcessingMode MTLFX::SpatialScalerBase::colorProcessingMode() const
{
    return _MTLFX_msg_MTLFX__SpatialScalerColorProcessingMode_colorProcessingMode((const void*)this, nullptr);
}

_MTLFX_INLINE MTL::Fence* MTLFX::SpatialScalerBase::fence() const
{
    return _MTLFX_msg_MTL__Fencep_fence((const void*)this, nullptr);
}

_MTLFX_INLINE void MTLFX::SpatialScalerBase::setFence(MTL::Fence* fence)
{
    _MTLFX_msg_v_setFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTLFX_INLINE void MTLFX::SpatialScaler::encodeToCommandBuffer(MTL::CommandBuffer* commandBuffer)
{
    _MTLFX_msg_v_encodeToCommandBuffer__MTL__CommandBufferp((const void*)this, nullptr, commandBuffer);
}
