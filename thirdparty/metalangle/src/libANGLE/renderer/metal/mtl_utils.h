//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_utils.h:
//    Declares utilities functions that create Metal shaders, convert from angle enums
//    to Metal enums and so on.
//

#ifndef LIBANGLE_RENDERER_METAL_MTL_UTILS_H_
#define LIBANGLE_RENDERER_METAL_MTL_UTILS_H_

#import <Metal/Metal.h>

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "libANGLE/Context.h"
#include "libANGLE/Texture.h"
#include "libANGLE/renderer/metal/mtl_format_utils.h"
#include "libANGLE/renderer/metal/mtl_resources.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"

namespace rx
{
namespace mtl
{

NS_ASSUME_NONNULL_BEGIN

angle::Result InitializeTextureContents(const gl::Context *context,
                                        const TextureRef &texture,
                                        const Format &textureObjFormat,
                                        const gl::ImageIndex &index);
// Same as above but using GPU clear operation instead of CPU.
// - channelsToInit parameter controls which channels will get their content initialized.
angle::Result InitializeTextureContentsGPU(const gl::Context *context,
                                           const TextureRef &texture,
                                           const Format &textureObjFormat,
                                           const gl::ImageIndex &index,
                                           MTLColorWriteMask channelsToInit);
// Same as above but for a depth/stencil texture.
angle::Result InitializeDepthStencilTextureContentsGPU(const gl::Context *context,
                                                       const TextureRef &texture,
                                                       const Format &textureObjFormat,
                                                       const gl::ImageIndex &index);

// Unified texture's per slice/depth texel reading function
angle::Result ReadTexturePerSliceBytes(const gl::Context *context,
                                       const TextureRef &texture,
                                       size_t bytesPerRow,
                                       const gl::Rectangle &fromRegion,
                                       uint32_t mipLevel,
                                       uint32_t sliceOrDepth,
                                       uint8_t *dataOut);

angle::Result ReadTexturePerSliceBytesToBuffer(const gl::Context *context,
                                               const TextureRef &texture,
                                               size_t bytesPerRow,
                                               const gl::Rectangle &fromRegion,
                                               uint32_t mipLevel,
                                               uint32_t sliceOrDepth,
                                               uint32_t dstOffset,
                                               const BufferRef &dstBuffer);

MTLViewport GetViewport(const gl::Rectangle &rect, double znear = 0, double zfar = 1);
MTLViewport GetViewportFlipY(const gl::Rectangle &rect,
                             NSUInteger screenHeight,
                             double znear = 0,
                             double zfar  = 1);
MTLViewport GetViewport(const gl::Rectangle &rect,
                        NSUInteger screenHeight,
                        bool flipY,
                        double znear = 0,
                        double zfar  = 1);
MTLScissorRect GetScissorRect(const gl::Rectangle &rect,
                              NSUInteger screenHeight = 0,
                              bool flipY              = false);

AutoObjCPtr<id<MTLLibrary>> CreateShaderLibrary(id<MTLDevice> metalDevice,
                                                const std::string &source,
                                                AutoObjCPtr<NSError *> *error);

AutoObjCPtr<id<MTLLibrary>> CreateShaderLibrary(id<MTLDevice> metalDevice,
                                                const char *source,
                                                size_t sourceLen,
                                                AutoObjCPtr<NSError *> *error);

AutoObjCPtr<id<MTLLibrary>> CreateShaderLibraryFromBinary(id<MTLDevice> metalDevice,
                                                          const uint8_t *binarySource,
                                                          size_t binarySourceLen,
                                                          AutoObjCPtr<NSError *> *error);

// Need to define invalid enum value since Metal doesn't define it
constexpr MTLTextureType MTLTextureTypeInvalid = static_cast<MTLTextureType>(NSUIntegerMax);
static_assert(sizeof(MTLTextureType) == sizeof(NSUInteger),
              "MTLTextureType is supposed to be based on NSUInteger");

constexpr MTLPrimitiveType MTLPrimitiveTypeInvalid = static_cast<MTLPrimitiveType>(NSUIntegerMax);
static_assert(sizeof(MTLPrimitiveType) == sizeof(NSUInteger),
              "MTLPrimitiveType is supposed to be based on NSUInteger");

constexpr MTLIndexType MTLIndexTypeInvalid = static_cast<MTLIndexType>(NSUIntegerMax);
static_assert(sizeof(MTLIndexType) == sizeof(NSUInteger),
              "MTLIndexType is supposed to be based on NSUInteger");

MTLTextureType GetTextureType(gl::TextureType glType);

MTLSamplerMinMagFilter GetFilter(GLenum filter);
MTLSamplerMipFilter GetMipmapFilter(GLenum filter);
MTLSamplerAddressMode GetSamplerAddressMode(GLenum wrap);

MTLBlendFactor GetBlendFactor(GLenum factor);
MTLBlendOperation GetBlendOp(GLenum op);

MTLCompareFunction GetCompareFunc(GLenum func);
MTLStencilOperation GetStencilOp(GLenum op);

MTLWinding GetFontfaceWinding(GLenum frontFaceMode, bool invert);

PrimitiveTopologyClass GetPrimitiveTopologyClass(gl::PrimitiveMode mode);
MTLPrimitiveType GetPrimitiveType(gl::PrimitiveMode mode);
MTLIndexType GetIndexType(gl::DrawElementsType type);

#if defined(__IPHONE_13_0) || defined(__MAC_10_15)
MTLTextureSwizzle GetTextureSwizzle(GLenum swizzle);
#endif

// Get color write mask for a specified format. Some formats such as RGB565 doesn't have alpha
// channel but is emulated by a RGBA8 format, we need to disable alpha write for this format.
// - emulatedChannelsOut: if the format is emulated, this pointer will store a true value.
MTLColorWriteMask GetEmulatedColorWriteMask(const mtl::Format &mtlFormat,
                                            bool *emulatedChannelsOut);
MTLColorWriteMask GetEmulatedColorWriteMask(const mtl::Format &mtlFormat);
bool IsFormatEmulated(const mtl::Format &mtlFormat);

// Useful to set clear color for texture originally having no alpha in GL, but backend's format
// has alpha channel.
MTLClearColor EmulatedAlphaClearColor(MTLClearColor color, MTLColorWriteMask colorMask);

static inline gl::Extents MTLSizeToGLExtents(const MTLSize &mtlSize)
{
    return gl::Extents(static_cast<int>(mtlSize.width), static_cast<int>(mtlSize.height),
                       static_cast<int>(mtlSize.depth));
}

static inline gl::Offset MTLOriginToGLOffset(const MTLOrigin &mtlOrigin)
{
    return gl::Offset(static_cast<int>(mtlOrigin.x), static_cast<int>(mtlOrigin.y),
                      static_cast<int>(mtlOrigin.z));
}

static inline gl::Box MTLRegionToGLBox(const MTLRegion &mtlRegion)
{
    return gl::Box(static_cast<int>(mtlRegion.origin.x), static_cast<int>(mtlRegion.origin.y),
                   static_cast<int>(mtlRegion.origin.z), static_cast<int>(mtlRegion.size.width),
                   static_cast<int>(mtlRegion.size.height), static_cast<int>(mtlRegion.size.depth));
}

static inline gl::Rectangle MTLRegionToGLRectangle(const MTLRegion &mtlRegion)
{
    return gl::Rectangle(static_cast<int>(mtlRegion.origin.x), static_cast<int>(mtlRegion.origin.y),
                         static_cast<int>(mtlRegion.size.width),
                         static_cast<int>(mtlRegion.size.height));
}


angle::Result TriangleFanBoundCheck(ContextMtl *context, size_t numTris);
angle::Result GetTriangleFanIndicesCount(ContextMtl *context,
                                         GLsizei vetexCount,
                                         uint32_t *numElemsOut);

NS_ASSUME_NONNULL_END
}  // namespace mtl
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_MTL_UTILS_H_ */
