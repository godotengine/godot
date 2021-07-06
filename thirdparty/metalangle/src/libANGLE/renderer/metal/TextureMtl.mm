
//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TextureMtl.mm:
//    Implements the class methods for TextureMtl.
//

#include "libANGLE/renderer/metal/TextureMtl.h"

#include "common/Color.h"
#include "common/MemoryBuffer.h"
#include "common/debug.h"
#include "common/mathutil.h"
#include "image_util/imageformats.h"
#include "libANGLE/Surface.h"
#include "libANGLE/renderer/metal/BufferMtl.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/FrameBufferMtl.h"
#include "libANGLE/renderer/metal/ImageMtl.h"
#include "libANGLE/renderer/metal/SamplerMtl.h"
#include "libANGLE/renderer/metal/SurfaceMtl.h"
#include "libANGLE/renderer/metal/mtl_common.h"
#include "libANGLE/renderer/metal/mtl_format_utils.h"
#include "libANGLE/renderer/metal/mtl_utils.h"
#include "libANGLE/renderer/renderer_utils.h"

namespace rx
{

namespace
{

gl::ImageIndex GetZeroLevelIndex(const mtl::TextureRef &image)
{
    switch (image->textureType())
    {
        case MTLTextureType2D:
            return gl::ImageIndex::Make2D(0);
        case MTLTextureTypeCube:
            return gl::ImageIndex::MakeFromType(gl::TextureType::CubeMap, 0);
        case MTLTextureType2DArray:
            return gl::ImageIndex::Make2DArray(0 /** entire layers */);
        case MTLTextureType2DMultisample:
            return gl::ImageIndex::Make2DMultisample();
        case MTLTextureType3D:
            return gl::ImageIndex::Make3D(0 /** entire layers */);
        default:
            UNREACHABLE();
            break;
    }

    return gl::ImageIndex();
}

// Slice is ignored if texture type is not Cube or 2D array
gl::ImageIndex GetSliceMipIndex(const mtl::TextureRef &image, uint32_t slice, uint32_t level)
{
    switch (image->textureType())
    {
        case MTLTextureType2D:
            return gl::ImageIndex::Make2D(level);
        case MTLTextureTypeCube: {
            auto cubeFace = static_cast<gl::TextureTarget>(
                static_cast<int>(gl::TextureTarget::CubeMapPositiveX) + slice);
            return gl::ImageIndex::MakeCubeMapFace(cubeFace, level);
        }
        case MTLTextureType2DArray:
            return gl::ImageIndex::Make2DArray(level, slice);
        case MTLTextureType2DMultisample:
            return gl::ImageIndex::Make2DMultisample();
        case MTLTextureType3D:
            return gl::ImageIndex::Make3D(level);
        default:
            UNREACHABLE();
            break;
    }

    return gl::ImageIndex();
}

GLuint GetImageLayerIndex(const gl::ImageIndex &index)
{
    switch (index.getType())
    {
        case gl::TextureType::_2D:
        case gl::TextureType::_2DMultisample:
            return 0;
        case gl::TextureType::CubeMap:
            return index.cubeMapFaceIndex();
        case gl::TextureType::_2DArray:
        case gl::TextureType::_3D:
            return index.getLayerIndex();
        default:
            UNREACHABLE();
    }

    return 0;
}

GLuint GetImageCubeFaceIndexOrZero(const gl::ImageIndex &index)
{
    switch (index.getType())
    {
        case gl::TextureType::CubeMap:
            return index.cubeMapFaceIndex();
        default:
            break;
    }

    return 0;
}

// Given texture type, get texture type of one image for a glTexImage call.
// For example, for texture 2d, one image is also texture 2d.
// for texture cube, one image is texture 2d.
gl::TextureType GetTextureImageType(gl::TextureType texType)
{
    switch (texType)
    {
        case gl::TextureType::CubeMap:
            return gl::TextureType::_2D;
        case gl::TextureType::_2D:
        case gl::TextureType::_2DArray:
        case gl::TextureType::_2DMultisample:
        case gl::TextureType::_3D:
            return texType;
        default:
            UNREACHABLE();
            return gl::TextureType::InvalidEnum;
    }
}

// D24X8 by default writes depth data to high 24 bits of 32 bit integers. However, Metal separate
// depth stencil blitting expects depth data to be in low 24 bits of the data.
void WriteDepthStencilToDepth24(const uint8_t *srcPtr, uint8_t *dstPtr)
{
    auto src = reinterpret_cast<const angle::DepthStencil *>(srcPtr);
    auto dst = reinterpret_cast<uint32_t *>(dstPtr);
    *dst     = gl::floatToNormalized<24, uint32_t>(static_cast<float>(src->depth));
}

angle::Result CopyTextureContentsToStagingBuffer(ContextMtl *contextMtl,
                                                 const angle::Format &textureAngleFormat,
                                                 const angle::Format &stagingAngleFormat,
                                                 rx::PixelWriteFunction pixelWriteFunctionOverride,
                                                 const MTLSize &regionSize,
                                                 const uint8_t *data,
                                                 size_t bytesPerRow,
                                                 size_t bytesPer2DImage,
                                                 size_t *bufferRowPitchOut,
                                                 size_t *buffer2DImageSizeOut,
                                                 mtl::BufferRef *bufferOut)
{
    size_t stagingBufferRowPitch    = regionSize.width * stagingAngleFormat.pixelBytes;
    size_t stagingBuffer2DImageSize = stagingBufferRowPitch * regionSize.height;
    size_t stagingBufferSize        = stagingBuffer2DImageSize * regionSize.depth;
    mtl::BufferRef stagingBuffer;
    ANGLE_TRY(mtl::Buffer::MakeBuffer(contextMtl, stagingBufferSize, nullptr, &stagingBuffer));

    uint8_t *pdst = stagingBuffer->map(contextMtl);

    if (textureAngleFormat.id == stagingAngleFormat.id)
    {
        for (NSUInteger d = 0; d < regionSize.depth; ++d)
        {
            for (NSUInteger r = 0; r < regionSize.height; ++r)
            {
                const uint8_t *pCopySrc = data + d * bytesPer2DImage + r * bytesPerRow;
                uint8_t *pCopyDst = pdst + d * stagingBuffer2DImageSize + r * stagingBufferRowPitch;
                memcpy(pCopyDst, pCopySrc, stagingBufferRowPitch);
            }
        }
    }
    else
    {
        rx::PixelWriteFunction pixelWriteFunction = pixelWriteFunctionOverride
                                                        ? pixelWriteFunctionOverride
                                                        : stagingAngleFormat.pixelWriteFunction;
        // This is only for depth & stencil case.
        ASSERT(textureAngleFormat.depthBits || textureAngleFormat.stencilBits);
        ASSERT(textureAngleFormat.pixelReadFunction && pixelWriteFunction);

        // cache to store read result of source pixel
        angle::DepthStencil depthStencilData;
        auto sourcePixelReadData = reinterpret_cast<uint8_t *>(&depthStencilData);
        ASSERT(textureAngleFormat.pixelBytes <= sizeof(depthStencilData));

        for (NSUInteger d = 0; d < regionSize.depth; ++d)
        {
            for (NSUInteger r = 0; r < regionSize.height; ++r)
            {
                for (NSUInteger c = 0; c < regionSize.width; ++c)
                {
                    const uint8_t *sourcePixelData = data + d * bytesPer2DImage + r * bytesPerRow +
                                                     c * textureAngleFormat.pixelBytes;

                    uint8_t *destPixelData = pdst + d * stagingBuffer2DImageSize +
                                             r * stagingBufferRowPitch +
                                             c * stagingAngleFormat.pixelBytes;

                    textureAngleFormat.pixelReadFunction(sourcePixelData, sourcePixelReadData);
                    pixelWriteFunction(sourcePixelReadData, destPixelData);
                }
            }
        }
    }

    stagingBuffer->unmap(contextMtl);

    *bufferOut            = stagingBuffer;
    *bufferRowPitchOut    = stagingBufferRowPitch;
    *buffer2DImageSizeOut = stagingBuffer2DImageSize;

    return angle::Result::Continue;
}

angle::Result UploadTextureContentsWithStagingBuffer(ContextMtl *contextMtl,
                                                     const angle::Format &textureAngleFormat,
                                                     MTLRegion region,
                                                     uint32_t mipmapLevel,
                                                     uint32_t slice,
                                                     const uint8_t *data,
                                                     size_t bytesPerRow,
                                                     size_t bytesPer2DImage,
                                                     const mtl::TextureRef &texture)
{
    ASSERT(texture && texture->valid());

    ASSERT(!texture->isCPUAccessible());

    ASSERT(!textureAngleFormat.depthBits || !textureAngleFormat.stencilBits);

    // Compressed texture is not supporte atm
    ASSERT(!textureAngleFormat.isBlock);

    // Copy data to staging buffer
    size_t stagingBufferRowPitch;
    size_t stagingBuffer2DImageSize;
    mtl::BufferRef stagingBuffer;
    ANGLE_TRY(CopyTextureContentsToStagingBuffer(
        contextMtl, textureAngleFormat, textureAngleFormat, textureAngleFormat.pixelWriteFunction,
        region.size, data, bytesPerRow, bytesPer2DImage, &stagingBufferRowPitch,
        &stagingBuffer2DImageSize, &stagingBuffer));

    // Copy staging buffer to texture.
    mtl::BlitCommandEncoder *encoder = contextMtl->getBlitCommandEncoder();
    encoder->copyBufferToTexture(stagingBuffer, 0, stagingBufferRowPitch, stagingBuffer2DImageSize,
                                 region.size, texture, slice, mipmapLevel, region.origin,
                                 MTLBlitOptionNone);

    return angle::Result::Continue;
}

// Packed depth stencil upload using staging buffer
angle::Result UploadPackedDepthStencilTextureContentsWithStagingBuffer(
    ContextMtl *contextMtl,
    const angle::Format &textureAngleFormat,
    MTLRegion region,
    uint32_t mipmapLevel,
    uint32_t slice,
    const uint8_t *data,
    size_t bytesPerRow,
    size_t bytesPer2DImage,
    const mtl::TextureRef &texture)
{
    ASSERT(texture && texture->valid());

    ASSERT(!texture->isCPUAccessible());

    ASSERT(textureAngleFormat.depthBits && textureAngleFormat.stencilBits);

    // We have to split the depth & stencil data into 2 buffers.
    angle::FormatID stagingDepthBufferFormatId;
    angle::FormatID stagingStencilBufferFormatId;
    // Custom depth write function. We cannot use those in imageformats.cpp since Metal has some
    // special cases.
    rx::PixelWriteFunction stagingDepthBufferWriteFunctionOverride = nullptr;

    switch (textureAngleFormat.id)
    {
        case angle::FormatID::D24_UNORM_S8_UINT:
            // D24_UNORM_X8_UINT writes depth data to high 24 bits. But Metal expects depth data to
            // be in low 24 bits.
            stagingDepthBufferFormatId              = angle::FormatID::D24_UNORM_X8_UINT;
            stagingDepthBufferWriteFunctionOverride = WriteDepthStencilToDepth24;
            stagingStencilBufferFormatId            = angle::FormatID::S8_UINT;
            break;
        case angle::FormatID::D32_FLOAT_S8X24_UINT:
            stagingDepthBufferFormatId   = angle::FormatID::D32_FLOAT;
            stagingStencilBufferFormatId = angle::FormatID::S8_UINT;
            break;
        default:
            ANGLE_MTL_UNREACHABLE(contextMtl);
    }

    const angle::Format &angleStagingDepthFormat = angle::Format::Get(stagingDepthBufferFormatId);
    const angle::Format &angleStagingStencilFormat =
        angle::Format::Get(stagingStencilBufferFormatId);

    size_t stagingDepthBufferRowPitch, stagingStencilBufferRowPitch;
    size_t stagingDepthBuffer2DImageSize, stagingStencilBuffer2DImageSize;
    mtl::BufferRef stagingDepthbuffer, stagingStencilBuffer;

    ANGLE_TRY(CopyTextureContentsToStagingBuffer(
        contextMtl, textureAngleFormat, angleStagingDepthFormat,
        stagingDepthBufferWriteFunctionOverride, region.size, data, bytesPerRow, bytesPer2DImage,
        &stagingDepthBufferRowPitch, &stagingDepthBuffer2DImageSize, &stagingDepthbuffer));

    ANGLE_TRY(CopyTextureContentsToStagingBuffer(
        contextMtl, textureAngleFormat, angleStagingStencilFormat, nullptr, region.size, data,
        bytesPerRow, bytesPer2DImage, &stagingStencilBufferRowPitch,
        &stagingStencilBuffer2DImageSize, &stagingStencilBuffer));

    mtl::BlitCommandEncoder *encoder = contextMtl->getBlitCommandEncoder();

    encoder->copyBufferToTexture(stagingDepthbuffer, 0, stagingDepthBufferRowPitch,
                                 stagingDepthBuffer2DImageSize, region.size, texture, slice,
                                 mipmapLevel, region.origin, MTLBlitOptionDepthFromDepthStencil);
    encoder->copyBufferToTexture(stagingStencilBuffer, 0, stagingStencilBufferRowPitch,
                                 stagingStencilBuffer2DImageSize, region.size, texture, slice,
                                 mipmapLevel, region.origin, MTLBlitOptionStencilFromDepthStencil);

    return angle::Result::Continue;
}

angle::Result UploadTextureContents(const gl::Context *context,
                                    const angle::Format &textureAngleFormat,
                                    const MTLRegion &region,
                                    uint32_t mipmapLevel,
                                    uint32_t slice,
                                    const uint8_t *data,
                                    size_t bytesPerRow,
                                    size_t bytesPer2DImage,
                                    const mtl::TextureRef &texture)
{
    ASSERT(texture && texture->valid());
    ContextMtl *contextMtl = mtl::GetImpl(context);

    if (texture->isCPUAccessible())
    {
        // If texture is CPU accessible, just call replaceRegion() directly.
        texture->replaceRegion(contextMtl, region, mipmapLevel, slice, data, bytesPerRow,
                               bytesPer2DImage);

        return angle::Result::Continue;
    }

    // Texture is not CPU accessible, we need to use staging buffer
    if (textureAngleFormat.depthBits && textureAngleFormat.stencilBits)
    {
        ANGLE_TRY(UploadPackedDepthStencilTextureContentsWithStagingBuffer(
            contextMtl, textureAngleFormat, region, mipmapLevel, slice, data, bytesPerRow,
            bytesPer2DImage, texture));
    }
    else
    {
        ANGLE_TRY(UploadTextureContentsWithStagingBuffer(contextMtl, textureAngleFormat, region,
                                                         mipmapLevel, slice, data, bytesPerRow,
                                                         bytesPer2DImage, texture));
    }

    return angle::Result::Continue;
}

ANGLE_MTL_UNUSED
GLenum OverrideSwizzleValue(const gl::Context *context,
                            GLenum swizzle,
                            const mtl::Format &format,
                            const gl::InternalFormat &glInternalFormat)
{
    if (format.actualAngleFormat().depthBits)
    {
        ASSERT(!format.swizzled);
        if (context->getState().getClientMajorVersion() >= 3 && glInternalFormat.sized)
        {
            // ES 3.0 spec: treat depth texture as red texture during sampling.
            if (swizzle == GL_GREEN || swizzle == GL_BLUE)
            {
                return GL_NONE;
            }
            else if (swizzle == GL_ALPHA)
            {
                return GL_ONE;
            }
        }
        else
        {
            // https://www.khronos.org/registry/OpenGL/extensions/OES/OES_depth_texture.txt
            // Treat depth texture as luminance texture during sampling.
            if (swizzle == GL_GREEN || swizzle == GL_BLUE)
            {
                return GL_RED;
            }
            else if (swizzle == GL_ALPHA)
            {
                return GL_ONE;
            }
        }
    }
    else if (format.swizzled)
    {
        // Combine the swizzles
        switch (swizzle)
        {
            case GL_RED:
                return format.swizzle[0];
            case GL_GREEN:
                return format.swizzle[1];
            case GL_BLUE:
                return format.swizzle[2];
            case GL_ALPHA:
                return format.swizzle[3];
            default:
                break;
        }
    }

    return swizzle;
}

}  // namespace

// TextureMtl implementation
TextureMtl::TextureMtl(const gl::TextureState &state) : TextureImpl(state) {}

TextureMtl::~TextureMtl() = default;

void TextureMtl::onDestroy(const gl::Context *context)
{
    releaseTexture(true);
}

void TextureMtl::releaseTexture(bool releaseImages)
{
    releaseTexture(releaseImages, false);
}

void TextureMtl::releaseTexture(bool releaseImages, bool releaseTextureObjectsOnly)
{

    if (releaseImages)
    {
        mTexImageDefs.clear();
    }
    else if (mNativeTexture)
    {
        // Release native texture but keep its old per face per mipmap level image views.
        retainImageDefinitions();
    }

    mNativeTexture             = nullptr;
    mNativeSwizzleSamplingView = nullptr;
    mBoundPBuffer              = nullptr;

    mImplicitMSTextures.clear();

    // Clear render target cache for each texture's image. We don't erase them because they
    // might still be referenced by a framebuffer.
    for (auto &sliceRenderTargets : mPerLayerRenderTargets)
    {
        for (RenderTargetMtl &mipRenderTarget : sliceRenderTargets.second)
        {
            mipRenderTarget.reset();
        }
    }

    for (mtl::TextureRef &view : mNativeLevelViews)
    {
        view.reset();
    }

    if (!releaseTextureObjectsOnly)
    {
        mMetalSamplerState = nil;
        mFormat            = mtl::Format();
        mIsPow2            = false;
    }
}

angle::Result TextureMtl::ensureTextureCreated(const gl::Context *context)
{
    if (mNativeTexture)
    {
        return angle::Result::Continue;
    }

    ContextMtl *contextMtl = mtl::GetImpl(context);

    // Create actual texture object:
    mCurrentBaseLevel = mState.getEffectiveBaseLevel();

    const GLuint mips  = mState.getMipmapMaxLevel() - mCurrentBaseLevel + 1;
    gl::ImageDesc desc = mState.getBaseLevelDesc();
    ANGLE_MTL_CHECK(contextMtl, desc.format.valid(), GL_INVALID_OPERATION);
    angle::FormatID angleFormatId =
        angle::Format::InternalFormatToID(desc.format.info->sizedInternalFormat);
    mFormat = contextMtl->getPixelFormat(angleFormatId);

    return createNativeTexture(context, mState.getType(), mips, desc.size);
}

angle::Result TextureMtl::createNativeTexture(const gl::Context *context,
                                              gl::TextureType type,
                                              GLuint mips,
                                              const gl::Extents &size)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);

    // Create actual texture object:
    mCurrentBaseLevel = mState.getEffectiveBaseLevel();
    mCurrentMaxLevel  = mState.getEffectiveMaxLevel();

    mIsPow2          = gl::isPow2(size.width) && gl::isPow2(size.height) && gl::isPow2(size.depth);
    mSlices          = 1;
    int numCubeFaces = 1;
    switch (type)
    {
        case gl::TextureType::_2D:
            ANGLE_TRY(mtl::Texture::Make2DTexture(contextMtl, mFormat, size.width, size.height,
                                                  mips,
                                                  /** renderTargetOnly */ false,
                                                  /** allowFormatView */ false, &mNativeTexture));
            break;
        case gl::TextureType::CubeMap:
            mSlices = numCubeFaces = 6;
            ANGLE_TRY(mtl::Texture::MakeCubeTexture(contextMtl, mFormat, size.width, mips,
                                                    /** renderTargetOnly */ false,
                                                    /** allowFormatView */ false, &mNativeTexture));
            break;
        case gl::TextureType::_3D:
            ANGLE_TRY(mtl::Texture::Make3DTexture(contextMtl, mFormat, size.width, size.height,
                                                  size.depth, mips,
                                                  /** renderTargetOnly */ false,
                                                  /** allowFormatView */ false, &mNativeTexture));
            break;
        case gl::TextureType::_2DArray:
            mSlices = size.depth;
            ANGLE_TRY(mtl::Texture::Make2DArrayTexture(
                contextMtl, mFormat, size.width, size.height, mips, mSlices,
                /** renderTargetOnly */ false,
                /** allowFormatView */ false, &mNativeTexture));
            break;
        default:
            UNREACHABLE();
    }

    bool hasEmulatedChannels = mtl::IsFormatEmulated(mFormat);

    // Transfer data from images to actual texture object
    for (int face = 0; face < numCubeFaces; ++face)
    {
        for (GLuint mip = 0; mip < mips; ++mip)
        {
            GLuint imageMipLevel             = mip + mCurrentBaseLevel;
            mtl::TextureRef &imageToTransfer = mTexImageDefs[face][imageMipLevel].image;

            // Only transfer if this mip & slice image has been defined and in correct size &
            // format.
            gl::Extents actualMipSize = mNativeTexture->size(mip);
            if (imageToTransfer && imageToTransfer->size() == actualMipSize &&
                imageToTransfer->pixelFormat() == mNativeTexture->pixelFormat())
            {
                mtl::BlitCommandEncoder *encoder = contextMtl->getBlitCommandEncoder();
                encoder->copyTexture(imageToTransfer, 0, 0, mNativeTexture, face, mip,
                                     imageToTransfer->arrayLength(), 1);
            }
            else
            {
                if (hasEmulatedChannels)
                {
                    GLint layerIndex = numCubeFaces > 1 ? face : gl::ImageIndex::kEntireLevel;
                    gl::ImageIndex mipIndex = gl::ImageIndex::MakeFromType(type, mip, layerIndex);
                    if (mFormat.getCaps().isRenderable())
                    {
                        ANGLE_TRY(mtl::InitializeTextureContentsGPU(
                            context, mNativeTexture, mFormat, mipIndex, MTLColorWriteMaskAll));
                    }
                    else
                    {
                        ANGLE_TRY(mtl::InitializeTextureContents(context, mNativeTexture, mFormat,
                                                                 mipIndex));
                    }
                }
            }

            imageToTransfer = nullptr;
        }
    }

    // Flush texture's GPU updates
    contextMtl->flushCommandBufer();

    // Create sampler state
    ANGLE_TRY(ensureSamplerStateCreated(context));

    return angle::Result::Continue;
}

angle::Result TextureMtl::ensureSamplerStateCreated(const gl::Context *context)
{
    if (mMetalSamplerState)
    {
        return angle::Result::Continue;
    }

    ContextMtl *contextMtl = mtl::GetImpl(context);
    DisplayMtl *displayMtl = contextMtl->getDisplay();

    mtl::SamplerDesc samplerDesc(mState.getSamplerState());

    if (mFormat.actualAngleFormat().depthBits &&
        !displayMtl->getFeatures().hasDepthTextureFiltering.enabled)
    {
        // On devices not supporting filtering for depth textures, we need to convert to nearest
        // here.
        samplerDesc.minFilter = MTLSamplerMinMagFilterNearest;
        samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
        if (samplerDesc.mipFilter != MTLSamplerMipFilterNotMipmapped)
        {
            samplerDesc.mipFilter = MTLSamplerMipFilterNearest;
        }

        samplerDesc.maxAnisotropy = 1;
    }

    mMetalSamplerState =
        displayMtl->getStateCache().getSamplerState(displayMtl->getMetalDevice(), samplerDesc);

    return angle::Result::Continue;
}

angle::Result TextureMtl::onBaseMaxLevelsChanged(const gl::Context *context)
{
    if (!mNativeTexture || (mCurrentBaseLevel == mState.getEffectiveBaseLevel() &&
                            mCurrentMaxLevel == mState.getEffectiveMaxLevel()))
    {
        return angle::Result::Continue;
    }

    ContextMtl *contextMtl = mtl::GetImpl(context);

    // Release native texture but keep old image definitions so that it can be recreated from old
    // image definitions with different base level
    releaseTexture(false, true);

    // Tell context to rebind textures
    contextMtl->invalidateCurrentTextures();

    return angle::Result::Continue;
}

angle::Result TextureMtl::ensureImageCreated(const gl::Context *context,
                                             const gl::ImageIndex &index)
{
    mtl::TextureRef &image = getImage(index);
    if (!image)
    {
        // Image at this level hasn't been defined yet. We need to define it:
        const gl::ImageDesc &desc = mState.getImageDesc(index);
        ANGLE_TRY(redefineImage(context, index, mFormat, desc.size, true));
    }
    return angle::Result::Continue;
}

angle::Result TextureMtl::ensureNativeLevelViewsCreated()
{
    ASSERT(mNativeTexture);
    const GLuint baseLevel = mState.getEffectiveBaseLevel();
    for (uint32_t mip = 0; mip < mNativeTexture->mipmapLevels(); ++mip)
    {
        if (mNativeLevelViews[mip])
        {
            continue;
        }

        if (mNativeTexture->textureType() != MTLTextureTypeCube &&
            mTexImageDefs[0][mip + baseLevel].image)
        {
            // Reuse texture image view.
            mNativeLevelViews[mip] = mTexImageDefs[0][mip + baseLevel].image;
        }
        else
        {
            mNativeLevelViews[mip] = mNativeTexture->createMipView(mip);
        }
    }
    return angle::Result::Continue;
}

mtl::TextureRef TextureMtl::createImageViewFromNativeTexture(GLuint cubeFaceOrZero,
                                                             GLuint nativeLevel)
{
    mtl::TextureRef image;
    if (mNativeTexture->textureType() == MTLTextureTypeCube)
    {
        // Cube texture's image is per face.
        image = mNativeTexture->createSliceMipView(cubeFaceOrZero, nativeLevel);
    }
    else
    {
        if (mNativeLevelViews[nativeLevel])
        {
            // Reuse the native level view
            image = mNativeLevelViews[nativeLevel];
        }
        else
        {
            image = mNativeTexture->createMipView(nativeLevel);
        }
    }

    return image;
}

void TextureMtl::retainImageDefinitions()
{
    if (!mNativeTexture)
    {
        return;
    }
    const GLuint mips = mNativeTexture->mipmapLevels();

    int numCubeFaces = 1;
    switch (mState.getType())
    {
        case gl::TextureType::CubeMap:
            numCubeFaces = 6;
            break;
        default:
            break;
    }

    // Create image view per cube face, per mip level
    for (int face = 0; face < numCubeFaces; ++face)
    {
        for (GLuint mip = 0; mip < mips; ++mip)
        {
            GLuint imageMipLevel         = mip + mCurrentBaseLevel;
            ImageDefinitionMtl &imageDef = mTexImageDefs[face][imageMipLevel];
            if (imageDef.image)
            {
                continue;
            }
            imageDef.image    = createImageViewFromNativeTexture(face, mip);
            imageDef.formatID = mFormat.intendedFormatId;
        }
    }
}

bool TextureMtl::isIndexWithinMinMaxLevels(const gl::ImageIndex &imageIndex) const
{
    return imageIndex.getLevelIndex() >= static_cast<GLint>(mState.getEffectiveBaseLevel()) &&
           imageIndex.getLevelIndex() <= static_cast<GLint>(mState.getEffectiveMaxLevel());
}

int TextureMtl::getNativeLevel(const gl::ImageIndex &imageIndex) const
{
    int baseLevel     = mState.getEffectiveBaseLevel();
    int adjustedLevel = imageIndex.getLevelIndex() - baseLevel;
    return adjustedLevel;
}

mtl::TextureRef &TextureMtl::getImage(const gl::ImageIndex &imageIndex)
{
    return getImageDefinition(imageIndex).image;
}

ImageDefinitionMtl &TextureMtl::getImageDefinition(const gl::ImageIndex &imageIndex)
{
    GLuint cubeFaceOrZero        = GetImageCubeFaceIndexOrZero(imageIndex);
    ImageDefinitionMtl &imageDef = mTexImageDefs[cubeFaceOrZero][imageIndex.getLevelIndex()];

    if (!imageDef.image && mNativeTexture)
    {
        // If native texture is already created, and the image at this index is not available,
        // then create a view of native texture at this index, so that modifications of the image
        // are reflected back to native texture's respective index.
        if (!isIndexWithinMinMaxLevels(imageIndex))
        {
            // Image below base level are skipped.
            return imageDef;
        }

        int nativeLevel   = getNativeLevel(imageIndex);
        imageDef.image    = createImageViewFromNativeTexture(cubeFaceOrZero, nativeLevel);
        imageDef.formatID = mFormat.intendedFormatId;
    }

    return imageDef;
}
RenderTargetMtl &TextureMtl::getRenderTarget(const gl::ImageIndex &imageIndex)
{
    ASSERT(imageIndex.getType() == gl::TextureType::_2D ||
           imageIndex.getType() == gl::TextureType::_2DMultisample || imageIndex.hasLayer());
    GLuint layer         = GetImageLayerIndex(imageIndex);
    RenderTargetMtl &rtt = mPerLayerRenderTargets[layer][imageIndex.getLevelIndex()];
    if (!rtt.getTexture())
    {
        // Lazy initialization of render target:
        mtl::TextureRef &image = getImage(imageIndex);
        if (image)
        {
            if (imageIndex.getType() == gl::TextureType::CubeMap)
            {
                // Cube map is special, the image is already the view of its layer
                rtt.set(image, 0, 0, mFormat);
            }
            else
            {
                rtt.set(image, 0, layer, mFormat);
            }
        }
    }
    return rtt;
}
mtl::TextureRef &TextureMtl::getImplicitMSTexture(const gl::ImageIndex &imageIndex)
{
    ASSERT(imageIndex.getType() == gl::TextureType::_2D || imageIndex.hasLayer());
    return mImplicitMSTextures[GetImageLayerIndex(imageIndex)][imageIndex.getLevelIndex()];
}

angle::Result TextureMtl::setImage(const gl::Context *context,
                                   const gl::ImageIndex &index,
                                   GLenum internalFormat,
                                   const gl::Extents &size,
                                   GLenum format,
                                   GLenum type,
                                   const gl::PixelUnpackState &unpack,
                                   const uint8_t *pixels)
{
    const gl::InternalFormat &formatInfo = gl::GetInternalFormatInfo(internalFormat, type);

    return setImageImpl(context, index, formatInfo, size, type, unpack, pixels);
}

angle::Result TextureMtl::setSubImage(const gl::Context *context,
                                      const gl::ImageIndex &index,
                                      const gl::Box &area,
                                      GLenum format,
                                      GLenum type,
                                      const gl::PixelUnpackState &unpack,
                                      gl::Buffer *unpackBuffer,
                                      const uint8_t *pixels)
{
    const gl::InternalFormat &formatInfo =
        gl::GetInternalFormatInfo(mFormat.intendedInternalFormat().internalFormat, type);

    return setSubImageImpl(context, index, area, formatInfo, type, unpack, unpackBuffer, pixels);
}

angle::Result TextureMtl::setCompressedImage(const gl::Context *context,
                                             const gl::ImageIndex &index,
                                             GLenum internalFormat,
                                             const gl::Extents &size,
                                             const gl::PixelUnpackState &unpack,
                                             size_t imageSize,
                                             const uint8_t *pixels)
{
    const gl::InternalFormat &formatInfo = gl::GetSizedInternalFormatInfo(internalFormat);

    return setImageImpl(context, index, formatInfo, size, GL_UNSIGNED_BYTE, unpack, pixels);
}

angle::Result TextureMtl::setCompressedSubImage(const gl::Context *context,
                                                const gl::ImageIndex &index,
                                                const gl::Box &area,
                                                GLenum format,
                                                const gl::PixelUnpackState &unpack,
                                                size_t imageSize,
                                                const uint8_t *pixels)
{

    const gl::InternalFormat &formatInfo = gl::GetInternalFormatInfo(format, GL_UNSIGNED_BYTE);

    const gl::State &glState = context->getState();
    gl::Buffer *unpackBuffer = glState.getTargetBuffer(gl::BufferBinding::PixelUnpack);

    return setSubImageImpl(context, index, area, formatInfo, GL_UNSIGNED_BYTE, unpack, unpackBuffer,
                           pixels);
}

angle::Result TextureMtl::copyImage(const gl::Context *context,
                                    const gl::ImageIndex &index,
                                    const gl::Rectangle &sourceArea,
                                    GLenum internalFormat,
                                    gl::Framebuffer *source)
{
    gl::Extents newImageSize(sourceArea.width, sourceArea.height, 1);
    const gl::InternalFormat &internalFormatInfo =
        gl::GetInternalFormatInfo(internalFormat, GL_UNSIGNED_BYTE);

    ContextMtl *contextMtl = mtl::GetImpl(context);
    angle::FormatID angleFormatId =
        angle::Format::InternalFormatToID(internalFormatInfo.sizedInternalFormat);
    const mtl::Format &mtlFormat = contextMtl->getPixelFormat(angleFormatId);

    ANGLE_TRY(redefineImage(context, index, mtlFormat, newImageSize, true));

    if (context->isWebGL())
    {
        ANGLE_TRY(initializeContents(context, index));
    }

    return copySubImageImpl(context, index, gl::Offset(0, 0, 0), sourceArea, internalFormatInfo,
                            source);
}

angle::Result TextureMtl::copySubImage(const gl::Context *context,
                                       const gl::ImageIndex &index,
                                       const gl::Offset &destOffset,
                                       const gl::Rectangle &sourceArea,
                                       gl::Framebuffer *source)
{
    const gl::InternalFormat &currentFormat = *mState.getImageDesc(index).format.info;
    return copySubImageImpl(context, index, destOffset, sourceArea, currentFormat, source);
}

angle::Result TextureMtl::copyTexture(const gl::Context *context,
                                      const gl::ImageIndex &index,
                                      GLenum internalFormat,
                                      GLenum type,
                                      size_t sourceLevel,
                                      bool unpackFlipY,
                                      bool unpackPremultiplyAlpha,
                                      bool unpackUnmultiplyAlpha,
                                      const gl::Texture *source)
{
    UNIMPLEMENTED();

    return angle::Result::Stop;
}

angle::Result TextureMtl::copySubTexture(const gl::Context *context,
                                         const gl::ImageIndex &index,
                                         const gl::Offset &destOffset,
                                         size_t sourceLevel,
                                         const gl::Box &sourceBox,
                                         bool unpackFlipY,
                                         bool unpackPremultiplyAlpha,
                                         bool unpackUnmultiplyAlpha,
                                         const gl::Texture *source)
{
    UNIMPLEMENTED();

    return angle::Result::Stop;
}

angle::Result TextureMtl::copyCompressedTexture(const gl::Context *context,
                                                const gl::Texture *source)
{
    UNIMPLEMENTED();

    return angle::Result::Stop;
}

angle::Result TextureMtl::setStorage(const gl::Context *context,
                                     gl::TextureType type,
                                     size_t mipmaps,
                                     GLenum internalFormat,
                                     const gl::Extents &size)
{
    ContextMtl *contextMtl               = mtl::GetImpl(context);
    const gl::InternalFormat &formatInfo = gl::GetSizedInternalFormatInfo(internalFormat);
    angle::FormatID angleFormatId =
        angle::Format::InternalFormatToID(formatInfo.sizedInternalFormat);
    const mtl::Format &mtlFormat = contextMtl->getPixelFormat(angleFormatId);

    return setStorageImpl(context, type, mipmaps, mtlFormat, size);
}

angle::Result TextureMtl::setStorageExternalMemory(const gl::Context *context,
                                                   gl::TextureType type,
                                                   size_t levels,
                                                   GLenum internalFormat,
                                                   const gl::Extents &size,
                                                   gl::MemoryObject *memoryObject,
                                                   GLuint64 offset)
{
    UNIMPLEMENTED();

    return angle::Result::Stop;
}

angle::Result TextureMtl::setStorageMultisample(const gl::Context *context,
                                                gl::TextureType type,
                                                GLsizei samples,
                                                GLint internalformat,
                                                const gl::Extents &size,
                                                bool fixedSampleLocations)
{
    UNIMPLEMENTED();

    return angle::Result::Stop;
}

angle::Result TextureMtl::setEGLImageTarget(const gl::Context *context,
                                            gl::TextureType type,
                                            egl::Image *image)
{
    releaseTexture(true);

    ContextMtl *contextMtl = mtl::GetImpl(context);

    ImageMtl *imageMtl = mtl::GetImpl(image);
    if (type != imageMtl->getImageTextureType())
    {
        return angle::Result::Stop;
    }

    mNativeTexture = imageMtl->getTexture();

    const angle::FormatID angleFormatId =
        angle::Format::InternalFormatToID(image->getFormat().info->sizedInternalFormat);
    mFormat = contextMtl->getPixelFormat(angleFormatId);

    mSlices = mNativeTexture->cubeFacesOrArrayLength();

    gl::Extents size = mNativeTexture->size();
    mIsPow2          = gl::isPow2(size.width) && gl::isPow2(size.height) && gl::isPow2(size.depth);
    ANGLE_TRY(ensureSamplerStateCreated(context));

    // Tell context to rebind textures
    contextMtl->invalidateCurrentTextures();

    return angle::Result::Continue;
}

angle::Result TextureMtl::setImageExternal(const gl::Context *context,
                                           gl::TextureType type,
                                           egl::Stream *stream,
                                           const egl::Stream::GLTextureDescription &desc)
{
    UNIMPLEMENTED();
    return angle::Result::Stop;
}

angle::Result TextureMtl::generateMipmap(const gl::Context *context)
{
    ANGLE_TRY(ensureTextureCreated(context));

    ContextMtl *contextMtl = mtl::GetImpl(context);
    if (!mNativeTexture)
    {
        return angle::Result::Continue;
    }

    const mtl::FormatCaps &caps = mFormat.getCaps();

    if (caps.writable &&
        !contextMtl->getDisplay()->getFeatures().forceNonCSBaseMipmapGeneration.enabled)
    {
        ANGLE_TRY(ensureNativeLevelViewsCreated());
        ANGLE_TRY(contextMtl->getDisplay()->getUtils().generateMipmapCS(contextMtl, mNativeTexture,
                                                                        &mNativeLevelViews));
    }
    else if (caps.filterable && caps.colorRenderable)
    {
        mtl::BlitCommandEncoder *blitEncoder = contextMtl->getBlitCommandEncoder();
        blitEncoder->generateMipmapsForTexture(mNativeTexture);
    }
    else
    {
        ANGLE_TRY(generateMipmapCPU(context));
    }

    return angle::Result::Continue;
}

angle::Result TextureMtl::generateMipmapCPU(const gl::Context *context)
{
    ASSERT(mNativeTexture && mNativeTexture->valid());

    ContextMtl *contextMtl           = mtl::GetImpl(context);
    const angle::Format &angleFormat = mFormat.actualAngleFormat();
    // This format must have mip generation function.
    ANGLE_MTL_TRY(contextMtl, angleFormat.mipGenerationFunction);

    for (uint32_t slice = 0; slice < mSlices; ++slice)
    {
        int maxMipLevel = static_cast<int>(mNativeTexture->mipmapLevels()) - 1;
        int firstLevel  = 0;

        uint32_t prevLevelWidth    = mNativeTexture->width();
        uint32_t prevLevelHeight   = mNativeTexture->height();
        uint32_t prevLevelDepth    = mNativeTexture->depth();
        size_t prevLevelRowPitch   = angleFormat.pixelBytes * prevLevelWidth;
        size_t prevLevelDepthPitch = prevLevelRowPitch * prevLevelHeight;
        std::unique_ptr<uint8_t[]> prevLevelData(new (std::nothrow)
                                                     uint8_t[prevLevelDepthPitch * prevLevelDepth]);
        ANGLE_CHECK_GL_ALLOC(contextMtl, prevLevelData);
        std::unique_ptr<uint8_t[]> dstLevelData;

        // Download base level data
        mNativeTexture->getBytes(
            contextMtl, prevLevelRowPitch, prevLevelDepthPitch,
            MTLRegionMake3D(0, 0, 0, prevLevelWidth, prevLevelHeight, prevLevelDepth), firstLevel,
            slice, prevLevelData.get());

        for (int mip = firstLevel + 1; mip <= maxMipLevel; ++mip)
        {
            uint32_t dstWidth  = mNativeTexture->width(mip);
            uint32_t dstHeight = mNativeTexture->height(mip);
            uint32_t dstDepth  = mNativeTexture->depth(mip);

            size_t dstRowPitch   = angleFormat.pixelBytes * dstWidth;
            size_t dstDepthPitch = dstRowPitch * dstHeight;
            size_t dstDataSize   = dstDepthPitch * dstDepth;
            if (!dstLevelData)
            {
                // Allocate once and reuse the buffer
                dstLevelData.reset(new (std::nothrow) uint8_t[dstDataSize]);
                ANGLE_CHECK_GL_ALLOC(contextMtl, dstLevelData);
            }

            // Generate mip level
            angleFormat.mipGenerationFunction(
                prevLevelWidth, prevLevelHeight, 1, prevLevelData.get(), prevLevelRowPitch,
                prevLevelDepthPitch, dstLevelData.get(), dstRowPitch, dstDepthPitch);

            // Upload to texture
            ANGLE_TRY(UploadTextureContents(
                context, angleFormat, MTLRegionMake3D(0, 0, 0, dstWidth, dstHeight, dstDepth), mip,
                slice, dstLevelData.get(), dstRowPitch, dstDepthPitch, mNativeTexture));

            prevLevelWidth      = dstWidth;
            prevLevelHeight     = dstHeight;
            prevLevelDepth      = dstDepth;
            prevLevelRowPitch   = dstRowPitch;
            prevLevelDepthPitch = dstDepthPitch;
            std::swap(prevLevelData, dstLevelData);
        }  // for mip level

    }  // For layers

    return angle::Result::Continue;
}

angle::Result TextureMtl::setBaseLevel(const gl::Context *context, GLuint baseLevel)
{
    return onBaseMaxLevelsChanged(context);
}

angle::Result TextureMtl::bindTexImage(const gl::Context *context, egl::Surface *surface)
{
    releaseTexture(true);

    mBoundPBuffer    = GetImplAs<OffscreenSurfaceMtl>(surface);
    mNativeTexture   = mBoundPBuffer->getColorTexture();
    mFormat          = mBoundPBuffer->getColorFormat();
    gl::Extents size = mNativeTexture->size();
    mIsPow2          = gl::isPow2(size.width) && gl::isPow2(size.height) && gl::isPow2(size.depth);
    ANGLE_TRY(ensureSamplerStateCreated(context));

    // Tell context to rebind textures
    ContextMtl *contextMtl = mtl::GetImpl(context);
    contextMtl->invalidateCurrentTextures();

    return angle::Result::Continue;
}

angle::Result TextureMtl::releaseTexImage(const gl::Context *context)
{
    releaseTexture(true);
    return angle::Result::Continue;
}

angle::Result TextureMtl::getAttachmentRenderTarget(const gl::Context *context,
                                                    GLenum binding,
                                                    const gl::ImageIndex &imageIndex,
                                                    GLsizei samples,
                                                    FramebufferAttachmentRenderTarget **rtOut)
{
    ANGLE_TRY(ensureTextureCreated(context));

    ContextMtl *contextMtl = mtl::GetImpl(context);
    ANGLE_MTL_TRY(contextMtl, mNativeTexture);

    switch (imageIndex.getType())
    {
        case gl::TextureType::_2D: {
            RenderTargetMtl &rtt = getRenderTarget(imageIndex);
            *rtOut               = &rtt;

            if (samples > 0 && rtt.getTexture() && isIndexWithinMinMaxLevels(imageIndex) &&
                mNativeTexture)
            {
                // This format must supports implicit resolve
                ANGLE_MTL_CHECK(contextMtl, mFormat.getCaps().resolve, GL_INVALID_VALUE);

                const gl::TextureCaps &textureCaps =
                    contextMtl->getTextureCaps().get(mFormat.intendedFormatId);
                samples = textureCaps.getNearestSamples(std::max(2, samples));
                ANGLE_MTL_CHECK(contextMtl, samples != 0, GL_INVALID_VALUE);

                // Create implicit MS texture.
                mtl::TextureRef &msTexture = getImplicitMSTexture(imageIndex);
                if (!msTexture || msTexture->samples() != static_cast<uint32_t>(samples))
                {
                    if (mBoundPBuffer)
                    {
                        // NOTE(hqle): mipmapped pbuffer is not supported yet.
                        ASSERT(imageIndex.getLevelIndex() == 0);
                        ANGLE_TRY(mBoundPBuffer->getAttachmentMSColorTexture(context, samples,
                                                                             &msTexture));
                    }
                    else
                    {
                        const gl::ImageDesc &desc = mState.getImageDesc(imageIndex);

                        ANGLE_TRY(mtl::Texture::Make2DMSTexture(
                            contextMtl, mFormat, desc.size.width, desc.size.height, samples,
                            /* renderTargetOnly */ true,
                            /* allowFormatView */ false, &msTexture));
                    }

                    ANGLE_TRY(checkForEmulatedChannels(context, mFormat, msTexture));
                }

                rtt.setImplicitMSTexture(msTexture);
            }
        }
        break;  // case gl::TextureType::_2D:
        default:
            *rtOut = &getRenderTarget(imageIndex);
            break;
    }

    return angle::Result::Continue;
}

angle::Result TextureMtl::syncState(const gl::Context *context,
                                    const gl::Texture::DirtyBits &dirtyBits)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    for (size_t dirtyBit : dirtyBits)
    {
        switch (dirtyBit)
        {
            case gl::Texture::DIRTY_BIT_COMPARE_MODE:
            case gl::Texture::DIRTY_BIT_COMPARE_FUNC:
                // Tell context to rebind textures so that ProgramMtl has a chance to verify
                // depth texture compare mode.
                contextMtl->invalidateCurrentTextures();
                // fall through
                OS_FALLTHROUGH;
            case gl::Texture::DIRTY_BIT_MIN_FILTER:
            case gl::Texture::DIRTY_BIT_MAG_FILTER:
            case gl::Texture::DIRTY_BIT_WRAP_S:
            case gl::Texture::DIRTY_BIT_WRAP_T:
            case gl::Texture::DIRTY_BIT_WRAP_R:
            case gl::Texture::DIRTY_BIT_MAX_ANISOTROPY:
            case gl::Texture::DIRTY_BIT_MIN_LOD:
            case gl::Texture::DIRTY_BIT_MAX_LOD:
            case gl::Texture::DIRTY_BIT_SRGB_DECODE:
            case gl::Texture::DIRTY_BIT_BORDER_COLOR:
                // Recreate sampler state
                mMetalSamplerState = nil;
                break;
            case gl::Texture::DIRTY_BIT_MAX_LEVEL:
            case gl::Texture::DIRTY_BIT_BASE_LEVEL:
                ANGLE_TRY(onBaseMaxLevelsChanged(context));
                break;
            case gl::Texture::DIRTY_BIT_SWIZZLE_RED:
            case gl::Texture::DIRTY_BIT_SWIZZLE_GREEN:
            case gl::Texture::DIRTY_BIT_SWIZZLE_BLUE:
            case gl::Texture::DIRTY_BIT_SWIZZLE_ALPHA: {
                // Recreate swizzle view.
                mNativeSwizzleSamplingView = nullptr;
            }
            break;
            default:
                break;
        }
    }

    ANGLE_TRY(ensureTextureCreated(context));
    ANGLE_TRY(ensureSamplerStateCreated(context));

    return angle::Result::Continue;
}

angle::Result TextureMtl::bindToShader(const gl::Context *context,
                                       mtl::RenderCommandEncoder *cmdEncoder,
                                       gl::ShaderType shaderType,
                                       gl::Sampler *sampler,
                                       int textureSlotIndex,
                                       int samplerSlotIndex)
{
    ASSERT(mNativeTexture);

    float minLodClamp;
    float maxLodClamp;
    id<MTLSamplerState> samplerState;

    if (!mNativeSwizzleSamplingView)
    {
#if defined(__IPHONE_13_0) || defined(__MAC_10_15)
        ContextMtl *contextMtl = mtl::GetImpl(context);

        if ((mState.getSwizzleState().swizzleRequired() || mFormat.actualAngleFormat().depthBits ||
             mFormat.swizzled) &&
            contextMtl->getDisplay()->getFeatures().hasTextureSwizzle.enabled)
        {
            const gl::InternalFormat &glInternalForamt = *mState.getBaseLevelDesc().format.info;

            MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(
                mtl::GetTextureSwizzle(OverrideSwizzleValue(
                    context, mState.getSwizzleState().swizzleRed, mFormat, glInternalForamt)),
                mtl::GetTextureSwizzle(OverrideSwizzleValue(
                    context, mState.getSwizzleState().swizzleGreen, mFormat, glInternalForamt)),
                mtl::GetTextureSwizzle(OverrideSwizzleValue(
                    context, mState.getSwizzleState().swizzleBlue, mFormat, glInternalForamt)),
                mtl::GetTextureSwizzle(OverrideSwizzleValue(
                    context, mState.getSwizzleState().swizzleAlpha, mFormat, glInternalForamt)));

            mNativeSwizzleSamplingView = mNativeTexture->createSwizzleView(swizzle);
        }
        else
#endif  // defined(__IPHONE_13_0) || defined(__MAC_10_15)
        {
            mNativeSwizzleSamplingView = mNativeTexture;
        }
    }

    if (!sampler)
    {
        samplerState = mMetalSamplerState;
        minLodClamp  = mState.getSamplerState().getMinLod();
        maxLodClamp  = mState.getSamplerState().getMaxLod();
    }
    else
    {
        SamplerMtl *samplerMtl = mtl::GetImpl(sampler);
        samplerState           = samplerMtl->getSampler(mtl::GetImpl(context));
        minLodClamp            = sampler->getSamplerState().getMinLod();
        maxLodClamp            = sampler->getSamplerState().getMaxLod();
    }

    minLodClamp = std::max(minLodClamp, 0.f);

    cmdEncoder->setTexture(shaderType, mNativeSwizzleSamplingView, textureSlotIndex);
    cmdEncoder->setSamplerState(shaderType, samplerState, minLodClamp, maxLodClamp,
                                samplerSlotIndex);

    return angle::Result::Continue;
}

angle::Result TextureMtl::redefineImage(const gl::Context *context,
                                        const gl::ImageIndex &index,
                                        const mtl::Format &mtlFormat,
                                        const gl::Extents &size,
                                        bool initEmulatedChannels)
{
    bool imageWithinLevelRange = false;

    if (isIndexWithinMinMaxLevels(index) && mNativeTexture && mNativeTexture->valid())
    {
        imageWithinLevelRange = true;
        int nativeLevel       = getNativeLevel(index);
        // Calculate the expected size for the index we are defining. If the size is different
        // from the given size, or the format is different, we are redefining the image so we
        // must release it.
        bool typeChanged = mNativeTexture->textureType() != mtl::GetTextureType(index.getType());
        if (mFormat != mtlFormat || size != mNativeTexture->size(nativeLevel) || typeChanged)
        {
            // Keep other images data if texture type hasn't been changed.
            releaseTexture(typeChanged);
        }
    }

    // Early-out on empty textures, don't create a zero-sized storage.
    if (size.empty())
    {
        return angle::Result::Continue;
    }

    ContextMtl *contextMtl = mtl::GetImpl(context);
    // Cache last defined image format:
    mFormat                      = mtlFormat;
    ImageDefinitionMtl &imageDef = getImageDefinition(index);

    // If native texture still exists, it means the size hasn't been changed, no need to create new
    // image
    if (mNativeTexture && imageDef.image && imageWithinLevelRange)
    {
        ASSERT(imageDef.image->textureType() ==
                   mtl::GetTextureType(GetTextureImageType(index.getType())) &&
               imageDef.formatID == mFormat.intendedFormatId && imageDef.image->size() == size);
    }
    else
    {
        imageDef.formatID = mtlFormat.intendedFormatId;
        // Create image to hold texture's data at this level & slice:
        switch (index.getType())
        {
            case gl::TextureType::_2D:
            case gl::TextureType::CubeMap:
                ANGLE_TRY(
                    mtl::Texture::Make2DTexture(contextMtl, mtlFormat, size.width, size.height, 1,
                                                /** renderTargetOnly */ false,
                                                /** allowFormatView */ false, &imageDef.image));
                break;
            case gl::TextureType::_2DArray:
                ANGLE_TRY(mtl::Texture::Make2DArrayTexture(
                    contextMtl, mtlFormat, size.width, size.height, 1, size.depth,
                    /** renderTargetOnly */ false,
                    /** allowFormatView */ false, &imageDef.image));
                break;
            case gl::TextureType::_3D:
                ANGLE_TRY(mtl::Texture::Make3DTexture(
                    contextMtl, mtlFormat, size.width, size.height, size.depth, 1,
                    /** renderTargetOnly */ false,
                    /** allowFormatView */ false, &imageDef.image));
                break;
            default:
                UNREACHABLE();
        }

        if (initEmulatedChannels)
        {
            // Make sure emulated channels are properly initialized
            ANGLE_TRY(checkForEmulatedChannels(context, mtlFormat, imageDef.image));
        }
    }

    // Tell context to rebind textures
    contextMtl->invalidateCurrentTextures();

    return angle::Result::Continue;
}

// If mipmaps = 0, this function will create full mipmaps texture.
angle::Result TextureMtl::setStorageImpl(const gl::Context *context,
                                         gl::TextureType type,
                                         size_t mipmaps,
                                         const mtl::Format &mtlFormat,
                                         const gl::Extents &size)
{
    // Don't need to hold old images data.
    releaseTexture(true);

    ContextMtl *contextMtl = mtl::GetImpl(context);

    // Tell context to rebind textures
    contextMtl->invalidateCurrentTextures();

    mFormat = mtlFormat;

    // Texture will be created later in ensureTextureCreated()

    return angle::Result::Continue;
}

angle::Result TextureMtl::setImageImpl(const gl::Context *context,
                                       const gl::ImageIndex &index,
                                       const gl::InternalFormat &formatInfo,
                                       const gl::Extents &size,
                                       GLenum type,
                                       const gl::PixelUnpackState &unpack,
                                       const uint8_t *pixels)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    angle::FormatID angleFormatId =
        angle::Format::InternalFormatToID(formatInfo.sizedInternalFormat);
    const mtl::Format &mtlFormat = contextMtl->getPixelFormat(angleFormatId);

    ANGLE_TRY(redefineImage(context, index, mtlFormat, size, /** initEmulatedChannels */ !pixels));

    // Early-out on empty textures, don't create a zero-sized storage.
    if (size.empty())
    {
        return angle::Result::Continue;
    }

    const gl::State &glState = context->getState();
    gl::Buffer *unpackBuffer = glState.getTargetBuffer(gl::BufferBinding::PixelUnpack);

    return setSubImageImpl(context, index, gl::Box(0, 0, 0, size.width, size.height, size.depth),
                           formatInfo, type, unpack, unpackBuffer, pixels);
}

angle::Result TextureMtl::setSubImageImpl(const gl::Context *context,
                                          const gl::ImageIndex &index,
                                          const gl::Box &area,
                                          const gl::InternalFormat &formatInfo,
                                          GLenum type,
                                          const gl::PixelUnpackState &unpack,
                                          gl::Buffer *unpackBuffer,
                                          const uint8_t *oriPixels)
{
    if (!oriPixels && !unpackBuffer)
    {
        // If source pointer is null, and there is not any PBO bound, then ignore.
        return angle::Result::Continue;
    }

    ContextMtl *contextMtl = mtl::GetImpl(context);

    angle::FormatID angleFormatId =
        angle::Format::InternalFormatToID(formatInfo.sizedInternalFormat);
    const mtl::Format &mtlSrcFormat = contextMtl->getPixelFormat(angleFormatId);

    if (mFormat.metalFormat != mtlSrcFormat.metalFormat)
    {
        ANGLE_MTL_CHECK(contextMtl, false, GL_INVALID_OPERATION);
    }

    ANGLE_TRY(ensureImageCreated(context, index));
    mtl::TextureRef &image = getImage(index);

    GLuint sourceRowPitch   = 0;
    GLuint sourceDepthPitch = 0;
    GLuint sourceSkipBytes  = 0;
    ANGLE_CHECK_GL_MATH(contextMtl, formatInfo.computeRowPitch(type, area.width, unpack.alignment,
                                                               unpack.rowLength, &sourceRowPitch));
    ANGLE_CHECK_GL_MATH(
        contextMtl, formatInfo.computeDepthPitch(area.height, unpack.imageHeight, sourceRowPitch,
                                                 &sourceDepthPitch));
    ANGLE_CHECK_GL_MATH(contextMtl,
                        formatInfo.computeSkipBytes(type, sourceRowPitch, sourceDepthPitch, unpack,
                                                    index.usesTex3D(), &sourceSkipBytes));

    // Check if partial image update is supported for this format
    if (!formatInfo.supportSubImage())
    {
        // area must be the whole mip level
        sourceRowPitch   = 0;
        gl::Extents size = image->size();
        if (area.x != 0 || area.y != 0 || area.width != size.width || area.height != size.height)
        {
            ANGLE_MTL_CHECK(contextMtl, false, GL_INVALID_OPERATION);
        }
    }

    const angle::Format &srcAngleFormat =
        angle::Format::Get(angle::Format::InternalFormatToID(formatInfo.sizedInternalFormat));

    const uint8_t *usablePixels = oriPixels + sourceSkipBytes;

    // Upload to texture
    if (index.getType() == gl::TextureType::_2DArray)
    {
        // OpenGL unifies texture array and texture 3d's box area by using z & depth as array start
        // index & length for texture array. However, Metal treats them differently. We need to
        // handle them in separate code.
        auto mtlRegion = MTLRegionMake3D(area.x, area.y, 0, area.width, area.height, 1);

        for (int slice = 0; slice < area.depth; ++slice)
        {
            int sliceIndex           = slice + area.z;
            const uint8_t *srcPixels = usablePixels + slice * sourceDepthPitch;
            ANGLE_TRY(setPerSliceSubImage(context, sliceIndex, mtlRegion, formatInfo, type,
                                          srcAngleFormat, sourceRowPitch, sourceDepthPitch,
                                          unpackBuffer, srcPixels, image));
        }
    }
    else
    {
        auto mtlRegion =
            MTLRegionMake3D(area.x, area.y, area.z, area.width, area.height, area.depth);

        ANGLE_TRY(setPerSliceSubImage(context, 0, mtlRegion, formatInfo, type, srcAngleFormat,
                                      sourceRowPitch, sourceDepthPitch, unpackBuffer, usablePixels,
                                      image));
    }

    return angle::Result::Continue;
}

angle::Result TextureMtl::setPerSliceSubImage(const gl::Context *context,
                                              int slice,
                                              const MTLRegion &mtlArea,
                                              const gl::InternalFormat &internalFormat,
                                              GLenum type,
                                              const angle::Format &pixelsAngleFormat,
                                              size_t pixelsRowPitch,
                                              size_t pixelsDepthPitch,
                                              gl::Buffer *unpackBuffer,
                                              const uint8_t *pixels,
                                              const mtl::TextureRef &image)
{
    // If source pixels are luminance or RGB8, we need to convert them to RGBA
    if (mFormat.needConversion(pixelsAngleFormat.id))
    {
        return convertAndSetPerSliceSubImage(context, slice, mtlArea, internalFormat, type,
                                             pixelsAngleFormat, pixelsRowPitch, pixelsDepthPitch,
                                             unpackBuffer, pixels, image);
    }

    // No conversion needed.
    ContextMtl *contextMtl = mtl::GetImpl(context);

    if (unpackBuffer)
    {
        uintptr_t offset = reinterpret_cast<uintptr_t>(pixels);
        if (offset % mFormat.actualAngleFormat().pixelBytes)
        {
            // offset is not divisible by pixelByte, use convertAndSetPerSliceSubImage() function.
            return convertAndSetPerSliceSubImage(context, slice, mtlArea, internalFormat, type,
                                                 pixelsAngleFormat, pixelsRowPitch,
                                                 pixelsDepthPitch, unpackBuffer, pixels, image);
        }

        BufferMtl *unpackBufferMtl = mtl::GetImpl(unpackBuffer);

        if (mFormat.hasDepthAndStencilBits())
        {
            // NOTE(hqle): packed depth & stencil texture cannot copy from buffer directly, needs
            // to split its depth & stencil data and copy separately.
            const uint8_t *clientData = unpackBufferMtl->getClientShadowCopyData(contextMtl);
            clientData += offset;
            ANGLE_TRY(UploadTextureContents(context, mFormat.actualAngleFormat(), mtlArea, 0, slice,
                                            clientData, pixelsRowPitch, pixelsDepthPitch, image));
        }
        else
        {
            // Use blit encoder to copy
            mtl::BlitCommandEncoder *blitEncoder = contextMtl->getBlitCommandEncoder();
            blitEncoder->copyBufferToTexture(
                unpackBufferMtl->getCurrentBuffer(), offset, pixelsRowPitch, pixelsDepthPitch,
                mtlArea.size, image, slice, 0, mtlArea.origin,
                mFormat.isPVRTC() ? mtl::kBlitOptionRowLinearPVRTC : MTLBlitOptionNone);
        }
    }
    else
    {
        // Upload texture data directly
        ANGLE_TRY(UploadTextureContents(context, mFormat.actualAngleFormat(), mtlArea, 0, slice,
                                        pixels, pixelsRowPitch, pixelsDepthPitch, image));
    }
    return angle::Result::Continue;
}

angle::Result TextureMtl::convertAndSetPerSliceSubImage(const gl::Context *context,
                                                        int slice,
                                                        const MTLRegion &mtlArea,
                                                        const gl::InternalFormat &internalFormat,
                                                        GLenum type,
                                                        const angle::Format &pixelsAngleFormat,
                                                        size_t pixelsRowPitch,
                                                        size_t pixelsDepthPitch,
                                                        gl::Buffer *unpackBuffer,
                                                        const uint8_t *pixels,
                                                        const mtl::TextureRef &image)
{
    ASSERT(image && image->valid());

    ContextMtl *contextMtl = mtl::GetImpl(context);

    if (unpackBuffer)
    {
        ANGLE_MTL_CHECK(contextMtl,
                        reinterpret_cast<uintptr_t>(pixels) <= std::numeric_limits<uint32_t>::max(),
                        GL_INVALID_OPERATION);

        uint32_t offset = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pixels));

        BufferMtl *unpackBufferMtl = mtl::GetImpl(unpackBuffer);
        if (!mFormat.getCaps().writable || mFormat.hasDepthOrStencilBits() ||
            mFormat.intendedAngleFormat().isBlock)
        {
            // Unsupported format, use CPU path.
            const uint8_t *clientData = unpackBufferMtl->getClientShadowCopyData(contextMtl);
            clientData += offset;
            ANGLE_TRY(convertAndSetPerSliceSubImage(context, slice, mtlArea, internalFormat, type,
                                                    pixelsAngleFormat, pixelsRowPitch,
                                                    pixelsDepthPitch, nullptr, clientData, image));
        }
        else
        {
            // Use compute shader
            mtl::CopyPixelsFromBufferParams params;
            params.buffer            = unpackBufferMtl->getCurrentBuffer();
            params.bufferStartOffset = offset;
            params.bufferRowPitch    = static_cast<uint32_t>(pixelsRowPitch);
            params.bufferDepthPitch  = static_cast<uint32_t>(pixelsDepthPitch);
            params.texture           = image;
            params.textureArea       = mtl::MTLRegionToGLBox(mtlArea);

            // If texture is not array, slice must be zero, if texture is array, mtlArea.origin.z
            // must be zero.
            // This is because this function uses Metal convention: where slice is only used for
            // array textures, and z layer of mtlArea.origin is only used for 3D textures.
            ASSERT(slice == 0 || params.textureArea.z == 0);

            // For mtl::RenderUtils we convert to OpenGL convention: z layer is used as either array
            // texture's slice or 3D texture's layer index.
            params.textureArea.z += slice;

            ANGLE_TRY(contextMtl->getDisplay()->getUtils().unpackPixelsFromBufferToTexture(
                contextMtl, pixelsAngleFormat, params));
        }
    }  // if (unpackBuffer)
    else
    {
        LoadImageFunctionInfo loadFunctionInfo = mFormat.textureLoadFunctions
                                                     ? mFormat.textureLoadFunctions(type)
                                                     : LoadImageFunctionInfo();
        const angle::Format &dstFormat = angle::Format::Get(mFormat.actualFormatId);
        const size_t dstRowPitch       = dstFormat.pixelBytes * mtlArea.size.width;
        const size_t dstDepthPitch     = dstRowPitch * mtlArea.size.height;

        // Check if original image data is compressed:
        if (mFormat.intendedAngleFormat().isBlock)
        {
            ASSERT(loadFunctionInfo.loadFunction);

            // Need to create a buffer to hold entire decompressed image.
            angle::MemoryBuffer decompressBuf;
            ANGLE_CHECK_GL_ALLOC(contextMtl,
                                 decompressBuf.resize(dstDepthPitch * mtlArea.size.depth));

            // Decompress
            loadFunctionInfo.loadFunction(
                mtlArea.size.width, mtlArea.size.height, mtlArea.size.depth, pixels, pixelsRowPitch,
                pixelsDepthPitch, decompressBuf.data(), dstRowPitch, dstDepthPitch);

            // Upload to texture
            ANGLE_TRY(UploadTextureContents(context, dstFormat, mtlArea, 0, slice,
                                            decompressBuf.data(), dstRowPitch, dstDepthPitch,
                                            image));
        }  // if (mFormat.intendedAngleFormat().isBlock)
        else
        {
            // Create scratch 2d buffer
            angle::MemoryBuffer conversionBuffer;
            ANGLE_CHECK_GL_ALLOC(contextMtl, conversionBuffer.resize(dstDepthPitch));

            // Convert layer by layer:
            MTLRegion mtlRect  = mtlArea;
            mtlRect.size.depth = 1;
            for (NSUInteger d = 0; d < mtlArea.size.depth; ++d)
            {
                mtlRect.origin.z = mtlArea.origin.z + d;

                const uint8_t *psrc = pixels + d * pixelsDepthPitch;

                // Convert pixels
                if (loadFunctionInfo.loadFunction)
                {
                    loadFunctionInfo.loadFunction(mtlRect.size.width, mtlRect.size.height, 1, psrc,
                                                  pixelsRowPitch, 0, conversionBuffer.data(),
                                                  dstRowPitch, 0);
                }
                else
                {
                    CopyImageCHROMIUM(psrc, pixelsRowPitch, pixelsAngleFormat.pixelBytes, 0,
                                      pixelsAngleFormat.pixelReadFunction, conversionBuffer.data(),
                                      dstRowPitch, dstFormat.pixelBytes, 0,
                                      dstFormat.pixelWriteFunction, internalFormat.format,
                                      dstFormat.componentType, mtlRect.size.width,
                                      mtlRect.size.height, 1, false, false, false);
                }

                // Upload to texture
                ANGLE_TRY(UploadTextureContents(context, dstFormat, mtlRect, 0, slice,
                                                conversionBuffer.data(), dstRowPitch, 0, image));
            }
        }  // if (mFormat.intendedAngleFormat().isBlock)
    }      // if (unpackBuffer)

    return angle::Result::Continue;
}

angle::Result TextureMtl::checkForEmulatedChannels(const gl::Context *context,
                                                   const mtl::Format &mtlFormat,
                                                   const mtl::TextureRef &texture)
{
    bool emulatedChannels = mtl::IsFormatEmulated(mtlFormat);

    // For emulated channels that GL texture intends to not have,
    // we need to initialize their content.
    if (emulatedChannels)
    {
        uint32_t mipmaps = texture->mipmapLevels();

        uint32_t layers = texture->cubeFacesOrArrayLength();
        for (uint32_t layer = 0; layer < layers; ++layer)
        {
            for (uint32_t mip = 0; mip < mipmaps; ++mip)
            {
                gl::ImageIndex index = GetSliceMipIndex(texture, layer, mip);

                ANGLE_TRY(mtl::InitializeTextureContents(context, texture, mFormat, index));
            }
        }
    }
    return angle::Result::Continue;
}

angle::Result TextureMtl::initializeContents(const gl::Context *context,
                                             const gl::ImageIndex &index)
{
    mtl::TextureRef &image = getImage(index);
    return mtl::InitializeTextureContents(context, image, mFormat, GetZeroLevelIndex(image));
}

angle::Result TextureMtl::copySubImageImpl(const gl::Context *context,
                                           const gl::ImageIndex &index,
                                           const gl::Offset &destOffset,
                                           const gl::Rectangle &sourceArea,
                                           const gl::InternalFormat &internalFormat,
                                           gl::Framebuffer *source)
{
    gl::Extents fbSize = source->getReadColorAttachment()->getSize();
    gl::Rectangle clippedSourceArea;
    if (!ClipRectangle(sourceArea, gl::Rectangle(0, 0, fbSize.width, fbSize.height),
                       &clippedSourceArea))
    {
        return angle::Result::Continue;
    }

    // If negative offsets are given, clippedSourceArea ensures we don't read from those offsets.
    // However, that changes the sourceOffset->destOffset mapping.  Here, destOffset is shifted by
    // the same amount as clipped to correct the error.
    const gl::Offset modifiedDestOffset(destOffset.x + clippedSourceArea.x - sourceArea.x,
                                        destOffset.y + clippedSourceArea.y - sourceArea.y, 0);

    ANGLE_TRY(ensureImageCreated(context, index));

    if (!mFormat.getCaps().isRenderable())
    {
        return copySubImageCPU(context, index, modifiedDestOffset, clippedSourceArea,
                               internalFormat, source);
    }

    // NOTE(hqle): Use compute shader.
    return copySubImageWithDraw(context, index, modifiedDestOffset, clippedSourceArea,
                                internalFormat, source);
}

angle::Result TextureMtl::copySubImageWithDraw(const gl::Context *context,
                                               const gl::ImageIndex &index,
                                               const gl::Offset &modifiedDestOffset,
                                               const gl::Rectangle &clippedSourceArea,
                                               const gl::InternalFormat &internalFormat,
                                               gl::Framebuffer *source)
{
    ContextMtl *contextMtl         = mtl::GetImpl(context);
    DisplayMtl *displayMtl         = contextMtl->getDisplay();
    FramebufferMtl *framebufferMtl = mtl::GetImpl(source);

    RenderTargetMtl *colorReadRT = framebufferMtl->getColorReadRenderTarget(context);

    if (!colorReadRT || !colorReadRT->getTexture())
    {
        // Is this an error?
        return angle::Result::Continue;
    }

    const RenderTargetMtl &imageRtt = getRenderTarget(index);

    mtl::RenderCommandEncoder *cmdEncoder = contextMtl->getRenderCommandEncoder(imageRtt);
    mtl::ColorBlitParams blitParams;

    blitParams.dstTextureSize = imageRtt.getTexture()->size(imageRtt.getLevelIndex());
    blitParams.dstRect        = gl::Rectangle(modifiedDestOffset.x, modifiedDestOffset.y,
                                       clippedSourceArea.width, clippedSourceArea.height);
    blitParams.dstScissorRect = blitParams.dstRect;

    blitParams.enabledBuffers.set(0);

    blitParams.src          = colorReadRT->getTexture();
    blitParams.srcLevel     = colorReadRT->getLevelIndex();
    blitParams.srcLayer     = colorReadRT->getLayerIndex();
    blitParams.srcRect      = clippedSourceArea;
    blitParams.srcYFlipped  = framebufferMtl->flipY();
    blitParams.dstLuminance = internalFormat.isLUMA();

    return displayMtl->getUtils().blitColorWithDraw(
        context, cmdEncoder, colorReadRT->getFormat()->actualAngleFormat(), blitParams);
}

angle::Result TextureMtl::copySubImageCPU(const gl::Context *context,
                                          const gl::ImageIndex &index,
                                          const gl::Offset &modifiedDestOffset,
                                          const gl::Rectangle &clippedSourceArea,
                                          const gl::InternalFormat &internalFormat,
                                          gl::Framebuffer *source)
{
    mtl::TextureRef &image = getImage(index);
    ASSERT(image && image->valid());

    ContextMtl *contextMtl         = mtl::GetImpl(context);
    FramebufferMtl *framebufferMtl = mtl::GetImpl(source);
    RenderTargetMtl *colorReadRT   = framebufferMtl->getColorReadRenderTarget(context);

    if (!colorReadRT || !colorReadRT->getTexture())
    {
        // Is this an error?
        return angle::Result::Continue;
    }

    const angle::Format &dstFormat = angle::Format::Get(mFormat.actualFormatId);
    const int dstRowPitch          = dstFormat.pixelBytes * clippedSourceArea.width;
    angle::MemoryBuffer conversionRow;
    ANGLE_CHECK_GL_ALLOC(contextMtl, conversionRow.resize(dstRowPitch));

    gl::Rectangle srcRowArea = gl::Rectangle(clippedSourceArea.x, 0, clippedSourceArea.width, 1);
    MTLRegion mtlDstRowArea  = MTLRegionMake2D(modifiedDestOffset.x, 0, clippedSourceArea.width, 1);
    uint32_t dstSlice        = 0;
    switch (index.getType())
    {
        case gl::TextureType::_2D:
        case gl::TextureType::CubeMap:
            dstSlice = 0;
            break;
        case gl::TextureType::_2DArray:
            ASSERT(index.hasLayer());
            dstSlice = index.getLayerIndex();
            break;
        case gl::TextureType::_3D:
            ASSERT(index.hasLayer());
            dstSlice               = 0;
            mtlDstRowArea.origin.z = index.getLayerIndex();
            break;
        default:
            UNREACHABLE();
    }

    // Copy row by row:
    for (int r = 0; r < clippedSourceArea.height; ++r)
    {
        mtlDstRowArea.origin.y = modifiedDestOffset.y + r;
        srcRowArea.y           = clippedSourceArea.y + r;

        PackPixelsParams packParams(srcRowArea, dstFormat, dstRowPitch, false, nullptr, 0);

        // Read pixels from framebuffer to memory:
        gl::Rectangle flippedSrcRowArea =
            framebufferMtl->getCorrectFlippedReadArea(context, srcRowArea);
        ANGLE_TRY(framebufferMtl->readPixelsImpl(context, flippedSrcRowArea, packParams,
                                                 framebufferMtl->getColorReadRenderTarget(context),
                                                 conversionRow.data()));

        // Upload to texture
        ANGLE_TRY(UploadTextureContents(context, dstFormat, mtlDstRowArea, 0, dstSlice,
                                        conversionRow.data(), dstRowPitch, 0, image));
    }

    return angle::Result::Continue;
}
}
