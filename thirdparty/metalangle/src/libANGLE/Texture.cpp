//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Texture.cpp: Implements the gl::Texture class. [OpenGL ES 2.0.24] section 3.7 page 63.

#include "libANGLE/Texture.h"

#include "common/mathutil.h"
#include "common/utilities.h"
#include "libANGLE/Config.h"
#include "libANGLE/Context.h"
#include "libANGLE/Image.h"
#include "libANGLE/State.h"
#include "libANGLE/Surface.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/TextureImpl.h"

namespace gl
{

namespace
{
bool IsPointSampled(const SamplerState &samplerState)
{
    return (samplerState.getMagFilter() == GL_NEAREST &&
            (samplerState.getMinFilter() == GL_NEAREST ||
             samplerState.getMinFilter() == GL_NEAREST_MIPMAP_NEAREST));
}

size_t GetImageDescIndex(TextureTarget target, size_t level)
{
    return IsCubeMapFaceTarget(target) ? (level * 6 + CubeMapTextureTargetToFaceIndex(target))
                                       : level;
}

InitState DetermineInitState(const Context *context, const uint8_t *pixels)
{
    // Can happen in tests.
    if (!context || !context->isRobustResourceInitEnabled())
        return InitState::Initialized;

    const auto &glState = context->getState();
    return (pixels == nullptr && glState.getTargetBuffer(gl::BufferBinding::PixelUnpack) == nullptr)
               ? InitState::MayNeedInit
               : InitState::Initialized;
}
}  // namespace

bool IsMipmapFiltered(const SamplerState &samplerState)
{
    switch (samplerState.getMinFilter())
    {
        case GL_NEAREST:
        case GL_LINEAR:
            return false;
        case GL_NEAREST_MIPMAP_NEAREST:
        case GL_LINEAR_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_LINEAR:
        case GL_LINEAR_MIPMAP_LINEAR:
            return true;
        default:
            UNREACHABLE();
            return false;
    }
}

SwizzleState::SwizzleState()
    : swizzleRed(GL_RED), swizzleGreen(GL_GREEN), swizzleBlue(GL_BLUE), swizzleAlpha(GL_ALPHA)
{}

SwizzleState::SwizzleState(GLenum red, GLenum green, GLenum blue, GLenum alpha)
    : swizzleRed(red), swizzleGreen(green), swizzleBlue(blue), swizzleAlpha(alpha)
{}

bool SwizzleState::swizzleRequired() const
{
    return swizzleRed != GL_RED || swizzleGreen != GL_GREEN || swizzleBlue != GL_BLUE ||
           swizzleAlpha != GL_ALPHA;
}

bool SwizzleState::operator==(const SwizzleState &other) const
{
    return swizzleRed == other.swizzleRed && swizzleGreen == other.swizzleGreen &&
           swizzleBlue == other.swizzleBlue && swizzleAlpha == other.swizzleAlpha;
}

bool SwizzleState::operator!=(const SwizzleState &other) const
{
    return !(*this == other);
}

TextureState::TextureState(TextureType type)
    : mType(type),
      mSamplerState(SamplerState::CreateDefaultForTarget(type)),
      mBaseLevel(0),
      mMaxLevel(1000),
      mDepthStencilTextureMode(GL_DEPTH_COMPONENT),
      mImmutableFormat(false),
      mImmutableLevels(0),
      mUsage(GL_NONE),
      mImageDescs((IMPLEMENTATION_MAX_TEXTURE_LEVELS + 1) * (type == TextureType::CubeMap ? 6 : 1)),
      mCropRect(0, 0, 0, 0),
      mGenerateMipmapHint(GL_FALSE),
      mInitState(InitState::MayNeedInit),
      mCachedSamplerFormat(SamplerFormat::InvalidEnum),
      mCachedSamplerCompareMode(GL_NONE),
      mCachedSamplerFormatValid(false)
{}

TextureState::~TextureState() {}

bool TextureState::swizzleRequired() const
{
    return mSwizzleState.swizzleRequired();
}

GLuint TextureState::getEffectiveBaseLevel() const
{
    if (mImmutableFormat)
    {
        // GLES 3.0.4 section 3.8.10
        return std::min(mBaseLevel, mImmutableLevels - 1);
    }
    // Some classes use the effective base level to index arrays with level data. By clamping the
    // effective base level to max levels these arrays need just one extra item to store properties
    // that should be returned for all out-of-range base level values, instead of needing special
    // handling for out-of-range base levels.
    return std::min(mBaseLevel, static_cast<GLuint>(IMPLEMENTATION_MAX_TEXTURE_LEVELS));
}

GLuint TextureState::getEffectiveMaxLevel() const
{
    if (mImmutableFormat)
    {
        // GLES 3.0.4 section 3.8.10
        GLuint clampedMaxLevel = std::max(mMaxLevel, getEffectiveBaseLevel());
        clampedMaxLevel        = std::min(clampedMaxLevel, mImmutableLevels - 1);
        return clampedMaxLevel;
    }
    return mMaxLevel;
}

GLuint TextureState::getMipmapMaxLevel() const
{
    const ImageDesc &baseImageDesc = getImageDesc(getBaseImageTarget(), getEffectiveBaseLevel());
    GLuint expectedMipLevels       = 0;
    if (mType == TextureType::_3D)
    {
        const int maxDim  = std::max(std::max(baseImageDesc.size.width, baseImageDesc.size.height),
                                    baseImageDesc.size.depth);
        expectedMipLevels = static_cast<GLuint>(log2(maxDim));
    }
    else
    {
        expectedMipLevels = static_cast<GLuint>(
            log2(std::max(baseImageDesc.size.width, baseImageDesc.size.height)));
    }

    return std::min<GLuint>(getEffectiveBaseLevel() + expectedMipLevels, getEffectiveMaxLevel());
}

bool TextureState::setBaseLevel(GLuint baseLevel)
{
    if (mBaseLevel != baseLevel)
    {
        mBaseLevel = baseLevel;
        return true;
    }
    return false;
}

bool TextureState::setMaxLevel(GLuint maxLevel)
{
    if (mMaxLevel != maxLevel)
    {
        mMaxLevel = maxLevel;
        return true;
    }

    return false;
}

// Tests for cube texture completeness. [OpenGL ES 2.0.24] section 3.7.10 page 81.
// According to [OpenGL ES 3.0.5] section 3.8.13 Texture Completeness page 160 any
// per-level checks begin at the base-level.
// For OpenGL ES2 the base level is always zero.
bool TextureState::isCubeComplete() const
{
    ASSERT(mType == TextureType::CubeMap);

    angle::EnumIterator<TextureTarget> face = kCubeMapTextureTargetMin;
    const ImageDesc &baseImageDesc          = getImageDesc(*face, getEffectiveBaseLevel());
    if (baseImageDesc.size.width == 0 || baseImageDesc.size.width != baseImageDesc.size.height)
    {
        return false;
    }

    ++face;

    for (; face != kAfterCubeMapTextureTargetMax; ++face)
    {
        const ImageDesc &faceImageDesc = getImageDesc(*face, getEffectiveBaseLevel());
        if (faceImageDesc.size.width != baseImageDesc.size.width ||
            faceImageDesc.size.height != baseImageDesc.size.height ||
            !Format::SameSized(faceImageDesc.format, baseImageDesc.format))
        {
            return false;
        }
    }

    return true;
}

const ImageDesc &TextureState::getBaseLevelDesc() const
{
    ASSERT(mType != TextureType::CubeMap || isCubeComplete());
    return getImageDesc(getBaseImageTarget(), getEffectiveBaseLevel());
}

void TextureState::setCrop(const gl::Rectangle &rect)
{
    mCropRect = rect;
}

const gl::Rectangle &TextureState::getCrop() const
{
    return mCropRect;
}

void TextureState::setGenerateMipmapHint(GLenum hint)
{
    mGenerateMipmapHint = hint;
}

GLenum TextureState::getGenerateMipmapHint() const
{
    return mGenerateMipmapHint;
}

SamplerFormat TextureState::computeRequiredSamplerFormat(const SamplerState &samplerState) const
{
    const ImageDesc &baseImageDesc = getImageDesc(getBaseImageTarget(), getEffectiveBaseLevel());
    if ((baseImageDesc.format.info->format == GL_DEPTH_COMPONENT ||
         baseImageDesc.format.info->format == GL_DEPTH_STENCIL) &&
        samplerState.getCompareMode() != GL_NONE)
    {
        return SamplerFormat::Shadow;
    }
    else
    {
        switch (baseImageDesc.format.info->componentType)
        {
            case GL_UNSIGNED_NORMALIZED:
            case GL_SIGNED_NORMALIZED:
            case GL_FLOAT:
                return SamplerFormat::Float;
            case GL_INT:
                return SamplerFormat::Signed;
            case GL_UNSIGNED_INT:
                return SamplerFormat::Unsigned;
            default:
                return SamplerFormat::InvalidEnum;
        }
    }
}

bool TextureState::computeSamplerCompleteness(const SamplerState &samplerState,
                                              const State &data) const
{
    if (mBaseLevel > mMaxLevel)
    {
        return false;
    }
    const ImageDesc &baseImageDesc = getImageDesc(getBaseImageTarget(), getEffectiveBaseLevel());
    if (baseImageDesc.size.width == 0 || baseImageDesc.size.height == 0 ||
        baseImageDesc.size.depth == 0)
    {
        return false;
    }
    // The cases where the texture is incomplete because base level is out of range should be
    // handled by the above condition.
    ASSERT(mBaseLevel < IMPLEMENTATION_MAX_TEXTURE_LEVELS || mImmutableFormat);

    if (mType == TextureType::CubeMap && baseImageDesc.size.width != baseImageDesc.size.height)
    {
        return false;
    }

    // According to es 3.1 spec, texture is justified as incomplete if sized internalformat is
    // unfilterable(table 20.11) and filter is not GL_NEAREST(8.16). The default value of minFilter
    // is NEAREST_MIPMAP_LINEAR and magFilter is LINEAR(table 20.11,). For multismaple texture,
    // filter state of multisample texture is ignored(11.1.3.3). So it shouldn't be judged as
    // incomplete texture. So, we ignore filtering for multisample texture completeness here.
    if (!IsMultisampled(mType) &&
        !baseImageDesc.format.info->filterSupport(data.getClientVersion(), data.getExtensions()) &&
        !IsPointSampled(samplerState))
    {
        return false;
    }
    bool npotSupport = data.getExtensions().textureNPOT || data.getClientMajorVersion() >= 3;
    if (!npotSupport)
    {
        if ((samplerState.getWrapS() != GL_CLAMP_TO_EDGE &&
             samplerState.getWrapS() != GL_CLAMP_TO_BORDER && !isPow2(baseImageDesc.size.width)) ||
            (samplerState.getWrapT() != GL_CLAMP_TO_EDGE &&
             samplerState.getWrapT() != GL_CLAMP_TO_BORDER && !isPow2(baseImageDesc.size.height)))
        {
            return false;
        }
    }

    if (mType != TextureType::_2DMultisample && IsMipmapFiltered(samplerState))
    {
        if (!npotSupport)
        {
            if (!isPow2(baseImageDesc.size.width) || !isPow2(baseImageDesc.size.height))
            {
                return false;
            }
        }

        if (!computeMipmapCompleteness())
        {
            return false;
        }
    }
    else
    {
        if (mType == TextureType::CubeMap && !isCubeComplete())
        {
            return false;
        }
    }

    // From GL_OES_EGL_image_external_essl3: If state is present in a sampler object bound to a
    // texture unit that would have been rejected by a call to TexParameter* for the texture bound
    // to that unit, the behavior of the implementation is as if the texture were incomplete. For
    // example, if TEXTURE_WRAP_S or TEXTURE_WRAP_T is set to anything but CLAMP_TO_EDGE on the
    // sampler object bound to a texture unit and the texture bound to that unit is an external
    // texture, the texture will be considered incomplete.
    // Sampler object state which does not affect sampling for the type of texture bound to a
    // texture unit, such as TEXTURE_WRAP_R for an external texture, does not affect completeness.
    if (mType == TextureType::External)
    {
        if (samplerState.getWrapS() != GL_CLAMP_TO_EDGE ||
            samplerState.getWrapT() != GL_CLAMP_TO_EDGE)
        {
            return false;
        }

        if (samplerState.getMinFilter() != GL_LINEAR && samplerState.getMinFilter() != GL_NEAREST)
        {
            return false;
        }
    }

    // OpenGLES 3.0.2 spec section 3.8.13 states that a texture is not mipmap complete if:
    // The internalformat specified for the texture arrays is a sized internal depth or
    // depth and stencil format (see table 3.13), the value of TEXTURE_COMPARE_-
    // MODE is NONE, and either the magnification filter is not NEAREST or the mini-
    // fication filter is neither NEAREST nor NEAREST_MIPMAP_NEAREST.
    if (!IsMultisampled(mType) && baseImageDesc.format.info->depthBits > 0 &&
        data.getClientMajorVersion() >= 3)
    {
        // Note: we restrict this validation to sized types. For the OES_depth_textures
        // extension, due to some underspecification problems, we must allow linear filtering
        // for legacy compatibility with WebGL 1.
        // See http://crbug.com/649200
        if (samplerState.getCompareMode() == GL_NONE && baseImageDesc.format.info->sized)
        {
            if ((samplerState.getMinFilter() != GL_NEAREST &&
                 samplerState.getMinFilter() != GL_NEAREST_MIPMAP_NEAREST) ||
                samplerState.getMagFilter() != GL_NEAREST)
            {
                return false;
            }
        }
    }

    // OpenGLES 3.1 spec section 8.16 states that a texture is not mipmap complete if:
    // The internalformat specified for the texture is DEPTH_STENCIL format, the value of
    // DEPTH_STENCIL_TEXTURE_MODE is STENCIL_INDEX, and either the magnification filter is
    // not NEAREST or the minification filter is neither NEAREST nor NEAREST_MIPMAP_NEAREST.
    // However, the ES 3.1 spec differs from the statement above, because it is incorrect.
    // See the issue at https://github.com/KhronosGroup/OpenGL-API/issues/33.
    // For multismaple texture, filter state of multisample texture is ignored(11.1.3.3).
    // So it shouldn't be judged as incomplete texture. So, we ignore filtering for multisample
    // texture completeness here.
    if (!IsMultisampled(mType) && baseImageDesc.format.info->depthBits > 0 &&
        mDepthStencilTextureMode == GL_STENCIL_INDEX)
    {
        if ((samplerState.getMinFilter() != GL_NEAREST &&
             samplerState.getMinFilter() != GL_NEAREST_MIPMAP_NEAREST) ||
            samplerState.getMagFilter() != GL_NEAREST)
        {
            return false;
        }
    }

    return true;
}

bool TextureState::computeMipmapCompleteness() const
{
    const GLuint maxLevel = getMipmapMaxLevel();

    for (GLuint level = getEffectiveBaseLevel(); level <= maxLevel; level++)
    {
        if (mType == TextureType::CubeMap)
        {
            for (TextureTarget face : AllCubeFaceTextureTargets())
            {
                if (!computeLevelCompleteness(face, level))
                {
                    return false;
                }
            }
        }
        else
        {
            if (!computeLevelCompleteness(NonCubeTextureTypeToTarget(mType), level))
            {
                return false;
            }
        }
    }

    return true;
}

bool TextureState::computeLevelCompleteness(TextureTarget target, size_t level) const
{
    ASSERT(level < IMPLEMENTATION_MAX_TEXTURE_LEVELS);

    if (mImmutableFormat)
    {
        return true;
    }

    const ImageDesc &baseImageDesc = getImageDesc(getBaseImageTarget(), getEffectiveBaseLevel());
    if (baseImageDesc.size.width == 0 || baseImageDesc.size.height == 0 ||
        baseImageDesc.size.depth == 0)
    {
        return false;
    }

    const ImageDesc &levelImageDesc = getImageDesc(target, level);
    if (levelImageDesc.size.width == 0 || levelImageDesc.size.height == 0 ||
        levelImageDesc.size.depth == 0)
    {
        return false;
    }

    if (!Format::SameSized(levelImageDesc.format, baseImageDesc.format))
    {
        return false;
    }

    ASSERT(level >= getEffectiveBaseLevel());
    const size_t relativeLevel = level - getEffectiveBaseLevel();
    if (levelImageDesc.size.width != std::max(1, baseImageDesc.size.width >> relativeLevel))
    {
        return false;
    }

    if (levelImageDesc.size.height != std::max(1, baseImageDesc.size.height >> relativeLevel))
    {
        return false;
    }

    if (mType == TextureType::_3D)
    {
        if (levelImageDesc.size.depth != std::max(1, baseImageDesc.size.depth >> relativeLevel))
        {
            return false;
        }
    }
    else if (mType == TextureType::_2DArray)
    {
        if (levelImageDesc.size.depth != baseImageDesc.size.depth)
        {
            return false;
        }
    }

    return true;
}

TextureTarget TextureState::getBaseImageTarget() const
{
    return mType == TextureType::CubeMap ? kCubeMapTextureTargetMin
                                         : NonCubeTextureTypeToTarget(mType);
}

ImageDesc::ImageDesc()
    : ImageDesc(Extents(0, 0, 0), Format::Invalid(), 0, GL_TRUE, InitState::Initialized)
{}

ImageDesc::ImageDesc(const Extents &size, const Format &format, const InitState initState)
    : size(size), format(format), samples(0), fixedSampleLocations(GL_TRUE), initState(initState)
{}

ImageDesc::ImageDesc(const Extents &size,
                     const Format &format,
                     const GLsizei samples,
                     const bool fixedSampleLocations,
                     const InitState initState)
    : size(size),
      format(format),
      samples(samples),
      fixedSampleLocations(fixedSampleLocations),
      initState(initState)
{}

GLint ImageDesc::getMemorySize() const
{
    // Assume allocated size is around width * height * depth * samples * pixelBytes
    angle::CheckedNumeric<GLint> levelSize = 1;
    levelSize *= format.info->pixelBytes;
    levelSize *= size.width;
    levelSize *= size.height;
    levelSize *= size.depth;
    levelSize *= std::max(samples, 1);
    return levelSize.ValueOrDefault(std::numeric_limits<GLint>::max());
}

const ImageDesc &TextureState::getImageDesc(TextureTarget target, size_t level) const
{
    size_t descIndex = GetImageDescIndex(target, level);
    ASSERT(descIndex < mImageDescs.size());
    return mImageDescs[descIndex];
}

void TextureState::setImageDesc(TextureTarget target, size_t level, const ImageDesc &desc)
{
    size_t descIndex = GetImageDescIndex(target, level);
    ASSERT(descIndex < mImageDescs.size());
    mImageDescs[descIndex] = desc;
    if (desc.initState == InitState::MayNeedInit)
    {
        mInitState = InitState::MayNeedInit;
    }
}

// Note that an ImageIndex that represents an entire level of a cube map corresponds to 6
// ImageDescs, so if the cube map is cube complete, we return the ImageDesc of the first cube
// face, and we don't allow using this function when the cube map is not cube complete.
const ImageDesc &TextureState::getImageDesc(const ImageIndex &imageIndex) const
{
    if (imageIndex.isEntireLevelCubeMap())
    {
        ASSERT(isCubeComplete());
        const GLint levelIndex = imageIndex.getLevelIndex();
        return getImageDesc(kCubeMapTextureTargetMin, levelIndex);
    }

    return getImageDesc(imageIndex.getTarget(), imageIndex.getLevelIndex());
}

void TextureState::setImageDescChain(GLuint baseLevel,
                                     GLuint maxLevel,
                                     Extents baseSize,
                                     const Format &format,
                                     InitState initState)
{
    for (GLuint level = baseLevel; level <= maxLevel; level++)
    {
        int relativeLevel = (level - baseLevel);
        Extents levelSize(std::max<int>(baseSize.width >> relativeLevel, 1),
                          std::max<int>(baseSize.height >> relativeLevel, 1),
                          (mType == TextureType::_2DArray)
                              ? baseSize.depth
                              : std::max<int>(baseSize.depth >> relativeLevel, 1));
        ImageDesc levelInfo(levelSize, format, initState);

        if (mType == TextureType::CubeMap)
        {
            for (TextureTarget face : AllCubeFaceTextureTargets())
            {
                setImageDesc(face, level, levelInfo);
            }
        }
        else
        {
            setImageDesc(NonCubeTextureTypeToTarget(mType), level, levelInfo);
        }
    }
}

void TextureState::setImageDescChainMultisample(Extents baseSize,
                                                const Format &format,
                                                GLsizei samples,
                                                bool fixedSampleLocations,
                                                InitState initState)
{
    ASSERT(mType == TextureType::_2DMultisample || mType == TextureType::_2DMultisampleArray);
    ImageDesc levelInfo(baseSize, format, samples, fixedSampleLocations, initState);
    setImageDesc(NonCubeTextureTypeToTarget(mType), 0, levelInfo);
}

void TextureState::clearImageDesc(TextureTarget target, size_t level)
{
    setImageDesc(target, level, ImageDesc());
}

void TextureState::clearImageDescs()
{
    for (size_t descIndex = 0; descIndex < mImageDescs.size(); descIndex++)
    {
        mImageDescs[descIndex] = ImageDesc();
    }
}

Texture::Texture(rx::GLImplFactory *factory, TextureID id, TextureType type)
    : RefCountObject(id),
      mState(type),
      mTexture(factory->createTexture(mState)),
      mImplObserver(this, rx::kTextureImageImplObserverMessageIndex),
      mLabel(),
      mBoundSurface(nullptr),
      mBoundStream(nullptr)
{
    mImplObserver.bind(mTexture);

    // Initially assume the implementation is dirty.
    mDirtyBits.set(DIRTY_BIT_IMPLEMENTATION);
}

void Texture::onDestroy(const Context *context)
{
    if (mBoundSurface)
    {
        ANGLE_SWALLOW_ERR(mBoundSurface->releaseTexImage(context, EGL_BACK_BUFFER));
        mBoundSurface = nullptr;
    }
    if (mBoundStream)
    {
        mBoundStream->releaseTextures();
        mBoundStream = nullptr;
    }

    (void)(orphanImages(context));

    if (mTexture)
    {
        mTexture->onDestroy(context);
    }
}

Texture::~Texture()
{
    SafeDelete(mTexture);
}

void Texture::setLabel(const Context *context, const std::string &label)
{
    mLabel = label;
    signalDirtyState(DIRTY_BIT_LABEL);
}

const std::string &Texture::getLabel() const
{
    return mLabel;
}

void Texture::setSwizzleRed(const Context *context, GLenum swizzleRed)
{
    mState.mSwizzleState.swizzleRed = swizzleRed;
    signalDirtyState(DIRTY_BIT_SWIZZLE_RED);
}

GLenum Texture::getSwizzleRed() const
{
    return mState.mSwizzleState.swizzleRed;
}

void Texture::setSwizzleGreen(const Context *context, GLenum swizzleGreen)
{
    mState.mSwizzleState.swizzleGreen = swizzleGreen;
    signalDirtyState(DIRTY_BIT_SWIZZLE_GREEN);
}

GLenum Texture::getSwizzleGreen() const
{
    return mState.mSwizzleState.swizzleGreen;
}

void Texture::setSwizzleBlue(const Context *context, GLenum swizzleBlue)
{
    mState.mSwizzleState.swizzleBlue = swizzleBlue;
    signalDirtyState(DIRTY_BIT_SWIZZLE_BLUE);
}

GLenum Texture::getSwizzleBlue() const
{
    return mState.mSwizzleState.swizzleBlue;
}

void Texture::setSwizzleAlpha(const Context *context, GLenum swizzleAlpha)
{
    mState.mSwizzleState.swizzleAlpha = swizzleAlpha;
    signalDirtyState(DIRTY_BIT_SWIZZLE_ALPHA);
}

GLenum Texture::getSwizzleAlpha() const
{
    return mState.mSwizzleState.swizzleAlpha;
}

void Texture::setMinFilter(const Context *context, GLenum minFilter)
{
    mState.mSamplerState.setMinFilter(minFilter);
    signalDirtyState(DIRTY_BIT_MIN_FILTER);
}

GLenum Texture::getMinFilter() const
{
    return mState.mSamplerState.getMinFilter();
}

void Texture::setMagFilter(const Context *context, GLenum magFilter)
{
    mState.mSamplerState.setMagFilter(magFilter);
    signalDirtyState(DIRTY_BIT_MAG_FILTER);
}

GLenum Texture::getMagFilter() const
{
    return mState.mSamplerState.getMagFilter();
}

void Texture::setWrapS(const Context *context, GLenum wrapS)
{
    mState.mSamplerState.setWrapS(wrapS);
    signalDirtyState(DIRTY_BIT_WRAP_S);
}

GLenum Texture::getWrapS() const
{
    return mState.mSamplerState.getWrapS();
}

void Texture::setWrapT(const Context *context, GLenum wrapT)
{
    mState.mSamplerState.setWrapT(wrapT);
    signalDirtyState(DIRTY_BIT_WRAP_T);
}

GLenum Texture::getWrapT() const
{
    return mState.mSamplerState.getWrapT();
}

void Texture::setWrapR(const Context *context, GLenum wrapR)
{
    mState.mSamplerState.setWrapR(wrapR);
    signalDirtyState(DIRTY_BIT_WRAP_R);
}

GLenum Texture::getWrapR() const
{
    return mState.mSamplerState.getWrapR();
}

void Texture::setMaxAnisotropy(const Context *context, float maxAnisotropy)
{
    mState.mSamplerState.setMaxAnisotropy(maxAnisotropy);
    signalDirtyState(DIRTY_BIT_MAX_ANISOTROPY);
}

float Texture::getMaxAnisotropy() const
{
    return mState.mSamplerState.getMaxAnisotropy();
}

void Texture::setMinLod(const Context *context, GLfloat minLod)
{
    mState.mSamplerState.setMinLod(minLod);
    signalDirtyState(DIRTY_BIT_MIN_LOD);
}

GLfloat Texture::getMinLod() const
{
    return mState.mSamplerState.getMinLod();
}

void Texture::setMaxLod(const Context *context, GLfloat maxLod)
{
    mState.mSamplerState.setMaxLod(maxLod);
    signalDirtyState(DIRTY_BIT_MAX_LOD);
}

GLfloat Texture::getMaxLod() const
{
    return mState.mSamplerState.getMaxLod();
}

void Texture::setCompareMode(const Context *context, GLenum compareMode)
{
    mState.mSamplerState.setCompareMode(compareMode);
    signalDirtyState(DIRTY_BIT_COMPARE_MODE);
}

GLenum Texture::getCompareMode() const
{
    return mState.mSamplerState.getCompareMode();
}

void Texture::setCompareFunc(const Context *context, GLenum compareFunc)
{
    mState.mSamplerState.setCompareFunc(compareFunc);
    signalDirtyState(DIRTY_BIT_COMPARE_FUNC);
}

GLenum Texture::getCompareFunc() const
{
    return mState.mSamplerState.getCompareFunc();
}

void Texture::setSRGBDecode(const Context *context, GLenum sRGBDecode)
{
    mState.mSamplerState.setSRGBDecode(sRGBDecode);
    signalDirtyState(DIRTY_BIT_SRGB_DECODE);
}

GLenum Texture::getSRGBDecode() const
{
    return mState.mSamplerState.getSRGBDecode();
}

const SamplerState &Texture::getSamplerState() const
{
    return mState.mSamplerState;
}

angle::Result Texture::setBaseLevel(const Context *context, GLuint baseLevel)
{
    if (mState.setBaseLevel(baseLevel))
    {
        ANGLE_TRY(mTexture->setBaseLevel(context, mState.getEffectiveBaseLevel()));
        signalDirtyState(DIRTY_BIT_BASE_LEVEL);
    }

    return angle::Result::Continue;
}

GLuint Texture::getBaseLevel() const
{
    return mState.mBaseLevel;
}

void Texture::setMaxLevel(const Context *context, GLuint maxLevel)
{
    if (mState.setMaxLevel(maxLevel))
    {
        signalDirtyState(DIRTY_BIT_MAX_LEVEL);
    }
}

GLuint Texture::getMaxLevel() const
{
    return mState.mMaxLevel;
}

void Texture::setDepthStencilTextureMode(const Context *context, GLenum mode)
{
    if (mState.mDepthStencilTextureMode != mode)
    {
        mState.mDepthStencilTextureMode = mode;
        signalDirtyState(DIRTY_BIT_DEPTH_STENCIL_TEXTURE_MODE);
    }
}

GLenum Texture::getDepthStencilTextureMode() const
{
    return mState.mDepthStencilTextureMode;
}

bool Texture::getImmutableFormat() const
{
    return mState.mImmutableFormat;
}

GLuint Texture::getImmutableLevels() const
{
    return mState.mImmutableLevels;
}

void Texture::setUsage(const Context *context, GLenum usage)
{
    mState.mUsage = usage;
    signalDirtyState(DIRTY_BIT_USAGE);
}

GLenum Texture::getUsage() const
{
    return mState.mUsage;
}

const TextureState &Texture::getTextureState() const
{
    return mState;
}

size_t Texture::getWidth(TextureTarget target, size_t level) const
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    return mState.getImageDesc(target, level).size.width;
}

size_t Texture::getHeight(TextureTarget target, size_t level) const
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    return mState.getImageDesc(target, level).size.height;
}

size_t Texture::getDepth(TextureTarget target, size_t level) const
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    return mState.getImageDesc(target, level).size.depth;
}

const Format &Texture::getFormat(TextureTarget target, size_t level) const
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    return mState.getImageDesc(target, level).format;
}

GLsizei Texture::getSamples(TextureTarget target, size_t level) const
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    return mState.getImageDesc(target, level).samples;
}

bool Texture::getFixedSampleLocations(TextureTarget target, size_t level) const
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    return mState.getImageDesc(target, level).fixedSampleLocations;
}

GLuint Texture::getMipmapMaxLevel() const
{
    return mState.getMipmapMaxLevel();
}

bool Texture::isMipmapComplete() const
{
    return mState.computeMipmapCompleteness();
}

egl::Surface *Texture::getBoundSurface() const
{
    return mBoundSurface;
}

egl::Stream *Texture::getBoundStream() const
{
    return mBoundStream;
}

GLint Texture::getMemorySize() const
{
    GLint implSize = mTexture->getMemorySize();
    if (implSize > 0)
    {
        return implSize;
    }

    angle::CheckedNumeric<GLint> size = 0;
    for (const ImageDesc &imageDesc : mState.mImageDescs)
    {
        size += imageDesc.getMemorySize();
    }
    return size.ValueOrDefault(std::numeric_limits<GLint>::max());
}

GLint Texture::getLevelMemorySize(TextureTarget target, GLint level) const
{
    GLint implSize = mTexture->getLevelMemorySize(target, level);
    if (implSize > 0)
    {
        return implSize;
    }

    return mState.getImageDesc(target, level).getMemorySize();
}

void Texture::signalDirtyStorage(InitState initState)
{
    mState.mInitState = initState;
    invalidateCompletenessCache();
    mState.mCachedSamplerFormatValid = false;
    onStateChange(angle::SubjectMessage::SubjectChanged);
}

void Texture::signalDirtyState(size_t dirtyBit)
{
    mDirtyBits.set(dirtyBit);
    invalidateCompletenessCache();
    mState.mCachedSamplerFormatValid = false;
    onStateChange(angle::SubjectMessage::DirtyBitsFlagged);
}

angle::Result Texture::setImage(Context *context,
                                const PixelUnpackState &unpackState,
                                TextureTarget target,
                                GLint level,
                                GLenum internalFormat,
                                const Extents &size,
                                GLenum format,
                                GLenum type,
                                const uint8_t *pixels)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, size.depth);

    ANGLE_TRY(mTexture->setImage(context, index, internalFormat, size, format, type, unpackState,
                                 pixels));

    InitState initState = DetermineInitState(context, pixels);
    mState.setImageDesc(target, level, ImageDesc(size, Format(internalFormat, type), initState));

    ANGLE_TRY(handleMipmapGenerationHint(context, level));

    signalDirtyStorage(initState);

    return angle::Result::Continue;
}

angle::Result Texture::setSubImage(Context *context,
                                   const PixelUnpackState &unpackState,
                                   Buffer *unpackBuffer,
                                   TextureTarget target,
                                   GLint level,
                                   const Box &area,
                                   GLenum format,
                                   GLenum type,
                                   const uint8_t *pixels)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    ANGLE_TRY(ensureSubImageInitialized(context, target, level, area));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, area.depth);

    ANGLE_TRY(mTexture->setSubImage(context, index, area, format, type, unpackState, unpackBuffer,
                                    pixels));

    ANGLE_TRY(handleMipmapGenerationHint(context, level));

    onStateChange(angle::SubjectMessage::ContentsChanged);

    return angle::Result::Continue;
}

angle::Result Texture::setCompressedImage(Context *context,
                                          const PixelUnpackState &unpackState,
                                          TextureTarget target,
                                          GLint level,
                                          GLenum internalFormat,
                                          const Extents &size,
                                          size_t imageSize,
                                          const uint8_t *pixels)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, size.depth);

    ANGLE_TRY(mTexture->setCompressedImage(context, index, internalFormat, size, unpackState,
                                           imageSize, pixels));

    InitState initState = DetermineInitState(context, pixels);
    mState.setImageDesc(target, level, ImageDesc(size, Format(internalFormat), initState));
    signalDirtyStorage(initState);

    return angle::Result::Continue;
}

angle::Result Texture::setCompressedSubImage(const Context *context,
                                             const PixelUnpackState &unpackState,
                                             TextureTarget target,
                                             GLint level,
                                             const Box &area,
                                             GLenum format,
                                             size_t imageSize,
                                             const uint8_t *pixels)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    ANGLE_TRY(ensureSubImageInitialized(context, target, level, area));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, area.depth);

    ANGLE_TRY(mTexture->setCompressedSubImage(context, index, area, format, unpackState, imageSize,
                                              pixels));

    onStateChange(angle::SubjectMessage::ContentsChanged);

    return angle::Result::Continue;
}

angle::Result Texture::copyImage(Context *context,
                                 TextureTarget target,
                                 GLint level,
                                 const Rectangle &sourceArea,
                                 GLenum internalFormat,
                                 Framebuffer *source)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    // Use the source FBO size as the init image area.
    Box destBox(0, 0, 0, sourceArea.width, sourceArea.height, 1);
    ANGLE_TRY(ensureSubImageInitialized(context, target, level, destBox));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, 1);

    ANGLE_TRY(mTexture->copyImage(context, index, sourceArea, internalFormat, source));

    const InternalFormat &internalFormatInfo =
        GetInternalFormatInfo(internalFormat, GL_UNSIGNED_BYTE);

    mState.setImageDesc(target, level,
                        ImageDesc(Extents(sourceArea.width, sourceArea.height, 1),
                                  Format(internalFormatInfo), InitState::Initialized));

    ANGLE_TRY(handleMipmapGenerationHint(context, level));

    // We need to initialize this texture only if the source attachment is not initialized.
    signalDirtyStorage(InitState::Initialized);

    return angle::Result::Continue;
}

angle::Result Texture::copySubImage(Context *context,
                                    const ImageIndex &index,
                                    const Offset &destOffset,
                                    const Rectangle &sourceArea,
                                    Framebuffer *source)
{
    ASSERT(TextureTargetToType(index.getTarget()) == mState.mType);

    Box destBox(destOffset.x, destOffset.y, destOffset.z, sourceArea.width, sourceArea.height, 1);
    ANGLE_TRY(
        ensureSubImageInitialized(context, index.getTarget(), index.getLevelIndex(), destBox));

    ANGLE_TRY(mTexture->copySubImage(context, index, destOffset, sourceArea, source));
    ANGLE_TRY(handleMipmapGenerationHint(context, index.getLevelIndex()));

    onStateChange(angle::SubjectMessage::ContentsChanged);

    return angle::Result::Continue;
}

angle::Result Texture::copyTexture(Context *context,
                                   TextureTarget target,
                                   GLint level,
                                   GLenum internalFormat,
                                   GLenum type,
                                   GLint sourceLevel,
                                   bool unpackFlipY,
                                   bool unpackPremultiplyAlpha,
                                   bool unpackUnmultiplyAlpha,
                                   Texture *source)
{
    ASSERT(TextureTargetToType(target) == mState.mType);
    ASSERT(source->getType() != TextureType::CubeMap);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    // Initialize source texture.
    // Note: we don't have a way to notify which portions of the image changed currently.
    ANGLE_TRY(source->ensureInitialized(context));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, ImageIndex::kEntireLevel);

    ANGLE_TRY(mTexture->copyTexture(context, index, internalFormat, type, sourceLevel, unpackFlipY,
                                    unpackPremultiplyAlpha, unpackUnmultiplyAlpha, source));

    const auto &sourceDesc =
        source->mState.getImageDesc(NonCubeTextureTypeToTarget(source->getType()), 0);
    const InternalFormat &internalFormatInfo = GetInternalFormatInfo(internalFormat, type);
    mState.setImageDesc(
        target, level,
        ImageDesc(sourceDesc.size, Format(internalFormatInfo), InitState::Initialized));

    signalDirtyStorage(InitState::Initialized);

    return angle::Result::Continue;
}

angle::Result Texture::copySubTexture(const Context *context,
                                      TextureTarget target,
                                      GLint level,
                                      const Offset &destOffset,
                                      GLint sourceLevel,
                                      const Box &sourceBox,
                                      bool unpackFlipY,
                                      bool unpackPremultiplyAlpha,
                                      bool unpackUnmultiplyAlpha,
                                      Texture *source)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    // Ensure source is initialized.
    ANGLE_TRY(source->ensureInitialized(context));

    Box destBox(destOffset.x, destOffset.y, destOffset.z, sourceBox.width, sourceBox.height,
                sourceBox.depth);
    ANGLE_TRY(ensureSubImageInitialized(context, target, level, destBox));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, sourceBox.depth);

    ANGLE_TRY(mTexture->copySubTexture(context, index, destOffset, sourceLevel, sourceBox,
                                       unpackFlipY, unpackPremultiplyAlpha, unpackUnmultiplyAlpha,
                                       source));

    onStateChange(angle::SubjectMessage::ContentsChanged);

    return angle::Result::Continue;
}

angle::Result Texture::copyCompressedTexture(Context *context, const Texture *source)
{
    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ANGLE_TRY(mTexture->copyCompressedTexture(context, source));

    ASSERT(source->getType() != TextureType::CubeMap && getType() != TextureType::CubeMap);
    const auto &sourceDesc =
        source->mState.getImageDesc(NonCubeTextureTypeToTarget(source->getType()), 0);
    mState.setImageDesc(NonCubeTextureTypeToTarget(getType()), 0, sourceDesc);

    return angle::Result::Continue;
}

angle::Result Texture::setStorage(Context *context,
                                  TextureType type,
                                  GLsizei levels,
                                  GLenum internalFormat,
                                  const Extents &size)
{
    ASSERT(type == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    mState.mImmutableFormat = true;
    mState.mImmutableLevels = static_cast<GLuint>(levels);

    ANGLE_TRY(mTexture->setStorage(context, type, levels, internalFormat, size));

    mState.clearImageDescs();
    mState.setImageDescChain(0, static_cast<GLuint>(levels - 1), size, Format(internalFormat),
                             InitState::MayNeedInit);

    // Changing the texture to immutable can trigger a change in the base and max levels:
    // GLES 3.0.4 section 3.8.10 pg 158:
    // "For immutable-format textures, levelbase is clamped to the range[0;levels],levelmax is then
    // clamped to the range[levelbase;levels].
    mDirtyBits.set(DIRTY_BIT_BASE_LEVEL);
    mDirtyBits.set(DIRTY_BIT_MAX_LEVEL);

    signalDirtyStorage(InitState::MayNeedInit);

    return angle::Result::Continue;
}

angle::Result Texture::setImageExternal(Context *context,
                                        TextureTarget target,
                                        GLint level,
                                        GLenum internalFormat,
                                        const Extents &size,
                                        GLenum format,
                                        GLenum type)
{
    ASSERT(TextureTargetToType(target) == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ImageIndex index = ImageIndex::MakeFromTarget(target, level, size.depth);

    ANGLE_TRY(mTexture->setImageExternal(context, index, internalFormat, size, format, type));

    InitState initState = InitState::Initialized;
    mState.setImageDesc(target, level, ImageDesc(size, Format(internalFormat, type), initState));

    ANGLE_TRY(handleMipmapGenerationHint(context, level));

    signalDirtyStorage(initState);

    return angle::Result::Continue;
}

angle::Result Texture::setStorageMultisample(Context *context,
                                             TextureType type,
                                             GLsizei samples,
                                             GLint internalFormat,
                                             const Extents &size,
                                             bool fixedSampleLocations)
{
    ASSERT(type == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ANGLE_TRY(mTexture->setStorageMultisample(context, type, samples, internalFormat, size,
                                              fixedSampleLocations));

    mState.mImmutableFormat = true;
    mState.mImmutableLevels = static_cast<GLuint>(1);
    mState.clearImageDescs();
    mState.setImageDescChainMultisample(size, Format(internalFormat), samples, fixedSampleLocations,
                                        InitState::MayNeedInit);

    signalDirtyStorage(InitState::MayNeedInit);

    return angle::Result::Continue;
}

angle::Result Texture::setStorageExternalMemory(Context *context,
                                                TextureType type,
                                                GLsizei levels,
                                                GLenum internalFormat,
                                                const Extents &size,
                                                MemoryObject *memoryObject,
                                                GLuint64 offset)
{
    ASSERT(type == mState.mType);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ANGLE_TRY(mTexture->setStorageExternalMemory(context, type, levels, internalFormat, size,
                                                 memoryObject, offset));

    mState.mImmutableFormat = true;
    mState.mImmutableLevels = static_cast<GLuint>(levels);
    mState.clearImageDescs();
    mState.setImageDescChain(0, static_cast<GLuint>(levels - 1), size, Format(internalFormat),
                             InitState::MayNeedInit);

    // Changing the texture to immutable can trigger a change in the base and max levels:
    // GLES 3.0.4 section 3.8.10 pg 158:
    // "For immutable-format textures, levelbase is clamped to the range[0;levels],levelmax is then
    // clamped to the range[levelbase;levels].
    mDirtyBits.set(DIRTY_BIT_BASE_LEVEL);
    mDirtyBits.set(DIRTY_BIT_MAX_LEVEL);

    signalDirtyStorage(InitState::Initialized);

    return angle::Result::Continue;
}

angle::Result Texture::generateMipmap(Context *context)
{
    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));

    // EGL_KHR_gl_image states that images are only orphaned when generating mipmaps if the texture
    // is not mip complete.
    if (!isMipmapComplete())
    {
        ANGLE_TRY(orphanImages(context));
    }

    const GLuint baseLevel = mState.getEffectiveBaseLevel();
    const GLuint maxLevel  = mState.getMipmapMaxLevel();

    if (maxLevel <= baseLevel)
    {
        return angle::Result::Continue;
    }

    if (hasAnyDirtyBit())
    {
        ANGLE_TRY(syncState(context));
    }

    // Clear the base image(s) immediately if needed
    if (context->isRobustResourceInitEnabled())
    {
        ImageIndexIterator it =
            ImageIndexIterator::MakeGeneric(mState.mType, baseLevel, baseLevel + 1,
                                            ImageIndex::kEntireLevel, ImageIndex::kEntireLevel);
        while (it.hasNext())
        {
            const ImageIndex index = it.next();
            const ImageDesc &desc  = mState.getImageDesc(index.getTarget(), index.getLevelIndex());

            if (desc.initState == InitState::MayNeedInit)
            {
                ANGLE_TRY(initializeContents(context, index));
            }
        }
    }

    ANGLE_TRY(mTexture->generateMipmap(context));

    // Propagate the format and size of the bsae mip to the smaller ones. Cube maps are guaranteed
    // to have faces of the same size and format so any faces can be picked.
    const ImageDesc &baseImageInfo = mState.getImageDesc(mState.getBaseImageTarget(), baseLevel);
    mState.setImageDescChain(baseLevel, maxLevel, baseImageInfo.size, baseImageInfo.format,
                             InitState::Initialized);

    signalDirtyStorage(InitState::Initialized);

    return angle::Result::Continue;
}

angle::Result Texture::bindTexImageFromSurface(Context *context, egl::Surface *surface)
{
    ASSERT(surface);

    if (mBoundSurface)
    {
        ANGLE_TRY(releaseTexImageFromSurface(context));
    }

    ANGLE_TRY(mTexture->bindTexImage(context, surface));
    mBoundSurface = surface;

    // Set the image info to the size and format of the surface
    ASSERT(mState.mType == TextureType::_2D || mState.mType == TextureType::Rectangle);
    Extents size(surface->getWidth(), surface->getHeight(), 1);
    ImageDesc desc(size, surface->getBindTexImageFormat(), InitState::Initialized);
    mState.setImageDesc(NonCubeTextureTypeToTarget(mState.mType), 0, desc);
    signalDirtyStorage(InitState::Initialized);
    return angle::Result::Continue;
}

angle::Result Texture::releaseTexImageFromSurface(const Context *context)
{
    ASSERT(mBoundSurface);
    mBoundSurface = nullptr;
    ANGLE_TRY(mTexture->releaseTexImage(context));

    // Erase the image info for level 0
    ASSERT(mState.mType == TextureType::_2D || mState.mType == TextureType::Rectangle);
    mState.clearImageDesc(NonCubeTextureTypeToTarget(mState.mType), 0);
    signalDirtyStorage(InitState::Initialized);
    return angle::Result::Continue;
}

void Texture::bindStream(egl::Stream *stream)
{
    ASSERT(stream);

    // It should not be possible to bind a texture already bound to another stream
    ASSERT(mBoundStream == nullptr);

    mBoundStream = stream;

    ASSERT(mState.mType == TextureType::External);
}

void Texture::releaseStream()
{
    ASSERT(mBoundStream);
    mBoundStream = nullptr;
}

angle::Result Texture::acquireImageFromStream(const Context *context,
                                              const egl::Stream::GLTextureDescription &desc)
{
    ASSERT(mBoundStream != nullptr);
    ANGLE_TRY(mTexture->setImageExternal(context, mState.mType, mBoundStream, desc));

    Extents size(desc.width, desc.height, 1);
    mState.setImageDesc(NonCubeTextureTypeToTarget(mState.mType), 0,
                        ImageDesc(size, Format(desc.internalFormat), InitState::Initialized));
    signalDirtyStorage(InitState::Initialized);
    return angle::Result::Continue;
}

angle::Result Texture::releaseImageFromStream(const Context *context)
{
    ASSERT(mBoundStream != nullptr);
    ANGLE_TRY(mTexture->setImageExternal(context, mState.mType, nullptr,
                                         egl::Stream::GLTextureDescription()));

    // Set to incomplete
    mState.clearImageDesc(NonCubeTextureTypeToTarget(mState.mType), 0);
    signalDirtyStorage(InitState::Initialized);
    return angle::Result::Continue;
}

angle::Result Texture::releaseTexImageInternal(Context *context)
{
    if (mBoundSurface)
    {
        // Notify the surface
        egl::Error eglErr = mBoundSurface->releaseTexImageFromTexture(context);
        // TODO(jmadill): Remove this once refactor is complete. http://anglebug.com/3041
        if (eglErr.isError())
        {
            context->handleError(GL_INVALID_OPERATION, "Error releasing tex image from texture",
                                 __FILE__, ANGLE_FUNCTION, __LINE__);
        }

        // Then, call the same method as from the surface
        ANGLE_TRY(releaseTexImageFromSurface(context));
    }
    return angle::Result::Continue;
}

angle::Result Texture::setEGLImageTarget(Context *context,
                                         TextureType type,
                                         egl::Image *imageTarget)
{
    ASSERT(type == mState.mType);
    ASSERT(type == TextureType::_2D || type == TextureType::External ||
           type == TextureType::CubeMap);

    // Release from previous calls to eglBindTexImage, to avoid calling the Impl after
    ANGLE_TRY(releaseTexImageInternal(context));
    ANGLE_TRY(orphanImages(context));

    ANGLE_TRY(mTexture->setEGLImageTarget(context, type, imageTarget));

    setTargetImage(context, imageTarget);

    Extents size(static_cast<int>(imageTarget->getWidth()),
                 static_cast<int>(imageTarget->getHeight()), 1);

    auto initState = imageTarget->sourceInitState();

    mState.clearImageDescs();
    if (type == TextureType::CubeMap)
    {
        for (int face = 0; face < 6; ++face)
        {
            mState.setImageDesc(FromGLenum<TextureTarget>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face), 0,
                                ImageDesc(size, imageTarget->getFormat(), initState));
        }
    }
    else
    {
        mState.setImageDesc(NonCubeTextureTypeToTarget(type), 0,
                            ImageDesc(size, imageTarget->getFormat(), initState));
    }
    signalDirtyStorage(initState);

    return angle::Result::Continue;
}

Extents Texture::getAttachmentSize(const ImageIndex &imageIndex) const
{
    // As an ImageIndex that represents an entire level of a cube map corresponds to 6 ImageDescs,
    // we only allow querying ImageDesc on a complete cube map, and this ImageDesc is exactly the
    // one that belongs to the first face of the cube map.
    if (imageIndex.isEntireLevelCubeMap())
    {
        // A cube map texture is cube complete if the following conditions all hold true:
        // - The levelbase arrays of each of the six texture images making up the cube map have
        //   identical, positive, and square dimensions.
        if (!mState.isCubeComplete())
        {
            return Extents();
        }
    }

    return mState.getImageDesc(imageIndex).size;
}

Format Texture::getAttachmentFormat(GLenum /*binding*/, const ImageIndex &imageIndex) const
{
    // As an ImageIndex that represents an entire level of a cube map corresponds to 6 ImageDescs,
    // we only allow querying ImageDesc on a complete cube map, and this ImageDesc is exactly the
    // one that belongs to the first face of the cube map.
    if (imageIndex.isEntireLevelCubeMap())
    {
        // A cube map texture is cube complete if the following conditions all hold true:
        // - The levelbase arrays were each specified with the same effective internal format.
        if (!mState.isCubeComplete())
        {
            return Format::Invalid();
        }
    }
    return mState.getImageDesc(imageIndex).format;
}

GLsizei Texture::getAttachmentSamples(const ImageIndex &imageIndex) const
{
    // We do not allow querying TextureTarget by an ImageIndex that represents an entire level of a
    // cube map (See comments in function TextureTypeToTarget() in ImageIndex.cpp).
    if (imageIndex.isEntireLevelCubeMap())
    {
        return 0;
    }

    return getSamples(imageIndex.getTarget(), imageIndex.getLevelIndex());
}

bool Texture::isRenderable(const Context *context,
                           GLenum binding,
                           const ImageIndex &imageIndex) const
{
    if (isEGLImageTarget())
    {
        return ImageSibling::isRenderable(context, binding, imageIndex);
    }
    return getAttachmentFormat(binding, imageIndex)
        .info->textureAttachmentSupport(context->getClientVersion(), context->getExtensions());
}

bool Texture::getAttachmentFixedSampleLocations(const ImageIndex &imageIndex) const
{
    // We do not allow querying TextureTarget by an ImageIndex that represents an entire level of a
    // cube map (See comments in function TextureTypeToTarget() in ImageIndex.cpp).
    if (imageIndex.isEntireLevelCubeMap())
    {
        return true;
    }

    // ES3.1 (section 9.4) requires that the value of TEXTURE_FIXED_SAMPLE_LOCATIONS should be
    // the same for all attached textures.
    return getFixedSampleLocations(imageIndex.getTarget(), imageIndex.getLevelIndex());
}

void Texture::setBorderColor(const Context *context, const ColorGeneric &color)
{
    mState.mSamplerState.setBorderColor(color);
    signalDirtyState(DIRTY_BIT_BORDER_COLOR);
}

const ColorGeneric &Texture::getBorderColor() const
{
    return mState.mSamplerState.getBorderColor();
}

void Texture::setCrop(const gl::Rectangle &rect)
{
    mState.setCrop(rect);
}

const gl::Rectangle &Texture::getCrop() const
{
    return mState.getCrop();
}

void Texture::setGenerateMipmapHint(GLenum hint)
{
    mState.setGenerateMipmapHint(hint);
}

GLenum Texture::getGenerateMipmapHint() const
{
    return mState.getGenerateMipmapHint();
}

void Texture::onAttach(const Context *context)
{
    addRef();
}

void Texture::onDetach(const Context *context)
{
    release(context);
}

GLuint Texture::getId() const
{
    return id().value;
}

GLuint Texture::getNativeID() const
{
    return mTexture->getNativeID();
}

angle::Result Texture::syncState(const Context *context)
{
    ASSERT(hasAnyDirtyBit());
    ANGLE_TRY(mTexture->syncState(context, mDirtyBits));
    mDirtyBits.reset();
    return angle::Result::Continue;
}

rx::FramebufferAttachmentObjectImpl *Texture::getAttachmentImpl() const
{
    return mTexture;
}

bool Texture::isSamplerComplete(const Context *context, const Sampler *optionalSampler)
{
    const auto &samplerState =
        optionalSampler ? optionalSampler->getSamplerState() : mState.mSamplerState;
    const auto &contextState = context->getState();

    if (contextState.getContextID() != mCompletenessCache.context ||
        !mCompletenessCache.samplerState.sameCompleteness(samplerState))
    {
        mCompletenessCache.context      = context->getState().getContextID();
        mCompletenessCache.samplerState = samplerState;
        mCompletenessCache.samplerComplete =
            mState.computeSamplerCompleteness(samplerState, contextState);
    }

    return mCompletenessCache.samplerComplete;
}

Texture::SamplerCompletenessCache::SamplerCompletenessCache()
    : context(0), samplerState(), samplerComplete(false)
{}

void Texture::invalidateCompletenessCache() const
{
    mCompletenessCache.context = 0;
}

angle::Result Texture::ensureInitialized(const Context *context)
{
    if (!context->isRobustResourceInitEnabled() || mState.mInitState == InitState::Initialized)
    {
        return angle::Result::Continue;
    }

    bool anyDirty = false;

    ImageIndexIterator it =
        ImageIndexIterator::MakeGeneric(mState.mType, 0, IMPLEMENTATION_MAX_TEXTURE_LEVELS + 1,
                                        ImageIndex::kEntireLevel, ImageIndex::kEntireLevel);
    while (it.hasNext())
    {
        const ImageIndex index = it.next();
        ImageDesc &desc =
            mState.mImageDescs[GetImageDescIndex(index.getTarget(), index.getLevelIndex())];
        if (desc.initState == InitState::MayNeedInit)
        {
            ASSERT(mState.mInitState == InitState::MayNeedInit);
            ANGLE_TRY(initializeContents(context, index));
            desc.initState = InitState::Initialized;
            anyDirty       = true;
        }
    }
    if (anyDirty)
    {
        signalDirtyStorage(InitState::Initialized);
    }
    mState.mInitState = InitState::Initialized;

    return angle::Result::Continue;
}

InitState Texture::initState(const ImageIndex &imageIndex) const
{
    // As an ImageIndex that represents an entire level of a cube map corresponds to 6 ImageDescs,
    // we need to check all the related ImageDescs.
    if (imageIndex.isEntireLevelCubeMap())
    {
        const GLint levelIndex = imageIndex.getLevelIndex();
        for (TextureTarget cubeFaceTarget : AllCubeFaceTextureTargets())
        {
            if (mState.getImageDesc(cubeFaceTarget, levelIndex).initState == InitState::MayNeedInit)
            {
                return InitState::MayNeedInit;
            }
        }
        return InitState::Initialized;
    }

    return mState.getImageDesc(imageIndex).initState;
}

void Texture::setInitState(const ImageIndex &imageIndex, InitState initState)
{
    // As an ImageIndex that represents an entire level of a cube map corresponds to 6 ImageDescs,
    // we need to update all the related ImageDescs.
    if (imageIndex.isEntireLevelCubeMap())
    {
        const GLint levelIndex = imageIndex.getLevelIndex();
        for (TextureTarget cubeFaceTarget : AllCubeFaceTextureTargets())
        {
            setInitState(ImageIndex::MakeCubeMapFace(cubeFaceTarget, levelIndex), initState);
        }
    }
    else
    {
        ImageDesc newDesc = mState.getImageDesc(imageIndex);
        newDesc.initState = initState;
        mState.setImageDesc(imageIndex.getTarget(), imageIndex.getLevelIndex(), newDesc);
    }
}

angle::Result Texture::ensureSubImageInitialized(const Context *context,
                                                 TextureTarget target,
                                                 size_t level,
                                                 const gl::Box &area)
{
    if (!context->isRobustResourceInitEnabled() || mState.mInitState == InitState::Initialized)
    {
        return angle::Result::Continue;
    }

    // Pre-initialize the texture contents if necessary.
    // TODO(jmadill): Check if area overlaps the entire texture.
    ImageIndex imageIndex =
        ImageIndex::MakeFromTarget(target, static_cast<GLint>(level), area.depth);
    const auto &desc = mState.getImageDesc(imageIndex);
    if (desc.initState == InitState::MayNeedInit)
    {
        ASSERT(mState.mInitState == InitState::MayNeedInit);
        bool coversWholeImage = area.x == 0 && area.y == 0 && area.z == 0 &&
                                area.width == desc.size.width && area.height == desc.size.height &&
                                area.depth == desc.size.depth;
        if (!coversWholeImage)
        {
            ANGLE_TRY(initializeContents(context, imageIndex));
        }
        setInitState(imageIndex, InitState::Initialized);
    }

    return angle::Result::Continue;
}

angle::Result Texture::handleMipmapGenerationHint(Context *context, int level)
{

    if (getGenerateMipmapHint() == GL_TRUE && level == 0)
    {
        ANGLE_TRY(generateMipmap(context));
    }

    return angle::Result::Continue;
}

void Texture::onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message)
{
    switch (message)
    {
        case angle::SubjectMessage::ContentsChanged:
            // ContentsChange is originates from TextureStorage11::resolveAndReleaseTexture
            // which resolves the underlying multisampled texture if it exists and so
            // Texture will signal dirty storage to invalidate its own cache and the
            // attached framebuffer's cache.
            signalDirtyStorage(InitState::Initialized);
            return;
        case angle::SubjectMessage::SubjectChanged:
            mDirtyBits.set(DIRTY_BIT_IMPLEMENTATION);
            signalDirtyState(DIRTY_BIT_IMPLEMENTATION);

            // Notify siblings that we are dirty.
            if (index == rx::kTextureImageImplObserverMessageIndex)
            {
                notifySiblings(message);
            }
            return;
        default:
            return;
    }
}

}  // namespace gl
