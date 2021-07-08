//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Texture.h: Defines the gl::Texture class [OpenGL ES 2.0.24] section 3.7 page 63.

#ifndef LIBANGLE_TEXTURE_H_
#define LIBANGLE_TEXTURE_H_

#include <map>
#include <vector>

#include "angle_gl.h"
#include "common/Optional.h"
#include "common/debug.h"
#include "common/utilities.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/Image.h"
#include "libANGLE/Observer.h"
#include "libANGLE/Stream.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/formatutils.h"

namespace egl
{
class Surface;
class Stream;
}  // namespace egl

namespace rx
{
class GLImplFactory;
class TextureImpl;
class TextureGL;
}  // namespace rx

namespace gl
{
class Framebuffer;
class MemoryObject;
class Sampler;
class State;
class Texture;

bool IsMipmapFiltered(const SamplerState &samplerState);

struct ImageDesc final
{
    ImageDesc();
    ImageDesc(const Extents &size, const Format &format, const InitState initState);
    ImageDesc(const Extents &size,
              const Format &format,
              const GLsizei samples,
              const bool fixedSampleLocations,
              const InitState initState);

    ImageDesc(const ImageDesc &other) = default;
    ImageDesc &operator=(const ImageDesc &other) = default;

    GLint getMemorySize() const;

    Extents size;
    Format format;
    GLsizei samples;
    bool fixedSampleLocations;

    // Needed for robust resource initialization.
    InitState initState;
};

struct SwizzleState final
{
    SwizzleState();
    SwizzleState(GLenum red, GLenum green, GLenum blue, GLenum alpha);
    SwizzleState(const SwizzleState &other) = default;
    SwizzleState &operator=(const SwizzleState &other) = default;

    bool swizzleRequired() const;

    bool operator==(const SwizzleState &other) const;
    bool operator!=(const SwizzleState &other) const;

    GLenum swizzleRed;
    GLenum swizzleGreen;
    GLenum swizzleBlue;
    GLenum swizzleAlpha;
};

// State from Table 6.9 (state per texture object) in the OpenGL ES 3.0.2 spec.
class TextureState final : private angle::NonCopyable
{
  public:
    TextureState(TextureType type);
    ~TextureState();

    bool swizzleRequired() const;
    GLuint getEffectiveBaseLevel() const;
    GLuint getEffectiveMaxLevel() const;

    // Returns the value called "q" in the GLES 3.0.4 spec section 3.8.10.
    GLuint getMipmapMaxLevel() const;

    // Returns true if base level changed.
    bool setBaseLevel(GLuint baseLevel);
    bool setMaxLevel(GLuint maxLevel);

    bool isCubeComplete() const;

    ANGLE_INLINE bool compatibleWithSamplerFormat(SamplerFormat format,
                                                  const SamplerState &samplerState) const
    {
        if (!mCachedSamplerFormatValid ||
            mCachedSamplerCompareMode != samplerState.getCompareMode())
        {
            mCachedSamplerFormat      = computeRequiredSamplerFormat(samplerState);
            mCachedSamplerCompareMode = samplerState.getCompareMode();
            mCachedSamplerFormatValid = true;
        }
        // Incomplete textures are compatible with any sampler format.
        return mCachedSamplerFormat == SamplerFormat::InvalidEnum || format == mCachedSamplerFormat;
    }

    const ImageDesc &getImageDesc(TextureTarget target, size_t level) const;
    const ImageDesc &getImageDesc(const ImageIndex &imageIndex) const;

    TextureType getType() const { return mType; }
    const SwizzleState &getSwizzleState() const { return mSwizzleState; }
    const SamplerState &getSamplerState() const { return mSamplerState; }
    GLenum getUsage() const { return mUsage; }
    GLenum getDepthStencilTextureMode() const { return mDepthStencilTextureMode; }
    bool isStencilMode() const { return mDepthStencilTextureMode == GL_STENCIL_INDEX; }

    // Returns the desc of the base level. Only valid for cube-complete/mip-complete textures.
    const ImageDesc &getBaseLevelDesc() const;

    // GLES1 emulation: For GL_OES_draw_texture
    void setCrop(const gl::Rectangle &rect);
    const gl::Rectangle &getCrop() const;

    // GLES1 emulation: Auto-mipmap generation is a texparameter
    void setGenerateMipmapHint(GLenum hint);
    GLenum getGenerateMipmapHint() const;

  private:
    // Texture needs access to the ImageDesc functions.
    friend class Texture;
    // TODO(jmadill): Remove TextureGL from friends.
    friend class rx::TextureGL;
    friend bool operator==(const TextureState &a, const TextureState &b);

    bool computeSamplerCompleteness(const SamplerState &samplerState, const State &data) const;
    bool computeMipmapCompleteness() const;
    bool computeLevelCompleteness(TextureTarget target, size_t level) const;
    SamplerFormat computeRequiredSamplerFormat(const SamplerState &samplerState) const;

    TextureTarget getBaseImageTarget() const;

    void setImageDesc(TextureTarget target, size_t level, const ImageDesc &desc);
    void setImageDescChain(GLuint baselevel,
                           GLuint maxLevel,
                           Extents baseSize,
                           const Format &format,
                           InitState initState);
    void setImageDescChainMultisample(Extents baseSize,
                                      const Format &format,
                                      GLsizei samples,
                                      bool fixedSampleLocations,
                                      InitState initState);

    void clearImageDesc(TextureTarget target, size_t level);
    void clearImageDescs();

    const TextureType mType;

    SwizzleState mSwizzleState;

    SamplerState mSamplerState;

    GLuint mBaseLevel;
    GLuint mMaxLevel;

    GLenum mDepthStencilTextureMode;

    bool mImmutableFormat;
    GLuint mImmutableLevels;

    // From GL_ANGLE_texture_usage
    GLenum mUsage;

    std::vector<ImageDesc> mImageDescs;

    // GLES1 emulation: Texture crop rectangle
    // For GL_OES_draw_texture
    gl::Rectangle mCropRect;

    // GLES1 emulation: Generate-mipmap hint per texture
    GLenum mGenerateMipmapHint;

    InitState mInitState;

    mutable SamplerFormat mCachedSamplerFormat;
    mutable GLenum mCachedSamplerCompareMode;
    mutable bool mCachedSamplerFormatValid;
};

bool operator==(const TextureState &a, const TextureState &b);
bool operator!=(const TextureState &a, const TextureState &b);

class Texture final : public RefCountObject<TextureID>,
                      public egl::ImageSibling,
                      public LabeledObject
{
  public:
    Texture(rx::GLImplFactory *factory, TextureID id, TextureType type);
    ~Texture() override;

    void onDestroy(const Context *context) override;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    TextureType getType() const { return mState.mType; }

    void setSwizzleRed(const Context *context, GLenum swizzleRed);
    GLenum getSwizzleRed() const;

    void setSwizzleGreen(const Context *context, GLenum swizzleGreen);
    GLenum getSwizzleGreen() const;

    void setSwizzleBlue(const Context *context, GLenum swizzleBlue);
    GLenum getSwizzleBlue() const;

    void setSwizzleAlpha(const Context *context, GLenum swizzleAlpha);
    GLenum getSwizzleAlpha() const;

    void setMinFilter(const Context *context, GLenum minFilter);
    GLenum getMinFilter() const;

    void setMagFilter(const Context *context, GLenum magFilter);
    GLenum getMagFilter() const;

    void setWrapS(const Context *context, GLenum wrapS);
    GLenum getWrapS() const;

    void setWrapT(const Context *context, GLenum wrapT);
    GLenum getWrapT() const;

    void setWrapR(const Context *context, GLenum wrapR);
    GLenum getWrapR() const;

    void setMaxAnisotropy(const Context *context, float maxAnisotropy);
    float getMaxAnisotropy() const;

    void setMinLod(const Context *context, GLfloat minLod);
    GLfloat getMinLod() const;

    void setMaxLod(const Context *context, GLfloat maxLod);
    GLfloat getMaxLod() const;

    void setCompareMode(const Context *context, GLenum compareMode);
    GLenum getCompareMode() const;

    void setCompareFunc(const Context *context, GLenum compareFunc);
    GLenum getCompareFunc() const;

    void setSRGBDecode(const Context *context, GLenum sRGBDecode);
    GLenum getSRGBDecode() const;

    const SamplerState &getSamplerState() const;

    angle::Result setBaseLevel(const Context *context, GLuint baseLevel);
    GLuint getBaseLevel() const;

    void setMaxLevel(const Context *context, GLuint maxLevel);
    GLuint getMaxLevel() const;

    void setDepthStencilTextureMode(const Context *context, GLenum mode);
    GLenum getDepthStencilTextureMode() const;

    bool getImmutableFormat() const;

    GLuint getImmutableLevels() const;

    void setUsage(const Context *context, GLenum usage);
    GLenum getUsage() const;

    void setBorderColor(const Context *context, const ColorGeneric &color);
    const ColorGeneric &getBorderColor() const;

    const TextureState &getTextureState() const;

    size_t getWidth(TextureTarget target, size_t level) const;
    size_t getHeight(TextureTarget target, size_t level) const;
    size_t getDepth(TextureTarget target, size_t level) const;
    GLsizei getSamples(TextureTarget target, size_t level) const;
    bool getFixedSampleLocations(TextureTarget target, size_t level) const;
    const Format &getFormat(TextureTarget target, size_t level) const;

    // Returns the value called "q" in the GLES 3.0.4 spec section 3.8.10.
    GLuint getMipmapMaxLevel() const;

    bool isMipmapComplete() const;

    angle::Result setImage(Context *context,
                           const PixelUnpackState &unpackState,
                           TextureTarget target,
                           GLint level,
                           GLenum internalFormat,
                           const Extents &size,
                           GLenum format,
                           GLenum type,
                           const uint8_t *pixels);
    angle::Result setSubImage(Context *context,
                              const PixelUnpackState &unpackState,
                              Buffer *unpackBuffer,
                              TextureTarget target,
                              GLint level,
                              const Box &area,
                              GLenum format,
                              GLenum type,
                              const uint8_t *pixels);

    angle::Result setCompressedImage(Context *context,
                                     const PixelUnpackState &unpackState,
                                     TextureTarget target,
                                     GLint level,
                                     GLenum internalFormat,
                                     const Extents &size,
                                     size_t imageSize,
                                     const uint8_t *pixels);
    angle::Result setCompressedSubImage(const Context *context,
                                        const PixelUnpackState &unpackState,
                                        TextureTarget target,
                                        GLint level,
                                        const Box &area,
                                        GLenum format,
                                        size_t imageSize,
                                        const uint8_t *pixels);

    angle::Result copyImage(Context *context,
                            TextureTarget target,
                            GLint level,
                            const Rectangle &sourceArea,
                            GLenum internalFormat,
                            Framebuffer *source);
    angle::Result copySubImage(Context *context,
                               const ImageIndex &index,
                               const Offset &destOffset,
                               const Rectangle &sourceArea,
                               Framebuffer *source);

    angle::Result copyTexture(Context *context,
                              TextureTarget target,
                              GLint level,
                              GLenum internalFormat,
                              GLenum type,
                              GLint sourceLevel,
                              bool unpackFlipY,
                              bool unpackPremultiplyAlpha,
                              bool unpackUnmultiplyAlpha,
                              Texture *source);
    angle::Result copySubTexture(const Context *context,
                                 TextureTarget target,
                                 GLint level,
                                 const Offset &destOffset,
                                 GLint sourceLevel,
                                 const Box &sourceBox,
                                 bool unpackFlipY,
                                 bool unpackPremultiplyAlpha,
                                 bool unpackUnmultiplyAlpha,
                                 Texture *source);
    angle::Result copyCompressedTexture(Context *context, const Texture *source);

    angle::Result setStorage(Context *context,
                             TextureType type,
                             GLsizei levels,
                             GLenum internalFormat,
                             const Extents &size);

    angle::Result setStorageMultisample(Context *context,
                                        TextureType type,
                                        GLsizei samples,
                                        GLint internalformat,
                                        const Extents &size,
                                        bool fixedSampleLocations);

    angle::Result setStorageExternalMemory(Context *context,
                                           TextureType type,
                                           GLsizei levels,
                                           GLenum internalFormat,
                                           const Extents &size,
                                           MemoryObject *memoryObject,
                                           GLuint64 offset);

    angle::Result setImageExternal(Context *context,
                                   TextureTarget target,
                                   GLint level,
                                   GLenum internalFormat,
                                   const Extents &size,
                                   GLenum format,
                                   GLenum type);

    angle::Result setEGLImageTarget(Context *context, TextureType type, egl::Image *imageTarget);

    angle::Result generateMipmap(Context *context);

    egl::Surface *getBoundSurface() const;
    egl::Stream *getBoundStream() const;

    GLint getMemorySize() const;
    GLint getLevelMemorySize(TextureTarget target, GLint level) const;

    void signalDirtyStorage(InitState initState);

    bool isSamplerComplete(const Context *context, const Sampler *optionalSampler);

    rx::TextureImpl *getImplementation() const { return mTexture; }

    // FramebufferAttachmentObject implementation
    Extents getAttachmentSize(const ImageIndex &imageIndex) const override;
    Format getAttachmentFormat(GLenum binding, const ImageIndex &imageIndex) const override;
    GLsizei getAttachmentSamples(const ImageIndex &imageIndex) const override;
    bool isRenderable(const Context *context,
                      GLenum binding,
                      const ImageIndex &imageIndex) const override;

    bool getAttachmentFixedSampleLocations(const ImageIndex &imageIndex) const;

    // GLES1 emulation
    void setCrop(const gl::Rectangle &rect);
    const gl::Rectangle &getCrop() const;
    void setGenerateMipmapHint(GLenum generate);
    GLenum getGenerateMipmapHint() const;

    void onAttach(const Context *context) override;
    void onDetach(const Context *context) override;

    // Used specifically for FramebufferAttachmentObject.
    GLuint getId() const override;

    GLuint getNativeID() const;

    // Needed for robust resource init.
    angle::Result ensureInitialized(const Context *context);
    InitState initState(const ImageIndex &imageIndex) const override;
    InitState initState() const { return mState.mInitState; }
    void setInitState(const ImageIndex &imageIndex, InitState initState) override;

    enum DirtyBitType
    {
        // Sampler state
        DIRTY_BIT_MIN_FILTER,
        DIRTY_BIT_MAG_FILTER,
        DIRTY_BIT_WRAP_S,
        DIRTY_BIT_WRAP_T,
        DIRTY_BIT_WRAP_R,
        DIRTY_BIT_MAX_ANISOTROPY,
        DIRTY_BIT_MIN_LOD,
        DIRTY_BIT_MAX_LOD,
        DIRTY_BIT_COMPARE_MODE,
        DIRTY_BIT_COMPARE_FUNC,
        DIRTY_BIT_SRGB_DECODE,
        DIRTY_BIT_BORDER_COLOR,

        // Texture state
        DIRTY_BIT_SWIZZLE_RED,
        DIRTY_BIT_SWIZZLE_GREEN,
        DIRTY_BIT_SWIZZLE_BLUE,
        DIRTY_BIT_SWIZZLE_ALPHA,
        DIRTY_BIT_BASE_LEVEL,
        DIRTY_BIT_MAX_LEVEL,
        DIRTY_BIT_DEPTH_STENCIL_TEXTURE_MODE,

        // Misc
        DIRTY_BIT_LABEL,
        DIRTY_BIT_USAGE,
        DIRTY_BIT_IMPLEMENTATION,

        DIRTY_BIT_COUNT,
    };
    using DirtyBits = angle::BitSet<DIRTY_BIT_COUNT>;

    angle::Result syncState(const Context *context);
    bool hasAnyDirtyBit() const { return mDirtyBits.any(); }

    // ObserverInterface implementation.
    void onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message) override;

  private:
    rx::FramebufferAttachmentObjectImpl *getAttachmentImpl() const override;

    // ANGLE-only method, used internally
    friend class egl::Surface;
    angle::Result bindTexImageFromSurface(Context *context, egl::Surface *surface);
    angle::Result releaseTexImageFromSurface(const Context *context);

    // ANGLE-only methods, used internally
    friend class egl::Stream;
    void bindStream(egl::Stream *stream);
    void releaseStream();
    angle::Result acquireImageFromStream(const Context *context,
                                         const egl::Stream::GLTextureDescription &desc);
    angle::Result releaseImageFromStream(const Context *context);

    void invalidateCompletenessCache() const;
    angle::Result releaseTexImageInternal(Context *context);

    angle::Result ensureSubImageInitialized(const Context *context,
                                            TextureTarget target,
                                            size_t level,
                                            const gl::Box &area);

    angle::Result handleMipmapGenerationHint(Context *context, int level);

    void signalDirtyState(size_t dirtyBit);

    TextureState mState;
    DirtyBits mDirtyBits;
    rx::TextureImpl *mTexture;
    angle::ObserverBinding mImplObserver;

    std::string mLabel;

    egl::Surface *mBoundSurface;
    egl::Stream *mBoundStream;

    struct SamplerCompletenessCache
    {
        SamplerCompletenessCache();

        // Context used to generate this cache entry
        ContextID context;

        // All values that affect sampler completeness that are not stored within
        // the texture itself
        SamplerState samplerState;

        // Result of the sampler completeness with the above parameters
        bool samplerComplete;
    };

    mutable SamplerCompletenessCache mCompletenessCache;
};

inline bool operator==(const TextureState &a, const TextureState &b)
{
    return a.mSwizzleState == b.mSwizzleState && a.mSamplerState == b.mSamplerState &&
           a.mBaseLevel == b.mBaseLevel && a.mMaxLevel == b.mMaxLevel &&
           a.mImmutableFormat == b.mImmutableFormat && a.mImmutableLevels == b.mImmutableLevels &&
           a.mUsage == b.mUsage;
}

inline bool operator!=(const TextureState &a, const TextureState &b)
{
    return !(a == b);
}
}  // namespace gl

#endif  // LIBANGLE_TEXTURE_H_
