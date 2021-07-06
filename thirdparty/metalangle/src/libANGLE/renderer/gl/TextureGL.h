//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// TextureGL.h: Defines the class interface for TextureGL.

#ifndef LIBANGLE_RENDERER_GL_TEXTUREGL_H_
#define LIBANGLE_RENDERER_GL_TEXTUREGL_H_

#include "libANGLE/Texture.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/TextureImpl.h"

namespace rx
{

class BlitGL;
class FunctionsGL;
class StateManagerGL;

struct LUMAWorkaroundGL
{
    bool enabled;
    GLenum workaroundFormat;

    LUMAWorkaroundGL();
    LUMAWorkaroundGL(bool enabled, GLenum workaroundFormat);
};

// Structure containing information about format and workarounds for each mip level of the
// TextureGL.
struct LevelInfoGL
{
    // Format of the data used in this mip level.
    GLenum sourceFormat;

    // Internal format used for the native call to define this texture
    GLenum nativeInternalFormat;

    // If this mip level requires sampler-state re-writing so that only a red channel is exposed.
    // In GLES 2.0, depth textures are treated as luminance, so we check the
    // context's major version when applying the depth swizzle.
    bool depthStencilWorkaround;

    // Information about luminance alpha texture workarounds in the core profile.
    LUMAWorkaroundGL lumaWorkaround;

    // If this texture level hides the fact that it has an alpha channel by setting the sampler
    // parameters to always sample 1.0.
    bool emulatedAlphaChannel;

    LevelInfoGL();
    LevelInfoGL(GLenum sourceFormat,
                GLenum nativeInternalFormat,
                bool depthStencilWorkaround,
                const LUMAWorkaroundGL &lumaWorkaround,
                bool emulatedAlphaChannel);
};

class TextureGL : public TextureImpl
{
  public:
    TextureGL(const gl::TextureState &state, GLuint id);
    ~TextureGL() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result setImage(const gl::Context *context,
                           const gl::ImageIndex &index,
                           GLenum internalFormat,
                           const gl::Extents &size,
                           GLenum format,
                           GLenum type,
                           const gl::PixelUnpackState &unpack,
                           const uint8_t *pixels) override;
    angle::Result setSubImage(const gl::Context *context,
                              const gl::ImageIndex &index,
                              const gl::Box &area,
                              GLenum format,
                              GLenum type,
                              const gl::PixelUnpackState &unpack,
                              gl::Buffer *unpackBuffer,
                              const uint8_t *pixels) override;

    angle::Result setCompressedImage(const gl::Context *context,
                                     const gl::ImageIndex &index,
                                     GLenum internalFormat,
                                     const gl::Extents &size,
                                     const gl::PixelUnpackState &unpack,
                                     size_t imageSize,
                                     const uint8_t *pixels) override;
    angle::Result setCompressedSubImage(const gl::Context *context,
                                        const gl::ImageIndex &index,
                                        const gl::Box &area,
                                        GLenum format,
                                        const gl::PixelUnpackState &unpack,
                                        size_t imageSize,
                                        const uint8_t *pixels) override;

    angle::Result copyImage(const gl::Context *context,
                            const gl::ImageIndex &index,
                            const gl::Rectangle &sourceArea,
                            GLenum internalFormat,
                            gl::Framebuffer *source) override;
    angle::Result copySubImage(const gl::Context *context,
                               const gl::ImageIndex &index,
                               const gl::Offset &destOffset,
                               const gl::Rectangle &sourceArea,
                               gl::Framebuffer *source) override;

    angle::Result copyTexture(const gl::Context *context,
                              const gl::ImageIndex &index,
                              GLenum internalFormat,
                              GLenum type,
                              size_t sourceLevel,
                              bool unpackFlipY,
                              bool unpackPremultiplyAlpha,
                              bool unpackUnmultiplyAlpha,
                              const gl::Texture *source) override;
    angle::Result copySubTexture(const gl::Context *context,
                                 const gl::ImageIndex &index,
                                 const gl::Offset &destOffset,
                                 size_t sourceLevel,
                                 const gl::Box &sourceBox,
                                 bool unpackFlipY,
                                 bool unpackPremultiplyAlpha,
                                 bool unpackUnmultiplyAlpha,
                                 const gl::Texture *source) override;
    angle::Result copySubTextureHelper(const gl::Context *context,
                                       gl::TextureTarget target,
                                       size_t level,
                                       const gl::Offset &destOffset,
                                       size_t sourceLevel,
                                       const gl::Rectangle &sourceArea,
                                       const gl::InternalFormat &destFormat,
                                       bool unpackFlipY,
                                       bool unpackPremultiplyAlpha,
                                       bool unpackUnmultiplyAlpha,
                                       const gl::Texture *source);

    angle::Result setStorage(const gl::Context *context,
                             gl::TextureType type,
                             size_t levels,
                             GLenum internalFormat,
                             const gl::Extents &size) override;

    angle::Result setStorageMultisample(const gl::Context *context,
                                        gl::TextureType type,
                                        GLsizei samples,
                                        GLint internalformat,
                                        const gl::Extents &size,
                                        bool fixedSampleLocations) override;

    angle::Result setStorageExternalMemory(const gl::Context *context,
                                           gl::TextureType type,
                                           size_t levels,
                                           GLenum internalFormat,
                                           const gl::Extents &size,
                                           gl::MemoryObject *memoryObject,
                                           GLuint64 offset) override;

    angle::Result setImageExternal(const gl::Context *context,
                                   const gl::ImageIndex &index,
                                   GLenum internalFormat,
                                   const gl::Extents &size,
                                   GLenum format,
                                   GLenum type) override;

    angle::Result setImageExternal(const gl::Context *context,
                                   gl::TextureType type,
                                   egl::Stream *stream,
                                   const egl::Stream::GLTextureDescription &desc) override;

    angle::Result generateMipmap(const gl::Context *context) override;

    angle::Result bindTexImage(const gl::Context *context, egl::Surface *surface) override;
    angle::Result releaseTexImage(const gl::Context *context) override;

    angle::Result setEGLImageTarget(const gl::Context *context,
                                    gl::TextureType type,
                                    egl::Image *image) override;

    GLint getNativeID() const override;

    GLuint getTextureID() const { return mTextureID; }

    gl::TextureType getType() const;

    angle::Result syncState(const gl::Context *context,
                            const gl::Texture::DirtyBits &dirtyBits) override;
    bool hasAnyDirtyBit() const;

    angle::Result setBaseLevel(const gl::Context *context, GLuint baseLevel) override;
    angle::Result setMaxLevel(const gl::Context *context, GLuint maxLevel);

    angle::Result initializeContents(const gl::Context *context,
                                     const gl::ImageIndex &imageIndex) override;

    angle::Result setMinFilter(const gl::Context *context, GLenum filter);
    angle::Result setMagFilter(const gl::Context *context, GLenum filter);

    angle::Result setSwizzle(const gl::Context *context, GLint swizzle[4]);

    GLenum getNativeInternalFormat(const gl::ImageIndex &index) const;
    bool hasEmulatedAlphaChannel(const gl::ImageIndex &index) const;

  private:
    angle::Result setImageHelper(const gl::Context *context,
                                 gl::TextureTarget target,
                                 size_t level,
                                 GLenum internalFormat,
                                 const gl::Extents &size,
                                 GLenum format,
                                 GLenum type,
                                 const uint8_t *pixels);
    // This changes the current pixel unpack state that will have to be reapplied.
    angle::Result reserveTexImageToBeFilled(const gl::Context *context,
                                            gl::TextureTarget target,
                                            size_t level,
                                            GLenum internalFormat,
                                            const gl::Extents &size,
                                            GLenum format,
                                            GLenum type);
    angle::Result setSubImageRowByRowWorkaround(const gl::Context *context,
                                                gl::TextureTarget target,
                                                size_t level,
                                                const gl::Box &area,
                                                GLenum format,
                                                GLenum type,
                                                const gl::PixelUnpackState &unpack,
                                                const gl::Buffer *unpackBuffer,
                                                const uint8_t *pixels);

    angle::Result setSubImagePaddingWorkaround(const gl::Context *context,
                                               gl::TextureTarget target,
                                               size_t level,
                                               const gl::Box &area,
                                               GLenum format,
                                               GLenum type,
                                               const gl::PixelUnpackState &unpack,
                                               const gl::Buffer *unpackBuffer,
                                               const uint8_t *pixels);

    angle::Result syncTextureStateSwizzle(const gl::Context *context,
                                          const FunctionsGL *functions,
                                          GLenum name,
                                          GLenum value,
                                          GLenum *outValue);

    void setLevelInfo(const gl::Context *context,
                      gl::TextureTarget target,
                      size_t level,
                      size_t levelCount,
                      const LevelInfoGL &levelInfo);
    void setLevelInfo(const gl::Context *context,
                      gl::TextureType type,
                      size_t level,
                      size_t levelCount,
                      const LevelInfoGL &levelInfo);
    const LevelInfoGL &getLevelInfo(gl::TextureTarget target, size_t level) const;
    const LevelInfoGL &getBaseLevelInfo() const;

    std::vector<LevelInfoGL> mLevelInfo;
    gl::Texture::DirtyBits mLocalDirtyBits;

    gl::SwizzleState mAppliedSwizzle;
    gl::SamplerState mAppliedSampler;
    GLuint mAppliedBaseLevel;
    GLuint mAppliedMaxLevel;

    GLuint mTextureID;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_TEXTUREGL_H_
