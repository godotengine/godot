//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TextureMtl.h:
//    Defines the class interface for TextureMtl, implementing TextureImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_TEXTUREMTL_H_
#define LIBANGLE_RENDERER_METAL_TEXTUREMTL_H_

#include <map>

#include "common/PackedEnums.h"
#include "libANGLE/renderer/TextureImpl.h"
#include "libANGLE/renderer/metal/RenderTargetMtl.h"
#include "libANGLE/renderer/metal/mtl_command_buffer.h"
#include "libANGLE/renderer/metal/mtl_resources.h"

namespace rx
{

class OffscreenSurfaceMtl;

struct ImageDefinitionMtl
{
    mtl::TextureRef image;
    angle::FormatID formatID = angle::FormatID::NONE;
};

class TextureMtl : public TextureImpl
{
  public:
    TextureMtl(const gl::TextureState &state);
    ~TextureMtl() override;
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

    angle::Result copyCompressedTexture(const gl::Context *context,
                                        const gl::Texture *source) override;

    angle::Result setStorage(const gl::Context *context,
                             gl::TextureType type,
                             size_t levels,
                             GLenum internalFormat,
                             const gl::Extents &size) override;

    angle::Result setStorageExternalMemory(const gl::Context *context,
                                           gl::TextureType type,
                                           size_t levels,
                                           GLenum internalFormat,
                                           const gl::Extents &size,
                                           gl::MemoryObject *memoryObject,
                                           GLuint64 offset) override;

    angle::Result setEGLImageTarget(const gl::Context *context,
                                    gl::TextureType type,
                                    egl::Image *image) override;

    angle::Result setImageExternal(const gl::Context *context,
                                   gl::TextureType type,
                                   egl::Stream *stream,
                                   const egl::Stream::GLTextureDescription &desc) override;

    angle::Result generateMipmap(const gl::Context *context) override;

    angle::Result setBaseLevel(const gl::Context *context, GLuint baseLevel) override;

    angle::Result bindTexImage(const gl::Context *context, egl::Surface *surface) override;
    angle::Result releaseTexImage(const gl::Context *context) override;

    angle::Result getAttachmentRenderTarget(const gl::Context *context,
                                            GLenum binding,
                                            const gl::ImageIndex &imageIndex,
                                            GLsizei samples,
                                            FramebufferAttachmentRenderTarget **rtOut) override;

    angle::Result syncState(const gl::Context *context,
                            const gl::Texture::DirtyBits &dirtyBits) override;

    angle::Result setStorageMultisample(const gl::Context *context,
                                        gl::TextureType type,
                                        GLsizei samples,
                                        GLint internalformat,
                                        const gl::Extents &size,
                                        bool fixedSampleLocations) override;

    angle::Result initializeContents(const gl::Context *context,
                                     const gl::ImageIndex &imageIndex) override;

    // The texture's data is initially initialized and stored in an array
    // of images through glTexImage*/glCopyTex* calls. During draw calls, the caller must make sure
    // the actual texture is created by calling this method to transfer the stored images data
    // to the actual texture.
    angle::Result ensureTextureCreated(const gl::Context *context);

    angle::Result bindToShader(const gl::Context *context,
                               mtl::RenderCommandEncoder *cmdEncoder,
                               gl::ShaderType shaderType,
                               gl::Sampler *sampler, /** nullable */
                               int textureSlotIndex,
                               int samplerSlotIndex);

    const mtl::Format &getFormat() const { return mFormat; }
    const mtl::TextureRef &getNativeTexture() const { return mNativeTexture; }

  private:
    void releaseTexture(bool releaseImages);
    void releaseTexture(bool releaseImages, bool releaseTextureObjectsOnly);
    angle::Result createNativeTexture(const gl::Context *context,
                                      gl::TextureType type,
                                      GLuint mips,
                                      const gl::Extents &size);
    angle::Result onBaseMaxLevelsChanged(const gl::Context *context);
    angle::Result ensureSamplerStateCreated(const gl::Context *context);
    // Ensure image at given index is created:
    angle::Result ensureImageCreated(const gl::Context *context, const gl::ImageIndex &index);
    // Ensure all image views at all faces/levels are retained.
    void retainImageDefinitions();
    mtl::TextureRef createImageViewFromNativeTexture(GLuint cubeFaceOrZero, GLuint nativeLevel);
    angle::Result ensureNativeLevelViewsCreated();
    angle::Result checkForEmulatedChannels(const gl::Context *context,
                                           const mtl::Format &mtlFormat,
                                           const mtl::TextureRef &texture);
    int getNativeLevel(const gl::ImageIndex &imageIndex) const;
    mtl::TextureRef &getImage(const gl::ImageIndex &imageIndex);
    ImageDefinitionMtl &getImageDefinition(const gl::ImageIndex &imageIndex);
    RenderTargetMtl &getRenderTarget(const gl::ImageIndex &imageIndex);
    mtl::TextureRef &getImplicitMSTexture(const gl::ImageIndex &imageIndex);
    bool isIndexWithinMinMaxLevels(const gl::ImageIndex &imageIndex) const;

    // If levels = 0, this function will create full mipmaps texture.
    angle::Result setStorageImpl(const gl::Context *context,
                                 gl::TextureType type,
                                 size_t levels,
                                 const mtl::Format &mtlFormat,
                                 const gl::Extents &size);

    angle::Result redefineImage(const gl::Context *context,
                                const gl::ImageIndex &index,
                                const mtl::Format &mtlFormat,
                                const gl::Extents &size,
                                bool initEmulatedChannels);

    angle::Result setImageImpl(const gl::Context *context,
                               const gl::ImageIndex &index,
                               const gl::InternalFormat &formatInfo,
                               const gl::Extents &size,
                               GLenum type,
                               const gl::PixelUnpackState &unpack,
                               const uint8_t *pixels);
    angle::Result setSubImageImpl(const gl::Context *context,
                                  const gl::ImageIndex &index,
                                  const gl::Box &area,
                                  const gl::InternalFormat &formatInfo,
                                  GLenum type,
                                  const gl::PixelUnpackState &unpack,
                                  gl::Buffer *unpackBuffer,
                                  const uint8_t *pixels);

    angle::Result copySubImageImpl(const gl::Context *context,
                                   const gl::ImageIndex &index,
                                   const gl::Offset &destOffset,
                                   const gl::Rectangle &sourceArea,
                                   const gl::InternalFormat &internalFormat,
                                   gl::Framebuffer *source);
    angle::Result copySubImageWithDraw(const gl::Context *context,
                                       const gl::ImageIndex &index,
                                       const gl::Offset &destOffset,
                                       const gl::Rectangle &sourceArea,
                                       const gl::InternalFormat &internalFormat,
                                       gl::Framebuffer *source);
    angle::Result copySubImageCPU(const gl::Context *context,
                                  const gl::ImageIndex &index,
                                  const gl::Offset &destOffset,
                                  const gl::Rectangle &sourceArea,
                                  const gl::InternalFormat &internalFormat,
                                  gl::Framebuffer *source);

    angle::Result setPerSliceSubImage(const gl::Context *context,
                                      int slice,
                                      const MTLRegion &mtlArea,
                                      const gl::InternalFormat &internalFormat,
                                      GLenum type,
                                      const angle::Format &pixelsAngleFormat,
                                      size_t pixelsRowPitch,
                                      size_t pixelsDepthPitch,
                                      gl::Buffer *unpackBuffer,
                                      const uint8_t *pixels,
                                      const mtl::TextureRef &image);

    // Convert pixels to suported format before uploading to texture
    angle::Result convertAndSetPerSliceSubImage(const gl::Context *context,
                                                int slice,
                                                const MTLRegion &mtlArea,
                                                const gl::InternalFormat &internalFormat,
                                                GLenum type,
                                                const angle::Format &pixelsAngleFormat,
                                                size_t pixelsRowPitch,
                                                size_t pixelsDepthPitch,
                                                gl::Buffer *unpackBuffer,
                                                const uint8_t *pixels,
                                                const mtl::TextureRef &image);

    angle::Result generateMipmapCPU(const gl::Context *context);

    mtl::Format mFormat;
    // The real texture used by Metal draw calls.
    mtl::TextureRef mNativeTexture;
    id<MTLSamplerState> mMetalSamplerState = nil;
    OffscreenSurfaceMtl *mBoundPBuffer     = nullptr;

    // Number of slices
    uint32_t mSlices = 1;

    // Stored images array defined by glTexImage/glCopy*.
    // Once the images array is complete, they will be transferred to real texture object.
    // NOTE:
    //  - The second dimension is indexed by configured base level + actual native level
    //  - For Cube map, there will be at most 6 entries in the map table, one for each face. This is
    //  because the Cube map's image is defined per face & per level.
    //  - For other texture types, there will be only one entry in the map table. All other textures
    //  except Cube map has texture image defined per level (all slices included).
    //  - These three variables' second dimension are indexed by image index (base level included).
    std::map<int, gl::TexLevelArray<ImageDefinitionMtl>> mTexImageDefs;
    std::map<int, gl::TexLevelArray<RenderTargetMtl>> mPerLayerRenderTargets;
    std::map<int, gl::TexLevelArray<mtl::TextureRef>> mImplicitMSTextures;

    // Mipmap views are indexed by native level (ignored base level):
    gl::TexLevelArray<mtl::TextureRef> mNativeLevelViews;

    // The swizzled view used for shader sampling.
    mtl::TextureRef mNativeSwizzleSamplingView;

    GLuint mCurrentBaseLevel = 0;
    GLuint mCurrentMaxLevel  = 1000;

    bool mIsPow2 = false;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_TEXTUREMTL_H_ */
