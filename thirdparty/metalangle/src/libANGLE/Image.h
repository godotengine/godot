//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Image.h: Defines the egl::Image class representing the EGLimage object.

#ifndef LIBANGLE_IMAGE_H_
#define LIBANGLE_IMAGE_H_

#include "common/angleutils.h"
#include "libANGLE/AttributeMap.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/formatutils.h"

#include <set>

namespace rx
{
class EGLImplFactory;
class ImageImpl;
class ExternalImageSiblingImpl;

// Used for distinguishing dirty bit messages from gl::Texture/rx::TexureImpl/gl::Image.
constexpr size_t kTextureImageImplObserverMessageIndex = 0;
constexpr size_t kTextureImageSiblingMessageIndex      = 1;
}  // namespace rx

namespace egl
{
class Image;
class Display;

// Only currently Renderbuffers and Textures can be bound with images. This makes the relationship
// explicit, and also ensures that an image sibling can determine if it's been initialized or not,
// which is important for the robust resource init extension with Textures and EGLImages.
class ImageSibling : public gl::FramebufferAttachmentObject, public angle::ObserverInterface
{
  public:
    ImageSibling();
    ~ImageSibling() override;

    bool isEGLImageTarget() const;
    gl::InitState sourceEGLImageInitState() const;
    void setSourceEGLImageInitState(gl::InitState initState) const;

    bool isRenderable(const gl::Context *context,
                      GLenum binding,
                      const gl::ImageIndex &imageIndex) const override;

    // ObserverInterface implementation
    void onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message) override
    {
        // default to no-op.
    }

  protected:
    // Set the image target of this sibling
    void setTargetImage(const gl::Context *context, egl::Image *imageTarget);

    // Orphan all EGL image sources and targets
    angle::Result orphanImages(const gl::Context *context);

    void notifySiblings(angle::SubjectMessage message);

  private:
    friend class Image;

    // Called from Image only to add a new source image
    void addImageSource(egl::Image *imageSource);

    // Called from Image only to remove a source image when the Image is being deleted
    void removeImageSource(egl::Image *imageSource);

    std::set<Image *> mSourcesOf;
    BindingPointer<Image> mTargetOf;
};

// Wrapper for EGLImage sources that are not owned by ANGLE, these often have to do
// platform-specific queries for format and size information.
class ExternalImageSibling : public ImageSibling
{
  public:
    ExternalImageSibling(rx::EGLImplFactory *factory,
                         const gl::Context *context,
                         EGLenum target,
                         EGLClientBuffer buffer,
                         const AttributeMap &attribs);
    ~ExternalImageSibling() override;

    void onDestroy(const egl::Display *display);

    Error initialize(const Display *display);

    gl::Extents getAttachmentSize(const gl::ImageIndex &imageIndex) const override;
    gl::Format getAttachmentFormat(GLenum binding, const gl::ImageIndex &imageIndex) const override;
    GLsizei getAttachmentSamples(const gl::ImageIndex &imageIndex) const override;
    bool isRenderable(const gl::Context *context,
                      GLenum binding,
                      const gl::ImageIndex &imageIndex) const override;
    bool isTextureable(const gl::Context *context) const;

    void onAttach(const gl::Context *context) override;
    void onDetach(const gl::Context *context) override;
    GLuint getId() const override;

    gl::InitState initState(const gl::ImageIndex &imageIndex) const override;
    void setInitState(const gl::ImageIndex &imageIndex, gl::InitState initState) override;

    rx::ExternalImageSiblingImpl *getImplementation() const;

  protected:
    rx::FramebufferAttachmentObjectImpl *getAttachmentImpl() const override;

  private:
    std::unique_ptr<rx::ExternalImageSiblingImpl> mImplementation;
};

struct ImageState : private angle::NonCopyable
{
    ImageState(EGLenum target, ImageSibling *buffer, const AttributeMap &attribs);
    ~ImageState();

    EGLLabelKHR label;
    EGLenum target;
    gl::ImageIndex imageIndex;
    ImageSibling *source;
    std::set<ImageSibling *> targets;

    gl::Format format;
    gl::Extents size;
    size_t samples;
    EGLenum sourceType;
};

class Image final : public RefCountObject, public LabeledObject
{
  public:
    Image(rx::EGLImplFactory *factory,
          const gl::Context *context,
          EGLenum target,
          ImageSibling *buffer,
          const AttributeMap &attribs);

    void onDestroy(const Display *display) override;
    ~Image() override;

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    const gl::Format &getFormat() const;
    bool isRenderable(const gl::Context *context) const;
    bool isTexturable(const gl::Context *context) const;
    size_t getWidth() const;
    size_t getHeight() const;
    size_t getSamples() const;

    Error initialize(const Display *display);

    rx::ImageImpl *getImplementation() const;

    bool orphaned() const;
    gl::InitState sourceInitState() const;
    void setInitState(gl::InitState initState);

  private:
    friend class ImageSibling;

    // Called from ImageSibling only notify the image that a new target sibling exists for state
    // tracking.
    void addTargetSibling(ImageSibling *sibling);

    // Called from ImageSibling only to notify the image that a sibling (source or target) has
    // been respecified and state tracking should be updated.
    angle::Result orphanSibling(const gl::Context *context, ImageSibling *sibling);

    void notifySiblings(const ImageSibling *notifier, angle::SubjectMessage message);

    ImageState mState;
    rx::ImageImpl *mImplementation;
    bool mOrphanedAndNeedsInit;
};
}  // namespace egl

#endif  // LIBANGLE_IMAGE_H_
