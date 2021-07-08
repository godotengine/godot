//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Stream.h: Defines the egl::Stream class, representing the stream
// where frames are streamed in. Implements EGLStreanKHR.

#ifndef LIBANGLE_STREAM_H_
#define LIBANGLE_STREAM_H_

#include <array>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "common/angleutils.h"
#include "libANGLE/AttributeMap.h"
#include "libANGLE/Debug.h"

namespace rx
{
class StreamProducerImpl;
}

namespace gl
{
class Context;
class Texture;
}  // namespace gl

namespace egl
{
class Display;
class Error;
class Thread;

class Stream final : public LabeledObject, angle::NonCopyable
{
  public:
    Stream(Display *display, const AttributeMap &attribs);
    ~Stream() override;

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    enum class ConsumerType
    {
        NoConsumer,
        GLTextureRGB,
        GLTextureYUV,
    };

    enum class ProducerType
    {
        NoProducer,
        D3D11Texture,
    };

    // A GL texture interpretation of a part of a producer frame. For use with GL texture consumers
    struct GLTextureDescription
    {
        unsigned int width;
        unsigned int height;
        unsigned int internalFormat;
        unsigned int mipLevels;
    };

    EGLenum getState() const;

    void setConsumerLatency(EGLint latency);
    EGLint getConsumerLatency() const;

    EGLuint64KHR getProducerFrame() const;
    EGLuint64KHR getConsumerFrame() const;

    void setConsumerAcquireTimeout(EGLint timeout);
    EGLint getConsumerAcquireTimeout() const;

    ConsumerType getConsumerType() const;
    ProducerType getProducerType() const;

    EGLint getPlaneCount() const;

    rx::StreamProducerImpl *getImplementation();

    // Consumer creation methods
    Error createConsumerGLTextureExternal(const AttributeMap &attributes, gl::Context *context);

    // Producer creation methods
    Error createProducerD3D11Texture(const AttributeMap &attributes);

    // Consumer methods
    Error consumerAcquire(const gl::Context *context);
    Error consumerRelease(const gl::Context *context);

    // Some consumers are bound to GL contexts. This validates that a given context is bound to the
    // stream's consumer
    bool isConsumerBoundToContext(const gl::Context *context) const;

    // Producer methods
    Error validateD3D11Texture(void *texture, const AttributeMap &attributes) const;
    Error postD3D11Texture(void *texture, const AttributeMap &attributes);

  private:
    EGLLabelKHR mLabel;

    // Associated display
    Display *mDisplay;

    // Producer Implementation
    rx::StreamProducerImpl *mProducerImplementation;

    // Associated GL context. Note that this is a weak pointer used for validation purposes only,
    // and should never be arbitrarily dereferenced without knowing the context still exists as it
    // can become dangling at any time.
    gl::Context *mContext;

    // EGL defined attributes
    EGLint mState;
    EGLuint64KHR mProducerFrame;
    EGLuint64KHR mConsumerFrame;
    EGLint mConsumerLatency;

    // EGL gltexture consumer attributes
    EGLint mConsumerAcquireTimeout;

    // EGL gltexture yuv consumer attributes
    EGLint mPlaneCount;
    struct PlaneTexture
    {
        EGLint textureUnit;
        gl::Texture *texture;
    };
    // Texture units and textures for all the planes
    std::array<PlaneTexture, 3> mPlanes;

    // Consumer and producer types
    ConsumerType mConsumerType;
    ProducerType mProducerType;

    // ANGLE-only method, used internally
    friend class gl::Texture;
    void releaseTextures();
};
}  // namespace egl

#endif  // LIBANGLE_STREAM_H_
