//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// StreamProducerImpl.h: Defines the abstract rx::StreamProducerImpl class.

#ifndef LIBANGLE_RENDERER_STREAMPRODUCERIMPL_H_
#define LIBANGLE_RENDERER_STREAMPRODUCERIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/Stream.h"

namespace rx
{

class StreamProducerImpl : angle::NonCopyable
{
  public:
    explicit StreamProducerImpl() {}
    virtual ~StreamProducerImpl() {}

    // Validates the ability for the producer to accept an arbitrary pointer to a frame. All
    // pointers should be validated through this function before being used to produce a frame.
    virtual egl::Error validateD3DTexture(void *pointer,
                                          const egl::AttributeMap &attributes) const = 0;

    // Constructs a frame from an arbitrary external pointer that points to producer specific frame
    // data. Replaces the internal frame with the new one.
    virtual void postD3DTexture(void *pointer, const egl::AttributeMap &attributes) = 0;

    // Returns an OpenGL texture interpretation of some frame attributes for the purpose of
    // constructing an OpenGL texture from a frame. Depending on the producer and consumer, some
    // frames may have multiple "planes" with different OpenGL texture representations.
    virtual egl::Stream::GLTextureDescription getGLFrameDescription(int planeIndex) = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_STREAMPRODUCERIMPL_H_
