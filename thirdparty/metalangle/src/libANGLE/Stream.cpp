//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Stream.cpp: Implements the egl::Stream class, representing the stream
// where frames are streamed in. Implements EGLStreanKHR.

#include "libANGLE/Stream.h"

#include <EGL/eglext.h>
#include <platform/Platform.h>

#include "common/debug.h"
#include "common/mathutil.h"
#include "common/platform.h"
#include "common/utilities.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/DisplayImpl.h"
#include "libANGLE/renderer/StreamProducerImpl.h"

namespace egl
{

Stream::Stream(Display *display, const AttributeMap &attribs)
    : mLabel(nullptr),
      mDisplay(display),
      mProducerImplementation(nullptr),
      mState(EGL_STREAM_STATE_CREATED_KHR),
      mProducerFrame(0),
      mConsumerFrame(0),
      mConsumerLatency(attribs.getAsInt(EGL_CONSUMER_LATENCY_USEC_KHR, 0)),
      mConsumerAcquireTimeout(attribs.getAsInt(EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, 0)),
      mPlaneCount(0),
      mConsumerType(ConsumerType::NoConsumer),
      mProducerType(ProducerType::NoProducer)
{
    for (auto &plane : mPlanes)
    {
        plane.textureUnit = -1;
        plane.texture     = nullptr;
    }
}

Stream::~Stream()
{
    SafeDelete(mProducerImplementation);
    for (auto &plane : mPlanes)
    {
        if (plane.texture != nullptr)
        {
            plane.texture->releaseStream();
        }
    }
}

void Stream::setLabel(EGLLabelKHR label)
{
    mLabel = label;
}

EGLLabelKHR Stream::getLabel() const
{
    return mLabel;
}

void Stream::setConsumerLatency(EGLint latency)
{
    mConsumerLatency = latency;
}

EGLint Stream::getConsumerLatency() const
{
    return mConsumerLatency;
}

EGLuint64KHR Stream::getProducerFrame() const
{
    return mProducerFrame;
}

EGLuint64KHR Stream::getConsumerFrame() const
{
    return mConsumerFrame;
}

EGLenum Stream::getState() const
{
    return mState;
}

void Stream::setConsumerAcquireTimeout(EGLint timeout)
{
    mConsumerAcquireTimeout = timeout;
}

EGLint Stream::getConsumerAcquireTimeout() const
{
    return mConsumerAcquireTimeout;
}

Stream::ProducerType Stream::getProducerType() const
{
    return mProducerType;
}

Stream::ConsumerType Stream::getConsumerType() const
{
    return mConsumerType;
}

EGLint Stream::getPlaneCount() const
{
    return mPlaneCount;
}

rx::StreamProducerImpl *Stream::getImplementation()
{
    return mProducerImplementation;
}

Error Stream::createConsumerGLTextureExternal(const AttributeMap &attributes, gl::Context *context)
{
    ASSERT(mState == EGL_STREAM_STATE_CREATED_KHR);
    ASSERT(mConsumerType == ConsumerType::NoConsumer);
    ASSERT(mProducerType == ProducerType::NoProducer);
    ASSERT(context != nullptr);

    const auto &glState = context->getState();
    EGLenum bufferType  = attributes.getAsInt(EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER);
    if (bufferType == EGL_RGB_BUFFER)
    {
        mPlanes[0].texture = glState.getTargetTexture(gl::TextureType::External);
        ASSERT(mPlanes[0].texture != nullptr);
        mPlanes[0].texture->bindStream(this);
        mConsumerType = ConsumerType::GLTextureRGB;
        mPlaneCount   = 1;
    }
    else
    {
        mPlaneCount = attributes.getAsInt(EGL_YUV_NUMBER_OF_PLANES_EXT, 2);
        ASSERT(mPlaneCount <= 3);
        for (EGLint i = 0; i < mPlaneCount; i++)
        {
            // Fetch all the textures
            mPlanes[i].textureUnit = attributes.getAsInt(EGL_YUV_PLANE0_TEXTURE_UNIT_NV + i, -1);
            if (mPlanes[i].textureUnit != EGL_NONE)
            {
                mPlanes[i].texture =
                    glState.getSamplerTexture(mPlanes[i].textureUnit, gl::TextureType::External);
                ASSERT(mPlanes[i].texture != nullptr);
            }
        }

        // Bind them to the stream
        for (EGLint i = 0; i < mPlaneCount; i++)
        {
            if (mPlanes[i].textureUnit != EGL_NONE)
            {
                mPlanes[i].texture->bindStream(this);
            }
        }
        mConsumerType = ConsumerType::GLTextureYUV;
    }

    mContext = context;
    mState   = EGL_STREAM_STATE_CONNECTING_KHR;

    return NoError();
}

Error Stream::createProducerD3D11Texture(const AttributeMap &attributes)
{
    ASSERT(mState == EGL_STREAM_STATE_CONNECTING_KHR);
    ASSERT(mConsumerType == ConsumerType::GLTextureRGB ||
           mConsumerType == ConsumerType::GLTextureYUV);
    ASSERT(mProducerType == ProducerType::NoProducer);

    mProducerImplementation =
        mDisplay->getImplementation()->createStreamProducerD3DTexture(mConsumerType, attributes);
    mProducerType = ProducerType::D3D11Texture;
    mState        = EGL_STREAM_STATE_EMPTY_KHR;

    return NoError();
}

// Called when the consumer of this stream starts using the stream
Error Stream::consumerAcquire(const gl::Context *context)
{
    ASSERT(mState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR ||
           mState == EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR);
    ASSERT(mConsumerType == ConsumerType::GLTextureRGB ||
           mConsumerType == ConsumerType::GLTextureYUV);
    ASSERT(mProducerType == ProducerType::D3D11Texture);

    mState = EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR;
    mConsumerFrame++;

    // Bind the planes to the gl textures
    for (int i = 0; i < mPlaneCount; i++)
    {
        if (mPlanes[i].texture != nullptr)
        {
            ANGLE_TRY(ResultToEGL(mPlanes[i].texture->acquireImageFromStream(
                context, mProducerImplementation->getGLFrameDescription(i))));
        }
    }

    return NoError();
}

Error Stream::consumerRelease(const gl::Context *context)
{
    ASSERT(mState == EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR ||
           mState == EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR);
    ASSERT(mConsumerType == ConsumerType::GLTextureRGB ||
           mConsumerType == ConsumerType::GLTextureYUV);
    ASSERT(mProducerType == ProducerType::D3D11Texture);

    // Release the images
    for (int i = 0; i < mPlaneCount; i++)
    {
        if (mPlanes[i].texture != nullptr)
        {
            ANGLE_TRY(ResultToEGL(mPlanes[i].texture->releaseImageFromStream(context)));
        }
    }

    return NoError();
}

bool Stream::isConsumerBoundToContext(const gl::Context *context) const
{
    ASSERT(context != nullptr);
    return (context == mContext);
}

Error Stream::validateD3D11Texture(void *texture, const AttributeMap &attributes) const
{
    ASSERT(mConsumerType == ConsumerType::GLTextureRGB ||
           mConsumerType == ConsumerType::GLTextureYUV);
    ASSERT(mProducerType == ProducerType::D3D11Texture);
    ASSERT(mProducerImplementation != nullptr);

    return mProducerImplementation->validateD3DTexture(texture, attributes);
}

Error Stream::postD3D11Texture(void *texture, const AttributeMap &attributes)
{
    ASSERT(mConsumerType == ConsumerType::GLTextureRGB ||
           mConsumerType == ConsumerType::GLTextureYUV);
    ASSERT(mProducerType == ProducerType::D3D11Texture);

    mProducerImplementation->postD3DTexture(texture, attributes);
    mProducerFrame++;

    mState = EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR;

    return NoError();
}

// This is called when a texture object associated with this stream is destroyed. Even if multiple
// textures are bound, one being destroyed invalidates the stream, so all the remaining textures
// will be released and the stream will be invalidated.
void Stream::releaseTextures()
{
    for (auto &plane : mPlanes)
    {
        if (plane.texture != nullptr)
        {
            plane.texture->releaseStream();
            plane.texture = nullptr;
        }
    }
    mConsumerType = ConsumerType::NoConsumer;
    mProducerType = ProducerType::NoProducer;
    mState        = EGL_STREAM_STATE_DISCONNECTED_KHR;
}

}  // namespace egl
