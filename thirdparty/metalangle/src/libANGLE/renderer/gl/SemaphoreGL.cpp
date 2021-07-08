//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "libANGLE/renderer/gl/SemaphoreGL.h"

#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/BufferGL.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/TextureGL.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"

namespace rx
{
namespace
{
void GatherNativeBufferIDs(const gl::BufferBarrierVector &bufferBarriers,
                           gl::BarrierVector<GLuint> *outIDs)
{
    outIDs->resize(bufferBarriers.size());
    for (GLuint bufferIdx = 0; bufferIdx < bufferBarriers.size(); bufferIdx++)
    {
        (*outIDs)[bufferIdx] = GetImplAs<BufferGL>(bufferBarriers[bufferIdx])->getBufferID();
    }
}

void GatherNativeTextureIDs(const gl::TextureBarrierVector &textureBarriers,
                            gl::BarrierVector<GLuint> *outIDs,
                            gl::BarrierVector<GLenum> *outLayouts)
{
    outIDs->resize(textureBarriers.size());
    outLayouts->resize(textureBarriers.size());
    for (GLuint textureIdx = 0; textureIdx < textureBarriers.size(); textureIdx++)
    {
        (*outIDs)[textureIdx] =
            GetImplAs<TextureGL>(textureBarriers[textureIdx].texture)->getTextureID();
        (*outLayouts)[textureIdx] = textureBarriers[textureIdx].layout;
    }
}

}  // anonymous namespace

SemaphoreGL::SemaphoreGL(GLuint semaphoreID) : mSemaphoreID(semaphoreID)
{
    ASSERT(mSemaphoreID != 0);
}

SemaphoreGL::~SemaphoreGL()
{
    ASSERT(mSemaphoreID == 0);
}

void SemaphoreGL::onDestroy(const gl::Context *context)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->deleteSemaphoresEXT(1, &mSemaphoreID);
    mSemaphoreID = 0;
}

angle::Result SemaphoreGL::importFd(gl::Context *context, gl::HandleType handleType, GLint fd)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->importSemaphoreFdEXT(mSemaphoreID, ToGLenum(handleType), fd);
    return angle::Result::Continue;
}

angle::Result SemaphoreGL::wait(gl::Context *context,
                                const gl::BufferBarrierVector &bufferBarriers,
                                const gl::TextureBarrierVector &textureBarriers)
{
    const FunctionsGL *functions = GetFunctionsGL(context);

    gl::BarrierVector<GLuint> bufferIDs(bufferBarriers.size());
    GatherNativeBufferIDs(bufferBarriers, &bufferIDs);

    gl::BarrierVector<GLuint> textureIDs(textureBarriers.size());
    gl::BarrierVector<GLenum> textureLayouts(textureBarriers.size());
    GatherNativeTextureIDs(textureBarriers, &textureIDs, &textureLayouts);
    ASSERT(textureIDs.size() == textureLayouts.size());

    functions->waitSemaphoreEXT(mSemaphoreID, static_cast<GLuint>(bufferIDs.size()),
                                bufferIDs.data(), static_cast<GLuint>(textureIDs.size()),
                                textureIDs.data(), textureLayouts.data());

    return angle::Result::Continue;
}

angle::Result SemaphoreGL::signal(gl::Context *context,
                                  const gl::BufferBarrierVector &bufferBarriers,
                                  const gl::TextureBarrierVector &textureBarriers)
{
    const FunctionsGL *functions = GetFunctionsGL(context);

    gl::BarrierVector<GLuint> bufferIDs(bufferBarriers.size());
    GatherNativeBufferIDs(bufferBarriers, &bufferIDs);

    gl::BarrierVector<GLuint> textureIDs(textureBarriers.size());
    gl::BarrierVector<GLenum> textureLayouts(textureBarriers.size());
    GatherNativeTextureIDs(textureBarriers, &textureIDs, &textureLayouts);
    ASSERT(textureIDs.size() == textureLayouts.size());

    functions->signalSemaphoreEXT(mSemaphoreID, static_cast<GLuint>(bufferIDs.size()),
                                  bufferIDs.data(), static_cast<GLuint>(textureIDs.size()),
                                  textureIDs.data(), textureLayouts.data());

    return angle::Result::Continue;
}

GLuint SemaphoreGL::getSemaphoreID() const
{
    return mSemaphoreID;
}
}  // namespace rx
