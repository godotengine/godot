//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClearMultiviewGL:
//   A helper for clearing multiview side-by-side and layered framebuffers.
//

#include "libANGLE/renderer/gl/ClearMultiviewGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"
#include "libANGLE/renderer/gl/TextureGL.h"

#include "libANGLE/Framebuffer.h"

namespace rx
{

ClearMultiviewGL::ClearMultiviewGL(const FunctionsGL *functions, StateManagerGL *stateManager)
    : mFunctions(functions), mStateManager(stateManager), mFramebuffer(0u)
{}

ClearMultiviewGL::~ClearMultiviewGL()
{
    if (mFramebuffer != 0u)
    {
        mFunctions->deleteFramebuffers(1, &mFramebuffer);
    }
}

void ClearMultiviewGL::clearMultiviewFBO(const gl::FramebufferState &state,
                                         const gl::Rectangle &scissorBase,
                                         ClearCommandType clearCommandType,
                                         GLbitfield mask,
                                         GLenum buffer,
                                         GLint drawbuffer,
                                         const uint8_t *values,
                                         GLfloat depth,
                                         GLint stencil)
{
    const gl::FramebufferAttachment *firstAttachment = state.getFirstNonNullAttachment();
    if (firstAttachment->isMultiview())
    {
        clearLayeredFBO(state, clearCommandType, mask, buffer, drawbuffer, values, depth, stencil);
    }
}

void ClearMultiviewGL::clearLayeredFBO(const gl::FramebufferState &state,
                                       ClearCommandType clearCommandType,
                                       GLbitfield mask,
                                       GLenum buffer,
                                       GLint drawbuffer,
                                       const uint8_t *values,
                                       GLfloat depth,
                                       GLint stencil)
{
    initializeResources();

    mStateManager->bindFramebuffer(GL_DRAW_FRAMEBUFFER, mFramebuffer);

    const gl::FramebufferAttachment *firstAttachment = state.getFirstNonNullAttachment();
    ASSERT(firstAttachment->isMultiview());

    const auto &drawBuffers = state.getDrawBufferStates();
    mFunctions->drawBuffers(static_cast<GLsizei>(drawBuffers.size()), drawBuffers.data());

    // Attach the new attachments and clear.
    int numViews      = firstAttachment->getNumViews();
    int baseViewIndex = firstAttachment->getBaseViewIndex();
    for (int i = 0; i < numViews; ++i)
    {
        attachTextures(state, baseViewIndex + i);
        genericClear(clearCommandType, mask, buffer, drawbuffer, values, depth, stencil);
    }

    detachTextures(state);
}

void ClearMultiviewGL::genericClear(ClearCommandType clearCommandType,
                                    GLbitfield mask,
                                    GLenum buffer,
                                    GLint drawbuffer,
                                    const uint8_t *values,
                                    GLfloat depth,
                                    GLint stencil)
{
    switch (clearCommandType)
    {
        case ClearCommandType::Clear:
            mFunctions->clear(mask);
            break;
        case ClearCommandType::ClearBufferfv:
            mFunctions->clearBufferfv(buffer, drawbuffer,
                                      reinterpret_cast<const GLfloat *>(values));
            break;
        case ClearCommandType::ClearBufferuiv:
            mFunctions->clearBufferuiv(buffer, drawbuffer,
                                       reinterpret_cast<const GLuint *>(values));
            break;
        case ClearCommandType::ClearBufferiv:
            mFunctions->clearBufferiv(buffer, drawbuffer, reinterpret_cast<const GLint *>(values));
            break;
        case ClearCommandType::ClearBufferfi:
            mFunctions->clearBufferfi(buffer, drawbuffer, depth, stencil);
            break;
        default:
            UNREACHABLE();
    }
}

void ClearMultiviewGL::attachTextures(const gl::FramebufferState &state, int layer)
{
    for (auto drawBufferId : state.getEnabledDrawBuffers())
    {
        const gl::FramebufferAttachment *attachment = state.getColorAttachment(drawBufferId);
        if (attachment == nullptr)
        {
            continue;
        }

        const auto &imageIndex = attachment->getTextureImageIndex();
        ASSERT(imageIndex.getType() == gl::TextureType::_2DArray);

        GLenum colorAttachment =
            static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + static_cast<int>(drawBufferId));
        const TextureGL *textureGL = GetImplAs<TextureGL>(attachment->getTexture());
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, colorAttachment,
                                            textureGL->getTextureID(), imageIndex.getLevelIndex(),
                                            layer);
    }

    const gl::FramebufferAttachment *depthStencilAttachment = state.getDepthStencilAttachment();
    const gl::FramebufferAttachment *depthAttachment        = state.getDepthAttachment();
    const gl::FramebufferAttachment *stencilAttachment      = state.getStencilAttachment();
    if (depthStencilAttachment != nullptr)
    {
        const auto &imageIndex = depthStencilAttachment->getTextureImageIndex();
        ASSERT(imageIndex.getType() == gl::TextureType::_2DArray);

        const TextureGL *textureGL = GetImplAs<TextureGL>(depthStencilAttachment->getTexture());
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                            textureGL->getTextureID(), imageIndex.getLevelIndex(),
                                            layer);
    }
    else if (depthAttachment != nullptr)
    {
        const auto &imageIndex = depthAttachment->getTextureImageIndex();
        ASSERT(imageIndex.getType() == gl::TextureType::_2DArray);

        const TextureGL *textureGL = GetImplAs<TextureGL>(depthAttachment->getTexture());
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                            textureGL->getTextureID(), imageIndex.getLevelIndex(),
                                            layer);
    }
    else if (stencilAttachment != nullptr)
    {
        const auto &imageIndex = stencilAttachment->getTextureImageIndex();
        ASSERT(imageIndex.getType() == gl::TextureType::_2DArray);

        const TextureGL *textureGL = GetImplAs<TextureGL>(stencilAttachment->getTexture());
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                                            textureGL->getTextureID(), imageIndex.getLevelIndex(),
                                            layer);
    }
}

void ClearMultiviewGL::detachTextures(const gl::FramebufferState &state)
{
    for (auto drawBufferId : state.getEnabledDrawBuffers())
    {
        const gl::FramebufferAttachment *attachment = state.getColorAttachment(drawBufferId);
        if (attachment == nullptr)
        {
            continue;
        }

        GLenum colorAttachment =
            static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + static_cast<int>(drawBufferId));
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, colorAttachment, 0, 0, 0);
    }

    const gl::FramebufferAttachment *depthStencilAttachment = state.getDepthStencilAttachment();
    const gl::FramebufferAttachment *depthAttachment        = state.getDepthAttachment();
    const gl::FramebufferAttachment *stencilAttachment      = state.getStencilAttachment();
    if (depthStencilAttachment != nullptr)
    {
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 0, 0,
                                            0);
    }
    else if (depthAttachment != nullptr)
    {
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0, 0);
    }
    else if (stencilAttachment != nullptr)
    {
        mFunctions->framebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, 0, 0, 0);
    }
}

void ClearMultiviewGL::initializeResources()
{
    if (mFramebuffer == 0u)
    {
        mFunctions->genFramebuffers(1, &mFramebuffer);
    }
    ASSERT(mFramebuffer != 0u);
}

}  // namespace rx
