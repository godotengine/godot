//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderGlsl/GLFramebuffer.h>

#include <MaterialXRenderGlsl/GlslProgram.h>
#include <MaterialXRenderGlsl/GlslRenderer.h>
#include <MaterialXRenderGlsl/GLTextureHandler.h>

#include <MaterialXRenderGlsl/External/Glad/glad.h>

MATERIALX_NAMESPACE_BEGIN

//
// GLFramebuffer methods
//

GLFramebufferPtr GLFramebuffer::create(unsigned int width, unsigned int height, unsigned channelCount, Image::BaseType baseType)
{
    return GLFramebufferPtr(new GLFramebuffer(width, height, channelCount, baseType));
}

GLFramebuffer::GLFramebuffer(unsigned int width, unsigned int height, unsigned int channelCount, Image::BaseType baseType) :
    _width(width),
    _height(height),
    _channelCount(channelCount),
    _baseType(baseType),
    _encodeSrgb(false),
    _framebuffer(0),
    _colorTexture(0),
    _depthTexture(0)
{
    if (!glGenFramebuffers)
    {
        gladLoadGL();
    }

    // Convert texture format to OpenGL.
    int glType, glFormat, glInternalFormat;
    GLTextureHandler::mapTextureFormatToGL(baseType, channelCount, true, glType, glFormat, glInternalFormat);

    // Create and bind framebuffer.
    glGenFramebuffers(1, &_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);

    // Create the offscreen color target and attach to the framebuffer.
    glGenTextures(1, &_colorTexture);
    glBindTexture(GL_TEXTURE_2D, _colorTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, _width, _height, 0, glFormat, glType, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _colorTexture, 0);

    // Create the offscreen depth target and attach to the framebuffer.
    glGenTextures(1, &_depthTexture);
    glBindTexture(GL_TEXTURE_2D, _depthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, _width, _height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depthTexture, 0);

    glBindTexture(GL_TEXTURE_2D, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
    glDrawBuffer(GL_NONE);

    // Validate the framebuffer.
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
        glDeleteFramebuffers(1, &_framebuffer);
        _framebuffer = GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID;

        string errorMessage;
        switch (status)
        {
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                errorMessage = "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                errorMessage = "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                errorMessage = "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                errorMessage = "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
                break;
            case GL_FRAMEBUFFER_UNSUPPORTED:
                errorMessage = "GL_FRAMEBUFFER_UNSUPPORTED";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                errorMessage = "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
                break;
            case GL_FRAMEBUFFER_UNDEFINED:
                errorMessage = "GL_FRAMEBUFFER_UNDEFINED";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
                errorMessage = "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
                break;
            default:
                errorMessage = std::to_string(status);
                break;
        }

        throw ExceptionRenderError("Frame buffer object setup failed: " + errorMessage);
    }

    // Unbind on cleanup
    glBindFramebuffer(GL_FRAMEBUFFER, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
}

GLFramebuffer::~GLFramebuffer()
{
    if (_framebuffer)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
        glDeleteTextures(1, &_colorTexture);
        glDeleteTextures(1, &_depthTexture);
        glDeleteFramebuffers(1, &_framebuffer);
    }
}

void GLFramebuffer::bind()
{
    if (!_framebuffer)
    {
        throw ExceptionRenderError("No framebuffer exists to bind");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
    GLenum colorList[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, colorList);

    if (_encodeSrgb)
    {
        glEnable(GL_FRAMEBUFFER_SRGB);
    }
    else
    {
        glDisable(GL_FRAMEBUFFER_SRGB);
    }

    glViewport(0, 0, _width, _height);
}

void GLFramebuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
    glDrawBuffer(GL_NONE);
}

ImagePtr GLFramebuffer::getColorImage(ImagePtr image)
{
    if (!image)
    {
        image = Image::create(_width, _height, _channelCount, _baseType);
        image->createResourceBuffer();
    }

    int glType, glFormat, glInternalFormat;
    GLTextureHandler::mapTextureFormatToGL(_baseType, _channelCount, false, glType, glFormat, glInternalFormat);

    bind();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, image->getWidth(), image->getHeight(), glFormat, glType, image->getResourceBuffer());
    unbind();

    return image;
}

void GLFramebuffer::blit()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _framebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDrawBuffer(GL_BACK);

    glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height,
                      GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

MATERIALX_NAMESPACE_END
