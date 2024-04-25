//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderGlsl/GlslRenderer.h>

#include <MaterialXRenderGlsl/External/Glad/glad.h>
#include <MaterialXRenderGlsl/GLContext.h>
#include <MaterialXRenderGlsl/GLUtil.h>

#include <MaterialXRenderHw/SimpleWindow.h>
#include <MaterialXRender/TinyObjLoader.h>
#include <MaterialXGenShader/HwShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

//
// GlslRenderer methods
//

GlslRendererPtr GlslRenderer::create(unsigned int width, unsigned int height, Image::BaseType baseType)
{
    return GlslRendererPtr(new GlslRenderer(width, height, baseType));
}

GlslRenderer::GlslRenderer(unsigned int width, unsigned int height, Image::BaseType baseType) :
    ShaderRenderer(width, height, baseType, MatrixConvention::OpenGL),
    _initialized(false),
    _screenColor(DEFAULT_SCREEN_COLOR_LIN_REC709)
{
    _program = GlslProgram::create();

    _geometryHandler = GeometryHandler::create();
    _geometryHandler->addLoader(TinyObjLoader::create());
}

void GlslRenderer::initialize(RenderContextHandle renderContextHandle)
{
    if (!_initialized)
    {
        // Create window
        _window = SimpleWindow::create();

        if (!_window->initialize("Renderer Window", _width, _height, nullptr))
        {
            throw ExceptionRenderError("Failed to initialize renderer window");
        }

        // Create offscreen context
        _context = GLContext::create(_window, (HardwareContextHandle) renderContextHandle);
        if (!_context)
        {
            throw ExceptionRenderError("Failed to create OpenGL context for renderer");
        }

        if (_context->makeCurrent())
        {
            // Initialize glad
            if (!gladLoadGL())
            {
                throw ExceptionRenderError("OpenGL support is required");
            }

            glClearStencil(0);

            _framebuffer = GLFramebuffer::create(_width, _height, 4, _baseType);
            _initialized = true;
        }
    }
}

void GlslRenderer::createProgram(ShaderPtr shader)
{
    if (!_context || !_context->makeCurrent())
    {
        throw ExceptionRenderError("Invalid OpenGL context in createProgram");
    }

    _program->setStages(shader);
    _program->build();
}

void GlslRenderer::createProgram(const StageMap& stages)
{
    if (!_context || !_context->makeCurrent())
    {
        throw ExceptionRenderError("Invalid OpenGL context in createProgram");
    }

    for (const auto& it : stages)
    {
        _program->addStage(it.first, it.second);
    }
    _program->build();
}

void GlslRenderer::renderTextureSpace(const Vector2& uvMin, const Vector2& uvMax)
{
    _program->bind();
    _program->bindTextures(_imageHandler);

    _framebuffer->bind();
    drawScreenSpaceQuad(uvMin, uvMax);
    _framebuffer->unbind();

    _program->unbind();
}

void GlslRenderer::validateInputs()
{
    if (!_context || !_context->makeCurrent())
    {
        throw ExceptionRenderError("Invalid OpenGL context in validateInputs");
    }

    // Check that the generated uniforms and attributes are valid
    _program->getUniformsList();
    _program->getAttributesList();
}

void GlslRenderer::updateUniform(const string& name, ConstValuePtr value)
{
    if (!_program->bind())
    {
        return;
    }

    _program->bindUniform(name, value);
}

void GlslRenderer::setSize(unsigned int width, unsigned int height)
{
    if (_context->makeCurrent())
    {
        if (!_framebuffer ||
             _framebuffer->getWidth() != width ||
             _framebuffer->getHeight() != height)
        {
            _framebuffer = GLFramebuffer::create(width, height, 4, _baseType);
        }
        _width = width;
        _height = height;
    }
}

void GlslRenderer::render()
{
    if (!_context || !_context->makeCurrent())
    {
        throw ExceptionRenderError("Invalid OpenGL context in render");
    }

    // Set up target
    _framebuffer->bind();

    glClearColor(_screenColor[0], _screenColor[1], _screenColor[2], 1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glDepthFunc(GL_LESS);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    try
    {
        // Bind program and input parameters
        if (_program)
        {
            // Check if we have any attributes to bind. If not then
            // there is nothing to draw
            if (!_program->hasActiveAttributes())
            {
                throw ExceptionRenderError("Program has no input vertex data");
            }
            else
            {
                // Bind the shader program.
                if (!_program->bind())
                {
                    throw ExceptionRenderError("Cannot bind inputs without a valid program");
                }

                // Update uniforms and attributes.
                _program->getUniformsList();
                _program->getAttributesList();

                // Bind shader properties.
                _program->bindViewInformation(_camera);
                _program->bindTextures(_imageHandler);
                _program->bindLighting(_lightHandler, _imageHandler);
                _program->bindTimeAndFrame();

                // Set blend state for the given material.
                bool isTransparent = _program->getShader()->hasAttribute(HW::ATTR_TRANSPARENT);
                if (isTransparent)
                {
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                }
                else
                {
                    glDisable(GL_BLEND);
                }

                // Bind each mesh and draw its partitions.
                for (MeshPtr mesh : _geometryHandler->getMeshes())
                {
                    _program->bindMesh(mesh);
                    for (size_t i = 0; i < mesh->getPartitionCount(); i++)
                    {
                        MeshPartitionPtr part = mesh->getPartition(i);
                        _program->bindPartition(part);
                        MeshIndexBuffer& indexData = part->getIndices();

                        if (isTransparent)
                        {
                            glEnable(GL_CULL_FACE);
                            glCullFace(GL_FRONT);
                            glDrawElements(GL_TRIANGLES, (GLsizei) indexData.size(), GL_UNSIGNED_INT, (void*) 0);
                            glCullFace(GL_BACK);
                            glDisable(GL_CULL_FACE);
                        }
                        glDrawElements(GL_TRIANGLES, (GLsizei) indexData.size(), GL_UNSIGNED_INT, (void*) 0);
                    }
                }

                // Unbind resources
                _imageHandler->unbindImages();
                _program->unbind();

                // Restore blend state.
                if (_program->getShader()->hasAttribute(HW::ATTR_TRANSPARENT))
                {
                    glDisable(GL_BLEND);
                }
            }
        }
    }
    catch (ExceptionRenderError& e)
    {
        _framebuffer->unbind();
        throw e;
    }

    // Unset target
    _framebuffer->unbind();
}

ImagePtr GlslRenderer::captureImage(ImagePtr image)
{
    return _framebuffer->getColorImage(image);
}

void GlslRenderer::drawScreenSpaceQuad(const Vector2& uvMin, const Vector2& uvMax)
{
    const float QUAD_VERTICES[] =
    {
         1.0f,  1.0f, 0.0f, uvMax[0], uvMax[1], // position, texcoord
         1.0f, -1.0f, 0.0f, uvMax[0], uvMin[1],
        -1.0f, -1.0f, 0.0f, uvMin[0], uvMin[1],
        -1.0f,  1.0f, 0.0f, uvMin[0], uvMax[1]
    };
    const unsigned int QUAD_INDICES[] =
    {
        0, 1, 3,
        1, 2, 3
    };
    const unsigned int VERTEX_STRIDE = 5;
    const unsigned int TEXCOORD_OFFSET = 3;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD_VERTICES), QUAD_VERTICES, GL_STATIC_DRAW);

    for (const auto& pair : _program->getAttributesList())
    {
        if (pair.first.find(HW::IN_POSITION) != std::string::npos)
        {
            glEnableVertexAttribArray(pair.second->location);
            glVertexAttribPointer(pair.second->location, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE * sizeof(float), (void*) 0);
        }

        if (pair.first.find(HW::IN_TEXCOORD + "_") != std::string::npos)
        {
            glEnableVertexAttribArray(pair.second->location);
            glVertexAttribPointer(pair.second->location, 2, GL_FLOAT, GL_FALSE, VERTEX_STRIDE * sizeof(float), (void*) (TEXCOORD_OFFSET * sizeof(float)));
        }
    }

    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(QUAD_INDICES), QUAD_INDICES, GL_STATIC_DRAW);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindVertexArray(GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
    glBindBuffer(GL_ARRAY_BUFFER, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);

    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    checkGlErrors("after draw screen-space quad");
}

MATERIALX_NAMESPACE_END
