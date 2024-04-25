//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GLSLRENDERER_H
#define MATERIALX_GLSLRENDERER_H

/// @file
/// GLSL code renderer

#include <MaterialXRenderGlsl/Export.h>

#include <MaterialXRenderGlsl/GLFramebuffer.h>
#include <MaterialXRenderGlsl/GlslProgram.h>
#include <MaterialXRenderGlsl/GLTextureHandler.h>

#include <MaterialXRender/ShaderRenderer.h>

MATERIALX_NAMESPACE_BEGIN

using GLContextPtr = std::shared_ptr<class GLContext>;
using SimpleWindowPtr = std::shared_ptr<class SimpleWindow>;

/// Shared pointer to a GlslRenderer
using GlslRendererPtr = std::shared_ptr<class GlslRenderer>;

/// @class GlslRenderer
/// Helper class for rendering generated GLSL code to produce images.
///
/// There are two main interfaces which can be used. One which takes in a HwShader and one which
/// allows for explicit setting of shader stage code.
///
/// The main services provided are:
///     - Validation: All shader stages are compiled and atteched to a GLSL shader program.
///     - Introspection: The compiled shader program is examined for uniforms and attributes.
///     - Binding: Uniforms and attributes which match the predefined variables generated the GLSL code generator
///       will have values assigned to this. This includes matrices, attribute streams, and textures.
///     - Rendering: The program with bound inputs will be used to drawing geometry to an offscreen buffer.
///     An interface is provided to save this offscreen buffer to disk using an externally defined image handler.
///
class MX_RENDERGLSL_API GlslRenderer : public ShaderRenderer
{
  public:
    /// Create a GLSL renderer instance
    static GlslRendererPtr create(unsigned int width = 512, unsigned int height = 512, Image::BaseType baseType = Image::BaseType::UINT8);

    /// Create a texture handler for OpenGL textures
    ImageHandlerPtr createImageHandler(ImageLoaderPtr imageLoader)
    {
        return GLTextureHandler::create(imageLoader);
    }

    /// Destructor
    virtual ~GlslRenderer() { }

    /// @name Setup
    /// @{

    /// Internal initialization of stages and OpenGL constructs
    /// required for program validation and rendering.
    /// An exception is thrown on failure.
    /// The exception will contain a list of initialization errors.
    /// @param renderContextHandle allows initializing the GlslRenderer with a Shared OpenGL Context
    void initialize(RenderContextHandle renderContextHandle = nullptr) override;

    /// @}
    /// @name Rendering
    /// @{

    /// Create GLSL program based on an input shader
    /// @param shader Input HwShader
    void createProgram(ShaderPtr shader) override;

    /// Create GLSL program based on shader stage source code.
    /// @param stages Map of name and source code for the shader stages.
    void createProgram(const StageMap& stages) override;

    /// Validate inputs for the program
    void validateInputs() override;

    /// Update the program with value of the uniform.
    void updateUniform(const string& name, ConstValuePtr value) override;

    /// Set the size of the rendered image
    void setSize(unsigned int width, unsigned int height) override;

    /// Render the current program to an offscreen buffer.
    void render() override;

    /// Render the current program in texture space to an off-screen buffer.
    void renderTextureSpace(const Vector2& uvMin, const Vector2& uvMax);

    /// @}
    /// @name Utilities
    /// @{

    /// Capture the current contents of the off-screen hardware buffer as an image.
    ImagePtr captureImage(ImagePtr image = nullptr) override;

    /// Return the GL frame buffer.
    GLFramebufferPtr getFramebuffer() const
    {
        return _framebuffer;
    }

    /// Return the GLSL program.
    GlslProgramPtr getProgram()
    {
        return _program;
    }

    /// Submit geometry for a screen-space quad.
    void drawScreenSpaceQuad(const Vector2& uvMin = Vector2(0.0f), const Vector2& uvMax = Vector2(1.0f));

    /// Set the screen background color.
    void setScreenColor(const Color3& screenColor)
    {
        _screenColor = screenColor;
    }

    /// Return the screen background color.
    Color3 getScreenColor() const
    {
        return _screenColor;
    }

    /// @}

  protected:
    GlslRenderer(unsigned int width, unsigned int height, Image::BaseType baseType);

  private:
    GlslProgramPtr _program;
    GLFramebufferPtr _framebuffer;

    bool _initialized;

    SimpleWindowPtr _window;
    GLContextPtr _context;
    Color3 _screenColor;
};

MATERIALX_NAMESPACE_END

#endif
