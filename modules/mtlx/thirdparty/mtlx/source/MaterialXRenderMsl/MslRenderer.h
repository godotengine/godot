//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MSLRENDERER_H
#define MATERIALX_MSLRENDERER_H

/// @file
/// MSL code renderer

#include <MaterialXRenderMsl/Export.h>

#include <MaterialXRenderMsl/MetalFramebuffer.h>
#include <MaterialXRenderMsl/MslPipelineStateObject.h>
#include <MaterialXRenderMsl/MetalTextureHandler.h>

#include <MaterialXRender/ShaderRenderer.h>

#import <Metal/Metal.h>

MATERIALX_NAMESPACE_BEGIN

using SimpleWindowPtr = std::shared_ptr<class SimpleWindow>;

/// Shared pointer to a MslRenderer
using MslRendererPtr = std::shared_ptr<class MslRenderer>;

/// @class MslRenderer
/// Helper class for rendering generated MSL code to produce images.
///
/// There are two main interfaces which can be used. One which takes in a HwShader and one which
/// allows for explicit setting of shader stage code.
///
/// The main services provided are:
///     - Validation: All shader stages are compiled and atteched to a MSL shader program.
///     - Introspection: The compiled shader program is examined for uniforms and attributes.
///     - Binding: Uniforms and attributes which match the predefined variables generated the MSL code generator
///       will have values assigned to this. This includes matrices, attribute streams, and textures.
///     - Rendering: The program with bound inputs will be used to drawing geometry to an offscreen buffer.
///     An interface is provided to save this offscreen buffer to disk using an externally defined image handler.
///
class MX_RENDERMSL_API MslRenderer : public ShaderRenderer
{
  public:
    /// Create a MSL renderer instance
    static MslRendererPtr create(unsigned int width = 512, unsigned int height = 512, Image::BaseType baseType = Image::BaseType::UINT8);
    
    /// Create a texture handler for Metal textures
    ImageHandlerPtr createImageHandler(ImageLoaderPtr imageLoader)
    {
        return MetalTextureHandler::create(_device, imageLoader);
    }
    
    /// Returns Metal Device used for rendering
    id<MTLDevice> getMetalDevice() const;

    /// Destructor
    virtual ~MslRenderer() { }

    /// @name Setup
    /// @{

    /// Internal initialization of stages and OpenGL constructs
    /// required for program validation and rendering.
    /// An exception is thrown on failure.
    /// The exception will contain a list of initialization errors.
    void initialize(RenderContextHandle renderContextHandle = nullptr) override;

    /// @}
    /// @name Rendering
    /// @{

    /// Create MSL program based on an input shader
    /// @param shader Input HwShader
    void createProgram(ShaderPtr shader) override;

    /// Create MSL program based on shader stage source code.
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

    /// Return the Metal frame buffer.
    MetalFramebufferPtr getFramebuffer() const
    {
        return _framebuffer;
    }

    /// Return the MSL program.
    MslProgramPtr getProgram()
    {
        return _program;
    }

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
    MslRenderer(unsigned int width, unsigned int height, Image::BaseType baseType);
    
    void triggerProgrammaticCapture();
    void stopProgrammaticCapture();
    
    void createFrameBuffer(bool encodeSrgb);
    
  private:
    MslProgramPtr        _program;

    id<MTLDevice>        _device = nil;
    id<MTLCommandQueue>  _cmdQueue = nil;
    id<MTLCommandBuffer> _cmdBuffer = nil;
    
    MetalFramebufferPtr  _framebuffer;

    bool _initialized;

    SimpleWindowPtr _window;
    Color3 _screenColor;
};

MATERIALX_NAMESPACE_END

#endif
