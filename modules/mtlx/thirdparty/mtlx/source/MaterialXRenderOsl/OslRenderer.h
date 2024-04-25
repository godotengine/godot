//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_OSLRENDERER_H
#define MATERIALX_OSLRENDERER_H

/// @file
/// OSL code renderer

#include <MaterialXRenderOsl/Export.h>

#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/ShaderRenderer.h>

MATERIALX_NAMESPACE_BEGIN

// Shared pointer to an OslRenderer
using OslRendererPtr = std::shared_ptr<class OslRenderer>;

/// @class OslRenderer
/// Helper class for rendering generated OSL code to produce images.
///
/// The main services provided are:
///     - Source code validation: Use of "oslc" to compile and test output results
///     - Introspection check: None at this time.
///     - Binding: None at this time.
///     - Render validation: Use of "testrender" to output rendered images. Assumes source compliation was success
///       as it depends on the existence of corresponding .oso files.
///
class MX_RENDEROSL_API OslRenderer : public ShaderRenderer
{
  public:
    /// Create an OSL renderer instance
    static OslRendererPtr create(unsigned int width = 512, unsigned int height = 512, Image::BaseType baseType = Image::BaseType::UINT8);

    /// Destructor
    virtual ~OslRenderer();

    /// Color closure OSL string
    static string OSL_CLOSURE_COLOR_STRING;

    /// @name Setup
    /// @{

    /// Internal initialization required for program validation and rendering.
    /// An exception is thrown on failure.
    /// The exception will contain a list of initialization errors.
    void initialize(RenderContextHandle renderContextHandle = nullptr) override;

    /// @}
    /// @name Rendering
    /// @{

    /// Create OSL program based on an input shader
    ///
    /// A valid executable and include path must be specified before calling this method.
    /// setOslCompilerExecutable(), and setOslIncludePath().
    ///
    /// Additionally setOslOutputFilePath() should be set to allow for output of .osl and .oso
    /// files to the appropriate path location to be used as input for render validation.
    ///
    /// If render validation is not required, then the same temporary name will be used for
    /// all shaders validated using this method.
    /// @param shader Input shader
    void createProgram(ShaderPtr shader) override;

    /// Create OSL program based on shader stage source code.
    /// @param stages Map of name and source code for the shader stages.
    void createProgram(const StageMap& stages) override;

    /// Validate inputs for the compiled OSL program.
    /// Note: Currently no validation has been implemented.
    void validateInputs() override;

    /// Set the size for rendered image
    void setSize(unsigned int width, unsigned int height) override;

    /// Render OSL program to disk.
    /// This is done by using either "testshade" or "testrender".
    /// Currently only "testshade" is supported.
    ///
    /// Usage of both executables requires compiled source (.oso) files as input.
    /// A shader output must be set before running this test via the setOslOutputName() method to
    /// ensure that the appropriate .oso files can be located.
    void render() override;

    /// @}
    /// @name Utilities
    /// @{

    /// Capture the current rendered output as an image.
    ImagePtr captureImage(ImagePtr image = nullptr) override;

    /// @}
    /// @name Compilation settings
    /// @{

    /// Set the path to the OSL executable. Note that it is assumed that this
    /// references the location of the oslc executable.
    /// @param executableFilePath Path to OSL compiler executable
    void setOslCompilerExecutable(const FilePath& executableFilePath)
    {
        _oslCompilerExecutable = executableFilePath;
    }

    /// Set the search locations for OSL include files.
    /// @param dirPath Include path(s) for the OSL compiler. This should include the
    /// path to stdosl.h.
    void setOslIncludePath(const FileSearchPath& dirPath)
    {
        _oslIncludePath = dirPath;
    }

    /// Set the location where compiled OSL files will reside.
    /// @param dirPath Path to output location
    void setOslOutputFilePath(const FilePath& dirPath)
    {
        _oslOutputFilePath = dirPath;
    }

    /// Set shader parameter strings to be added to the scene XML file. These
    /// strings will set parameter overrides for the shader.
    void setShaderParameterOverrides(const StringVec& parameterOverrides)
    {
        _oslShaderParameterOverrides = parameterOverrides;
    }

    /// Set shader parameter strings to be added to the scene XML file. These
    /// strings will set parameter overrides for the shader.
    void setEnvShaderParameterOverrides(const StringVec& parameterOverrides)
    {
        _envOslShaderParameterOverrides = parameterOverrides;
    }

    /// Set the OSL shader output.
    /// This is used during render validation if "testshade" or "testrender" is executed.
    /// For testrender this value is used to replace the %shader_output% token in the
    /// input scene file.
    /// @param outputName Name of shader output
    /// @param outputType The MaterialX type of the output
    void setOslShaderOutput(const string& outputName, const string& outputType)
    {
        _oslShaderOutputName = outputName;
        _oslShaderOutputType = outputType;
    }

    /// Set the path to the OSL shading tester. Note that it is assumed that this
    /// references the location of the "testshade" executable.
    /// @param executableFilePath Path to OSL "testshade" executable
    void setOslTestShadeExecutable(const FilePath& executableFilePath)
    {
        _oslTestShadeExecutable = executableFilePath;
    }

    /// Set the path to the OSL rendering tester. Note that it is assumed that this
    /// references the location of the "testrender" executable.
    /// @param executableFilePath Path to OSL "testrender" executable
    void setOslTestRenderExecutable(const FilePath& executableFilePath)
    {
        _oslTestRenderExecutable = executableFilePath;
    }

    /// Set the XML scene file to use for testrender. This is a template file
    /// with the following tokens for replacement:
    ///     - %shader% : which will be replaced with the name of the shader to use
    ///     - %shader_output% : which will be replace with the name of the shader output to use
    /// @param templateFilePath Scene file name
    void setOslTestRenderSceneTemplateFile(const FilePath& templateFilePath)
    {
        _oslTestRenderSceneTemplateFile = templateFilePath;
    }

    /// Set the name of the shader to be used for the input XML scene file.
    /// The value is used to replace the %shader% token in the file.
    /// @param shaderName Name of shader
    void setOslShaderName(const string& shaderName)
    {
        _oslShaderName = shaderName;
    }

    /// Set the search path for dependent shaders (.oso files) which are used
    /// when rendering with testrender.
    /// @param dirPath Path to location containing .oso files.
    void setOslUtilityOSOPath(const FilePath& dirPath)
    {
        _oslUtilityOSOPath = dirPath;
    }

    /// Used to toggle to either use testrender or testshade during render validation
    /// By default testshade is used.
    /// @param useTestRender Indicate whether to use testrender.
    void useTestRender(bool useTestRender)
    {
        _useTestRender = useTestRender;
    }

    /// Set the number of rays per pixel to be used for lit surfaces.
    void setRaysPerPixelLit(int rays)
    {
        _raysPerPixelLit = rays;
    }

    /// Set the number of rays per pixel to be used for unlit surfaces.
    void setRaysPerPixelUnlit(int rays)
    {
        _raysPerPixelUnlit = rays;
    }

    ///
    /// Compile OSL code stored in a file. Will throw an exception if an error occurs.
    /// @param oslFilePath OSL file path.
    void compileOSL(const FilePath& oslFilePath);

    /// @}

  protected:
    ///
    /// Shade using OSO input file. Will throw an exception if an error occurs.
    /// @param dirPath Path to location containing input .oso file.
    /// @param shaderName Name of OSL shader. A corresponding .oso file is assumed to exist in the output path folder.
    /// @param outputName Name of OSL shader output to use.
    void shadeOSL(const FilePath& dirPath, const string& shaderName, const string& outputName);

    ///
    /// Render using OSO input file. Will throw an exception if an error occurs.
    /// @param dirPath Path to location containing input .oso file.
    /// @param shaderName Name of OSL shader. A corresponding .oso file is assumed to exist in the output path folder.
    /// @param outputName Name of OSL shader output to use.
    void renderOSL(const FilePath& dirPath, const string& shaderName, const string& outputName);

    /// Constructor
    OslRenderer(unsigned int width, unsigned int height, Image::BaseType baseType);

  private:
    FilePath _oslCompilerExecutable;
    FileSearchPath _oslIncludePath;
    FilePath _oslOutputFilePath;
    FilePath _oslOutputFileName;

    FilePath _oslTestShadeExecutable;
    FilePath _oslTestRenderExecutable;
    FilePath _oslTestRenderSceneTemplateFile;
    string _oslShaderName;
    StringVec _oslShaderParameterOverrides;
    StringVec _envOslShaderParameterOverrides;
    string _oslShaderOutputName;
    string _oslShaderOutputType;
    FilePath _oslUtilityOSOPath;
    bool _useTestRender;
    int _raysPerPixelLit;
    int _raysPerPixelUnlit;
};

MATERIALX_NAMESPACE_END

#endif
