//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef RENDER_UTIL_H
#define RENDER_UTIL_H

#include <MaterialXTest/MaterialXGenShader/GenShaderUtil.h>

#include <MaterialXRender/Mesh.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/Timer.h>
#include <MaterialXRender/Util.h>

#define LOG_TO_FILE

namespace mx = MaterialX;

// Utilities for running render tests.
//
// Execution uses existing code generator instances to produce the code and corresponding renderer
// instance to check the validity of the generated code by compiling and / or rendering
// the code to produce images on disk.
//
// Input uniform and stream checking as well as node implementation coverage and profiling
// can also be performed depending on the options enabled.
//
// See the test suite file "_options.mtlx" which is parsed during validaiton to
// restrive validation options.
//
namespace RenderUtil
{

// Per language profile times
//
class LanguageProfileTimes
{
  public:
    void print(const std::string& label, std::ostream& output) const
    {
        output << label << std::endl;
        output << "\tTotal: " << totalTime << " seconds" << std::endl;;
        output << "\tSetup: " << setupTime << " seconds" << std::endl;;
        output << "\tTransparency: " << transparencyTime << " seconds" << std::endl;;
        output << "\tGeneration: " << generationTime << " seconds" << std::endl;;
        output << "\tCompile: " << compileTime << " seconds" << std::endl;
        output << "\tRender: " << renderTime << " seconds" << std::endl;
        output << "\tI/O: " << ioTime << " seconds" << std::endl;
        output << "\tImage save: " << imageSaveTime << " seconds" << std::endl;
    }
    double totalTime = 0.0;
    double setupTime = 0.0;
    double transparencyTime = 0.0;
    double generationTime = 0.0;
    double compileTime = 0.0;
    double renderTime = 0.0;
    double ioTime = 0.0;
    double imageSaveTime = 0.0;
};

// Render validation profiling structure
//
class RenderProfileTimes
{
  public:
    void print(std::ostream& output) const
    {
        output << "Overall time: " << languageTimes.totalTime << " seconds" << std::endl;
        output << "\tI/O time: " << ioTime << " seconds" << std::endl;
        output << "\tValidation time: " << validateTime << " seconds" << std::endl;
        output << "\tRenderable search time: " << renderableSearchTime << " seconds" << std::endl;

        languageTimes.print("Profile Times:", output);

        output << "Elements tested: " << elementsTested << std::endl;
    }

    LanguageProfileTimes languageTimes;
    double totalTime = 0;
    double ioTime = 0.0;
    double validateTime = 0.0;
    double renderableSearchTime = 0.0;
    unsigned int elementsTested = 0;
};

// Base class used for performing compilation and render tests for a given
// shading language and target.
//
class ShaderRenderTester
{
  public:
    ShaderRenderTester(mx::ShaderGeneratorPtr shaderGenerator);
    virtual ~ShaderRenderTester();

    bool validate(const mx::FilePath optionsFilePath);

    void setEmitColorTransforms(bool val)
    {
        _emitColorTransforms = val;
    }

  protected:
    // Check if testing should be performed based in input options
#if defined(MATERIALX_TEST_RENDER)
    virtual bool runTest(const GenShaderUtil::TestSuiteOptions& testOptions)
    {
        return (testOptions.targets.count(_shaderGenerator->getTarget()) > 0);
    }
#else
    virtual bool runTest(const GenShaderUtil::TestSuiteOptions& /*testOptions*/)
    {
        return false;
    }
#endif

    // Load dependencies
    void loadDependentLibraries(GenShaderUtil::TestSuiteOptions options, mx::FileSearchPath searchPath,
                             mx::DocumentPtr& dependLib);

    // Load any additional libraries requird by the generator
    virtual void loadAdditionalLibraries(mx::DocumentPtr /*dependLib*/,
                                         GenShaderUtil::TestSuiteOptions& /*options*/) {};

    //
    // Code generation methods
    //

    // Register any lights used by the generation context
    virtual void registerLights(mx::DocumentPtr /*dependLib*/,
                                const GenShaderUtil::TestSuiteOptions &/*options*/,
                                mx::GenContext& /*context*/) {};

    //
    // Code validation methods (compile and render)
    //

    // Create a renderer for the generated code
    virtual void createRenderer(std::ostream& log) = 0;

    // Run the renderer
    virtual bool runRenderer(const std::string& shaderName,
        mx::TypedElementPtr element,
        mx::GenContext& context,
        mx::DocumentPtr doc,
        std::ostream& log,
        const GenShaderUtil::TestSuiteOptions& testOptions,
        RenderProfileTimes& profileTimes,
        const mx::FileSearchPath& imageSearchPath,
        const std::string& outputPath = ".",
        mx::ImageVec* imageVec = nullptr) = 0;

    // Save an image
    virtual bool saveImage(const mx::FilePath&, mx::ConstImagePtr, bool) const { return false;  };

    // Create a list of generation options based on unit test options
    // These options will override the original generation context options.
    void getGenerationOptions(const GenShaderUtil::TestSuiteOptions& testOptions,
                              const mx::GenOptions& originalOptions,
                              std::vector<mx::GenOptions>& optionsList);

    // Print execution summary
    void printRunLog(const RenderProfileTimes &profileTimes,
                     const GenShaderUtil::TestSuiteOptions& options,
                     std::ostream& stream,
                     mx::DocumentPtr dependLib);

    // If these streams don't exist add them for testing purposes
    void addAdditionalTestStreams(mx::MeshPtr mesh);

    // Generator to use
    mx::ShaderGeneratorPtr _shaderGenerator;
    // Whether to resolve image file name references before code generation
    bool _resolveImageFilenames;
    mx::StringResolverPtr _customFilenameResolver;

    // Color management information
    mx::ColorManagementSystemPtr _colorManagementSystem;
    mx::FilePath _colorManagementConfigFile;
    bool _emitColorTransforms;
};

} // namespace RenderUtil

#endif
