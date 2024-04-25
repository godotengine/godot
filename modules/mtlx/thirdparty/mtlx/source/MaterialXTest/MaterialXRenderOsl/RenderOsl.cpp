//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>
#include <MaterialXTest/MaterialXRender/RenderUtil.h>

#include <MaterialXRenderOsl/OslRenderer.h>

#include <MaterialXRender/StbImageLoader.h>
#if defined(MATERIALX_BUILD_OIIO)
#include <MaterialXRender/OiioImageLoader.h>
#endif

#include <MaterialXGenOsl/OslShaderGenerator.h>

#include <MaterialXFormat/Util.h>

namespace mx = MaterialX;

namespace
{

//
// Define local overrides for the tangent frame in shader generation, aligning conventions
// between MaterialXRender and testrender.
//

class TangentOsl : public mx::ShaderNodeImpl
{
  public:
    static mx::ShaderNodeImplPtr create()
    {
        return std::make_shared<TangentOsl>();
    }

    void emitFunctionCall(const  mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();

        DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL)
        {
            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(node.getOutput(), true, false, context, stage);
            shadergen.emitString(" = normalize(vector(N[2], 0, -N[0]))", stage);
            shadergen.emitLineEnd(stage);
        }
    }
};

class BitangentOsl : public mx::ShaderNodeImpl
{
  public:
    static mx::ShaderNodeImplPtr create()
    {
        return std::make_shared<BitangentOsl>();
    }

    void emitFunctionCall(const  mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();

        DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL)
        {
            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(node.getOutput(), true, false, context, stage);
            shadergen.emitString(" = normalize(cross(N, vector(N[2], 0, -N[0])))", stage);
            shadergen.emitLineEnd(stage);
        }
    }
};

} // anonymous namespace

class OslShaderRenderTester : public RenderUtil::ShaderRenderTester
{
  public:
    explicit OslShaderRenderTester(mx::ShaderGeneratorPtr shaderGenerator) :
        RenderUtil::ShaderRenderTester(shaderGenerator)
    {
        // Preprocess to resolve to absolute image file names 
        // and all non-POSIX separators must be converted to POSIX ones (this only affects running on Windows)
        _resolveImageFilenames = true;
        _customFilenameResolver = mx::StringResolver::create();
        _customFilenameResolver->setFilenameSubstitution("\\\\", "/");
        _customFilenameResolver->setFilenameSubstitution("\\", "/");

    }

  protected:
    void createRenderer(std::ostream& log) override;

    bool runRenderer(const std::string& shaderName,
                     mx::TypedElementPtr element,
                     mx::GenContext& context,
                     mx::DocumentPtr doc,
                     std::ostream& log,
                     const GenShaderUtil::TestSuiteOptions& testOptions,
                     RenderUtil::RenderProfileTimes& profileTimes,
                     const mx::FileSearchPath& imageSearchPath,
                     const std::string& outputPath = ".",
                     mx::ImageVec* imageVec = nullptr) override;

    bool saveImage(const mx::FilePath& filePath, mx::ConstImagePtr image, bool /*verticalFlip*/) const override
    {
        return _renderer->getImageHandler()->saveImage(filePath, image, false);
    }

    mx::ImageLoaderPtr _imageLoader;
    mx::OslRendererPtr _renderer;
};

// Renderer setup
void OslShaderRenderTester::createRenderer(std::ostream& log)
{
    bool initialized = false;

    _renderer = mx::OslRenderer::create();
    _imageLoader = mx::StbImageLoader::create();

    // Set up additional utilities required to run OSL testing including
    // oslc and testrender paths and OSL include path
    //
    const std::string oslcExecutable(MATERIALX_OSL_BINARY_OSLC);
    _renderer->setOslCompilerExecutable(oslcExecutable);
    const std::string testRenderExecutable(MATERIALX_OSL_BINARY_TESTRENDER);
    _renderer->setOslTestRenderExecutable(testRenderExecutable);
    _renderer->setOslIncludePath(mx::FileSearchPath(MATERIALX_OSL_INCLUDE_PATH));

    try
    {
        _renderer->initialize();

        mx::StbImageLoaderPtr stbLoader = mx::StbImageLoader::create();
        mx::ImageHandlerPtr imageHandler = mx::ImageHandler::create(stbLoader);
        imageHandler->setSearchPath(mx::getDefaultDataSearchPath());
#if defined(MATERIALX_BUILD_OIIO)
        mx::OiioImageLoaderPtr oiioLoader = mx::OiioImageLoader::create();
        imageHandler->addLoader(oiioLoader);
#endif
        _renderer->setImageHandler(imageHandler);
        _renderer->setLightHandler(nullptr);
        initialized = true;

        // Pre-compile some required shaders for testrender
        if (!oslcExecutable.empty() && !testRenderExecutable.empty())
        {
            mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
            mx::FilePath shaderPath = searchPath.find("resources/Utilities/");
            _renderer->setOslOutputFilePath(shaderPath);

            const std::string OSL_EXTENSION("osl");
            for (const mx::FilePath& filename : shaderPath.getFilesInDirectory(OSL_EXTENSION))
            {
                _renderer->compileOSL(shaderPath / filename);
            }

            // Set the search path for these compiled shaders.
            _renderer->setOslUtilityOSOPath(shaderPath);
        }
    }
    catch (mx::ExceptionRenderError& e)
    {
        for (const auto& error : e.errorLog())
        {
            log << e.what() << " " << error << std::endl;
        }
    }
    REQUIRE(initialized);
}

// Renderer execution
bool OslShaderRenderTester::runRenderer(const std::string& shaderName,
                                         mx::TypedElementPtr element,
                                         mx::GenContext& context,
                                         mx::DocumentPtr doc,
                                         std::ostream& log,
                                         const GenShaderUtil::TestSuiteOptions& testOptions,
                                         RenderUtil::RenderProfileTimes& profileTimes,
                                         const mx::FileSearchPath&,
                                         const std::string& outputPath,
                                         mx::ImageVec* imageVec)
{
    std::cout << "Validating OSL rendering for: " << doc->getSourceUri() << std::endl;

    mx::ScopedTimer totalOSLTime(&profileTimes.languageTimes.totalTime);

    mx::ShaderGenerator& shadergen = context.getShaderGenerator();

    // Perform validation if requested
    if (testOptions.validateElementToRender)
    {
        std::string message;
        if (!element->validate(&message))
        {
            log << "Element is invalid: " << message << std::endl;
            return false;
        }
    }

    std::vector<mx::GenOptions> optionsList;
    getGenerationOptions(testOptions, context.getOptions(), optionsList);

    if (element && doc)
    {
        log << "------------ Run OSL validation with element: " << element->getNamePath() << "-------------------" << std::endl;

        for (const auto& options : optionsList)
        {
            profileTimes.elementsTested++;

            mx::ShaderPtr shader;
            try
            {
                mx::ScopedTimer genTimer(&profileTimes.languageTimes.generationTime);
                mx::GenOptions& contextOptions = context.getOptions();
                contextOptions = options;
                contextOptions.targetColorSpaceOverride = "lin_rec709";

                // Apply local overrides for shader generation.
                shadergen.registerImplementation("IM_tangent_vector3_" + mx::OslShaderGenerator::TARGET, TangentOsl::create);
                shadergen.registerImplementation("IM_bitangent_vector3_" + mx::OslShaderGenerator::TARGET, BitangentOsl::create);

                shader = shadergen.generate(shaderName, element, context);
            }
            catch (mx::Exception& e)
            {
                log << ">> " << e.what() << "\n";
                shader = nullptr;
            }
            CHECK(shader != nullptr);
            if (shader == nullptr)
            {
                log << ">> Failed to generate shader\n";
                return false;
            }
            CHECK(shader->getSourceCode().length() > 0);

            std::string shaderPath;
            mx::FilePath outputFilePath = outputPath;
            // Use separate directory for reduced output
            if (options.shaderInterfaceType == mx::SHADER_INTERFACE_REDUCED)
            {
                outputFilePath = outputFilePath / mx::FilePath("reduced");
            }

            // Note: mkdir will fail if the directory already exists which is ok.
            {
                mx::ScopedTimer ioDir(&profileTimes.languageTimes.ioTime);
                outputFilePath.createDirectory();
            }

            shaderPath = mx::FilePath(outputFilePath) / mx::FilePath(shaderName);

            // Write out osl file
            if (testOptions.dumpGeneratedCode)
            {
                mx::ScopedTimer ioTimer(&profileTimes.languageTimes.ioTime);
                std::ofstream file;
                file.open(shaderPath + ".osl");
                file << shader->getSourceCode();
                file.close();
            }

            if (!testOptions.compileCode)
            {
                return false;
            }

            // Validate
            bool validated = false;
            try
            {
                // Set renderer properties.
                _renderer->setOslOutputFilePath(outputFilePath);
                _renderer->setOslShaderName(shaderName);
                _renderer->setRaysPerPixelLit(testOptions.enableReferenceQuality ? 8 : 4);
                _renderer->setRaysPerPixelUnlit(testOptions.enableReferenceQuality ? 2 : 1);

                // Validate compilation
                {
                    mx::ScopedTimer compileTimer(&profileTimes.languageTimes.compileTime);
                    _renderer->createProgram(shader);
                }

                if (testOptions.renderImages)
                {
                    _renderer->setSize(static_cast<unsigned int>(testOptions.renderSize[0]), static_cast<unsigned int>(testOptions.renderSize[1]));

                    const mx::ShaderStage& stage = shader->getStage(mx::Stage::PIXEL);

                    // Bind IBL image name overrides.
                    mx::StringVec envOverrides;
                    std::string envmap_filename("string envmap_filename \"");
                    envmap_filename += testOptions.radianceIBLPath;
                    envmap_filename += "\";\n";                    
                    envOverrides.push_back(envmap_filename);

                    _renderer->setEnvShaderParameterOverrides(envOverrides);

                    const mx::VariableBlock& outputs = stage.getOutputBlock(mx::OSL::OUTPUTS);
                    if (outputs.size() > 0)
                    {
                        const mx::ShaderPort* output = outputs[0];
                        const mx::TypeSyntax& typeSyntax = shadergen.getSyntax().getTypeSyntax(output->getType());

                        const std::string& outputName = output->getVariable();
                        const std::string& outputType = typeSyntax.getTypeAlias().empty() ? typeSyntax.getName() : typeSyntax.getTypeAlias();
                        const std::string& sceneTemplateFile = "scene_template.xml";

                        // Set shader output name and type to use
                        _renderer->setOslShaderOutput(outputName, outputType);

                        // Set scene template file. For now we only have the constant color scene file
                        mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
                        mx::FilePath sceneTemplatePath = searchPath.find("resources/Utilities/");
                        sceneTemplatePath = sceneTemplatePath / sceneTemplateFile;
                        _renderer->setOslTestRenderSceneTemplateFile(sceneTemplatePath.asString());

                        // Validate rendering
                        {
                            mx::ScopedTimer renderTimer(&profileTimes.languageTimes.renderTime);
                            _renderer->render();
                            if (imageVec)
                            {
                                imageVec->push_back(_renderer->captureImage());
                            }
                        }
                    }
                    else
                    {
                        CHECK(false);
                        log << ">> Shader has no output to render from\n";
                    }
                }

                validated = true;
            }
            catch (mx::ExceptionRenderError& e)
            {
                // Always dump shader on error
                std::ofstream file;
                file.open(shaderPath + ".osl");
                file << shader->getSourceCode();
                file.close();

                for (const auto& error : e.errorLog())
                {
                    log << e.what() << " " << error << std::endl;
                }
                log << ">> Refer to shader code in dump file: " << shaderPath << ".osl file" << std::endl;
            }
            catch (mx::Exception& e)
            {
                log << e.what();
            }
            CHECK(validated);
        }
    }

    return true;
}

TEST_CASE("Render: OSL TestSuite", "[renderosl]")
{
    if (std::string(MATERIALX_OSL_BINARY_OSLC).empty() &&
        std::string(MATERIALX_OSL_BINARY_TESTRENDER).empty())
    {
        INFO("Skipping the OSL test suite as its executable locations haven't been set.");
        return;
    }

    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePath optionsFilePath = searchPath.find("resources/Materials/TestSuite/_options.mtlx");

    OslShaderRenderTester renderTester(mx::OslShaderGenerator::create());
    renderTester.validate(optionsFilePath);
}
