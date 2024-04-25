//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderOsl/OslRenderer.h>

#include <MaterialXGenOsl/OslShaderGenerator.h>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <fstream>

MATERIALX_NAMESPACE_BEGIN

string OslRenderer::OSL_CLOSURE_COLOR_STRING("closure color");

//
// OslRenderer methods
//

OslRendererPtr OslRenderer::create(unsigned int width, unsigned int height, Image::BaseType baseType)
{
    return std::shared_ptr<OslRenderer>(new OslRenderer(width, height, baseType));
}

OslRenderer::OslRenderer(unsigned int width, unsigned int height, Image::BaseType baseType) :
    ShaderRenderer(width, height, baseType),
    _useTestRender(true),
    _raysPerPixelLit(1),
    _raysPerPixelUnlit(1)
{
}

OslRenderer::~OslRenderer()
{
}

void OslRenderer::setSize(unsigned int width, unsigned int height)
{
    if (_width != width || _height != height)
    {
        _width = width;
        _height = height;
    }
}

void OslRenderer::initialize(RenderContextHandle)
{
    if (_oslIncludePath.isEmpty())
    {
        throw ExceptionRenderError("OSL validation include path is empty");
    }
    if (_oslTestShadeExecutable.isEmpty() && _oslCompilerExecutable.isEmpty())
    {
        throw ExceptionRenderError("OSL validation executables not set");
    }
}

void OslRenderer::renderOSL(const FilePath& dirPath, const string& shaderName, const string& outputName)
{
    // If command options missing, skip testing.
    if (_oslTestRenderExecutable.isEmpty() || _oslIncludePath.isEmpty() ||
        _oslTestRenderSceneTemplateFile.isEmpty() || _oslUtilityOSOPath.isEmpty())
    {
        throw ExceptionRenderError("Command input arguments are missing");
    }

    static const StringSet RENDERABLE_TYPES = { "float", "color", "vector", "closure color", "color4", "vector2", "vector4" };
    static const StringSet REMAPPABLE_TYPES = { "color4", "vector2", "vector4" };

    // If the output type is not which can be supported for rendering then skip testing.
    if (RENDERABLE_TYPES.count(_oslShaderOutputType) == 0)
    {
        throw ExceptionRenderError("Output type to render is not supported: " + _oslShaderOutputType);
    }

    const bool isColorClosure = _oslShaderOutputType == "closure color";
    const bool isRemappable = REMAPPABLE_TYPES.count(_oslShaderOutputType) != 0;

    // Determine the shader path from output path and shader name
    FilePath shaderFilePath(dirPath);
    shaderFilePath = shaderFilePath / shaderName;
    string shaderPath = shaderFilePath.asString();

    // Set output image name.
    string outputFileName = shaderPath + "_osl.png";
    _oslOutputFileName = outputFileName;

    // Use a known error file name to check
    string errorFile(shaderPath + "_render_errors.txt");
    const string redirectString(" 2>&1");

    // Read in scene template and replace the applicable tokens to have a valid ShaderGroup.
    // Write to local file to use as input for rendering.
    std::ifstream sceneTemplateStream(_oslTestRenderSceneTemplateFile);
    string sceneTemplateString;
    sceneTemplateString.assign(std::istreambuf_iterator<char>(sceneTemplateStream),
                               std::istreambuf_iterator<char>());

    // Get final output to use in the shader
    const string CLOSURE_PASSTHROUGH_SHADER_STRING("closure_passthrough");
    const string CONSTANT_COLOR_SHADER_STRING("constant_color");
    const string CONSTANT_COLOR_SHADER_PREFIX_STRING("constant_");
    string outputShader = isColorClosure ? CLOSURE_PASSTHROUGH_SHADER_STRING :
        (isRemappable ? CONSTANT_COLOR_SHADER_PREFIX_STRING + _oslShaderOutputType : CONSTANT_COLOR_SHADER_STRING);

    // Perform token replacement
    const string ENVIRONMENT_SHADER_PARAMETER_OVERRIDES("%environment_shader_parameter_overrides%");
    const string OUTPUT_SHADER_TYPE_STRING("%output_shader_type%");
    const string OUTPUT_SHADER_INPUT_STRING("%output_shader_input%");
    const string OUTPUT_SHADER_INPUT_VALUE_STRING("Cin");
    const string INPUT_SHADER_TYPE_STRING("%input_shader_type%");
    const string INPUT_SHADER_PARAMETER_OVERRIDES("%input_shader_parameter_overrides%");
    const string INPUT_SHADER_OUTPUT_STRING("%input_shader_output%");
    const string BACKGROUND_COLOR_STRING("%background_color%");

    StringMap replacementMap;
    replacementMap[OUTPUT_SHADER_TYPE_STRING] = outputShader;
    replacementMap[OUTPUT_SHADER_INPUT_STRING] = OUTPUT_SHADER_INPUT_VALUE_STRING;
    replacementMap[INPUT_SHADER_TYPE_STRING] = shaderName;
    string overrideString;
    for (const auto& param : _oslShaderParameterOverrides)
    {
        overrideString.append(param);
    }
    string envOverrideString;
    for (const auto& param : _envOslShaderParameterOverrides)
    {
        envOverrideString.append(param);
    }
    replacementMap[INPUT_SHADER_PARAMETER_OVERRIDES] = overrideString;
    replacementMap[ENVIRONMENT_SHADER_PARAMETER_OVERRIDES] = envOverrideString;
    replacementMap[INPUT_SHADER_OUTPUT_STRING] = outputName;
    replacementMap[BACKGROUND_COLOR_STRING] = std::to_string(DEFAULT_SCREEN_COLOR_LIN_REC709[0]) + " " +
                                              std::to_string(DEFAULT_SCREEN_COLOR_LIN_REC709[1]) + " " +
                                              std::to_string(DEFAULT_SCREEN_COLOR_LIN_REC709[2]);
    string sceneString = replaceSubstrings(sceneTemplateString, replacementMap);
    if ((sceneString == sceneTemplateString) || sceneTemplateString.empty())
    {
        throw ExceptionRenderError("Scene template file: " + _oslTestRenderSceneTemplateFile.asString() +
                                   " does not include proper tokens for rendering");
    }

    // Set the working directory for rendering.
    FileSearchPath searchPath = getDefaultDataSearchPath();
    FilePath rootPath = searchPath.isEmpty() ? FilePath() : searchPath[0];
    FilePath origWorkingPath = FilePath::getCurrentPath();
    rootPath.setCurrentPath();

    // Write scene file
    const string sceneFileName("scene_template.xml");
    std::ofstream shaderFileStream;
    shaderFileStream.open(sceneFileName);
    if (shaderFileStream.is_open())
    {
        shaderFileStream << sceneString;
        shaderFileStream.close();
    }

    // Set oso file paths
    string osoPaths(_oslUtilityOSOPath);
    osoPaths += PATH_LIST_SEPARATOR + dirPath.asString();
    osoPaths += PATH_LIST_SEPARATOR + dirPath.getParentPath().asString();

    // Build and run render command
    string command(_oslTestRenderExecutable);
    command += " " + sceneFileName;
    command += " " + outputFileName;
    command += " -r " + std::to_string(_width) + " " + std::to_string(_height);
    command += " --path " + osoPaths;
    command += " -aa " + std::to_string(isColorClosure ? _raysPerPixelLit : _raysPerPixelUnlit);
    command += " > " + errorFile + redirectString;

    // Repeat the render command to allow for sporadic errors.
    int returnValue = 0;
    for (int i = 0; i < 5; i++)
    {
        returnValue = std::system(command.c_str());
        if (!returnValue)
        {
            break;
        }
    }

    // Restore the working directory after rendering.
    origWorkingPath.setCurrentPath();

    // Report errors on a non-zero return value.
    if (returnValue)
    {
        std::ifstream errorStream(errorFile);
        StringVec result;
        string line;
        unsigned int errCount = 0;
        while (std::getline(errorStream, line))
        {
            if (errCount++ > 10)
            {
                break;
            }
            result.push_back(line);
        }

        StringVec errors;
        errors.push_back("Errors reported in renderOSL:");
        for (size_t i = 0; i < result.size(); i++)
        {
            errors.push_back(result[i]);
        }
        errors.push_back("Command string: " + command);
        errors.push_back("Command return code: " + std::to_string(returnValue));
        throw ExceptionRenderError("OSL rendering error", errors);
    }
}

void OslRenderer::shadeOSL(const FilePath& dirPath, const string& shaderName, const string& outputName)
{
    // If no command and include path specified then skip checking.
    if (_oslTestShadeExecutable.isEmpty() || _oslIncludePath.isEmpty())
    {
        return;
    }

    FilePath shaderFilePath(dirPath);
    shaderFilePath = shaderFilePath / shaderName;
    string shaderPath = shaderFilePath.asString();

    // Set output image name.
    string outputFileName = shaderPath + ".testshade.png";
    _oslOutputFileName = outputFileName;

    // Use a known error file name to check
    string errorFile(shaderPath + "_shade_errors.txt");
    const string redirectString(" 2>&1");

    string command(_oslTestShadeExecutable);
    command += " " + shaderPath;
    command += " -o " + outputName + " " + outputFileName;
    command += " -g 256 256";
    command += " > " + errorFile + redirectString;

    int returnValue = std::system(command.c_str());

    // There is no "silent" or "quiet" mode for testshade so we must parse the lines
    // to check if there were any error lines which are not the success line.
    // Note: This is currently hard-coded to a specific value. If testshade
    // modifies this then this hard-coded string must also be modified.
    // The formatted string is "Output <outputName> to <outputFileName>".
    std::ifstream errorStream(errorFile);
    StringVec results;
    string line;
    string successfulOutputSubString("Output " + outputName + " to " +
                                           outputFileName);
    while (std::getline(errorStream, line))
    {
        if (!line.empty() &&
            line.find(successfulOutputSubString) == string::npos)
        {
            results.push_back(line);
        }
    }

    if (!results.empty())
    {
        StringVec errors;
        errors.push_back("Errors reported in shadeOSL:");
        for (const auto& resultLine : results)
        {
            errors.push_back(resultLine);
        }
        errors.push_back("Command string: " + command);
        errors.push_back("Command return code: " + std::to_string(returnValue));
        throw ExceptionRenderError("OSL rendering error", errors);
    }
}

void OslRenderer::compileOSL(const FilePath& oslFilePath)
{
    // If no command and include path specified then skip checking.
    if (_oslCompilerExecutable.isEmpty() || _oslIncludePath.isEmpty())
    {
        return;
    }

    FilePath outputFileName = oslFilePath;
    outputFileName.removeExtension();
    outputFileName.addExtension("oso");

    // Use a known error file name to check
    string errorFile(oslFilePath.asString() + "_compile_errors.txt");
    const string redirectString(" 2>&1");

    // Run the command and get back the result. If non-empty string throw exception with error
    string command = _oslCompilerExecutable.asString() + " -q ";
    for (FilePath p : _oslIncludePath)
    { 
        command += " -I\"" + p.asString() + "\" ";
    }
    command += oslFilePath.asString() + " -o " + outputFileName.asString() + " > " + errorFile + redirectString;

    int returnValue = std::system(command.c_str());

    std::ifstream errorStream(errorFile);
    string result;
    result.assign(std::istreambuf_iterator<char>(errorStream),
                  std::istreambuf_iterator<char>());

    if (!result.empty())
    {
        StringVec errors;
        errors.push_back("Command string: " + command);
        errors.push_back("Command return code: " + std::to_string(returnValue));
        errors.push_back("Shader failed to compile:");
        errors.push_back(result);
        throw ExceptionRenderError("OSL compilation error", errors);
    }
}

void OslRenderer::createProgram(ShaderPtr shader)
{
    StageMap stages = { {Stage::PIXEL, shader->getStage(Stage::PIXEL).getSourceCode()} };
    createProgram(stages);
}

void OslRenderer::createProgram(const StageMap& stages)
{
    // There is only one stage in an OSL shader so only
    // the first stage is examined.
    if (stages.empty() || stages.begin()->second.empty())
    {
        throw ExceptionRenderError("No shader code to validate");
    }

    bool haveCompiler = !_oslCompilerExecutable.isEmpty() && !_oslIncludePath.isEmpty();
    if (!haveCompiler)
    {
        throw ExceptionRenderError("No OSL compiler specified for validation");
    }

    // Dump string to disk. For OSL assume shader is in stage 0 slot.
    FilePath filePath(_oslOutputFilePath);
    filePath = filePath  / _oslShaderName;
    string fileName = filePath.asString();
    if (fileName.empty())
    {
        fileName = "_osl_temp.osl";
    }
    else
    {
        fileName += ".osl";
    }

    // TODO: Seems testrender will crash currently when trying to convert to "object" space.
    // Thus we replace all instances of "object" with "world" to avoid issues.
    StringMap spaceMap;
    spaceMap["\"object\""] = "\"world\"";
    string oslCode = replaceSubstrings(stages.begin()->second, spaceMap);

    std::ofstream file;
    file.open(fileName);
    file << oslCode;
    file.close();

    // Try compiling the code
    compileOSL(fileName);
}

void OslRenderer::validateInputs()
{
    throw ExceptionRenderError("OSL input validation is not yet supported");
}

void OslRenderer::render()
{
    if (_oslOutputFilePath.isEmpty())
    {
        throw ExceptionRenderError("OSL output file path string has not been specified");
    }
    if (_oslShaderOutputName.empty())
    {
        throw ExceptionRenderError("OSL shader output name has not been specified");
    }

    _oslOutputFileName.assign(EMPTY_STRING);

    // Use testshade
    if (!_useTestRender)
    {
        shadeOSL(_oslOutputFilePath, _oslShaderName, _oslShaderOutputName);
    }

    // Use testrender
    else
    {
        if (_oslShaderName.empty())
        {
            throw ExceptionRenderError("OSL shader name has not been specified");
        }
        renderOSL(_oslOutputFilePath, _oslShaderName, _oslShaderOutputName);
    }
}

ImagePtr OslRenderer::captureImage(ImagePtr)
{
    // As rendering goes to disk need to read the image back from disk
    if (!_imageHandler || _oslOutputFileName.isEmpty())
    {
        throw ExceptionRenderError("Failed to read image: " + _oslOutputFileName.asString());
    }

    ImagePtr returnImage = _imageHandler->acquireImage(_oslOutputFileName);
    if (!returnImage)
    {
        throw ExceptionRenderError("Failed to save image to file: " + _oslOutputFileName.asString());
    }

    return returnImage;
}

MATERIALX_NAMESPACE_END
