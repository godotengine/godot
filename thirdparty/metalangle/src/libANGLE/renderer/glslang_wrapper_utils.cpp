//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Wrapper for Khronos glslang compiler.
//

#include "libANGLE/renderer/glslang_wrapper_utils.h"

// glslang has issues with some specific warnings.
ANGLE_DISABLE_EXTRA_SEMI_WARNING

// glslang's version of ShaderLang.h, not to be confused with ANGLE's.
#include <glslang/Public/ShaderLang.h>

// Other glslang includes.
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>

ANGLE_REENABLE_EXTRA_SEMI_WARNING

#include <array>
#include <numeric>

#include "common/FixedVector.h"
#include "common/string_utils.h"
#include "common/utilities.h"
#include "libANGLE/Caps.h"
#include "libANGLE/ProgramLinkedResources.h"

#define ANGLE_GLSLANG_CHECK(CALLBACK, TEST, ERR) \
    do                                           \
    {                                            \
        if (ANGLE_UNLIKELY(!(TEST)))             \
        {                                        \
            return CALLBACK(ERR);                \
        }                                        \
                                                 \
    } while (0)

namespace rx
{
namespace
{
constexpr char kMarkerStart[]               = "@@ ";
constexpr char kQualifierMarkerBegin[]      = "@@ QUALIFIER-";
constexpr char kLayoutMarkerBegin[]         = "@@ LAYOUT-";
constexpr char kXfbDeclMarkerBegin[]        = "@@ XFB-DECL";
constexpr char kXfbOutMarkerBegin[]         = "@@ XFB-OUT";
constexpr char kMarkerEnd[]                 = " @@";
constexpr char kParamsBegin                 = '(';
constexpr char kParamsEnd                   = ')';
constexpr char kUniformQualifier[]          = "uniform";
constexpr char kSSBOQualifier[]             = "buffer";
constexpr char kUnusedBlockSubstitution[]   = "struct";
constexpr char kUnusedUniformSubstitution[] = "// ";
constexpr char kVersionDefine[]             = "#version 450 core\n";
constexpr char kLineRasterDefine[]          = "#define ANGLE_ENABLE_LINE_SEGMENT_RASTERIZATION\n";
constexpr char kXfbEmuMacro[]               = "ANGLE_ENABLE_XFB_EMULATION";
constexpr char kXfbEmuDefine[]              = "#define ANGLE_ENABLE_XFB_EMULATION\n";

template <size_t N>
constexpr size_t ConstStrLen(const char (&)[N])
{
    static_assert(N > 0, "C++ shouldn't allow N to be zero");

    // The length of a string defined as a char array is the size of the array minus 1 (the
    // terminating '\0').
    return N - 1;
}

void GetBuiltInResourcesFromCaps(const gl::Caps &caps, TBuiltInResource *outBuiltInResources)
{
    outBuiltInResources->maxDrawBuffers                   = caps.maxDrawBuffers;
    outBuiltInResources->maxAtomicCounterBindings         = caps.maxAtomicCounterBufferBindings;
    outBuiltInResources->maxAtomicCounterBufferSize       = caps.maxAtomicCounterBufferSize;
    outBuiltInResources->maxClipPlanes                    = caps.maxClipPlanes;
    outBuiltInResources->maxCombinedAtomicCounterBuffers  = caps.maxCombinedAtomicCounterBuffers;
    outBuiltInResources->maxCombinedAtomicCounters        = caps.maxCombinedAtomicCounters;
    outBuiltInResources->maxCombinedImageUniforms         = caps.maxCombinedImageUniforms;
    outBuiltInResources->maxCombinedTextureImageUnits     = caps.maxCombinedTextureImageUnits;
    outBuiltInResources->maxCombinedShaderOutputResources = caps.maxCombinedShaderOutputResources;
    outBuiltInResources->maxComputeWorkGroupCountX        = caps.maxComputeWorkGroupCount[0];
    outBuiltInResources->maxComputeWorkGroupCountY        = caps.maxComputeWorkGroupCount[1];
    outBuiltInResources->maxComputeWorkGroupCountZ        = caps.maxComputeWorkGroupCount[2];
    outBuiltInResources->maxComputeWorkGroupSizeX         = caps.maxComputeWorkGroupSize[0];
    outBuiltInResources->maxComputeWorkGroupSizeY         = caps.maxComputeWorkGroupSize[1];
    outBuiltInResources->maxComputeWorkGroupSizeZ         = caps.maxComputeWorkGroupSize[2];
    outBuiltInResources->minProgramTexelOffset            = caps.minProgramTexelOffset;
    outBuiltInResources->maxFragmentUniformVectors        = caps.maxFragmentUniformVectors;
    outBuiltInResources->maxFragmentInputComponents       = caps.maxFragmentInputComponents;
    outBuiltInResources->maxGeometryInputComponents       = caps.maxGeometryInputComponents;
    outBuiltInResources->maxGeometryOutputComponents      = caps.maxGeometryOutputComponents;
    outBuiltInResources->maxGeometryOutputVertices        = caps.maxGeometryOutputVertices;
    outBuiltInResources->maxGeometryTotalOutputComponents = caps.maxGeometryTotalOutputComponents;
    outBuiltInResources->maxLights                        = caps.maxLights;
    outBuiltInResources->maxProgramTexelOffset            = caps.maxProgramTexelOffset;
    outBuiltInResources->maxVaryingComponents             = caps.maxVaryingComponents;
    outBuiltInResources->maxVaryingVectors                = caps.maxVaryingVectors;
    outBuiltInResources->maxVertexAttribs                 = caps.maxVertexAttributes;
    outBuiltInResources->maxVertexOutputComponents        = caps.maxVertexOutputComponents;
    outBuiltInResources->maxVertexUniformVectors          = caps.maxVertexUniformVectors;
    outBuiltInResources->maxClipDistances                 = caps.maxClipDistances;
}

class IntermediateShaderSource final : angle::NonCopyable
{
  public:
    void init(const std::string &source);
    bool empty() const { return mTokens.empty(); }

    bool findTokenName(const std::string &name);
    // Find @@ LAYOUT-name(extra, args) @@ and replace it with:
    //
    //     layout(specifier, extra, args)
    //
    // or if |specifier| is empty:
    //
    //     layout(extra, args)
    //
    void insertLayoutSpecifier(const std::string &name, const std::string &specifier);

    // Find @@ QUALIFIER-name(other qualifiers) @@ and replace it with:
    //
    //      specifier other qualifiers
    //
    // or if |specifier| is empty, with nothing.
    //
    void insertQualifierSpecifier(const std::string &name, const std::string &specifier);

    // Replace @@ XFB-DECL @@ with |decl|.
    void insertTransformFeedbackDeclaration(const std::string &&decl);

    // Replace @@ XFB-OUT @@ with |output| code block.
    void insertTransformFeedbackOutput(const std::string &&output);

    // Remove @@ LAYOUT-name(*) @@ and @@ QUALIFIER-name(*) @@ altogether, optionally replacing them
    // with something to make sure the shader still compiles.
    void eraseLayoutAndQualifierSpecifiers(const std::string &name, const std::string &replacement);

    // Get the transformed shader source as one string.
    std::string getShaderSource();

  private:
    enum class TokenType
    {
        // A piece of shader source code.
        Text,
        // Block corresponding to @@ QUALIFIER-abc(other qualifiers) @@
        Qualifier,
        // Block corresponding to @@ LAYOUT-abc(extra, args) @@
        Layout,
        // Block corresponding to @@ XFB-DECL @@
        TransformFeedbackDeclaration,
        // Block corresponding to @@ XFB-OUT @@
        TransformFeedbackOutput,
    };

    struct Token
    {
        TokenType type;
        // |text| contains some shader code if Text, or the id of macro ("abc" in examples above)
        // being replaced if Qualifier or Layout.
        std::string text;
        // If Qualifier or Layout, this contains extra parameters passed in parentheses, if any.
        std::string args;
    };

    void addTextBlock(std::string &&text);
    void addLayoutBlock(std::string &&name, std::string &&args);
    void addQualifierBlock(std::string &&name, std::string &&args);
    void addTransformFeedbackDeclarationBlock();
    void addTransformFeedbackOutputBlock();

    void replaceSingleMacro(TokenType type, const std::string &&text);

    std::vector<Token> mTokens;
};

void IntermediateShaderSource::addTextBlock(std::string &&text)
{
    if (!text.empty())
    {
        Token token = {TokenType::Text, std::move(text), ""};
        mTokens.emplace_back(std::move(token));
    }
}

void IntermediateShaderSource::addLayoutBlock(std::string &&name, std::string &&args)
{
    ASSERT(!name.empty());
    Token token = {TokenType::Layout, std::move(name), std::move(args)};
    mTokens.emplace_back(std::move(token));
}

void IntermediateShaderSource::addQualifierBlock(std::string &&name, std::string &&args)
{
    ASSERT(!name.empty());
    Token token = {TokenType::Qualifier, std::move(name), std::move(args)};
    mTokens.emplace_back(std::move(token));
}

void IntermediateShaderSource::addTransformFeedbackDeclarationBlock()
{
    Token token = {TokenType::TransformFeedbackDeclaration, "", ""};
    mTokens.emplace_back(std::move(token));
}

void IntermediateShaderSource::addTransformFeedbackOutputBlock()
{
    Token token = {TokenType::TransformFeedbackOutput, "", ""};
    mTokens.emplace_back(std::move(token));
}

size_t ExtractNameAndArgs(const std::string &source,
                          size_t cur,
                          std::string *nameOut,
                          std::string *argsOut)
{
    *nameOut = angle::GetPrefix(source, cur, kParamsBegin);

    // There should always be an extra args list (even if empty, for simplicity).
    size_t readCount = nameOut->length() + 1;
    *argsOut         = angle::GetPrefix(source, cur + readCount, kParamsEnd);
    readCount += argsOut->length() + 1;

    return readCount;
}

void IntermediateShaderSource::init(const std::string &source)
{
    size_t cur = 0;

    // Split the source into Text, Layout and Qualifier blocks for efficient macro expansion.
    while (cur < source.length())
    {
        // Create a Text block for the code up to the first marker.
        std::string text = angle::GetPrefix(source, cur, kMarkerStart);
        cur += text.length();

        addTextBlock(std::move(text));

        if (cur >= source.length())
        {
            break;
        }

        if (source.compare(cur, ConstStrLen(kQualifierMarkerBegin), kQualifierMarkerBegin) == 0)
        {
            cur += ConstStrLen(kQualifierMarkerBegin);

            // Get the id and arguments of the macro and add a qualifier block.
            std::string name, args;
            cur += ExtractNameAndArgs(source, cur, &name, &args);
            addQualifierBlock(std::move(name), std::move(args));
        }
        else if (source.compare(cur, ConstStrLen(kLayoutMarkerBegin), kLayoutMarkerBegin) == 0)
        {
            cur += ConstStrLen(kLayoutMarkerBegin);

            // Get the id and arguments of the macro and add a layout block.
            std::string name, args;
            cur += ExtractNameAndArgs(source, cur, &name, &args);
            addLayoutBlock(std::move(name), std::move(args));
        }
        else if (source.compare(cur, ConstStrLen(kXfbDeclMarkerBegin), kXfbDeclMarkerBegin) == 0)
        {
            cur += ConstStrLen(kXfbDeclMarkerBegin);
            addTransformFeedbackDeclarationBlock();
        }
        else if (source.compare(cur, ConstStrLen(kXfbOutMarkerBegin), kXfbOutMarkerBegin) == 0)
        {
            cur += ConstStrLen(kXfbOutMarkerBegin);
            addTransformFeedbackOutputBlock();
        }
        else
        {
            // If reached here, @@ was met in the shader source itself which would have been a
            // compile error.
            UNREACHABLE();
        }

        // There should always be a closing marker at this point.
        ASSERT(source.compare(cur, ConstStrLen(kMarkerEnd), kMarkerEnd) == 0);

        // Continue from after the closing of this macro.
        cur += ConstStrLen(kMarkerEnd);
    }
}

bool IntermediateShaderSource::findTokenName(const std::string &name)
{
    for (Token &block : mTokens)
    {
        if (block.text == name)
        {
            return true;
        }
    }
    return false;
}

void IntermediateShaderSource::insertLayoutSpecifier(const std::string &name,
                                                     const std::string &specifier)
{
    for (Token &block : mTokens)
    {
        if (block.type == TokenType::Layout && block.text == name)
        {
            const char *separator = specifier.empty() || block.args.empty() ? "" : ", ";

            block.type = TokenType::Text;
            block.text = "layout(" + block.args + separator + specifier + ")";
            break;
        }
    }
}

void IntermediateShaderSource::insertQualifierSpecifier(const std::string &name,
                                                        const std::string &specifier)
{
    for (Token &block : mTokens)
    {
        if (block.type == TokenType::Qualifier && block.text == name)
        {
            block.type = TokenType::Text;
            block.text = specifier;
            if (!specifier.empty() && !block.args.empty())
            {
                block.text += " " + block.args;
            }
            break;
        }
    }
}

void IntermediateShaderSource::replaceSingleMacro(TokenType type, const std::string &&text)
{
    for (Token &block : mTokens)
    {
        if (block.type == type)
        {
            block.type = TokenType::Text;
            block.text = std::move(text);
            break;
        }
    }
}

void IntermediateShaderSource::insertTransformFeedbackDeclaration(const std::string &&decl)
{
    replaceSingleMacro(TokenType::TransformFeedbackDeclaration, std::move(decl));
}

void IntermediateShaderSource::insertTransformFeedbackOutput(const std::string &&output)
{
    replaceSingleMacro(TokenType::TransformFeedbackOutput, std::move(output));
}

void IntermediateShaderSource::eraseLayoutAndQualifierSpecifiers(const std::string &name,
                                                                 const std::string &replacement)
{
    for (Token &block : mTokens)
    {
        if (block.type == TokenType::Text || block.text != name)
        {
            continue;
        }

        block.text = block.type == TokenType::Layout ? "" : replacement;
        block.type = TokenType::Text;
    }
}

std::string IntermediateShaderSource::getShaderSource()
{
    std::string shaderSource;

    for (Token &block : mTokens)
    {
        // All blocks should have been replaced.
        ASSERT(block.type == TokenType::Text);
        shaderSource += block.text;
    }

    return shaderSource;
}

std::string GetMappedSamplerNameOld(const std::string &originalName)
{
    std::string samplerName = gl::ParseResourceName(originalName, nullptr);

    // Samplers in structs are extracted.
    std::replace(samplerName.begin(), samplerName.end(), '.', '_');

    // Samplers in arrays of structs are also extracted.
    std::replace(samplerName.begin(), samplerName.end(), '[', '_');
    samplerName.erase(std::remove(samplerName.begin(), samplerName.end(), ']'), samplerName.end());
    return samplerName;
}

template <typename OutputIter, typename ImplicitIter>
uint32_t CountExplicitOutputs(OutputIter outputsBegin,
                              OutputIter outputsEnd,
                              ImplicitIter implicitsBegin,
                              ImplicitIter implicitsEnd)
{
    auto reduce = [implicitsBegin, implicitsEnd](uint32_t count, const sh::ShaderVariable &var) {
        bool isExplicit = std::find(implicitsBegin, implicitsEnd, var.name) == implicitsEnd;
        return count + isExplicit;
    };

    return std::accumulate(outputsBegin, outputsEnd, 0, reduce);
}

std::string GenerateTransformFeedbackVaryingOutput(const gl::TransformFeedbackVarying &varying,
                                                   const gl::UniformTypeInfo &info,
                                                   size_t strideBytes,
                                                   size_t offset,
                                                   const std::string &bufferIndex)
{
    std::ostringstream result;

    ASSERT(strideBytes % 4 == 0);
    size_t stride = strideBytes / 4;

    const size_t arrayIndexStart = varying.arrayIndex == GL_INVALID_INDEX ? 0 : varying.arrayIndex;
    const size_t arrayIndexEnd   = arrayIndexStart + varying.size();

    for (size_t arrayIndex = arrayIndexStart; arrayIndex < arrayIndexEnd; ++arrayIndex)
    {
        for (int col = 0; col < info.columnCount; ++col)
        {
            for (int row = 0; row < info.rowCount; ++row)
            {
                result << "xfbOut" << bufferIndex << "[ANGLEUniforms.xfbBufferOffsets["
                       << bufferIndex
                       << "] + (gl_VertexIndex + gl_InstanceIndex * "
                          "ANGLEUniforms.xfbVerticesPerDraw) * "
                       << stride << " + " << offset << "] = " << info.glslAsFloat << "("
                       << varying.mappedName;

                if (varying.isArray())
                {
                    result << "[" << arrayIndex << "]";
                }

                if (info.columnCount > 1)
                {
                    result << "[" << col << "]";
                }

                if (info.rowCount > 1)
                {
                    result << "[" << row << "]";
                }

                result << ");\n";
                ++offset;
            }
        }
    }

    return result.str();
}

void GenerateTransformFeedbackOutputs(const GlslangSourceOptions &options,
                                      const gl::ProgramState &programState,
                                      IntermediateShaderSource *vertexShader)
{
    const std::vector<gl::TransformFeedbackVarying> &varyings =
        programState.getLinkedTransformFeedbackVaryings();
    const std::vector<GLsizei> &bufferStrides = programState.getTransformFeedbackStrides();
    const bool isInterleaved =
        programState.getTransformFeedbackBufferMode() == GL_INTERLEAVED_ATTRIBS;
    const size_t bufferCount = isInterleaved ? 1 : varyings.size();

    const std::string xfbSet = Str(options.uniformsAndXfbDescriptorSetIndex);
    std::vector<std::string> xfbIndices(bufferCount);

    const std::string ifdef = "#ifdef " + std::string(kXfbEmuMacro) + "\n";

    // Wrap XFB emulation in #ifdef guard.
    std::string xfbDecl = ifdef;

    for (uint32_t bufferIndex = 0; bufferIndex < bufferCount; ++bufferIndex)
    {
        const std::string xfbBinding = Str(options.xfbBindingIndexStart + bufferIndex);
        xfbIndices[bufferIndex]      = Str(bufferIndex);

        xfbDecl += "layout(set = " + xfbSet + ", binding = " + xfbBinding + ") buffer xfbBuffer" +
                   xfbIndices[bufferIndex] + " { float xfbOut" + xfbIndices[bufferIndex] +
                   "[]; };\n";
    }
    xfbDecl += "#endif\n";

    std::string xfbOut = ifdef;
    xfbOut += "if (ANGLEUniforms.xfbActiveUnpaused != 0)\n{\n";
    size_t outputOffset = 0;
    for (size_t varyingIndex = 0; varyingIndex < varyings.size(); ++varyingIndex)
    {
        const size_t bufferIndex                    = isInterleaved ? 0 : varyingIndex;
        const gl::TransformFeedbackVarying &varying = varyings[varyingIndex];

        // For every varying, output to the respective buffer packed.  If interleaved, the output is
        // always to the same buffer, but at different offsets.
        const gl::UniformTypeInfo &info = gl::GetUniformTypeInfo(varying.type);
        xfbOut += GenerateTransformFeedbackVaryingOutput(varying, info, bufferStrides[bufferIndex],
                                                         outputOffset, xfbIndices[bufferIndex]);

        if (isInterleaved)
        {
            outputOffset += info.columnCount * info.rowCount * varying.size();
        }
    }
    xfbOut += "}\n";
    xfbOut += "#endif\n";

    vertexShader->insertTransformFeedbackDeclaration(std::move(xfbDecl));
    vertexShader->insertTransformFeedbackOutput(std::move(xfbOut));
}

void AssignAttributeLocations(const gl::ProgramState &programState,
                              IntermediateShaderSource *shaderSource)
{
    ASSERT(!shaderSource->empty());

    // Parse attribute locations and replace them in the vertex shader.
    // See corresponding code in OutputVulkanGLSL.cpp.
    for (const sh::ShaderVariable &attribute : programState.getProgramInputs())
    {
        // Warning: If we end up supporting ES 3.0 shaders and up, Program::linkAttributes is
        // going to bring us all attributes in this list instead of only the active ones.
        ASSERT(attribute.active);

        std::string locationString = "location = " + Str(attribute.location);
        shaderSource->insertLayoutSpecifier(attribute.name, locationString);
        shaderSource->insertQualifierSpecifier(attribute.name, "in");
    }
}

void AssignOutputLocations(const gl::ProgramState &programState,
                           IntermediateShaderSource *fragmentSource)
{
    // Parse output locations and replace them in the fragment shader.
    // See corresponding code in OutputVulkanGLSL.cpp.
    // TODO(syoussefi): Add support for EXT_blend_func_extended.  http://anglebug.com/3385
    const auto &outputLocations                      = programState.getOutputLocations();
    const auto &outputVariables                      = programState.getOutputVariables();
    const std::array<std::string, 3> implicitOutputs = {"gl_FragDepth", "gl_SampleMask",
                                                        "gl_FragStencilRefARB"};
    for (const gl::VariableLocation &outputLocation : outputLocations)
    {
        if (outputLocation.arrayIndex == 0 && outputLocation.used() && !outputLocation.ignored)
        {
            const sh::ShaderVariable &outputVar = outputVariables[outputLocation.index];

            std::string name = outputVar.name;
            std::string locationString;
            if (outputVar.location != -1)
            {
                locationString = "location = " + Str(outputVar.location);
            }
            else if (std::find(implicitOutputs.begin(), implicitOutputs.end(), name) ==
                     implicitOutputs.end())
            {
                // If there is only one output, it is allowed not to have a location qualifier, in
                // which case it defaults to 0.  GLSL ES 3.00 spec, section 4.3.8.2.
                ASSERT(CountExplicitOutputs(outputVariables.begin(), outputVariables.end(),
                                            implicitOutputs.begin(), implicitOutputs.end()) == 1);
                locationString = "location = 0";
            }

            fragmentSource->insertLayoutSpecifier(name, locationString);
        }
    }
}

void AssignVaryingLocations(const gl::ProgramState &programState,
                            const gl::ProgramLinkedResources &resources,
                            IntermediateShaderSource *outStageSource,
                            IntermediateShaderSource *inStageSource)
{
    // Assign varying locations.
    for (const gl::PackedVaryingRegister &varyingReg : resources.varyingPacking.getRegisterList())
    {
        const auto &varying = *varyingReg.packedVarying;

        // In Vulkan GLSL, struct fields are not allowed to have location assignments.  The varying
        // of a struct type is thus given a location equal to the one assigned to its first field.
        if (varying.isStructField() && varying.fieldIndex > 0)
        {
            continue;
        }

        // Similarly, assign array varying locations to the assigned location of the first element.
        if (varying.isArrayElement() && varying.arrayIndex != 0)
        {
            continue;
        }

        std::string locationString = "location = " + Str(varyingReg.registerRow);
        if (varyingReg.registerColumn > 0)
        {
            ASSERT(!varying.varying->isStruct());
            ASSERT(!gl::IsMatrixType(varying.varying->type));
            locationString += ", component = " + Str(varyingReg.registerColumn);
        }

        // In the following:
        //
        //     struct S { vec4 field; };
        //     out S varStruct;
        //
        // "varStruct" is found through |parentStructName|, with |varying->name| being "field".  In
        // such a case, use |parentStructName|.
        const std::string &name =
            varying.isStructField() ? varying.parentStructName : varying.varying->name;

        // Varings are from 3 stage of shader sources
        // To match pair of (out - in) qualifier, varying should be in the pair of shader source
        if (!outStageSource->findTokenName(name) || !inStageSource->findTokenName(name))
        {
            // Pair can be unmatching at transform feedback case,
            // But it requires qualifier.
            if (!varying.vertexOnly)
                continue;
        }

        outStageSource->insertLayoutSpecifier(name, locationString);
        inStageSource->insertLayoutSpecifier(name, locationString);

        const char *outQualifier = "out";
        const char *inQualifier  = "in";
        switch (varying.interpolation)
        {
            case sh::INTERPOLATION_SMOOTH:
                break;
            case sh::INTERPOLATION_CENTROID:
                outQualifier = "centroid out";
                inQualifier  = "centroid in";
                break;
            case sh::INTERPOLATION_FLAT:
                outQualifier = "flat out";
                inQualifier  = "flat in";
                break;
            default:
                UNREACHABLE();
        }
        outStageSource->insertQualifierSpecifier(name, outQualifier);
        inStageSource->insertQualifierSpecifier(name, inQualifier);
    }

    // Substitute layout and qualifier strings for the position varying. Use the first free
    // varying register after the packed varyings.
    constexpr char kVaryingName[] = "ANGLEPosition";
    std::stringstream layoutStream;
    layoutStream << "location = " << (resources.varyingPacking.getMaxSemanticIndex() + 1);
    const std::string layout = layoutStream.str();

    outStageSource->insertLayoutSpecifier(kVaryingName, layout);
    inStageSource->insertLayoutSpecifier(kVaryingName, layout);

    outStageSource->insertQualifierSpecifier(kVaryingName, "out");
    inStageSource->insertQualifierSpecifier(kVaryingName, "in");
}

void AssignUniformBindings(const GlslangSourceOptions &options,
                           gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    // Bind the default uniforms for vertex and fragment shaders.
    // See corresponding code in OutputVulkanGLSL.cpp.
    const std::string uniformsDescriptorSet =
        "set = " + Str(options.uniformsAndXfbDescriptorSetIndex);

    constexpr char kDefaultUniformsBlockName[] = "defaultUniforms";
    uint32_t bindingIndex                      = 0;
    for (IntermediateShaderSource &shaderSource : *shaderSources)
    {
        if (!shaderSource.empty())
        {
            std::string defaultUniformsBinding =
                uniformsDescriptorSet + ", binding = " + Str(bindingIndex++);

            shaderSource.insertLayoutSpecifier(kDefaultUniformsBlockName, defaultUniformsBinding);
        }
    }

    // Substitute layout and qualifier strings for the driver uniforms block.
    const std::string driverBlockLayoutString =
        "set = " + Str(options.driverUniformsDescriptorSetIndex) + ", binding = 0";
    constexpr char kDriverBlockName[] = "ANGLEUniformBlock";

    for (IntermediateShaderSource &shaderSource : *shaderSources)
    {
        shaderSource.insertLayoutSpecifier(kDriverBlockName, driverBlockLayoutString);
        shaderSource.insertQualifierSpecifier(kDriverBlockName, kUniformQualifier);
    }
}

// Helper to go through shader stages and substitute layout and qualifier macros.
void AssignResourceBinding(gl::ShaderBitSet activeShaders,
                           const std::string &name,
                           const std::string &bindingString,
                           const char *qualifier,
                           const char *unusedSubstitution,
                           gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    for (const gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        IntermediateShaderSource &shaderSource = (*shaderSources)[shaderType];
        if (!shaderSource.empty())
        {
            if (activeShaders[shaderType])
            {
                shaderSource.insertLayoutSpecifier(name, bindingString);
                shaderSource.insertQualifierSpecifier(name, qualifier);
            }
            else
            {
                shaderSource.eraseLayoutAndQualifierSpecifiers(name, unusedSubstitution);
            }
        }
    }
}

uint32_t AssignInterfaceBlockBindings(const GlslangSourceOptions &options,
                                      const std::vector<gl::InterfaceBlock> &blocks,
                                      const char *qualifier,
                                      uint32_t bindingStart,
                                      gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    const std::string resourcesDescriptorSet =
        "set = " + Str(options.shaderResourceDescriptorSetIndex);

    uint32_t bindingIndex = bindingStart;
    for (const gl::InterfaceBlock &block : blocks)
    {
        if (!block.isArray || block.arrayElement == 0)
        {
            const std::string bindingString =
                resourcesDescriptorSet + ", binding = " + Str(bindingIndex++);

            AssignResourceBinding(block.activeShaders(), block.name, bindingString, qualifier,
                                  kUnusedBlockSubstitution, shaderSources);
        }
    }

    return bindingIndex;
}

uint32_t AssignAtomicCounterBufferBindings(const GlslangSourceOptions &options,
                                           const std::vector<gl::AtomicCounterBuffer> &buffers,
                                           const char *qualifier,
                                           uint32_t bindingStart,
                                           gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    if (buffers.size() == 0)
    {
        return bindingStart;
    }

    constexpr char kAtomicCounterBlockName[] = "ANGLEAtomicCounters";
    const std::string bindingString = "set = " + Str(options.shaderResourceDescriptorSetIndex) +
                                      ", binding = " + Str(bindingStart);

    for (const gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        IntermediateShaderSource &shaderSource = (*shaderSources)[shaderType];
        if (!shaderSource.empty())
        {
            // All atomic counter buffers are placed under one binding shared between all stages.
            shaderSource.insertLayoutSpecifier(kAtomicCounterBlockName, bindingString);
            shaderSource.insertQualifierSpecifier(kAtomicCounterBlockName, qualifier);
        }
    }

    return bindingStart + 1;
}

uint32_t AssignImageBindings(const GlslangSourceOptions &options,
                             const std::vector<gl::LinkedUniform> &uniforms,
                             const gl::RangeUI &imageUniformRange,
                             uint32_t bindingStart,
                             gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    const std::string resourcesDescriptorSet =
        "set = " + Str(options.shaderResourceDescriptorSetIndex);

    uint32_t bindingIndex = bindingStart;
    for (unsigned int uniformIndex : imageUniformRange)
    {
        const gl::LinkedUniform &imageUniform = uniforms[uniformIndex];
        const std::string bindingString =
            resourcesDescriptorSet + ", binding = " + Str(bindingIndex++);

        std::string name = imageUniform.name;
        if (name.back() == ']')
        {
            name = name.substr(0, name.find('['));
        }

        AssignResourceBinding(imageUniform.activeShaders(), name, bindingString, kUniformQualifier,
                              kUnusedUniformSubstitution, shaderSources);
    }

    return bindingIndex;
}

void AssignNonTextureBindings(const GlslangSourceOptions &options,
                              const gl::ProgramState &programState,
                              gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    uint32_t bindingStart = 0;

    const std::vector<gl::InterfaceBlock> &uniformBlocks = programState.getUniformBlocks();
    bindingStart = AssignInterfaceBlockBindings(options, uniformBlocks, kUniformQualifier,
                                                bindingStart, shaderSources);

    const std::vector<gl::InterfaceBlock> &storageBlocks = programState.getShaderStorageBlocks();
    bindingStart = AssignInterfaceBlockBindings(options, storageBlocks, kSSBOQualifier,
                                                bindingStart, shaderSources);

    const std::vector<gl::AtomicCounterBuffer> &atomicCounterBuffers =
        programState.getAtomicCounterBuffers();
    bindingStart = AssignAtomicCounterBufferBindings(options, atomicCounterBuffers, kSSBOQualifier,
                                                     bindingStart, shaderSources);

    const std::vector<gl::LinkedUniform> &uniforms = programState.getUniforms();
    const gl::RangeUI &imageUniformRange           = programState.getImageUniformRange();
    bindingStart =
        AssignImageBindings(options, uniforms, imageUniformRange, bindingStart, shaderSources);
}

void AssignTextureBindings(const GlslangSourceOptions &options,
                           bool useOldRewriteStructSamplers,
                           const gl::ProgramState &programState,
                           gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    const std::string texturesDescriptorSet = "set = " + Str(options.textureDescriptorSetIndex);

    // Assign textures to a descriptor set and binding.
    uint32_t bindingIndex                          = 0;
    const std::vector<gl::LinkedUniform> &uniforms = programState.getUniforms();

    for (unsigned int uniformIndex : programState.getSamplerUniformRange())
    {
        const gl::LinkedUniform &samplerUniform = uniforms[uniformIndex];

        if (!useOldRewriteStructSamplers &&
            gl::SamplerNameContainsNonZeroArrayElement(samplerUniform.name))
        {
            continue;
        }

        const std::string bindingString =
            texturesDescriptorSet + ", binding = " + Str(bindingIndex++);

        // Samplers in structs are extracted and renamed.
        const std::string samplerName = useOldRewriteStructSamplers
                                            ? GetMappedSamplerNameOld(samplerUniform.name)
                                            : GlslangGetMappedSamplerName(samplerUniform.name);

        AssignResourceBinding(samplerUniform.activeShaders(), samplerName, bindingString,
                              kUniformQualifier, kUnusedUniformSubstitution, shaderSources);
    }
}

void CleanupUnusedEntities(bool useOldRewriteStructSamplers,
                           const gl::ProgramState &programState,
                           const gl::ProgramLinkedResources &resources,
                           gl::ShaderType shaderType,
                           gl::ShaderMap<IntermediateShaderSource> *shaderSources)
{
    gl::Shader *shader               = programState.getAttachedShader(shaderType);
    IntermediateShaderSource &source = (*shaderSources)[shaderType];
    if (!source.empty())
    {
        ASSERT(shader != nullptr);

        // The attributes in the programState could have been filled with active attributes only
        // depending on the shader version. If there is inactive attributes left, we have to remove
        // their @@ QUALIFIER and @@ LAYOUT markers.
        for (const sh::ShaderVariable &attribute : shader->getAllAttributes())
        {
            if (attribute.active)
            {
                continue;
            }

            source.eraseLayoutAndQualifierSpecifiers(attribute.name, "");
        }
    }

    // Remove all the markers for unused varyings.
    for (const std::string &varyingName : resources.varyingPacking.getInactiveVaryingNames())
    {
        for (IntermediateShaderSource &shaderSource : *shaderSources)
        {
            shaderSource.eraseLayoutAndQualifierSpecifiers(varyingName, "");
        }
    }

    // Remove all the markers for unused interface blocks, and replace them with |struct|.
    for (const std::string &unusedInterfaceBlock : resources.unusedInterfaceBlocks)
    {
        for (IntermediateShaderSource &shaderSource : *shaderSources)
        {
            shaderSource.eraseLayoutAndQualifierSpecifiers(unusedInterfaceBlock,
                                                           kUnusedBlockSubstitution);
        }
    }

    // Comment out unused uniforms.  This relies on the fact that the shader compiler outputs
    // uniforms to a single line.
    for (const gl::UnusedUniform &unusedUniform : resources.unusedUniforms)
    {
        std::string uniformName = unusedUniform.isSampler
                                      ? useOldRewriteStructSamplers
                                            ? GetMappedSamplerNameOld(unusedUniform.name)
                                            : GlslangGetMappedSamplerName(unusedUniform.name)
                                      : unusedUniform.name;

        for (IntermediateShaderSource &shaderSource : *shaderSources)
        {
            shaderSource.eraseLayoutAndQualifierSpecifiers(uniformName, kUnusedUniformSubstitution);
        }
    }
}

constexpr gl::ShaderMap<EShLanguage> kShLanguageMap = {
    {gl::ShaderType::Vertex, EShLangVertex},
    {gl::ShaderType::Geometry, EShLangGeometry},
    {gl::ShaderType::Fragment, EShLangFragment},
    {gl::ShaderType::Compute, EShLangCompute},
};

angle::Result GetShaderSpirvCode(GlslangErrorCallback callback,
                                 const gl::Caps &glCaps,
                                 const gl::ShaderMap<std::string> &shaderSources,
                                 gl::ShaderMap<std::vector<uint32_t>> *shaderCodeOut)
{
    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);

    TBuiltInResource builtInResources(glslang::DefaultTBuiltInResource);
    GetBuiltInResourcesFromCaps(glCaps, &builtInResources);

    glslang::TShader vertexShader(EShLangVertex);
    glslang::TShader fragmentShader(EShLangFragment);
    glslang::TShader geometryShader(EShLangGeometry);
    glslang::TShader computeShader(EShLangCompute);

    gl::ShaderMap<glslang::TShader *> shaders = {
        {gl::ShaderType::Vertex, &vertexShader},
        {gl::ShaderType::Fragment, &fragmentShader},
        {gl::ShaderType::Geometry, &geometryShader},
        {gl::ShaderType::Compute, &computeShader},
    };
    glslang::TProgram program;

    for (const gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        if (shaderSources[shaderType].empty())
        {
            continue;
        }

        const char *shaderString = shaderSources[shaderType].c_str();
        int shaderLength         = static_cast<int>(shaderSources[shaderType].size());

        glslang::TShader *shader = shaders[shaderType];
        shader->setStringsWithLengths(&shaderString, &shaderLength, 1);
        shader->setEntryPoint("main");

        bool result = shader->parse(&builtInResources, 450, ECoreProfile, false, false, messages);
        if (!result)
        {
            ERR() << "Internal error parsing Vulkan shader corresponding to " << shaderType << ":\n"
                  << shader->getInfoLog() << "\n"
                  << shader->getInfoDebugLog() << "\n";
            ANGLE_GLSLANG_CHECK(callback, false, GlslangError::InvalidShader);
        }

        program.addShader(shader);
    }

    bool linkResult = program.link(messages);
    if (!linkResult)
    {
        ERR() << "Internal error linking Vulkan shaders:\n" << program.getInfoLog() << "\n";
        ANGLE_GLSLANG_CHECK(callback, false, GlslangError::InvalidShader);
    }

    for (const gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        if (shaderSources[shaderType].empty())
        {
            continue;
        }

        glslang::TIntermediate *intermediate = program.getIntermediate(kShLanguageMap[shaderType]);
        glslang::GlslangToSpv(*intermediate, (*shaderCodeOut)[shaderType]);
    }

    return angle::Result::Continue;
}
}  // anonymous namespace

void GlslangInitialize()
{
    int result = ShInitialize();
    ASSERT(result != 0);
}

void GlslangRelease()
{
    int result = ShFinalize();
    ASSERT(result != 0);
}

std::string GlslangGetMappedSamplerName(const std::string &originalName)
{
    std::string samplerName = originalName;

    // Samplers in structs are extracted.
    std::replace(samplerName.begin(), samplerName.end(), '.', '_');

    // Remove array elements
    auto out = samplerName.begin();
    for (auto in = samplerName.begin(); in != samplerName.end(); in++)
    {
        if (*in == '[')
        {
            while (*in != ']')
            {
                in++;
                ASSERT(in != samplerName.end());
            }
        }
        else
        {
            *out++ = *in;
        }
    }

    samplerName.erase(out, samplerName.end());

    return samplerName;
}

void GlslangGetShaderSource(const GlslangSourceOptions &options,
                            bool useOldRewriteStructSamplers,
                            const gl::ProgramState &programState,
                            const gl::ProgramLinkedResources &resources,
                            gl::ShaderMap<std::string> *shaderSourcesOut)
{
    gl::ShaderMap<IntermediateShaderSource> intermediateSources;

    for (const gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        gl::Shader *glShader = programState.getAttachedShader(shaderType);
        if (glShader)
        {
            intermediateSources[shaderType].init(glShader->getTranslatedSource());
        }
    }

    IntermediateShaderSource *vertexSource   = &intermediateSources[gl::ShaderType::Vertex];
    IntermediateShaderSource *fragmentSource = &intermediateSources[gl::ShaderType::Fragment];
    IntermediateShaderSource *geometrySource = &intermediateSources[gl::ShaderType::Geometry];

    if (!geometrySource->empty())
    {
        AssignOutputLocations(programState, fragmentSource);
        AssignVaryingLocations(programState, resources, geometrySource, fragmentSource);
        if (!vertexSource->empty())
        {
            AssignAttributeLocations(programState, vertexSource);
            AssignVaryingLocations(programState, resources, vertexSource, geometrySource);
        }
    }
    else if (!vertexSource->empty())
    {
        AssignAttributeLocations(programState, vertexSource);
        AssignOutputLocations(programState, fragmentSource);
        AssignVaryingLocations(programState, resources, vertexSource, fragmentSource);
    }
    else if (!fragmentSource->empty())
    {
        AssignAttributeLocations(programState, fragmentSource);
        AssignOutputLocations(programState, fragmentSource);
        AssignVaryingLocations(programState, resources, vertexSource, fragmentSource);
    }
    AssignUniformBindings(options, &intermediateSources);
    AssignTextureBindings(options, useOldRewriteStructSamplers, programState, &intermediateSources);
    AssignNonTextureBindings(options, programState, &intermediateSources);

    for (const auto shaderType : gl::kAllGraphicsShaderTypes)
    {
        CleanupUnusedEntities(useOldRewriteStructSamplers, programState, resources, shaderType,
                              &intermediateSources);
    }

    // Write transform feedback output code.
    if (!vertexSource->empty())
    {
        if (programState.getLinkedTransformFeedbackVaryings().empty())
        {
            vertexSource->insertTransformFeedbackDeclaration("");
            vertexSource->insertTransformFeedbackOutput("");
        }
        else
        {
            GenerateTransformFeedbackOutputs(options, programState, vertexSource);
        }
    }

    for (const gl::ShaderType shaderType : gl::AllShaderTypes())
    {
        (*shaderSourcesOut)[shaderType] = intermediateSources[shaderType].getShaderSource();
    }
}

angle::Result GlslangGetShaderSpirvCode(GlslangErrorCallback callback,
                                        const gl::Caps &glCaps,
                                        bool enableLineRasterEmulation,
                                        bool enableXfbEmulation,
                                        const gl::ShaderMap<std::string> &shaderSources,
                                        gl::ShaderMap<std::vector<uint32_t>> *shaderCodeOut)
{
    if (enableLineRasterEmulation || enableXfbEmulation)
    {
        ASSERT(shaderSources[gl::ShaderType::Compute].empty());

        gl::ShaderMap<std::string> patchedSources = shaderSources;

        std::string defines = kVersionDefine;
        if (enableLineRasterEmulation)
        {
            defines += kLineRasterDefine;
        }
        if (enableXfbEmulation)
        {
            defines += kXfbEmuDefine;
        }

        for (const gl::ShaderType shaderType : gl::AllShaderTypes())
        {
            std::string &src = patchedSources[shaderType];
            if (src.empty())
            {
                continue;
            }

            ANGLE_GLSLANG_CHECK(callback, angle::ReplaceSubstring(&src, kVersionDefine, defines),
                                GlslangError::InvalidShader);
        }

        return GetShaderSpirvCode(callback, glCaps, patchedSources, shaderCodeOut);
    }
    else
    {
        return GetShaderSpirvCode(callback, glCaps, shaderSources, shaderCodeOut);
    }
}
}  // namespace rx
