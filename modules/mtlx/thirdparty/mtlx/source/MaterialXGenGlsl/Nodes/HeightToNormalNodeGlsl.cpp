//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/Nodes/HeightToNormalNodeGlsl.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

/// Name of filter function to call to compute normals from input samples
const string filterFunctionName = "mx_normal_from_samples_sobel";

/// Name of function to compute sample size in uv space. Takes uv, filter size, and filter offset
/// as input, and return a 2 channel vector as output
const string sampleSizeFunctionUV = "mx_compute_sample_size_uv";

const unsigned int sampleCount = 9;
const unsigned int filterWidth = 3;
const float filterSize = 1.0;
const float filterOffset = 0.0;

} // anonymous namespace

ShaderNodeImplPtr HeightToNormalNodeGlsl::create()
{
    return std::make_shared<HeightToNormalNodeGlsl>();
}

void HeightToNormalNodeGlsl::createVariables(const ShaderNode&, GenContext&, Shader&) const
{
    // Default filter kernels from ConvolutionNode are not used by this derived class.
}

void HeightToNormalNodeGlsl::computeSampleOffsetStrings(const string& sampleSizeName, const string& offsetTypeString,
                                                        unsigned int, StringVec& offsetStrings) const
{
    // Build a 3x3 grid of samples that are offset by the provided sample size
    for (int row = -1; row <= 1; row++)
    {
        for (int col = -1; col <= 1; col++)
        {
            offsetStrings.push_back(" + " + sampleSizeName + " * " + offsetTypeString + "(" + std::to_string(float(col)) + "," + std::to_string(float(row)) + ")");
        }
    }
}

bool HeightToNormalNodeGlsl::acceptsInputType(const TypeDesc* type) const
{
    // Only support inputs which are float scalar
    return (*type == *Type::FLOAT && type->isScalar());
}

void HeightToNormalNodeGlsl::emitFunctionDefinition(const ShaderNode&, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Emit sampling functions
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLibraryInclude("stdlib/genglsl/lib/mx_sampling.glsl", context, stage);
        shadergen.emitLineBreak(stage);
    }
}

void HeightToNormalNodeGlsl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        const ShaderInput* inInput = node.getInput("in");
        const ShaderInput* scaleInput = node.getInput("scale");

        if (!inInput || !scaleInput)
        {
            throw ExceptionShaderGenError("Node '" + node.getName() + "' is not a valid heighttonormal node");
        }

        // Create the input "samples". This means to emit the calls to
        // compute the sames and return a set of strings containaing
        // the variables to assign to the sample grid.
        //
        StringVec sampleStrings;
        emitInputSamplesUV(node, sampleCount, filterWidth,
                           filterSize, filterOffset, sampleSizeFunctionUV,
                           context, stage, sampleStrings);

        const ShaderOutput* output = node.getOutput();

        // Emit code to evaluate samples.
        //
        string sampleName(output->getVariable() + "_samples");
        shadergen.emitLine("float " + sampleName + "[" + std::to_string(sampleCount) + "]", stage);
        for (unsigned int i = 0; i < sampleCount; i++)
        {
            shadergen.emitLine(sampleName + "[" + std::to_string(i) + "] = " + sampleStrings[i], stage);
        }
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, false, context, stage);
        shadergen.emitString(" = " + filterFunctionName, stage);
        shadergen.emitString("(" + sampleName + ", ", stage);
        shadergen.emitInput(scaleInput, context, stage);
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

const string& HeightToNormalNodeGlsl::getTarget() const
{
    return GlslShaderGenerator::TARGET;
}

MATERIALX_NAMESPACE_END
