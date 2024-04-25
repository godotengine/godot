//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/HeightToNormalNodeMdl.h>

#include <MaterialXGenMdl/MdlShaderGenerator.h>

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

ShaderNodeImplPtr HeightToNormalNodeMdl::create()
{
    return std::make_shared<HeightToNormalNodeMdl>();
}

void HeightToNormalNodeMdl::computeSampleOffsetStrings(const string& sampleSizeName, const string& offsetTypeString,
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

bool HeightToNormalNodeMdl::acceptsInputType(const TypeDesc* type) const
{
    // Only support inputs which are float scalar
    return (*type == *Type::FLOAT && type->isScalar());
}

void HeightToNormalNodeMdl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
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
        string arrayDeclaration = "float[" + std::to_string(sampleCount) + "]";
        shadergen.emitLine(arrayDeclaration + " " + sampleName + " = " + arrayDeclaration + "(", stage, false);
        for (unsigned int i = 0; i < sampleCount; i++)
        {
            shadergen.emitLineBegin(stage);
            shadergen.emitString("    " + sampleStrings[i], stage);
            if (i + 1 < sampleCount)
            {
                shadergen.emitLine(",", stage, false);
            }
        }
        shadergen.emitLine(")", stage);
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, false, context, stage);
        shadergen.emitString(" = " + filterFunctionName, stage);
        shadergen.emitString("(" + sampleName + ", ", stage);
        shadergen.emitInput(scaleInput, context, stage);
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

const string& HeightToNormalNodeMdl::getTarget() const
{
    return MdlShaderGenerator::TARGET;
}

MATERIALX_NAMESPACE_END
