//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/BlurNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Name of function to compute sample size in uv space. Takes uv, filter size, and filter offset
/// as input, and return a 2 channel vector as output
const string BlurNode::_sampleSizeFunctionUV = "mx_compute_sample_size_uv";

const float BlurNode::_filterSize = 1.0;
const float BlurNode::_filterOffset = 0.0;

const string BlurNode::BOX_FILTER = "box";
const string BlurNode::GAUSSIAN_FILTER = "gaussian";
const string BlurNode::BOX_WEIGHTS_VARIABLE = "c_box_filter_weights";
const string BlurNode::GAUSSIAN_WEIGHTS_VARIABLE = "c_gaussian_filter_weights";
const string BlurNode::FILTER_LIST = "box,gaussian";
const string BlurNode::IN_STRING = "in";
const string BlurNode::FILTER_TYPE_STRING = "filtertype";
const string BlurNode::FILTER_SIZE_STRING = "size";

BlurNode::BlurNode() :
    ConvolutionNode()
{
}

void BlurNode::computeSampleOffsetStrings(const string& sampleSizeName, const string& offsetTypeString,
                                          unsigned int filterWidth, StringVec& offsetStrings) const
{
    int w = static_cast<int>(filterWidth) / 2;
    // Build a NxN grid of samples that are offset by the provided sample size
    for (int row = -w; row <= w; row++)
    {
        for (int col = -w; col <= w; col++)
        {
            offsetStrings.push_back(" + " + sampleSizeName + " * " + offsetTypeString + "(" + std::to_string(float(col)) + "," + std::to_string(float(row)) + ")");
        }
    }
}

bool BlurNode::acceptsInputType(const TypeDesc* type) const
{
    // Float 1-4 is acceptable as input
    return ((*type == *Type::FLOAT && type->isScalar()) ||
            type->isFloat2() || type->isFloat3() || type->isFloat4());
}

void BlurNode::outputSampleArray(const ShaderGenerator& shadergen, ShaderStage& stage, const TypeDesc* inputType,
                                 const string& sampleName, const StringVec& sampleStrings) const
{
    const string MX_MAX_SAMPLE_COUNT_STRING("MX_MAX_SAMPLE_COUNT");

    const Syntax& syntax = shadergen.getSyntax();
    const string& inputTypeString = syntax.getTypeName(inputType);

    shadergen.emitLine(inputTypeString + " " + sampleName + "[" + MX_MAX_SAMPLE_COUNT_STRING + "]", stage);
    for (size_t i = 0; i < sampleStrings.size(); i++)
    {
        shadergen.emitLine(sampleName + "[" + std::to_string(i) + "] = " + sampleStrings[i], stage);
    }
}

void BlurNode::emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        emitSamplingFunctionDefinition(node, context, stage);
    }
}

void BlurNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        const ShaderInput* inInput = node.getInput(IN_STRING);
        const Syntax& syntax = shadergen.getSyntax();

        // Get input type name string
        const string& inputTypeString = inInput && acceptsInputType(inInput->getType()) ? syntax.getTypeName(inInput->getType()) : EMPTY_STRING;

        const ShaderInput* filterTypeInput = node.getInput(FILTER_TYPE_STRING);
        if (!filterTypeInput || inputTypeString.empty())
        {
            throw ExceptionShaderGenError("Node '" + node.getName() + "' is not a valid Blur node");
        }

        // Compute width of filter. Default is 1 which just means one 1x1 upstream samples
        const ShaderInput* sizeInput = node.getInput(FILTER_SIZE_STRING);
        unsigned int filterWidth = 1;
        unsigned int arrayOffset = 0;
        if (sizeInput)
        {
            float sizeInputValue = sizeInput->getValue()->asA<float>();
            if (sizeInputValue > 0.0f)
            {
                if (sizeInputValue <= 0.333f)
                {
                    filterWidth = 3;
                    arrayOffset = 1;
                }
                else if (sizeInputValue <= 0.666f)
                {
                    filterWidth = 5;
                    arrayOffset = 10;
                }
                else
                {
                    filterWidth = 7;
                    arrayOffset = 35;
                }
            }
        }

        // Sample count is square of filter size
        const unsigned int sampleCount = filterWidth * filterWidth;

        // Emit samples
        // Note: The maximum sample count MX_MAX_SAMPLE_COUNT is defined in the shader code and
        // is assumed to be 49 (7x7 kernel). If this changes the filter size logic here
        // needs to be adjusted.
        //
        StringVec sampleStrings;
        emitInputSamplesUV(node, sampleCount, filterWidth,
                           _filterSize, _filterOffset, _sampleSizeFunctionUV,
                           context, stage, sampleStrings);

        // There should always be at least 1 sample
        if (sampleStrings.empty())
        {
            throw ExceptionShaderGenError("Node '" + node.getName() + "' cannot compute upstream samples");
        }

        const ShaderOutput* output = node.getOutput();

        if (sampleCount > 1)
        {
            const string MX_CONVOLUTION_PREFIX_STRING("mx_convolution_");
            const string SAMPLES_POSTFIX_STRING("_samples");

            // Set up sample array
            string sampleName(output->getVariable() + SAMPLES_POSTFIX_STRING);
            outputSampleArray(shadergen, stage, inInput->getType(), sampleName, sampleStrings);

            // Emit code to evaluate using input sample and weight arrays.
            // The function to call depends on input type.
            //
            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(output, true, false, context, stage);
            shadergen.emitLineEnd(stage);

            // Emit branching code to compute result based on filter type
            //
            shadergen.emitLineBegin(stage);
            shadergen.emitString("if (", stage);
            shadergen.emitInput(filterTypeInput, context, stage);
            // Remap enumeration for comparison as needed
            std::pair<const TypeDesc*, ValuePtr> result;
            string emitValue = "\"" + GAUSSIAN_FILTER + "\"";
            if (syntax.remapEnumeration(GAUSSIAN_FILTER, Type::STRING, FILTER_LIST, result))
            {
                emitValue = syntax.getValue(result.first, *(result.second));
            }
            shadergen.emitString(" == " + emitValue + ")", stage);
            shadergen.emitLineEnd(stage, false);

            shadergen.emitScopeBegin(stage);
            {
                string filterFunctionName = MX_CONVOLUTION_PREFIX_STRING + inputTypeString;
                shadergen.emitLineBegin(stage);
                shadergen.emitString(output->getVariable(), stage);
                shadergen.emitString(" = " + filterFunctionName, stage);
                shadergen.emitString("(" + sampleName + ", " +
                                         GAUSSIAN_WEIGHTS_VARIABLE + ", " +
                                         std::to_string(arrayOffset) + ", " +
                                         std::to_string(sampleCount) +
                                         ")",
                                     stage);
                shadergen.emitLineEnd(stage);
            }
            shadergen.emitScopeEnd(stage);
            shadergen.emitLine("else", stage, false);
            shadergen.emitScopeBegin(stage);
            {
                string filterFunctionName = MX_CONVOLUTION_PREFIX_STRING + inputTypeString;
                shadergen.emitLineBegin(stage);
                shadergen.emitString(output->getVariable(), stage);
                shadergen.emitString(" = " + filterFunctionName, stage);
                shadergen.emitString("(" + sampleName + ", " +
                                         BOX_WEIGHTS_VARIABLE + ", " +
                                         std::to_string(arrayOffset) + ", " +
                                         std::to_string(sampleCount) +
                                         ")",
                                     stage);
                shadergen.emitLineEnd(stage);
            }
            shadergen.emitScopeEnd(stage);
        }
        else
        {
            // This is just a pass-through of the upstream sample if any,
            // or the constant value on the node.
            //
            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(output, true, false, context, stage);
            shadergen.emitString(" = " + sampleStrings[0], stage);
            shadergen.emitLineEnd(stage);
        }
    }
}

MATERIALX_NAMESPACE_END
