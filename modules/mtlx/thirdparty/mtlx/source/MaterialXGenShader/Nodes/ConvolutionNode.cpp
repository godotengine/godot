//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/ConvolutionNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string SAMPLE2D_INPUT = "texcoord";
const string SAMPLE3D_INPUT = "position";

} // anonymous namespace

const std::array<float, 3> GAUSSIAN_KERNEL_3 = {
    0.27901f, 0.44198f, 0.27901f // Sigma 1
};
const std::array<float, 5> GAUSSIAN_KERNEL_5 = {
    0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f // Sigma 1
};
const std::array<float, 7> GAUSSIAN_KERNEL_7 = {
    0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f // Sigma 1
};

ConvolutionNode::ConvolutionNode()
{
}

void ConvolutionNode::createVariables(const ShaderNode&, GenContext&, Shader& shader) const
{
    ShaderStage& stage = shader.getStage(Stage::PIXEL);
    VariableBlock& constants = stage.getConstantBlock();

    // Create constant for box weights
    const float boxWeight3x3 = 1.0f / 9.0f;
    const float boxWeight5x5 = 1.0f / 25.0f;
    const float boxWeight7x7 = 1.0f / 49.0f;
    vector<float> boxWeightArray;
    boxWeightArray.push_back(1.0f);
    for (unsigned int i = 0; i < 9; i++)
    {
        boxWeightArray.push_back(boxWeight3x3);
    }
    for (unsigned int i = 0; i < 25; i++)
    {
        boxWeightArray.push_back(boxWeight5x5);
    }
    for (unsigned int i = 0; i < 49; i++)
    {
        boxWeightArray.push_back(boxWeight7x7);
    }
    constants.add(Type::FLOATARRAY, "c_box_filter_weights", Value::createValue<vector<float>>(boxWeightArray));

    // Create constant for Gaussian weights
    vector<float> gaussianWeightArray;
    gaussianWeightArray.push_back(1.0f);
    for (unsigned int y = 0; y < 3; y++)
    {
        for (unsigned int x = 0; x < 3; x++)
        {
            gaussianWeightArray.push_back(GAUSSIAN_KERNEL_3[y] * GAUSSIAN_KERNEL_3[x]);
        }
    }
    for (unsigned int y = 0; y < 5; y++)
    {
        for (unsigned int x = 0; x < 5; x++)
        {
            gaussianWeightArray.push_back(GAUSSIAN_KERNEL_5[y] * GAUSSIAN_KERNEL_5[x]);
        }
    }
    for (unsigned int y = 0; y < 7; y++)
    {
        for (unsigned int x = 0; x < 7; x++)
        {
            gaussianWeightArray.push_back(GAUSSIAN_KERNEL_7[y] * GAUSSIAN_KERNEL_7[x]);
        }
    }
    constants.add(Type::FLOATARRAY, "c_gaussian_filter_weights", Value::createValue<vector<float>>(gaussianWeightArray));
}

/// Get input which is used for sampling. If there is none
/// then a null pointer is returned.
const ShaderInput* ConvolutionNode::getSamplingInput(const ShaderNode& node) const
{
    // Determine if this input can be sampled
    if (node.hasClassification(ShaderNode::Classification::SAMPLE2D))
    {
        const ShaderInput* input = node.getInput(SAMPLE2D_INPUT);
        return *input->getType() == *Type::VECTOR2 ? input : nullptr;
    }
    else if (node.hasClassification(ShaderNode::Classification::SAMPLE3D))
    {
        const ShaderInput* input = node.getInput(SAMPLE3D_INPUT);
        return *input->getType() == *Type::VECTOR3 ? input : nullptr;
    }
    return nullptr;
}

void ConvolutionNode::emitInputSamplesUV(const ShaderNode& node,
                                         unsigned int sampleCount, unsigned int filterWidth,
                                         float filterSize, float filterOffset,
                                         const string& sampleSizeFunctionUV,
                                         GenContext& context, ShaderStage& stage,
                                         StringVec& sampleStrings) const
{
    sampleStrings.clear();

    const ShaderGenerator& shadergen = context.getShaderGenerator();

    // Check for an upstream node to sample
    const ShaderInput* inInput = node.getInput("in");
    const ShaderOutput* inConnection = inInput ? inInput->getConnection() : nullptr;

    if (inConnection && inConnection->getType() && acceptsInputType(inConnection->getType()))
    {
        const ShaderNode* upstreamNode = inConnection->getNode();
        if (upstreamNode && upstreamNode->hasClassification(ShaderNode::Classification::SAMPLE2D))
        {
            const ShaderNodeImpl& impl = upstreamNode->getImplementation();
            const ShaderOutput* upstreamOutput = upstreamNode->getOutput();
            if (upstreamOutput)
            {
                // Find out which input needs to be sampled multiple times
                // If the sample count is 1 then the sample code has already been emitted
                const ShaderInput* samplingInput = (sampleCount > 1) ? getSamplingInput(*upstreamNode) : nullptr;

                // TODO: For now we only support uv space sampling
                if (samplingInput && samplingInput->getType() != Type::VECTOR2)
                {
                    samplingInput = nullptr;
                }

                if (samplingInput)
                {
                    // This is not exposed. Assume a filter size of 1 with no offset

                    const ShaderOutput* output = node.getOutput();

                    // Emit code to compute sample size
                    //
                    string sampleInputValue = shadergen.getUpstreamResult(samplingInput, context);

                    const string sampleSizeName(output->getVariable() + "_sample_size");
                    const string vec2TypeString = shadergen.getSyntax().getTypeName(Type::VECTOR2);
                    string sampleCall(vec2TypeString + " " + sampleSizeName + " = " +
                                      sampleSizeFunctionUV + "(" +
                                      sampleInputValue + "," +
                                      std::to_string(filterSize) + "," +
                                      std::to_string(filterOffset) + ")");
                    shadergen.emitLine(sampleCall, stage);

                    // Build the sample offset strings. This is dependent on
                    // the derived class to determine where samples are located
                    // and to generate the strings required to offset from the center
                    // sample. The sample size is passed over.
                    //
                    StringVec inputVec2Suffix;
                    if (sampleCount > 1)
                    {
                        computeSampleOffsetStrings(sampleSizeName, vec2TypeString,
                                                   filterWidth, inputVec2Suffix);
                    }

                    // Emit outputs for sample input
                    for (unsigned int i = 0; i < sampleCount; i++)
                    {
                        // Add an input name suffix.
                        context.addInputSuffix(samplingInput, inputVec2Suffix[i]);

                        // Add a output name suffix for the emit call
                        string outputSuffix("_" + output->getVariable() + std::to_string(i));
                        context.addOutputSuffix(upstreamOutput, outputSuffix);

                        impl.emitFunctionCall(*upstreamNode, context, stage);

                        // Remove suffixes
                        context.removeInputSuffix(samplingInput);
                        context.removeOutputSuffix(upstreamOutput);

                        // Keep track of the output name with the suffix
                        sampleStrings.push_back(upstreamOutput->getVariable() + outputSuffix);
                    }
                }
                else
                {
                    // Use the same input for all samples.
                    // Note: This does not recomputed the output, but instead it reuses
                    // the output variable.
                    for (unsigned int i = 0; i < sampleCount; i++)
                    {
                        // Call the unmodified function
                        sampleStrings.push_back(upstreamOutput->getVariable());
                    }
                }
            }
        }
    }
    else
    {
        if (!inInput || !inInput->getValue())
        {
            throw ExceptionShaderGenError("No connection or value found on node: '" + node.getName() + "'");
        }
    }

    // Build a set of samples with constant values
    if (sampleStrings.empty())
    {
        const string inValueString = shadergen.getUpstreamResult(inInput, context);
        for (unsigned int i = 0; i < sampleCount; i++)
        {
            sampleStrings.push_back(inValueString);
        }
    }
}

MATERIALX_NAMESPACE_END
