//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>

#include <MaterialXGenMdl/Nodes/BlurNodeMdl.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr BlurNodeMdl::create()
{
    return std::make_shared<BlurNodeMdl>();
}

void BlurNodeMdl::outputSampleArray(const ShaderGenerator& shadergen, ShaderStage& stage, const TypeDesc* inputType,
                                    const string& sampleName, const StringVec& sampleStrings) const
{
    const Syntax& syntax = shadergen.getSyntax();
    const string& inputTypeString = syntax.getTypeName(inputType);
    const string& inputTypeDefaultValue = syntax.getDefaultValue(inputType);

    const size_t maxSampleCount = 49;
    const string arrayDeclaration = inputTypeString + "[49]";
    shadergen.emitLine(arrayDeclaration + " " + sampleName + " = " + arrayDeclaration, stage, false);
    shadergen.emitScopeBegin(stage, Syntax::PARENTHESES);

    const size_t sampleCount = sampleStrings.size();
    for (size_t i = 0; i < sampleCount; i++)
    {
        shadergen.emitLineBegin(stage);
        shadergen.emitString(sampleStrings[i], stage);
        if (i + 1 < maxSampleCount)
        {
            shadergen.emitString(",", stage);
        }
        shadergen.emitLineEnd(stage, false);
    }
    // We must fill out the whole array to have a valid MDL syntax.
    // So padd it with dummy default values.
    for (size_t i = sampleCount; i < maxSampleCount; i++)
    {
        shadergen.emitLineBegin(stage);
        shadergen.emitString(inputTypeDefaultValue, stage);
        if (i + 1 < maxSampleCount)
        {
            shadergen.emitString(",", stage);
        }
        shadergen.emitLineEnd(stage, false);
    }
    shadergen.emitScopeEnd(stage, true);
}

// No definitions needed for blur in MDL so override the base class with an empty function definition emitter.
void BlurNodeMdl::emitSamplingFunctionDefinition(const ShaderNode&, GenContext&, ShaderStage&) const
{
}

void BlurNodeMdl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        const ShaderInput* inInput = node.getInput(IN_STRING);

        const Syntax& syntax = shadergen.getSyntax();

        // Get input type name string
        const string& inputTypeString = inInput && acceptsInputType(inInput->getType()) ?
                                                   syntax.getTypeName(inInput->getType()) :
                                                   EMPTY_STRING;

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

            // Emit branching code to compute result based on filter type
            //
            shadergen.emitString(" = ", stage);
            shadergen.emitInput(filterTypeInput, context, stage);
            // Remap enumeration for comparison as needed
            std::pair<const TypeDesc*, ValuePtr> result;
            string emitValue = "\"" + GAUSSIAN_FILTER + "\"";
            if (syntax.remapEnumeration(GAUSSIAN_FILTER, Type::STRING, FILTER_LIST, result))
            {
                emitValue = syntax.getValue(result.first, *(result.second));
            }
            shadergen.emitString(" == " + emitValue + " ? ", stage);
            {
                string filterFunctionName = MX_CONVOLUTION_PREFIX_STRING + inputTypeString;
                shadergen.emitString(filterFunctionName + "(" + sampleName + ", " +
                                         GAUSSIAN_WEIGHTS_VARIABLE + ", " +
                                         std::to_string(arrayOffset) + ", " +
                                         std::to_string(sampleCount) +
                                         ")",
                                     stage);
            }
            shadergen.emitString(" : ", stage);
            {
                string filterFunctionName = MX_CONVOLUTION_PREFIX_STRING + inputTypeString;
                shadergen.emitString(filterFunctionName + "(" + sampleName + ", " +
                                         BOX_WEIGHTS_VARIABLE + ", " +
                                         std::to_string(arrayOffset) + ", " +
                                         std::to_string(sampleCount) +
                                         ")",
                                     stage);
            }
            shadergen.emitLineEnd(stage);
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
