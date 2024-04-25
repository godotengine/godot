//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/SourceCodeNodeMdl.h>
#include <MaterialXGenMdl/MdlSyntax.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Util.h>

#include <numeric>

MATERIALX_NAMESPACE_BEGIN

namespace // anonymous
{
const string MARKER_MDL_VERSION_SUFFIX = "MDL_VERSION_SUFFIX";

StringVec replaceSourceCodeMarkers(const string& nodeName, const string& soureCode, std::function<string(const string&)> lambda)
{
    // An inline function call
    // Replace tokens of the format "{{<var>}}"
    static const string prefix("{{");
    static const string postfix("}}");

    size_t pos = 0;
    size_t i = soureCode.find_first_of(prefix);
    StringVec code;
    while (i != string::npos)
    {
        code.push_back(soureCode.substr(pos, i - pos));
        size_t j = soureCode.find_first_of(postfix, i + 2);
        if (j == string::npos)
        {
            throw ExceptionShaderGenError("Malformed inline expression in implementation for node " + nodeName);
        }
        const string marker = soureCode.substr(i + 2, j - i - 2);
        code.push_back(lambda(marker));
        pos = j + 2;
        i = soureCode.find_first_of(prefix, pos);
    }
    code.push_back(soureCode.substr(pos));
    return code;
}

} // anonymous namespace

ShaderNodeImplPtr SourceCodeNodeMdl::create()
{
    return std::make_shared<SourceCodeNodeMdl>();
}

void SourceCodeNodeMdl::initialize(const InterfaceElement& element, GenContext& context)
{
    SourceCodeNode::initialize(element, context);

    const Implementation& impl = static_cast<const Implementation&>(element);
    NodeDefPtr nodeDef = impl.getNodeDef();
    if (!nodeDef)
    {
        throw ExceptionShaderGenError("Can't find nodedef for implementation element " + element.getName());
    }

    _returnStruct = EMPTY_STRING;
    if (nodeDef->getOutputCount() > 1)
    {
        if (_functionName.empty())
        {
            size_t pos = _functionSource.find_first_of('(');
            string functionName = _functionSource.substr(0, pos);

            const ShaderGenerator& shadergen = context.getShaderGenerator();
            const MdlShaderGenerator& shadergenMdl = static_cast<const MdlShaderGenerator&>(shadergen);
            const string versionSuffix = shadergenMdl.getMdlVersionFilenameSuffix(context);
            StringVec code = replaceSourceCodeMarkers(getName(), functionName, [&versionSuffix](const string& marker)
                {
                    return marker == MARKER_MDL_VERSION_SUFFIX ? versionSuffix : EMPTY_STRING;
                });
            functionName = std::accumulate(code.begin(), code.end(), EMPTY_STRING);
            _returnStruct = functionName + "__result";
        }
        else
        {
            _returnStruct = _functionName + "__result";
        }
    }
}

void SourceCodeNodeMdl::emitFunctionDefinition(const ShaderNode&, GenContext&, ShaderStage&) const
{
}

void SourceCodeNodeMdl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const MdlShaderGenerator& shadergenMdl = static_cast<const MdlShaderGenerator&>(shadergen);
        if (_inlined)
        {
            const string versionSuffix = shadergenMdl.getMdlVersionFilenameSuffix(context);
            StringVec code = replaceSourceCodeMarkers(node.getName(), _functionSource,
                [&shadergenMdl, &context, &node, &versionSuffix](const string& marker)
                {
                    // Special handling for the version suffix of MDL source code modules.
                    if (marker == MARKER_MDL_VERSION_SUFFIX)
                    {
                        return versionSuffix;
                    }
                    // Insert inputs based on parameter names.
                    else
                    {
                        const ShaderInput* input = node.getInput(marker);
                        if (!input)
                        {
                            throw ExceptionShaderGenError("Could not find an input named '" + marker +
                                                          "' on node '" + node.getName() + "'");
                        }

                        return shadergenMdl.getUpstreamResult(input, context);
                    }
                });

            if (!_returnStruct.empty())
            {
                // Emit the struct multioutput.
                const string resultVariableName = node.getName() + "_result";
                shadergen.emitLineBegin(stage);
                shadergen.emitString(_returnStruct + " " + resultVariableName + " = ", stage);
            }
            else
            {
                // Emit the single output.
                shadergen.emitLineBegin(stage);
                shadergen.emitOutput(node.getOutput(0), true, false, context, stage);
                shadergen.emitString(" = ", stage);
            }

            for (const string& c : code)
            {
                shadergen.emitString(c, stage);
            }
            shadergen.emitLineEnd(stage);
        }
        else
        {
            // An ordinary source code function call

            if (!_returnStruct.empty())
            {
                // Emit the struct multioutput.
                const string resultVariableName = node.getName() + "_result";
                shadergen.emitLineBegin(stage);
                shadergen.emitString(_returnStruct + " " + resultVariableName + " = ", stage);
            }
            else
            {
                // Emit the single output.
                shadergen.emitLineBegin(stage);
                shadergen.emitOutput(node.getOutput(0), true, false, context, stage);
                shadergen.emitString(" = ", stage);
            }

            // Emit function name.
            shadergen.emitString(_functionName + "(", stage);

            // Emit all inputs on the node.
            string delim = "";
            for (ShaderInput* input : node.getInputs())
            {
                shadergen.emitString(delim, stage);
                shadergen.emitInput(input, context, stage);
                delim = ", ";
            }

            // End function call
            shadergen.emitString(")", stage);
            shadergen.emitLineEnd(stage);
        }
    }
}

MATERIALX_NAMESPACE_END
