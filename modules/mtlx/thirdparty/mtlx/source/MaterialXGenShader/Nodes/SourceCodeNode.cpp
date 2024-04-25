//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/SourceCodeNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXFormat/Util.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const string INLINE_VARIABLE_PREFIX("{{");
const string INLINE_VARIABLE_SUFFIX("}}");

} // anonymous namespace

ShaderNodeImplPtr SourceCodeNode::create()
{
    return std::make_shared<SourceCodeNode>();
}

void SourceCodeNode::initialize(const InterfaceElement& element, GenContext& context)
{
    ShaderNodeImpl::initialize(element, context);

    if (!element.isA<Implementation>())
    {
        throw ExceptionShaderGenError("Element '" + element.getName() + "' is not an Implementation element");
    }

    const Implementation& impl = static_cast<const Implementation&>(element);

    // Get source code from either an attribute or a file.
    _functionSource = impl.getAttribute("sourcecode");
    if (_functionSource.empty())
    {
        FilePath localPath = FilePath(impl.getActiveSourceUri()).getParentPath();
        _sourceFilename = context.resolveSourceFile(impl.getAttribute("file"), localPath);
        _functionSource = readFile(_sourceFilename);
        if (_functionSource.empty())
        {
            throw ExceptionShaderGenError("Failed to get source code from file '" + _sourceFilename.asString() +
                                          "' used by implementation '" + impl.getName() + "'");
        }
    }

    // Find the function name to use
    // If no function is given the source will be inlined.
    _functionName = impl.getAttribute("function");
    _inlined = _functionName.empty();
    if (!_inlined)
    {
        // Make sure the function name is valid.
        string validFunctionName = _functionName;
        context.getShaderGenerator().getSyntax().makeValidName(validFunctionName);
        if (_functionName != validFunctionName)
        {
            throw ExceptionShaderGenError("Function name '" + _functionName +
                                          "' used by implementation '" + impl.getName() + "' is not a valid identifier.");
        }
    }
    else
    {
        _functionSource = replaceSubstrings(_functionSource, { { "\n", "" } });
    }

    // Set hash using the function name.
    // TODO: Could be improved to include the full function signature.
    _hash = std::hash<string>{}(_functionName);
}

void SourceCodeNode::emitFunctionDefinition(const ShaderNode&, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Emit function definition for non-inlined functions
        if (!_functionSource.empty())
        {
            if (!_sourceFilename.isEmpty())
            {
                stage.addSourceDependency(_sourceFilename);
            }
            if (!_inlined)
            {
                const ShaderGenerator& shadergen = context.getShaderGenerator();
                shadergen.emitBlock(_functionSource, _sourceFilename, context, stage);
                shadergen.emitLineBreak(stage);
            }
        }
    }
}

void SourceCodeNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        if (_inlined)
        {
            // An inline function call

            size_t pos = 0;
            size_t i = _functionSource.find(INLINE_VARIABLE_PREFIX);
            StringSet variableNames;
            StringVec code;
            while (i != string::npos)
            {
                code.push_back(_functionSource.substr(pos, i - pos));
                size_t j = _functionSource.find(INLINE_VARIABLE_SUFFIX, i + 2);
                if (j == string::npos)
                {
                    throw ExceptionShaderGenError("Malformed inline expression in implementation for node " + node.getName());
                }

                const string variable = _functionSource.substr(i + 2, j - i - 2);
                const ShaderInput* input = node.getInput(variable);
                if (!input)
                {
                    throw ExceptionShaderGenError("Could not find an input named '" + variable +
                                                  "' on node '" + node.getName() + "'");
                }

                if (input->getConnection())
                {
                    code.push_back(shadergen.getUpstreamResult(input, context));
                }
                else
                {
                    string variableName = node.getName() + "_" + input->getName() + "_tmp";
                    if (!variableNames.count(variableName))
                    {
                        ShaderPort v(nullptr, input->getType(), variableName, input->getValue());
                        shadergen.emitLineBegin(stage);
                        const Syntax& syntax = shadergen.getSyntax();
                        const string valueStr = (v.getValue() ? syntax.getValue(v.getType(), *v.getValue()) : syntax.getDefaultValue(v.getType()));
                        const string& qualifier = syntax.getConstantQualifier();
                        string str = qualifier.empty() ? EMPTY_STRING : qualifier + " ";
                        str += syntax.getTypeName(v.getType()) + " " + v.getVariable();
                        str += valueStr.empty() ? EMPTY_STRING : " = " + valueStr;
                        shadergen.emitString(str, stage);
                        shadergen.emitLineEnd(stage);
                        variableNames.insert(variableName);
                    }
                    code.push_back(variableName);
                }

                pos = j + 2;
                i = _functionSource.find(INLINE_VARIABLE_PREFIX, pos);
            }
            code.push_back(_functionSource.substr(pos));

            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(node.getOutput(), true, false, context, stage);
            shadergen.emitString(" = ", stage);
            for (const string& c : code)
            {
                shadergen.emitString(c, stage);
            }
            shadergen.emitLineEnd(stage);
        }
        else
        {
            // An ordinary source code function call

            // Declare the output variables.
            emitOutputVariables(node, context, stage);

            shadergen.emitLineBegin(stage);
            string delim = "";

            // Emit function name.
            shadergen.emitString(_functionName + "(", stage);

            // Emit all inputs on the node.
            for (ShaderInput* input : node.getInputs())
            {
                shadergen.emitString(delim, stage);
                shadergen.emitInput(input, context, stage);
                delim = ", ";
            }

            // Emit node outputs.
            for (size_t i = 0; i < node.numOutputs(); ++i)
            {
                shadergen.emitString(delim, stage);
                shadergen.emitOutput(node.getOutput(i), false, false, context, stage);
                delim = ", ";
            }

            // End function call
            shadergen.emitString(")", stage);
            shadergen.emitLineEnd(stage);
        }
    }
}

MATERIALX_NAMESPACE_END
