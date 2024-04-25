//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Nodes/ConvertNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/TypeDesc.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr ConvertNode::create()
{
    return std::make_shared<ConvertNode>();
}

void ConvertNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    using ConvertTable = std::unordered_map<const TypeDesc*, std::unordered_map<const TypeDesc*, string>>;

    static const ConvertTable CONVERT_TABLE({ { Type::COLOR3,
                                                { { Type::VECTOR3, string("rgb") },
                                                  { Type::COLOR4, string("rgb1") } } },
                                              { Type::COLOR4,
                                                { { Type::VECTOR4, string("rgba") },
                                                  { Type::COLOR3, string("rgb") } } },
                                              { Type::VECTOR2,
                                                { { Type::VECTOR3, string("xy0") } } },
                                              { Type::VECTOR3,
                                                { { Type::COLOR3, string("xyz") },
                                                  { Type::VECTOR4, string("xyz1") },
                                                  { Type::VECTOR2, string("xy") } } },
                                              { Type::VECTOR4,
                                                { { Type::COLOR4, string("xyzw") },
                                                  { Type::VECTOR3, string("xyz") } } },
                                              { Type::FLOAT,
                                                { { Type::COLOR3, string("rrr") },
                                                  { Type::COLOR4, string("rrrr") },
                                                  { Type::VECTOR2, string("rr") },
                                                  { Type::VECTOR3, string("rrr") },
                                                  { Type::VECTOR4, string("rrrr") } } } });

    static const string IN_STRING("in");

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        const ShaderInput* in = node.getInput(IN_STRING);
        const ShaderOutput* out = node.getOutput();
        if (!in || !out)
        {
            throw ExceptionShaderGenError("Node '" + node.getName() + "' is not a valid convert node");
        }
        if (!in->getConnection() && !in->getValue())
        {
            throw ExceptionShaderGenError("No connection or value found to convert on node '" + node.getName() + "'");
        }

        string result;

        // Handle supported scalar type conversions.
        if (in->getType()->isScalar() && out->getType()->isScalar())
        {
            result = shadergen.getUpstreamResult(in, context);
            result = shadergen.getSyntax().getTypeName(out->getType()) + "(" + result + ")";
        }
        // Handle supported vector type conversions.
        else
        {
            // Search the conversion table for a swizzle pattern to use.
            const string* swizzle = nullptr;
            auto i = CONVERT_TABLE.find(in->getType());
            if (i != CONVERT_TABLE.end())
            {
                auto j = i->second.find(out->getType());
                if (j != i->second.end())
                {
                    swizzle = &j->second;
                }
            }
            if (!swizzle || swizzle->empty())
            {
                throw ExceptionShaderGenError("Conversion from '" + in->getType()->getName() + "' to '" + out->getType()->getName() + "' is not supported by convert node");
            }

            // If the input is unconnected we must declare a local variable
            // for it first, in order to swizzle it below.
            string variableName;
            if (!in->getConnection())
            {
                variableName = in->getVariable();
                string variableValue = in->getValue() ? shadergen.getSyntax().getValue(in->getType(), *in->getValue()) : shadergen.getSyntax().getDefaultValue(in->getType());
                shadergen.emitLine(shadergen.getSyntax().getTypeName(in->getType()) + " " + variableName + " = " + variableValue, stage);
            }
            else
            {
                variableName = shadergen.getUpstreamResult(in, context);
            }

            const TypeDesc* type = in->getConnection() ? in->getConnection()->getType() : in->getType();
            result = shadergen.getSyntax().getSwizzledVariable(variableName, type, *swizzle, node.getOutput()->getType());
        }

        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = " + result, stage);
        shadergen.emitLineEnd(stage);
    }
}

MATERIALX_NAMESPACE_END
