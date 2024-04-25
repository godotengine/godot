//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/CombineNodeMdl.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

ShaderNodeImplPtr CombineNodeMdl::create()
{
    return std::make_shared<CombineNodeMdl>();
}

void CombineNodeMdl::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Custom handling for color3 type input, all other types
        // can are handled by our parent class below.
        // Custom handling is needed since in MDL color must be converted
        // to float3 before accessing its sub-component.

        const ShaderInput* in1 = node.getInput(0);
        const ShaderOutput* out = node.getOutput();
        if (!in1 || !out)
        {
            throw ExceptionShaderGenError("Node '" + node.getName() + "' is not a valid convert node");
        }

        if (*in1->getType() == *Type::COLOR3)
        {
            const ShaderInput* in2 = node.getInput(1);
            if (!in2 || *in2->getType() != *Type::FLOAT)
            {
                throw ExceptionShaderGenError("Node '" + node.getName() + "' is not a valid convert node");
            }

            const ShaderGenerator& shadergen = context.getShaderGenerator();

            // If in1 is unconnected we must declare a local variable
            // for it first, in order to access components from it below.
            string in1Variable = in1->getConnection() ? in1->getConnection()->getVariable() : in1->getVariable();
            if (!in1->getConnection())
            {
                string variableValue = in1->getValue() ? shadergen.getSyntax().getValue(in1->getType(), *in1->getValue()) : shadergen.getSyntax().getDefaultValue(in1->getType());
                shadergen.emitLine(shadergen.getSyntax().getTypeName(in1->getType()) + " " + in1Variable + " = " + variableValue, stage);
            }

            // Get the value components to use for constructing the new value.
            StringVec valueComponents;

            // Get components from in1
            const StringVec& in1Members = shadergen.getSyntax().getTypeSyntax(in1->getType()).getMembers();
            size_t memberSize = in1Members.size();
            if (memberSize)
            {
                valueComponents.resize(memberSize + 1);
                for (size_t i = 0; i < memberSize; i++)
                {
                    valueComponents[i] = "float3(" + in1Variable + ")" + in1Members[i];
                }
            }
            else
            {
                memberSize = 1;
                valueComponents.resize(2);
                valueComponents[0] = in1Variable;
            }
            // Get component from in2
            valueComponents[memberSize] = shadergen.getUpstreamResult(in2, context);

            // Let the TypeSyntax construct the value from the components.
            const TypeSyntax& outTypeSyntax = shadergen.getSyntax().getTypeSyntax(out->getType());
            const string result = outTypeSyntax.getValue(valueComponents, false);

            shadergen.emitLineBegin(stage);
            shadergen.emitOutput(node.getOutput(), true, false, context, stage);
            shadergen.emitString(" = " + result, stage);
            shadergen.emitLineEnd(stage);
        }
        else
        {
            CombineNode::emitFunctionCall(node, context, stage);
        }
    }
}

MATERIALX_NAMESPACE_END
