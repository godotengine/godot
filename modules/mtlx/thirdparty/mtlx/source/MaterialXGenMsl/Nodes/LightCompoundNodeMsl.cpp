//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMsl/Nodes/LightCompoundNodeMsl.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>

#include <MaterialXGenShader/Util.h>

MATERIALX_NAMESPACE_BEGIN

LightCompoundNodeMsl::LightCompoundNodeMsl() :
    _lightUniforms(HW::LIGHT_DATA, EMPTY_STRING)
{
}

ShaderNodeImplPtr LightCompoundNodeMsl::create()
{
    return std::make_shared<LightCompoundNodeMsl>();
}

const string& LightCompoundNodeMsl::getTarget() const
{
    return MslShaderGenerator::TARGET;
}

void LightCompoundNodeMsl::initialize(const InterfaceElement& element, GenContext& context)
{
    CompoundNode::initialize(element, context);

    // Store light uniforms for all inputs on the interface
    const NodeGraph& graph = static_cast<const NodeGraph&>(element);
    NodeDefPtr nodeDef = graph.getNodeDef();
    for (InputPtr input : nodeDef->getActiveInputs())
    {
        _lightUniforms.add(TypeDesc::get(input->getType()), input->getName());
    }
}

void LightCompoundNodeMsl::createVariables(const ShaderNode&, GenContext& context, Shader& shader) const
{
    // Create variables for all child nodes
    for (ShaderNode* childNode : _rootGraph->getNodes())
    {
        childNode->getImplementation().createVariables(*childNode, context, shader);
    }

    ShaderStage& ps = shader.getStage(Stage::PIXEL);
    VariableBlock& lightData = ps.getUniformBlock(HW::LIGHT_DATA);

    // Create all light uniforms
    for (size_t i = 0; i < _lightUniforms.size(); ++i)
    {
        ShaderPort* u = const_cast<ShaderPort*>(_lightUniforms[i]);
        lightData.add(u->getSelf());
    }

    const MslShaderGenerator& shadergen = static_cast<const MslShaderGenerator&>(context.getShaderGenerator());
    shadergen.addStageLightingUniforms(context, ps);
}

void LightCompoundNodeMsl::emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const MslShaderGenerator& shadergen = static_cast<const MslShaderGenerator&>(context.getShaderGenerator());

        // Emit functions for all child nodes
        shadergen.emitFunctionDefinitions(*_rootGraph, context, stage);

        // Find any closure contexts used by this node
        // and emit the function for each context.
        vector<ClosureContext*> ccts;
        shadergen.getClosureContexts(node, ccts);
        if (ccts.empty())
        {
            emitFunctionDefinition(nullptr, context, stage);
        }
        else
        {
            for (ClosureContext* cct : ccts)
            {
                emitFunctionDefinition(cct, context, stage);
            }
        }
    }
}

void LightCompoundNodeMsl::emitFunctionDefinition(ClosureContext* cct, GenContext& context, ShaderStage& stage) const
{
    const MslShaderGenerator& shadergen = static_cast<const MslShaderGenerator&>(context.getShaderGenerator());

    // Emit function signature
    if (cct)
    {
        // Use the first output for classifying node type for the closure context.
        // This is only relevent for closures, and they only have a single output.
        const TypeDesc* nodeType = _rootGraph->getOutputSocket()->getType();
        shadergen.emitLine("void " + _functionName + cct->getSuffix(nodeType) + "(LightData light, float3 position, out lightshader result)", stage, false);
    }
    else
    {
        shadergen.emitLine("void " + _functionName + "(LightData light, float3 position, out lightshader result)", stage, false);
    }

    shadergen.emitFunctionBodyBegin(*_rootGraph, context, stage);

    // Emit all texturing nodes. These are inputs to any
    // closure/shader nodes and need to be emitted first.
    shadergen.emitFunctionCalls(*_rootGraph, context, stage, ShaderNode::Classification::TEXTURE);

    // Emit function calls for all light shader nodes.
    // These will internally emit their closure function calls.
    if (cct)
    {
        context.pushClosureContext(cct);
        shadergen.emitFunctionCalls(*_rootGraph, context, stage, ShaderNode::Classification::SHADER | ShaderNode::Classification::LIGHT);
        context.popClosureContext();
    }
    else
    {
        shadergen.emitFunctionCalls(*_rootGraph, context, stage, ShaderNode::Classification::SHADER | ShaderNode::Classification::LIGHT);
    }

    shadergen.emitFunctionBodyEnd(*_rootGraph, context, stage);
}

void LightCompoundNodeMsl::emitFunctionCall(const ShaderNode&, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLine(_functionName + "(light, position, result)", stage);
    }
}

MATERIALX_NAMESPACE_END
