//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/Nodes/ClosureLayerNodeMdl.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/TypeDesc.h>

MATERIALX_NAMESPACE_BEGIN

const string StringConstantsMdl::TOP = "top";
const string StringConstantsMdl::BASE = "base";
const string StringConstantsMdl::FG = "fg";
const string StringConstantsMdl::BG = "bg";
const string StringConstantsMdl::IN1 = "in1";
const string StringConstantsMdl::IN2 = "in2";

const string StringConstantsMdl::THICKNESS = "thickness";
const string StringConstantsMdl::IOR = "ior";
const string StringConstantsMdl::THIN_FILM_THICKNESS = "thinfilm_thickness";
const string StringConstantsMdl::THIN_FILM_IOR = "thinfilm_ior";

const string StringConstantsMdl::EMPTY = "";

ShaderNodeImplPtr ClosureLayerNodeMdl::create()
{
    return std::make_shared<ClosureLayerNodeMdl>();
}

void ClosureLayerNodeMdl::emitFunctionCall(const ShaderNode& _node, GenContext& context, ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();

    ShaderNode& node = const_cast<ShaderNode&>(_node);

    ShaderInput* topInput = node.getInput(StringConstantsMdl::TOP);
    ShaderInput* baseInput = node.getInput(StringConstantsMdl::BASE);
    ShaderOutput* output = node.getOutput();

    //
    // 1. Handle the BSDF-over-VDF case
    //
    if (*baseInput->getType() == *Type::VDF)
    {
        // Make sure we have a top BSDF connected.
        if (!topInput->getConnection())
        {
            // No top BSDF so just emit an empty material.
            shadergen.emitLine("material " + output->getVariable() + " = material()", stage);
            return;
        }

        // Emit function call for top node if it's a sibling node
        // and not the graph interface.
        ShaderNode* top = topInput->getConnection()->getNode();
        if (top->getParent() == node.getParent())
        {
            shadergen.emitFunctionCall(*top, context, stage);
        }

        // Emit function call for base node if it's a sibling node
        // and not the graph interface.
        if (baseInput->getConnection())
        {
            ShaderNode* base = baseInput->getConnection()->getNode();
            if (base->getParent() == node.getParent())
            {
                shadergen.emitFunctionCall(*base, context, stage);
            }
        }

        const string t = shadergen.getUpstreamResult(topInput, context);
        const string b = shadergen.getUpstreamResult(baseInput, context);

        // Join the BSDF and VDF into a single material.
        shadergen.emitLine("material " + output->getVariable() +
                           " = material(surface: " + t + ".surface, backface: " + t +
                           ".backface, ior: " + t + ".ior, volume: " + b + ".volume)", stage);

        return;
    }

    //
    // 2. Handle the BSDF-over-BSDF case
    //

    // Make sure the layer is fully connected.
    if (!(topInput->getConnection() && baseInput->getConnection()))
    {
        // Just emit an empty material.
        shadergen.emitLine("material " + output->getVariable() + " = material()", stage);
        return;
    }

    ShaderNode* top = topInput->getConnection()->getNode();
    ShaderNode* base = baseInput->getConnection()->getNode();

    // Make sure top BSDF is a sibling node and not the graph interface.
    if (top->getParent() != node.getParent())
    {
        shadergen.emitComment("Warning: MDL has no support for layering BSDFs through a graph interface. Only the top BSDF will used.", stage);
        shadergen.emitLine("material " + output->getVariable() + " = " + shadergen.getUpstreamResult(topInput, context), stage);
        return;
    }

    // Special composition based on the layers.

    // Handle the layering of thin film onto another node separately.
    // This reads the parameters from the thin film bsdf and passes them to base.
    if (top->hasClassification(ShaderNode::Classification::THINFILM))
    {
        emitBsdfOverBsdfFunctionCalls_thinFilm(node, context, stage, shadergen, top, base, output);
        return;
    }

    // Otherwise, if the layer is carrying thin film parameters already,
    // they are pushed further down to the top and base node if they supported it.
    ShaderInput* layerNodeThicknessInput = node.getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* layerNodeIorInput = node.getInput(StringConstantsMdl::THIN_FILM_IOR);

    ShaderInput* topNodeThicknessInput = top->getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* topNodeIorInput = top->getInput(StringConstantsMdl::THIN_FILM_IOR);
    bool breakTopConnection = false;
    if (layerNodeThicknessInput && layerNodeIorInput && topNodeThicknessInput && topNodeIorInput)
    {
        topNodeThicknessInput->makeConnection(layerNodeThicknessInput->getConnection());
        topNodeIorInput->makeConnection(layerNodeIorInput->getConnection());
        breakTopConnection = true;
    }
    ShaderInput* baseNodeThicknessInput = base->getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* baseNodeIorInput = base->getInput(StringConstantsMdl::THIN_FILM_IOR);
    bool breakBaseConnection = false;
    if (layerNodeThicknessInput && layerNodeIorInput && baseNodeThicknessInput && baseNodeIorInput)
    {
        baseNodeThicknessInput->makeConnection(layerNodeThicknessInput->getConnection());
        baseNodeIorInput->makeConnection(layerNodeIorInput->getConnection());
        breakBaseConnection = true;
    }

    // note, this called for all layering operations independent of thin film
    emitBsdfOverBsdfFunctionCalls(node, context, stage, shadergen, top, base, output);

    if (breakTopConnection)
    {
        topNodeThicknessInput->breakConnection();
        topNodeIorInput->breakConnection();
    }
    if (breakBaseConnection)
    {
        baseNodeThicknessInput->breakConnection();
        baseNodeIorInput->breakConnection();
    }
}

void ClosureLayerNodeMdl::emitBsdfOverBsdfFunctionCalls(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage,
    const ShaderGenerator& shadergen,
    ShaderNode* top,
    ShaderNode* base,
    ShaderOutput* output) const
{
    // transport the base bsdf further than one layer
    ShaderNode* baseReceiverNode = top;
    while (true)
    {
        // if the top node is again a layer, we don't want to override the base
        // parameter but instead aim for the base parameter of layers base
        if (baseReceiverNode->hasClassification(ShaderNode::Classification::LAYER))
        {
            baseReceiverNode = top->getInput(StringConstantsMdl::BASE)->getConnection()->getNode();
        }
        else
        {
            // we stop at elemental bsdfs
            // TODO handle mix, add, and multiply
            break;
        }
    }

    // Only a subset of the MaterialX BSDF nodes can be layered vertically in MDL.
    // This is because MDL only supports layering through BSDF nesting with a base
    // input, and it's only possible to do this workaround on a subset of the BSDFs.
    // So if the top BSDF doesn't have a base input, we can only emit the top BSDF
    // without any base layering.
    ShaderInput* topNodeBaseInput = baseReceiverNode->getInput(StringConstantsMdl::BASE);
    if (!topNodeBaseInput)
    {
        shadergen.emitComment("Warning: MDL has no support for layering BSDF nodes without a base input. Only the top BSDF will used.", stage);

        // Change the state so we emit the top BSDF function
        // with output variable name from the layer node itself.
        ScopedSetVariableName setVariable(output->getVariable(), top->getOutput());

        // Make the call.
        if (top->getParent() == node.getParent())
        {
            top->getImplementation().emitFunctionCall(*top, context, stage);
        }
        return;
    }

    // Emit the base BSDF function call.
    // Make sure it's a sibling node and not the graph interface.
    if (base->getParent() == node.getParent())
    {
        shadergen.emitFunctionCall(*base, context, stage);
    }
    // Emit the layer operation with the top BSDF function call.
    // Change the state so we emit the top BSDF function with
    // base BSDF connection and output variable name from the
    // layer operator itself.
    topNodeBaseInput->makeConnection(base->getOutput());
    ScopedSetVariableName setVariable(output->getVariable(), top->getOutput());

    // Make the call.
    if (top->getParent() == node.getParent())
    {
        top->getImplementation().emitFunctionCall(*top, context, stage);
    }

    // Restore state.
    topNodeBaseInput->breakConnection();
}

void ClosureLayerNodeMdl::emitBsdfOverBsdfFunctionCalls_thinFilm(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage,
    const ShaderGenerator& shadergen,
    ShaderNode* top,
    ShaderNode* base,
    ShaderOutput* output) const
{
    ShaderInput* thinFilmThicknessInput = top->getInput(StringConstantsMdl::THICKNESS);
    ShaderInput* thinFilmIorInput = top->getInput(StringConstantsMdl::IOR);

    ShaderInput* baseNodeThicknessInput = base->getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* baseNodeIorInput = base->getInput(StringConstantsMdl::THIN_FILM_IOR);

    // Make sure the base node has thickness and IOR inputs for thin film.
    if (!baseNodeThicknessInput || !baseNodeIorInput)
    {
        shadergen.emitComment("Warning: The base node does not have parameters to transport thin-film thickness and IOR.", stage);

        // Change the state so we emit the base BSDF function without thin film
        // with output variable name from the layer node itself.
        ScopedSetVariableName setVariable(output->getVariable(), base->getOutput());

        // Make the call.
        if (base->getParent() == node.getParent())
        {
            base->getImplementation().emitFunctionCall(*base, context, stage);
        }
        return;
    }

    // Emit the base operation with the thin-film parameters of the top node
    // pushed down to the base node.
    baseNodeThicknessInput->makeConnection(thinFilmThicknessInput->getConnection());
    baseNodeIorInput->makeConnection(thinFilmIorInput->getConnection());

    // Change the output of this node to the output of base.
    // This basically removes the top as it is not needed anymore.
    ScopedSetVariableName setVariable(output->getVariable(), base->getOutput());

    // Make the call.
    if (base->getParent() == node.getParent())
    {
        base->getImplementation().emitFunctionCall(*base, context, stage);
    }

    // Restore state.
    baseNodeThicknessInput->breakConnection();
    baseNodeIorInput->breakConnection();
}

ShaderNodeImplPtr LayerableNodeMdl::create()
{
    return std::make_shared<LayerableNodeMdl>();
}

void LayerableNodeMdl::addInputs(ShaderNode& node, GenContext& /*context*/) const
{
    // Add the input to hold base layer BSDF.
    node.addInput(StringConstantsMdl::BASE, Type::BSDF);
}

ShaderNodeImplPtr ThinFilmReceiverNodeMdl::create()
{
    return std::make_shared<ThinFilmReceiverNodeMdl>();
}

void ThinFilmCombineNodeMdl::emitFunctionCall(const ShaderNode& _node, GenContext& context, ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();
    ShaderNode& node = const_cast<ShaderNode&>(_node);

    ShaderInput* topInput = node.getInput(getOperatorName(0));
    ShaderInput* baseInput = node.getInput(getOperatorName(1));

    ShaderNode* top = topInput->getConnection()->getNode();
    ShaderNode* base = baseInput->getConnection()->getNode();

    // Otherwise, if the combine node is carrying thin film parameters already,
    // they are pushed further down to the top and base node if they supported it.
    ShaderInput* layerNodeThicknessInput = node.getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* layerNodeIorInput = node.getInput(StringConstantsMdl::THIN_FILM_IOR);

    ShaderInput* topNodeThicknessInput = top->getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* topNodeIorInput = top->getInput(StringConstantsMdl::THIN_FILM_IOR);
    bool breakTopConnection = false;
    if (layerNodeThicknessInput && layerNodeIorInput && topNodeThicknessInput && topNodeIorInput)
    {
        topNodeThicknessInput->makeConnection(layerNodeThicknessInput->getConnection());
        topNodeIorInput->makeConnection(layerNodeIorInput->getConnection());
        breakTopConnection = true;
    }
    ShaderInput* baseNodeThicknessInput = base->getInput(StringConstantsMdl::THIN_FILM_THICKNESS);
    ShaderInput* baseNodeIorInput = base->getInput(StringConstantsMdl::THIN_FILM_IOR);
    bool breakBaseConnection = false;
    if (layerNodeThicknessInput && layerNodeIorInput && baseNodeThicknessInput && baseNodeIorInput)
    {
        baseNodeThicknessInput->makeConnection(layerNodeThicknessInput->getConnection());
        baseNodeIorInput->makeConnection(layerNodeIorInput->getConnection());
        breakBaseConnection = true;
    }

    // Emit the fore and background calls.
    // Make sure it's a sibling node and not the graph interface.
    if (top->getParent() == node.getParent())
    {
        shadergen.emitFunctionCall(*top, context, stage);
    }
    if (base->getParent() == node.getParent())
    {
        shadergen.emitFunctionCall(*base, context, stage);
    }

    // Note, this is called for all combine operations independent of thin film.
    Base::emitFunctionCall(_node, context, stage);

    if (breakTopConnection)
    {
        topNodeThicknessInput->breakConnection();
        topNodeIorInput->breakConnection();
    }
    if (breakBaseConnection)
    {
        baseNodeThicknessInput->breakConnection();
        baseNodeIorInput->breakConnection();
    }
}

ShaderNodeImplPtr MixBsdfNodeMdl::create()
{
    return std::make_shared<MixBsdfNodeMdl>();
}

const string& MixBsdfNodeMdl::getOperatorName(size_t index) const
{
    switch (index)
    {
        case 0:
            return StringConstantsMdl::FG;
        case 1:
            return StringConstantsMdl::BG;
        default:
            return StringConstantsMdl::EMPTY;
    }
}

ShaderNodeImplPtr AddOrMultiplyBsdfNodeMdl::create()
{
    return std::make_shared<AddOrMultiplyBsdfNodeMdl>();
}

const string& AddOrMultiplyBsdfNodeMdl::getOperatorName(size_t index) const
{
    switch (index)
    {
        case 0:
            return StringConstantsMdl::IN1;
        case 1:
            return StringConstantsMdl::IN2;
        default:
            return StringConstantsMdl::EMPTY;
    }
}

MATERIALX_NAMESPACE_END
