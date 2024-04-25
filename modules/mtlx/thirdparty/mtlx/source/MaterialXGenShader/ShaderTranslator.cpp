//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/ShaderTranslator.h>

#include <MaterialXCore/Material.h>

MATERIALX_NAMESPACE_BEGIN

//
// ShaderTranslator methods
//

void ShaderTranslator::connectTranslationInputs(NodePtr shader, NodeDefPtr translationNodeDef)
{
    vector<InputPtr> origInputs = shader->getInputs();
    std::set<OutputPtr> origOutputs;
    for (InputPtr shaderInput : origInputs)
    {
        if (translationNodeDef->getInput(shaderInput->getName()))
        {
            InputPtr input = _translationNode->addInput(shaderInput->getName(), shaderInput->getType());

            OutputPtr connectedOutput = shaderInput->getConnectedOutput();
            if (connectedOutput)
            {
                NodePtr connectedNode = connectedOutput->getConnectedNode();

                // Nodes with world-space outputs are skipped, with translation being applied to
                // the node directly upstream.
                NodePtr worldSpaceNode = connectsToWorldSpaceNode(connectedOutput);
                if (worldSpaceNode)
                {
                    NodePtr upstreamNode = worldSpaceNode->getConnectedNode("in");
                    if (upstreamNode)
                    {
                        connectedNode = upstreamNode;
                    }
                }

                input->setConnectedNode(connectedNode);
                origOutputs.insert(connectedOutput);
            }
            else if (shaderInput->hasValueString())
            {
                input->setValueString(shaderInput->getValueString());
            }
            else
            {
                throw Exception("Shader input has no associated output or value " + shaderInput->getName());
            }

            string colorSpace = shaderInput->getActiveColorSpace();
            if (!colorSpace.empty())
            {
                input->setColorSpace(colorSpace);
            }
            if (shaderInput->hasUnit())
            {
                input->setUnit(shaderInput->getUnit());
                input->setUnitType(shaderInput->getUnitType());
            }
        }
    }

    for (InputPtr input : origInputs)
    {
        shader->removeInput(input->getName());
    }
    for (OutputPtr output : origOutputs)
    {
        _graph->removeOutput(output->getName());
    }
}

void ShaderTranslator::connectTranslationOutputs(NodePtr shader)
{
    DocumentPtr doc = shader->getDocument();
    InterfaceElementPtr implement = _translationNode->getImplementation();
    NodeGraphPtr translationGraph = implement ? implement->asA<NodeGraph>() : nullptr;
    if (!translationGraph)
    {
        throw Exception("No graph implementation for " + _translationNode->getCategory() + " was found");
    }

    // Iterate through outputs of the translation graph.
    for (OutputPtr translationGraphOutput : translationGraph->getOutputs())
    {
        // Convert output name to input name, using a hardcoded naming convention for now.
        string outputName = translationGraphOutput->getName();
        size_t pos = outputName.find("_out");
        if (pos == string::npos)
        {
            throw Exception("Translation graph output " + outputName + " does not end with '_out'");
        }
        string inputName = outputName.substr(0, pos);

        // Determine the node and output representing this translated stream.
        NodePtr translatedStreamNode = _translationNode;
        string translatedStreamOutput = outputName;

        // Nodes with world-space outputs are moved outside of their containing graph,
        // providing greater flexibility in texture baking.
        NodePtr worldSpaceNode = connectsToWorldSpaceNode(translationGraphOutput);
        if (worldSpaceNode)
        {
            InputPtr nodeInput = worldSpaceNode->getInput("in");
            if (nodeInput && nodeInput->hasInterfaceName())
            {
                InputPtr interfaceInput = _translationNode->getInput(nodeInput->getInterfaceName());
                NodePtr sourceNode = interfaceInput ? interfaceInput->getConnectedNode() : nullptr;
                if (!sourceNode)
                {
                    continue;
                }
                translatedStreamNode = _graph->addNode(worldSpaceNode->getCategory(), worldSpaceNode->getName(), worldSpaceNode->getType());
                translatedStreamNode->setConnectedNode("in", sourceNode);
                translatedStreamOutput = EMPTY_STRING;
            }
        }

        // Create translated output.
        OutputPtr translatedOutput = _graph->getOutput(outputName);
        if (!translatedOutput)
        {
            translatedOutput = _graph->addOutput(outputName, translationGraphOutput->getType());
        }
        translatedOutput->setConnectedNode(translatedStreamNode);
        if (!translatedStreamOutput.empty())
        {
            translatedOutput->setOutputString(translatedStreamOutput);
        }

        // Add translated shader input.
        InputPtr translatedShaderInput = shader->getInput(inputName);
        if (!translatedShaderInput)
        {
            translatedShaderInput = shader->addInput(inputName, translationGraphOutput->getType());
        }
        translatedShaderInput->setConnectedOutput(translatedOutput);
    }
}

void ShaderTranslator::translateShader(NodePtr shader, const string& destCategory)
{
    _graph = nullptr;
    _translationNode = nullptr;

    if (!shader)
    {
        return;
    }

    const string& sourceCategory = shader->getCategory();
    if (sourceCategory == destCategory)
    {
        throw Exception("The source shader \"" + shader->getNamePath() + "\" category is already \"" + destCategory + "\"");
    }

    DocumentPtr doc = shader->getDocument();
    vector<OutputPtr> referencedOutputs = getConnectedOutputs(shader);
    ElementPtr referencedParent = !referencedOutputs.empty() ? referencedOutputs[0]->getParent() : nullptr;
    NodeGraphPtr referencedGraph = referencedParent ? referencedParent->asA<NodeGraph>() : nullptr;
    _graph = referencedGraph ? referencedGraph : doc->addNodeGraph();

    string translateNodeString = sourceCategory + "_to_" + destCategory;
    vector<NodeDefPtr> matchingNodeDefs = doc->getMatchingNodeDefs(translateNodeString);
    if (matchingNodeDefs.empty())
    {
        throw Exception("Shader translation requires a translation nodedef named " + translateNodeString);
    }
    NodeDefPtr translationNodeDef = matchingNodeDefs[0];
    _translationNode = _graph->addNodeInstance(translationNodeDef);

    connectTranslationInputs(shader, translationNodeDef);
    shader->setCategory(destCategory);
    shader->removeAttribute(InterfaceElement::NODE_DEF_ATTRIBUTE);
    connectTranslationOutputs(shader);
}

void ShaderTranslator::translateAllMaterials(DocumentPtr doc, const string& destCategory)
{
    vector<TypedElementPtr> materialNodes = findRenderableMaterialNodes(doc);
    for (auto elem : materialNodes)
    {
        NodePtr materialNode = elem->asA<Node>();
        if (!materialNode)
        {
            continue;
        }
        for (NodePtr shaderNode : getShaderNodes(materialNode))
        {
            translateShader(shaderNode, destCategory);
        }
    }
}

MATERIALX_NAMESPACE_END
