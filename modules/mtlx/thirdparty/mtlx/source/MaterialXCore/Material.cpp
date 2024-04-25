//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Material.h>

MATERIALX_NAMESPACE_BEGIN

vector<NodePtr> getShaderNodes(NodePtr materialNode, const string& nodeType, const string& target)
{
    vector<NodePtr> shaderNodeVec;
    std::set<NodePtr> shaderNodeSet;

    vector<InputPtr> inputs = materialNode->getActiveInputs();
    for (InputPtr input : inputs)
    {
        // Scan for a node directly connected to the input.
        // Note that this will handle traversing through interfacename associations.
        NodePtr shaderNode = input->getConnectedNode();
        if (shaderNode && !shaderNodeSet.count(shaderNode))
        {
            if (!nodeType.empty() && shaderNode->getType() != nodeType)
            {
                continue;
            }

            if (!target.empty())
            {
                NodeDefPtr nodeDef = shaderNode->getNodeDef(target);
                if (!nodeDef)
                {
                    continue;
                }
            }

            shaderNodeVec.push_back(shaderNode);
            shaderNodeSet.insert(shaderNode);
        }
        else if (input->hasNodeGraphString())
        {
            // Check upstream nodegraph connected to the input.
            // If no explicit output name given then scan all outputs on the nodegraph.
            ElementPtr parent = materialNode->getParent();
            NodeGraphPtr nodeGraph = parent->getChildOfType<NodeGraph>(input->getNodeGraphString());
            if (!nodeGraph)
            {
                continue;
            }
            vector<OutputPtr> outputs;
            if (input->hasOutputString())
            {
                outputs.push_back(nodeGraph->getOutput(input->getOutputString()));
            }
            else
            {
                outputs = nodeGraph->getOutputs();
            }
            for (OutputPtr output : outputs)
            {
                NodePtr upstreamNode = output->getConnectedNode();
                if (upstreamNode && !shaderNodeSet.count(upstreamNode))
                {
                    if (!target.empty() && !upstreamNode->getNodeDef(target))
                    {
                        continue;
                    }
                    shaderNodeVec.push_back(upstreamNode);
                    shaderNodeSet.insert(upstreamNode);
                }
            }
        }
    }

    if (inputs.empty())
    {
        // Try to find material nodes in the implementation graph if any.
        // If a target is specified the nodedef for the given target is searched for.
        NodeDefPtr materialNodeDef = materialNode->getNodeDef(target);
        if (materialNodeDef)
        {
            InterfaceElementPtr impl = materialNodeDef->getImplementation();
            if (impl && impl->isA<NodeGraph>())
            {
                NodeGraphPtr implGraph = impl->asA<NodeGraph>();
                for (OutputPtr defOutput : materialNodeDef->getOutputs())
                {
                    if (defOutput->getType() == MATERIAL_TYPE_STRING)
                    {
                        OutputPtr implGraphOutput = implGraph->getOutput(defOutput->getName());
                        for (GraphIterator it = implGraphOutput->traverseGraph().begin(); it != GraphIterator::end(); ++it)
                        {
                            ElementPtr upstreamElem = it.getUpstreamElement();
                            if (!upstreamElem)
                            {
                                it.setPruneSubgraph(true);
                                continue;
                            }
                            NodePtr upstreamNode = upstreamElem->asA<Node>();
                            if (upstreamNode && upstreamNode->getType() == MATERIAL_TYPE_STRING)
                            {
                                for (NodePtr shaderNode : getShaderNodes(upstreamNode, nodeType, target))
                                {
                                    if (!shaderNodeSet.count(shaderNode))
                                    {
                                        shaderNodeVec.push_back(shaderNode);
                                        shaderNodeSet.insert(shaderNode);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return shaderNodeVec;
}

vector<OutputPtr> getConnectedOutputs(NodePtr node)
{
    vector<OutputPtr> outputVec;
    std::set<OutputPtr> outputSet;
    for (InputPtr input : node->getInputs())
    {
        OutputPtr output = input->getConnectedOutput();
        if (output && !outputSet.count(output))
        {
            outputVec.push_back(output);
            outputSet.insert(output);
        }
    }
    return outputVec;
}

MATERIALX_NAMESPACE_END
