//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Node.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Material.h>

#include <deque>

MATERIALX_NAMESPACE_BEGIN

const string Backdrop::CONTAINS_ATTRIBUTE = "contains";
const string Backdrop::WIDTH_ATTRIBUTE = "width";
const string Backdrop::HEIGHT_ATTRIBUTE = "height";

//
// Node methods
//

void Node::setConnectedNode(const string& inputName, ConstNodePtr node)
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        input = addInput(inputName);
    }
    if (node)
    {
        const string& type = node->getType();
        if (type != MULTI_OUTPUT_TYPE_STRING)
        {
            input->setType(type);
        }
    }
    input->setConnectedNode(node);
}

NodePtr Node::getConnectedNode(const string& inputName) const
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        return NodePtr();
    }
    return input->getConnectedNode();
}

void Node::setConnectedNodeName(const string& inputName, const string& nodeName)
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        input = addInput(inputName);
    }
    input->setNodeName(nodeName);
}

string Node::getConnectedNodeName(const string& inputName) const
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        return EMPTY_STRING;
    }
    return input->getNodeName();
}

void Node::setConnectedOutput(const string& inputName, OutputPtr output)
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        input = addInput(inputName, DEFAULT_TYPE_STRING);
    }
    if (output)
    {
        input->setType(output->getType());
    }
    input->setConnectedOutput(output);
}

OutputPtr Node::getConnectedOutput(const string& inputName) const
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        return OutputPtr();
    }
    return input->getConnectedOutput();
}

NodeDefPtr Node::getNodeDef(const string& target, bool allowRoughMatch) const
{
    if (hasNodeDefString())
    {
        return resolveNameReference<NodeDef>(getNodeDefString());
    }
    vector<NodeDefPtr> nodeDefs = getDocument()->getMatchingNodeDefs(getQualifiedName(getCategory()));
    vector<NodeDefPtr> secondary = getDocument()->getMatchingNodeDefs(getCategory());
    vector<NodeDefPtr> roughMatches;
    nodeDefs.insert(nodeDefs.end(), secondary.begin(), secondary.end());
    for (NodeDefPtr nodeDef : nodeDefs)
    {
        if (!targetStringsMatch(nodeDef->getTarget(), target) ||
            !nodeDef->isVersionCompatible(getVersionString()) ||
            nodeDef->getType() != getType())
        {
            continue;
        }
        if (!hasExactInputMatch(nodeDef))
        {
            if (allowRoughMatch)
            {
                roughMatches.push_back(nodeDef);
            }
            continue;
        }
        return nodeDef;
    }
    if (!roughMatches.empty())
    {
        return roughMatches[0];
    }
    return NodeDefPtr();
}

Edge Node::getUpstreamEdge(size_t index) const
{
    if (index < getUpstreamEdgeCount())
    {
        InputPtr input = getInputs()[index];
        ElementPtr upstreamNode = input->getConnectedNode();
        if (upstreamNode)
        {
            return Edge(getSelfNonConst(), input, upstreamNode);
        }
    }

    return NULL_EDGE;
}

OutputPtr Node::getNodeDefOutput(ElementPtr connectingElement)
{
    string outputName;
    const PortElementPtr port = connectingElement->asA<PortElement>();
    if (port)
    {
        // The connecting element is an input/output port,
        // so get the name of the output connected to this
        // port. If no explicit output is specified this will
        // return an empty string which is handled below.
        outputName = port->getOutputString();

        // Handle case where it's an input to a top level output
        InputPtr connectedInput = connectingElement->asA<Input>();
        OutputPtr output;
        if (connectedInput)
        {
            InputPtr interfaceInput = connectedInput->getInterfaceInput();
            if (interfaceInput)
            {
                output = interfaceInput->getConnectedOutput();
                outputName = interfaceInput->getOutputString();
            }
            else
            {
                output = connectedInput->getConnectedOutput();
            }
        }
        if (output)
        {
            if (connectedInput ||
                output->getParent() == output->getDocument())
            {
                if (!output->getOutputString().empty())
                {
                    outputName = output->getOutputString();
                }
            }
        }
    }
    if (!outputName.empty())
    {
        // Find this output on our nodedef.
        NodeDefPtr nodeDef = getNodeDef();
        if (nodeDef)
        {
            return nodeDef->getActiveOutput(outputName);
        }
    }
    return OutputPtr();
}

vector<PortElementPtr> Node::getDownstreamPorts() const
{
    vector<PortElementPtr> downstreamPorts;
    for (PortElementPtr port : getDocument()->getMatchingPorts(getQualifiedName(getName())))
    {
        if (port->getConnectedNode() == getSelf())
        {
            downstreamPorts.push_back(port);
        }
    }
    std::sort(downstreamPorts.begin(), downstreamPorts.end(), [](const ConstElementPtr& a, const ConstElementPtr& b)
    {
        return a->getName() > b->getName();
    });
    return downstreamPorts;
}

bool Node::validate(string* message) const
{
    bool res = true;
    validateRequire(!getCategory().empty(), res, message, "Node element is missing a category");
    validateRequire(hasType(), res, message, "Node element is missing a type");

    NodeDefPtr nodeDef = getNodeDef(EMPTY_STRING, true);
    if (nodeDef)
    {
        string matchMessage;
        bool exactMatch = hasExactInputMatch(nodeDef, &matchMessage);
        validateRequire(exactMatch, res, message, "Node interface error: " + matchMessage);
    }
    else
    {
        bool categoryDeclared = !getDocument()->getMatchingNodeDefs(getCategory()).empty();
        validateRequire(!categoryDeclared, res, message, "Node interface doesn't support this output type");
    }

    return InterfaceElement::validate(message) && res;
}

//
// GraphElement methods
//

NodePtr GraphElement::addMaterialNode(const string& name, ConstNodePtr shaderNode)
{
    bool isVolumeShader = shaderNode && shaderNode->getType() == VOLUME_SHADER_TYPE_STRING;
    string category = isVolumeShader ? VOLUME_MATERIAL_NODE_STRING : SURFACE_MATERIAL_NODE_STRING;
    NodePtr materialNode = addNode(category, name, MATERIAL_TYPE_STRING);
    if (shaderNode)
    {
        InputPtr input = materialNode->addInput(shaderNode->getType(), shaderNode->getType());
        input->setConnectedNode(shaderNode);
    }
    return materialNode;
}

void GraphElement::flattenSubgraphs(const string& target, NodePredicate filter)
{
    vector<NodePtr> nodeQueue = getNodes();
    while (!nodeQueue.empty())
    {
        // Determine which nodes require processing, and precompute declarations
        // and graph implementations for these nodes.
        using PortElementVec = vector<PortElementPtr>;
        std::vector<NodePtr> processNodeVec;
        std::unordered_map<NodePtr, NodeGraphPtr> graphImplMap;
        std::unordered_map<NodePtr, ConstInterfaceElementPtr> declarationMap;
        std::unordered_map<NodePtr, PortElementVec> downstreamPortMap;
        for (NodePtr node : nodeQueue)
        {
            if (filter && !filter(node))
            {
                continue;
            }

            InterfaceElementPtr implement = node->getImplementation(target);
            if (implement && implement->isA<NodeGraph>())
            {
                processNodeVec.push_back(node);
                graphImplMap[node] = implement->asA<NodeGraph>();
                declarationMap[node] = node->getDeclaration(target);
                downstreamPortMap[node] = node->getDownstreamPorts();
                for (NodePtr sourceSubNode : implement->asA<NodeGraph>()->getNodes())
                {
                    downstreamPortMap[sourceSubNode] = sourceSubNode->getDownstreamPorts();
                }
            }
        }
        nodeQueue.clear();

        // Iterate through nodes with graph implementations.
        for (NodePtr processNode : processNodeVec)
        {
            NodeGraphPtr sourceSubGraph = graphImplMap[processNode];
            std::unordered_map<NodePtr, NodePtr> subNodeMap;

            // Create a new instance of each original subnode.
            for (NodePtr sourceSubNode : sourceSubGraph->getNodes())
            {
                string origName = sourceSubNode->getName();
                string destName = createValidChildName(origName);
                NodePtr destSubNode = addNode(sourceSubNode->getCategory(), destName);

                destSubNode->copyContentFrom(sourceSubNode);
                setChildIndex(destSubNode->getName(), getChildIndex(processNode->getName()));

                // Store the mapping between subgraphs.
                subNodeMap[sourceSubNode] = destSubNode;

                // Add the subnode to the queue, allowing processing of nested subgraphs.
                nodeQueue.push_back(destSubNode);
            }

            // Update properties of generated subnodes.
            for (const auto& subNodePair : subNodeMap)
            {
                NodePtr sourceSubNode = subNodePair.first;
                NodePtr destSubNode = subNodePair.second;

                // Update node connections.
                for (PortElementPtr sourcePort : downstreamPortMap[sourceSubNode])
                {
                    if (sourcePort->isA<Input>())
                    {
                        auto it = subNodeMap.find(sourcePort->getParent()->asA<Node>());
                        if (it != subNodeMap.end())
                        {
                            InputPtr processNodeInput = it->second->getInput(sourcePort->getName());
                            if (processNodeInput)
                            {
                                processNodeInput->setNodeName(destSubNode->getName());
                            }
                        }
                    }
                    else if (sourcePort->isA<Output>())
                    {
                        for (PortElementPtr processNodePort : downstreamPortMap[processNode])
                        {
                            processNodePort->setNodeName(destSubNode->getName());
                        }
                    }
                }

                // Transfer interface properties.
                for (InputPtr destInput : destSubNode->getInputs())
                {
                    if (destInput->hasInterfaceName())
                    {
                        InputPtr sourceInput = processNode->getInput(destInput->getInterfaceName());
                        if (sourceInput)
                        {
                            destInput->copyContentFrom(sourceInput);
                        }
                        else
                        {
                            ConstInterfaceElementPtr declaration = declarationMap[processNode];
                            InputPtr declInput = declaration ? declaration->getActiveInput(destInput->getInterfaceName()) : nullptr;
                            if (declInput)
                            {
                                if (declInput->hasValueString())
                                {
                                    destInput->setValueString(declInput->getValueString());
                                }
                                if (declInput->hasDefaultGeomPropString())
                                {
                                    ConstGeomPropDefPtr geomPropDef = getDocument()->getGeomPropDef(declInput->getDefaultGeomPropString());
                                    if (geomPropDef)
                                    {
                                        destInput->setConnectedNode(addGeomNode(geomPropDef, "geomNode"));
                                    }
                                }
                            }
                        }
                        destInput->removeAttribute(ValueElement::INTERFACE_NAME_ATTRIBUTE);
                    }
                }
            }

            // Update downstream ports with connections to subgraph outputs.
            for (PortElementPtr downstreamPort : downstreamPortMap[processNode])
            {
                if (downstreamPort->hasOutputString())
                {
                    OutputPtr subGraphOutput = sourceSubGraph->getOutput(downstreamPort->getOutputString());
                    if (subGraphOutput)
                    {
                        string destName = subGraphOutput->getNodeName();
                        NodePtr sourceSubNode = sourceSubGraph->getNode(destName);
                        NodePtr destNode = sourceSubNode ? subNodeMap[sourceSubNode] : nullptr;
                        if (destNode)
                        {
                            destName = destNode->getName();
                        }
                        downstreamPort->setNodeName(destName);
                        downstreamPort->setOutputString(EMPTY_STRING);
                    }
                }
            }

            // The processed node has been replaced, so remove it from the graph.
            removeNode(processNode->getName());
        }
    }
}

vector<ElementPtr> GraphElement::topologicalSort() const
{
    // Calculate a topological order of the children, using Kahn's algorithm
    // to avoid recursion.
    //
    // Running time: O(numNodes + numEdges).

    const vector<ElementPtr>& children = getChildren();

    // Calculate in-degrees for all children.
    std::unordered_map<ElementPtr, size_t> inDegree(children.size());
    std::deque<ElementPtr> childQueue;
    for (ElementPtr child : children)
    {
        size_t connectionCount = 0;
        for (size_t i = 0; i < child->getUpstreamEdgeCount(); ++i)
        {
            Edge upstreamEdge = child->getUpstreamEdge(i);
            if (upstreamEdge)
            {
                if (upstreamEdge.getUpstreamElement())
                {
                    ElementPtr elem = upstreamEdge.getUpstreamElement()->getParent();
                    if (elem == child->getParent())
                    {
                        connectionCount++;
                    }
                }
            }
        }

        inDegree[child] = connectionCount;

        // Enqueue children with in-degree 0.
        if (connectionCount == 0)
        {
            childQueue.push_back(child);
        }
    }

    vector<ElementPtr> result;
    while (!childQueue.empty())
    {
        // Pop the queue and add to topological order.
        ElementPtr child = childQueue.front();
        childQueue.pop_front();
        result.push_back(child);

        // Find connected nodes and decrease their in-degree,
        // adding node to the queue if in-degrees becomes 0.
        if (child->isA<Node>())
        {
            for (PortElementPtr port : child->asA<Node>()->getDownstreamPorts())
            {
                const ElementPtr downstreamElem = port->isA<Output>() ? port : port->getParent();
                if (inDegree[downstreamElem] > 1)
                {
                    inDegree[downstreamElem]--;
                }
                else
                {
                    inDegree[downstreamElem] = 0;
                    childQueue.push_back(downstreamElem);
                }
            }
        }
    }

    return result;
}

NodePtr GraphElement::addGeomNode(ConstGeomPropDefPtr geomPropDef, const string& namePrefix)
{
    string geomNodeName = namePrefix + "_" + geomPropDef->getName();
    NodePtr geomNode = getNode(geomNodeName);
    if (!geomNode)
    {
        geomNode = addNode(geomPropDef->getGeomProp(), geomNodeName, geomPropDef->getType());
        if (geomPropDef->hasAttribute(GeomPropDef::SPACE_ATTRIBUTE))
        {
            geomNode->setInputValue(GeomPropDef::SPACE_ATTRIBUTE, geomPropDef->getAttribute(GeomPropDef::SPACE_ATTRIBUTE));
        }
        if (geomPropDef->hasAttribute(GeomPropDef::INDEX_ATTRIBUTE))
        {
            geomNode->setInputValue(GeomPropDef::INDEX_ATTRIBUTE, geomPropDef->getAttribute(GeomPropDef::INDEX_ATTRIBUTE), getTypeString<int>());
        }
    }
    return geomNode;
}

string GraphElement::asStringDot() const
{
    string dot = "digraph {\n";

    // Create a unique name for each child element.
    vector<ElementPtr> children = topologicalSort();
    StringMap nameMap;
    StringSet nameSet;
    for (ElementPtr elem : children)
    {
        string uniqueName = elem->getCategory();
        while (nameSet.count(uniqueName))
        {
            uniqueName = incrementName(uniqueName);
        }
        nameMap[elem->getName()] = uniqueName;
        nameSet.insert(uniqueName);
    }

    // Write out all nodes.
    for (ElementPtr elem : children)
    {
        NodePtr node = elem->asA<Node>();
        if (node)
        {
            dot += "    \"" + nameMap[node->getName()] + "\" ";
            NodeDefPtr nodeDef = node->getNodeDef();
            const string& nodeGroup = nodeDef ? nodeDef->getNodeGroup() : EMPTY_STRING;
            if (nodeGroup == NodeDef::CONDITIONAL_NODE_GROUP)
            {
                dot += "[shape=diamond];\n";
            }
            else
            {
                dot += "[shape=box];\n";
            }
        }
    }

    // Write out all connections.
    std::set<Edge> processedEdges;
    StringSet processedInterfaces;
    for (OutputPtr output : getOutputs())
    {
        for (Edge edge : output->traverseGraph())
        {
            if (!processedEdges.count(edge))
            {
                ElementPtr upstreamElem = edge.getUpstreamElement();
                ElementPtr downstreamElem = edge.getDownstreamElement();
                ElementPtr connectingElem = edge.getConnectingElement();

                dot += "    \"" + nameMap[upstreamElem->getName()];
                dot += "\" -> \"" + nameMap[downstreamElem->getName()];
                dot += "\" [label=\"";
                dot += connectingElem ? connectingElem->getName() : EMPTY_STRING;
                dot += "\"];\n";

                NodePtr upstreamNode = upstreamElem->asA<Node>();
                if (upstreamNode && !processedInterfaces.count(upstreamNode->getName()))
                {
                    for (InputPtr input : upstreamNode->getInputs())
                    {
                        if (input->hasInterfaceName())
                        {
                            dot += "    \"" + input->getInterfaceName();
                            dot += "\" -> \"" + nameMap[upstreamElem->getName()];
                            dot += "\" [label=\"";
                            dot += input->getName();
                            dot += "\"];\n";
                        }
                    }
                    processedInterfaces.insert(upstreamNode->getName());
                }

                processedEdges.insert(edge);
            }
        }
    }

    dot += "}\n";

    return dot;
}

//
// NodeGraph methods
//

vector<OutputPtr> NodeGraph::getMaterialOutputs() const
{
    vector<OutputPtr> materialOutputs;
    for (auto graphOutput : getActiveOutputs())
    {
        if (graphOutput->getType() == MATERIAL_TYPE_STRING)
        {
            NodePtr node = graphOutput->getConnectedNode();
            if (node && node->getType() == MATERIAL_TYPE_STRING)
            {
                materialOutputs.push_back(graphOutput);
            }
        }
    }
    return materialOutputs;
}

void NodeGraph::setNodeDef(ConstNodeDefPtr nodeDef)
{
    if (nodeDef)
    {
        setNodeDefString(nodeDef->getName());
    }
    else
    {
        removeAttribute(NODE_DEF_ATTRIBUTE);
    }
}

InputPtr Node::addInputFromNodeDef(const string& inputName)
{
    InputPtr nodeInput = getInput(inputName);
    if (!nodeInput)
    {
        NodeDefPtr nodeDef = getNodeDef();
        InputPtr nodeDefInput = nodeDef ? nodeDef->getActiveInput(inputName) : nullptr;
        if (nodeDefInput)
        {
            nodeInput = addInput(nodeDefInput->getName(), nodeDefInput->getType());
            if (nodeDefInput->hasValueString())
            {
                nodeInput->setValueString(nodeDefInput->getValueString());
            }
        }
    }
    return nodeInput;
}

void Node::addInputsFromNodeDef()
{
    NodeDefPtr nodeDef = getNodeDef();
    if (nodeDef)
    {
        for (InputPtr nodeDefInput : nodeDef->getActiveInputs())
        {
            const string& inputName = nodeDefInput->getName();
            InputPtr nodeInput = getInput(inputName);
            if (!nodeInput)
            {
                nodeInput = addInput(inputName, nodeDefInput->getType());
                if (nodeDefInput->hasValueString())
                {
                    nodeInput->setValueString(nodeDefInput->getValueString());
                }
            }
        }
    }
}

void NodeGraph::addInterfaceName(const string& inputPath, const string& interfaceName)
{
    NodeDefPtr nodeDef = getNodeDef();
    if (!nodeDef)
    {
        throw Exception("Cannot declare an interface for a nodegraph which is not associated with a node definition: " + getName());
    }
    if (nodeDef->getChild(interfaceName))
    {
        throw Exception("Interface: " + interfaceName + " has already been declared on the node definition: " + nodeDef->getName());
    }

    ElementPtr elem = getDescendant(inputPath);
    InputPtr input = elem ? elem->asA<Input>() : nullptr;
    if (input && !input->getConnectedNode())
    {
        input->setInterfaceName(interfaceName);
        InputPtr nodeDefInput = nodeDef->getInput(interfaceName);
        if (!nodeDefInput)
        {
            nodeDefInput = nodeDef->addInput(interfaceName, input->getType());
        }
        if (input->hasValue())
        {
            nodeDefInput->setValueString(input->getValueString());
            input->removeAttribute(Input::VALUE_ATTRIBUTE);
        }
    }
}

void NodeGraph::removeInterfaceName(const string& inputPath)
{
    ElementPtr desc = getDescendant(inputPath);
    InputPtr input = desc ? desc->asA<Input>() : nullptr;
    if (input)
    {
        const string& interfaceName = input->getInterfaceName();
        getNodeDef()->removeChild(interfaceName);
        input->setInterfaceName(EMPTY_STRING);
    }
}

void NodeGraph::modifyInterfaceName(const string& inputPath, const string& interfaceName)
{
    ElementPtr desc = getDescendant(inputPath);
    InputPtr input = desc ? desc->asA<Input>() : nullptr;
    if (input)
    {
        const string& previousName = input->getInterfaceName();
        ElementPtr previousChild = getNodeDef()->getChild(previousName);
        if (previousChild)
        {
            previousChild->setName(interfaceName);
        }
        input->setInterfaceName(interfaceName);
    }
}

NodeDefPtr NodeGraph::getNodeDef() const
{
    NodeDefPtr nodedef = resolveNameReference<NodeDef>(getNodeDefString());
    // If not directly defined look for an implementation which has a nodedef association
    if (!nodedef)
    {
        for (auto impl : getDocument()->getImplementations())
        {
            if (impl->getNodeGraph() == getQualifiedName(getName()))
            {
                nodedef = impl->getNodeDef();
            }
        }
    }
    return nodedef;
}

InterfaceElementPtr NodeGraph::getImplementation() const
{
    NodeDefPtr nodedef = getNodeDef();
    return nodedef ? nodedef->getImplementation() : InterfaceElementPtr();
}

vector<PortElementPtr> NodeGraph::getDownstreamPorts() const
{
    vector<PortElementPtr> downstreamPorts;
    for (PortElementPtr port : getDocument()->getMatchingPorts(getQualifiedName(getName())))
    {
        ElementPtr node = port->getParent();
        ElementPtr graph = node ? node->getParent() : nullptr;
        if (graph && graph->isA<GraphElement>() && graph == getParent())
        {
            downstreamPorts.push_back(port);
        }
    }
    std::sort(downstreamPorts.begin(), downstreamPorts.end(), [](const ConstElementPtr& a, const ConstElementPtr& b)
    {
        return a->getName() > b->getName();
    });
    return downstreamPorts;
}

bool NodeGraph::validate(string* message) const
{
    bool res = true;

    validateRequire(!hasVersionString(), res, message, "NodeGraph elements do not support version strings");
    if (hasNodeDefString())
    {
        NodeDefPtr nodeDef = getNodeDef();
        validateRequire(nodeDef != nullptr, res, message, "NodeGraph implementation refers to non-existent NodeDef");
        if (nodeDef)
        {
            vector<OutputPtr> graphOutputs = getOutputs();
            vector<OutputPtr> nodeDefOutputs = nodeDef->getActiveOutputs();
            validateRequire(graphOutputs.size() == nodeDefOutputs.size(), res, message, "NodeGraph implementation has a different number of outputs than its NodeDef");
            if (graphOutputs.size() == 1 && nodeDefOutputs.size() == 1)
            {
                validateRequire(graphOutputs[0]->getType() == nodeDefOutputs[0]->getType(), res, message, "NodeGraph implementation has a different output type than its NodeDef");
            }
        }
    }

    return GraphElement::validate(message) && res;
}

ConstInterfaceElementPtr NodeGraph::getDeclaration(const string&) const
{
    ConstNodeDefPtr nodeDef = getNodeDef();
    if (nodeDef)
    {
        return nodeDef;
    }
    if (!hasNodeDefString())
    {
        return getSelf()->asA<InterfaceElement>();
    }
    return nullptr;
}

//
// Backdrop methods
//

void Backdrop::setContainsElements(const vector<ConstTypedElementPtr>& elems)
{
    if (!elems.empty())
    {
        StringVec stringVec;
        for (ConstTypedElementPtr elem : elems)
        {
            stringVec.push_back(elem->getName());
        }
        setTypedAttribute(CONTAINS_ATTRIBUTE, stringVec);
    }
    else
    {
        removeAttribute(CONTAINS_ATTRIBUTE);
    }
}

vector<TypedElementPtr> Backdrop::getContainsElements() const
{
    vector<TypedElementPtr> vec;
    ConstGraphElementPtr graph = getAncestorOfType<GraphElement>();
    if (graph)
    {
        for (const string& str : getTypedAttribute<StringVec>(CONTAINS_ATTRIBUTE))
        {
            TypedElementPtr elem = graph->getChildOfType<TypedElement>(str);
            if (elem)
            {
                vec.push_back(elem);
            }
        }
    }
    return vec;
}

bool Backdrop::validate(string* message) const
{
    bool res = true;
    if (hasContainsString())
    {
        StringVec stringVec = getTypedAttribute<StringVec>("contains");
        vector<TypedElementPtr> elemVec = getContainsElements();
        validateRequire(stringVec.size() == elemVec.size(), res, message, "Invalid element in contains string");
    }
    return Element::validate(message) && res;
}

MATERIALX_NAMESPACE_END
