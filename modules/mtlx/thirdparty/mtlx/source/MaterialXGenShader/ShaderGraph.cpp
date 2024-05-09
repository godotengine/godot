//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/ShaderGraph.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

#include <iostream>
#include <queue>

MATERIALX_NAMESPACE_BEGIN

//
// ShaderGraph methods
//

ShaderGraph::ShaderGraph(const ShaderGraph* parent, const string& name, ConstDocumentPtr document, const StringSet& reservedWords) :
    ShaderNode(parent, name),
    _document(document)
{
    // Add all reserved words as taken identifiers
    for (const string& n : reservedWords)
    {
        _identifiers[n] = 1;
    }
}

void ShaderGraph::addInputSockets(const InterfaceElement& elem, GenContext& context)
{
    for (InputPtr input : elem.getActiveInputs())
    {
        ShaderGraphInputSocket* inputSocket = nullptr;
        ValuePtr portValue = input->getResolvedValue();
        const string& portValueString = portValue ? portValue->getValueString() : EMPTY_STRING;
        std::pair<const TypeDesc*, ValuePtr> enumResult;
        const string& enumNames = input->getAttribute(ValueElement::ENUM_ATTRIBUTE);
        const TypeDesc* portType = TypeDesc::get(input->getType());
        if (context.getShaderGenerator().getSyntax().remapEnumeration(portValueString, portType, enumNames, enumResult))
        {
            inputSocket = addInputSocket(input->getName(), enumResult.first);
            inputSocket->setValue(enumResult.second);
        }
        else
        {
            inputSocket = addInputSocket(input->getName(), portType);
            if (!portValueString.empty())
            {
                inputSocket->setValue(portValue);
            }
        }
        if (input->getIsUniform())
        {
            inputSocket->setUniform();
        }
        GeomPropDefPtr geomprop = input->getDefaultGeomProp();
        if (geomprop)
        {
            inputSocket->setGeomProp(geomprop->getName());
        }
    }
}

void ShaderGraph::addOutputSockets(const InterfaceElement& elem)
{
    for (const OutputPtr& output : elem.getActiveOutputs())
    {
        ShaderGraphOutputSocket* outputSocket = addOutputSocket(output->getName(), TypeDesc::get(output->getType()));
        outputSocket->setChannels(output->getChannels());
    }
    if (numOutputSockets() == 0)
    {
        addOutputSocket("out", TypeDesc::get(elem.getType()));
    }
}

void ShaderGraph::createConnectedNodes(const ElementPtr& downstreamElement,
                                       const ElementPtr& upstreamElement,
                                       ElementPtr connectingElement,
                                       GenContext& context)
{
    // Create the node if it doesn't exists
    NodePtr upstreamNode = upstreamElement->asA<Node>();
    if (!upstreamNode)
    {
        throw ExceptionShaderGenError("Upstream element to connect is not a node '" +
                                      upstreamElement->getName() + "'");
    }
    const string& newNodeName = upstreamNode->getName();
    ShaderNode* newNode = getNode(newNodeName);
    if (!newNode)
    {
        newNode = createNode(upstreamNode, context);
    }

    //
    // Make connections
    //

    // Find the output to connect to.
    if (!connectingElement && downstreamElement->isA<Output>())
    {
        // Edge case for having an output downstream but no connecting
        // element (input) reported upstream. In this case we set the
        // output itself as connecting element, which handles finding
        // the nodedef output in case of a multioutput node upstream.
        connectingElement = downstreamElement->asA<Output>();
    }
    OutputPtr nodeDefOutput = connectingElement ? upstreamNode->getNodeDefOutput(connectingElement) : nullptr;
    ShaderOutput* output = nodeDefOutput ? newNode->getOutput(nodeDefOutput->getName()) : newNode->getOutput();
    if (!output)
    {
        throw ExceptionShaderGenError("Could not find an output named '" + (nodeDefOutput ? nodeDefOutput->getName() : string("out")) +
                                      "' on upstream node '" + upstreamNode->getName() + "'");
    }

    // Check if it was a node downstream
    NodePtr downstreamNode = downstreamElement->asA<Node>();
    if (downstreamNode)
    {
        // We have a node downstream
        ShaderNode* downstream = getNode(downstreamNode->getName());
        if (downstream && connectingElement)
        {
            ShaderInput* input = downstream->getInput(connectingElement->getName());
            if (!input)
            {
                throw ExceptionShaderGenError("Could not find an input named '" + connectingElement->getName() +
                                              "' on downstream node '" + downstream->getName() + "'");
            }
            input->makeConnection(output);
        }
        else
        {
            throw ExceptionShaderGenError("Could not find downstream node ' " + downstreamNode->getName() + "'");
        }
    }
    else
    {
        // Not a node, then it must be an output
        ShaderGraphOutputSocket* outputSocket = getOutputSocket(downstreamElement->getName());
        if (outputSocket)
        {
            outputSocket->makeConnection(output);
        }
    }
}

void ShaderGraph::addUpstreamDependencies(const Element& root, GenContext& context)
{
    std::set<ElementPtr> processedOutputs;

    for (Edge edge : root.traverseGraph())
    {
        ElementPtr upstreamElement = edge.getUpstreamElement();
        if (!upstreamElement)
        {
            continue;
        }

        ElementPtr downstreamElement = edge.getDownstreamElement();

        // Early out if downstream element is an output that
        // we have already processed. This might happen since
        // we perform jumps over output elements below.
        if (processedOutputs.count(downstreamElement))
        {
            continue;
        }

        // If upstream is an output jump to the actual node connected to the output.
        if (upstreamElement->isA<Output>())
        {
            // Record this output so we don't process it again when it
            // shows up as a downstream element in the next iteration.
            processedOutputs.insert(upstreamElement);

            upstreamElement = upstreamElement->asA<Output>()->getConnectedNode();
            if (!upstreamElement)
            {
                continue;
            }
        }

        createConnectedNodes(downstreamElement,
                             upstreamElement,
                             edge.getConnectingElement(),
                             context);
    }
}

void ShaderGraph::addDefaultGeomNode(ShaderInput* input, const GeomPropDef& geomprop, GenContext& context)
{
    const string geomNodeName = "geomprop_" + geomprop.getName();
    ShaderNode* node = getNode(geomNodeName);

    if (!node)
    {
        // Find the nodedef for the geometric node referenced by the geomprop. Use the type of the
        // input here and ignore the type of the geomprop. They are required to have the same type.
        string geomNodeDefName = "ND_" + geomprop.getGeomProp() + "_" + input->getType()->getName();
        NodeDefPtr geomNodeDef = _document->getNodeDef(geomNodeDefName);
        if (!geomNodeDef)
        {
            throw ExceptionShaderGenError("Could not find a nodedef named '" + geomNodeDefName +
                                          "' for defaultgeomprop on input '" + input->getFullName() + "'");
        }

        ShaderNodePtr geomNode = ShaderNode::create(this, geomNodeName, *geomNodeDef, context);
        addNode(geomNode);

        // Set node inputs if given.
        const string& namePath = geomprop.getNamePath();
        const string& space = geomprop.getSpace();
        if (!space.empty())
        {
            ShaderInput* spaceInput = geomNode->getInput(GeomPropDef::SPACE_ATTRIBUTE);
            ValueElementPtr nodeDefSpaceInput = geomNodeDef->getActiveValueElement(GeomPropDef::SPACE_ATTRIBUTE);
            if (spaceInput && nodeDefSpaceInput)
            {
                std::pair<const TypeDesc*, ValuePtr> enumResult;
                const string& enumNames = nodeDefSpaceInput->getAttribute(ValueElement::ENUM_ATTRIBUTE);
                const TypeDesc* portType = TypeDesc::get(nodeDefSpaceInput->getType());
                if (context.getShaderGenerator().getSyntax().remapEnumeration(space, portType, enumNames, enumResult))
                {
                    spaceInput->setValue(enumResult.second);
                }
                else
                {
                    spaceInput->setValue(Value::createValue<string>(space));
                }
                spaceInput->setPath(namePath);
            }
        }
        const string& index = geomprop.getIndex();
        if (!index.empty())
        {
            ShaderInput* indexInput = geomNode->getInput("index");
            if (indexInput)
            {
                indexInput->setValue(Value::createValue<string>(index));
                indexInput->setPath(namePath);
            }
        }
        const string& geomProp = geomprop.getGeomProp();
        if (!geomProp.empty())
        {
            ShaderInput* geomPropInput = geomNode->getInput(GeomPropDef::GEOM_PROP_ATTRIBUTE);
            if (geomPropInput)
            {
                geomPropInput->setValue(Value::createValue<string>(geomProp));
                geomPropInput->setPath(namePath);
            }
        }

        node = geomNode.get();

        // Assign a unique variable name for the node output.
        const Syntax& syntax = context.getShaderGenerator().getSyntax();
        ShaderOutput* output = node->getOutput();
        string variable = output->getFullName();
        variable = syntax.getVariableName(variable, output->getType(), _identifiers);
        output->setVariable(variable);
    }

    input->makeConnection(node->getOutput());
}

void ShaderGraph::addColorTransformNode(ShaderInput* input, const ColorSpaceTransform& transform, GenContext& context)
{
    if (input->getConnection() && !input->isBindInput())
    {
        // Ignore connected inputs, which are unaffected by color space bindings.
        return;
    }

    ColorManagementSystemPtr colorManagementSystem = context.getShaderGenerator().getColorManagementSystem();
    if (!colorManagementSystem)
    {
        return;
    }
    const string colorTransformNodeName = input->getFullName() + "_cm";
    ShaderNodePtr colorTransformNodePtr = colorManagementSystem->createNode(this, transform, colorTransformNodeName, context);

    if (colorTransformNodePtr)
    {
        addNode(colorTransformNodePtr);

        ShaderNode* colorTransformNode = colorTransformNodePtr.get();
        ShaderOutput* colorTransformNodeOutput = colorTransformNode->getOutput(0);

        ShaderInput* shaderInput = colorTransformNode->getInput(0);
        shaderInput->setVariable(input->getFullName());
        shaderInput->setValue(input->getValue());
        shaderInput->setPath(input->getPath());
        shaderInput->setUnit(EMPTY_STRING);

        if (input->isBindInput())
        {
            ShaderOutput* oldConnection = input->getConnection();
            shaderInput->makeConnection(oldConnection);
        }

        input->makeConnection(colorTransformNodeOutput);
    }
}

void ShaderGraph::addColorTransformNode(ShaderOutput* output, const ColorSpaceTransform& transform, GenContext& context)
{
    ColorManagementSystemPtr colorManagementSystem = context.getShaderGenerator().getColorManagementSystem();
    if (!colorManagementSystem)
    {
        return;
    }

    const string colorTransformNodeName = output->getFullName() + "_cm";
    ShaderNodePtr colorTransformNodePtr = colorManagementSystem->createNode(this, transform, colorTransformNodeName, context);

    if (colorTransformNodePtr)
    {
        addNode(colorTransformNodePtr);

        ShaderNode* colorTransformNode = colorTransformNodePtr.get();
        ShaderOutput* colorTransformNodeOutput = colorTransformNode->getOutput(0);

        ShaderInputVec inputs = output->getConnections();
        for (ShaderInput* input : inputs)
        {
            input->breakConnection();
            input->makeConnection(colorTransformNodeOutput);
        }

        // Connect the node to the upstream output
        ShaderInput* colorTransformNodeInput = colorTransformNode->getInput(0);
        colorTransformNodeInput->makeConnection(output);
    }
}

void ShaderGraph::addUnitTransformNode(ShaderInput* input, const UnitTransform& transform, GenContext& context)
{
    if (input->getConnection() && !input->isBindInput())
    {
        // Ignore connected inputs, which are unaffected by unit bindings.
        return;
    }

    UnitSystemPtr unitSystem = context.getShaderGenerator().getUnitSystem();
    if (!unitSystem)
    {
        return;
    }
    const string unitTransformNodeName = input->getFullName() + "_unit";
    ShaderNodePtr unitTransformNodePtr = unitSystem->createNode(this, transform, unitTransformNodeName, context);

    if (unitTransformNodePtr)
    {
        addNode(unitTransformNodePtr);

        ShaderNode* unitTransformNode = unitTransformNodePtr.get();
        ShaderOutput* unitTransformNodeOutput = unitTransformNode->getOutput(0);

        ShaderInput* shaderInput = unitTransformNode->getInput(0);
        shaderInput->setVariable(input->getFullName());
        shaderInput->setValue(input->getValue());
        shaderInput->setPath(input->getPath());
        shaderInput->setUnit(input->getUnit());
        shaderInput->setColorSpace(input->getColorSpace());

        if (input->isBindInput())
        {
            ShaderOutput* oldConnection = input->getConnection();
            shaderInput->makeConnection(oldConnection);
        }

        input->makeConnection(unitTransformNodeOutput);
    }
}

void ShaderGraph::addUnitTransformNode(ShaderOutput* output, const UnitTransform& transform, GenContext& context)
{
    UnitSystemPtr unitSystem = context.getShaderGenerator().getUnitSystem();
    if (!unitSystem)
    {
        return;
    }

    const string unitTransformNodeName = output->getFullName() + "_unit";
    ShaderNodePtr unitTransformNodePtr = unitSystem->createNode(this, transform, unitTransformNodeName, context);

    if (unitTransformNodePtr)
    {
        addNode(unitTransformNodePtr);

        ShaderNode* unitTransformNode = unitTransformNodePtr.get();
        ShaderOutput* unitTransformNodeOutput = unitTransformNode->getOutput(0);

        ShaderInputVec inputs = output->getConnections();
        for (ShaderInput* input : inputs)
        {
            input->breakConnection();
            input->makeConnection(unitTransformNodeOutput);
        }

        // Connect the node to the upstream output
        ShaderInput* unitTransformNodeInput = unitTransformNode->getInput(0);
        unitTransformNodeInput->makeConnection(output);
    }
}

ShaderGraphPtr ShaderGraph::create(const ShaderGraph* parent, const NodeGraph& nodeGraph, GenContext& context)
{
    NodeDefPtr nodeDef = nodeGraph.getNodeDef();
    if (!nodeDef)
    {
        throw ExceptionShaderGenError("Can't find nodedef '" + nodeGraph.getNodeDefString() + "' referenced by nodegraph '" + nodeGraph.getName() + "'");
    }

    string graphName = nodeGraph.getName();
    context.getShaderGenerator().getSyntax().makeValidName(graphName);
    ShaderGraphPtr graph = std::make_shared<ShaderGraph>(parent, graphName, nodeGraph.getDocument(), context.getReservedWords());

    // Clear classification
    graph->_classification = 0;

    // Create input sockets from the nodedef
    graph->addInputSockets(*nodeDef, context);

    // Create output sockets from the nodegraph
    graph->addOutputSockets(nodeGraph);

    // Traverse all outputs and create all internal nodes
    for (OutputPtr graphOutput : nodeGraph.getActiveOutputs())
    {
        graph->addUpstreamDependencies(*graphOutput, context);
    }

    // Finalize the graph
    graph->finalize(context);

    return graph;
}

ShaderGraphPtr ShaderGraph::create(const ShaderGraph* parent, const string& name, ElementPtr element, GenContext& context)
{
    ShaderGraphPtr graph;
    ElementPtr root;

    if (element->isA<Output>())
    {
        OutputPtr output = element->asA<Output>();
        ElementPtr outputParent = output->getParent();

        InterfaceElementPtr interface;
        if (outputParent->isA<NodeGraph>())
        {
            // A nodegraph output.
            NodeGraphPtr nodeGraph = outputParent->asA<NodeGraph>();
            NodeDefPtr nodeDef = nodeGraph->getNodeDef();
            if (nodeDef)
            {
                interface = nodeDef;
            }
            else
            {
                interface = nodeGraph;
            }
        }
        else if (outputParent->isA<Document>())
        {
            // A free floating output.
            outputParent = output->getConnectedNode();
            interface = outputParent ? outputParent->asA<InterfaceElement>() : nullptr;
        }
        if (!interface)
        {
            throw ExceptionShaderGenError("Given output '" + output->getName() + "' has no interface valid for shader generation");
        }

        graph = std::make_shared<ShaderGraph>(parent, name, element->getDocument(), context.getReservedWords());

        // Clear classification
        graph->_classification = 0;

        // Create input sockets
        graph->addInputSockets(*interface, context);

        // Create the given output socket
        ShaderGraphOutputSocket* outputSocket = graph->addOutputSocket(output->getName(), TypeDesc::get(output->getType()));
        outputSocket->setPath(output->getNamePath());
        outputSocket->setChannels(output->getChannels());
        const string& outputUnit = output->getUnit();
        if (!outputUnit.empty())
        {
            outputSocket->setUnit(outputUnit);
        }
        const string& outputColorSpace = output->getColorSpace();
        if (!outputColorSpace.empty())
        {
            outputSocket->setColorSpace(outputColorSpace);
        }

        // Start traversal from this output
        root = output;
    }

    else if (element->isA<Node>())
    {
        NodePtr node = element->asA<Node>();
        NodeDefPtr nodeDef = node->getNodeDef();
        if (!nodeDef)
        {
            throw ExceptionShaderGenError("Could not find a nodedef for node '" + node->getName() + "'");
        }

        graph = std::make_shared<ShaderGraph>(parent, name, element->getDocument(), context.getReservedWords());

        // Create input sockets
        graph->addInputSockets(*nodeDef, context);

        // Create output sockets
        graph->addOutputSockets(*nodeDef);

        // Create this shader node in the graph.
        ShaderNodePtr newNode = ShaderNode::create(graph.get(), node->getName(), *nodeDef, context);
        graph->addNode(newNode);

        // Share metadata.
        graph->setMetadata(newNode->getMetadata());

        // Connect it to the graph outputs
        for (size_t i = 0; i < newNode->numOutputs(); ++i)
        {
            ShaderGraphOutputSocket* outputSocket = graph->getOutputSocket(i);
            outputSocket->makeConnection(newNode->getOutput(i));
            outputSocket->setPath(node->getNamePath());
        }

        // Handle node input ports
        for (const InputPtr& nodedefInput : nodeDef->getActiveInputs())
        {
            ShaderGraphInputSocket* inputSocket = graph->getInputSocket(nodedefInput->getName());
            ShaderInput* input = newNode->getInput(nodedefInput->getName());
            if (!inputSocket || !input)
            {
                throw ExceptionShaderGenError("Node input '" + nodedefInput->getName() + "' doesn't match an existing input on graph '" + graph->getName() + "'");
            }

            // Copy data from node element to shadergen representation
            InputPtr nodeInput = node->getInput(nodedefInput->getName());
            if (nodeInput)
            {
                ValuePtr value = nodeInput->getResolvedValue();
                if (value)
                {
                    const string& valueString = value->getValueString();
                    std::pair<const TypeDesc*, ValuePtr> enumResult;
                    const TypeDesc* type = TypeDesc::get(nodedefInput->getType());
                    const string& enumNames = nodedefInput->getAttribute(ValueElement::ENUM_ATTRIBUTE);
                    if (context.getShaderGenerator().getSyntax().remapEnumeration(valueString, type, enumNames, enumResult))
                    {
                        inputSocket->setValue(enumResult.second);
                    }
                    else
                    {
                        inputSocket->setValue(value);
                    }
                }

                input->setBindInput();
                const string path = nodeInput->getNamePath();
                if (!path.empty())
                {
                    inputSocket->setPath(path);
                    input->setPath(path);
                }
                const string& unit = nodeInput->getUnit();
                if (!unit.empty())
                {
                    inputSocket->setUnit(unit);
                    input->setUnit(unit);
                }
                const string& colorSpace = nodeInput->getColorSpace();
                if (!colorSpace.empty())
                {
                    inputSocket->setColorSpace(colorSpace);
                    input->setColorSpace(colorSpace);
                }
            }

            // Connect graph socket to the node input
            inputSocket->makeConnection(input);

            // Share metadata.
            inputSocket->setMetadata(input->getMetadata());
        }

        // Apply color and unit transforms to each input.
        graph->applyInputTransforms(node, newNode, context);

        // Set root for upstream dependency traversal
        root = node;
    }

    if (!graph)
    {
        throw ExceptionShaderGenError("Shader generation from element '" + element->getName() + "' of type '" + element->getCategory() + "' is not supported");
    }

    // Traverse and create all dependencies upstream
    if (root && context.getOptions().addUpstreamDependencies)
    {
        graph->addUpstreamDependencies(*root, context);
    }

    graph->finalize(context);

    return graph;
}

void ShaderGraph::applyInputTransforms(ConstNodePtr node, ShaderNodePtr shaderNode, GenContext& context)
{
    ColorManagementSystemPtr colorManagementSystem = context.getShaderGenerator().getColorManagementSystem();
    UnitSystemPtr unitSystem = context.getShaderGenerator().getUnitSystem();

    const string& targetColorSpace = context.getOptions().targetColorSpaceOverride.empty() ?
                                     _document->getActiveColorSpace() :
                                     context.getOptions().targetColorSpaceOverride;
    const string& targetDistanceUnit = context.getOptions().targetDistanceUnit;

    for (InputPtr input : node->getInputs())
    {
        if (input->hasValue() || input->hasInterfaceName())
        {
            string sourceColorSpace = input->getActiveColorSpace();
            if (input->getType() == FILENAME_TYPE_STRING && node->isColorType())
            {
                // Adjust the source color space for filename interface names.
                if (input->hasInterfaceName())
                {
                    for (ConstNodePtr parentNode : context.getParentNodes())
                    {
                        if (!parentNode->isColorType())
                        {
                            InputPtr interfaceInput = parentNode->getInput(input->getInterfaceName());
                            string interfaceColorSpace = interfaceInput ? interfaceInput->getActiveColorSpace() : EMPTY_STRING;
                            if (!interfaceColorSpace.empty())
                            {
                                sourceColorSpace = interfaceColorSpace;
                            }
                        }
                    }
                }

                ShaderOutput* shaderOutput = shaderNode->getOutput();
                populateColorTransformMap(colorManagementSystem, shaderOutput, sourceColorSpace, targetColorSpace, false);
                populateUnitTransformMap(unitSystem, shaderOutput, input, targetDistanceUnit, false);
            }
            else
            {
                ShaderInput* shaderInput = shaderNode->getInput(input->getName());
                populateColorTransformMap(colorManagementSystem, shaderInput, sourceColorSpace, targetColorSpace, true);
                populateUnitTransformMap(unitSystem, shaderInput, input, targetDistanceUnit, true);
            }
        }
    }
}

ShaderNode* ShaderGraph::createNode(ConstNodePtr node, GenContext& context)
{
    NodeDefPtr nodeDef = node->getNodeDef();
    if (!nodeDef)
    {
        throw ExceptionShaderGenError("Could not find a nodedef for node '" + node->getName() + "'");
    }

    // Create this node in the graph.
    context.pushParentNode(node);
    ShaderNodePtr newNode = ShaderNode::create(this, node->getName(), *nodeDef, context);
    newNode->initialize(*node, *nodeDef, context);
    _nodeMap[node->getName()] = newNode;
    _nodeOrder.push_back(newNode.get());
    context.popParentNode();

    // Check if any of the node inputs should be connected to the graph interface
    for (ValueElementPtr elem : node->getChildrenOfType<ValueElement>())
    {
        const string& interfaceName = elem->getInterfaceName();
        if (!interfaceName.empty())
        {
            ShaderGraphInputSocket* inputSocket = getInputSocket(interfaceName);
            if (inputSocket)
            {
                ShaderInput* input = newNode->getInput(elem->getName());

                if (input)
                {
                    input->makeConnection(inputSocket);
                }
            }
        }
    }

    // Handle the "defaultgeomprop" directives on the nodedef inputs.
    // Create and connect default geometric nodes on unconnected inputs.
    for (const InputPtr& nodeDefInput : nodeDef->getActiveInputs())
    {
        ShaderInput* input = newNode->getInput(nodeDefInput->getName());
        InputPtr nodeInput = node->getInput(nodeDefInput->getName());

        const string& connection = nodeInput ? nodeInput->getNodeName() : EMPTY_STRING;
        if (connection.empty() && !input->getConnection())
        {
            GeomPropDefPtr geomprop = nodeDefInput->getDefaultGeomProp();
            if (geomprop)
            {
                addDefaultGeomNode(input, *geomprop, context);
            }
        }
    }

    // Apply color and unit transforms to each input.
    applyInputTransforms(node, newNode, context);

    return newNode.get();
}

ShaderGraphInputSocket* ShaderGraph::addInputSocket(const string& name, const TypeDesc* type)
{
    return ShaderNode::addOutput(name, type);
}

ShaderGraphOutputSocket* ShaderGraph::addOutputSocket(const string& name, const TypeDesc* type)
{
    return ShaderNode::addInput(name, type);
}

ShaderGraphEdgeIterator ShaderGraph::traverseUpstream(ShaderOutput* output)
{
    return ShaderGraphEdgeIterator(output);
}

void ShaderGraph::addNode(ShaderNodePtr node)
{
    _nodeMap[node->getName()] = node;
    _nodeOrder.push_back(node.get());
}

ShaderNode* ShaderGraph::getNode(const string& name)
{
    auto it = _nodeMap.find(name);
    return it != _nodeMap.end() ? it->second.get() : nullptr;
}

const ShaderNode* ShaderGraph::getNode(const string& name) const
{
    return const_cast<ShaderGraph*>(this)->getNode(name);
}

void ShaderGraph::finalize(GenContext& context)
{
    // Allow node implementations to update the classification
    // on its node instances
    for (ShaderNode* node : getNodes())
    {
        node->getImplementation().addClassification(*node);
    }

    // Add classification according to root node
    ShaderGraphOutputSocket* outputSocket = getOutputSocket();
    if (outputSocket->getConnection())
    {
        addClassification(outputSocket->getConnection()->getNode()->getClassification());
    }

    // Insert color transformation nodes where needed
    if (context.getOptions().emitColorTransforms)
    {
        for (const auto& it : _inputColorTransformMap)
        {
            addColorTransformNode(it.first, it.second, context);
        }
        for (const auto& it : _outputColorTransformMap)
        {
            addColorTransformNode(it.first, it.second, context);
        }
    }
    _inputColorTransformMap.clear();
    _outputColorTransformMap.clear();

    // Insert unit transformation nodes where needed
    for (const auto& it : _inputUnitTransformMap)
    {
        addUnitTransformNode(it.first, it.second, context);
    }
    for (const auto& it : _outputUnitTransformMap)
    {
        addUnitTransformNode(it.first, it.second, context);
    }
    _inputUnitTransformMap.clear();
    _outputUnitTransformMap.clear();

    // Optimize the graph, removing redundant paths.
    optimize(context);

    // Sort the nodes in topological order.
    topologicalSort();

    if (context.getOptions().shaderInterfaceType == SHADER_INTERFACE_COMPLETE)
    {
        // Publish all node inputs that has not been connected already.
        for (const ShaderNode* node : getNodes())
        {
            for (ShaderInput* input : node->getInputs())
            {
                if (!input->getConnection())
                {
                    // Check if the type is editable otherwise we can't
                    // publish the input as an editable uniform.
                    if (input->getType()->isEditable() && node->isEditable(*input))
                    {
                        // Use a consistent naming convention: <nodename>_<inputname>
                        // so application side can figure out what uniforms to set
                        // when node inputs change on application side.
                        const string interfaceName = node->getName() + "_" + input->getName();

                        ShaderGraphInputSocket* inputSocket = getInputSocket(interfaceName);
                        if (!inputSocket)
                        {
                            inputSocket = addInputSocket(interfaceName, input->getType());
                            inputSocket->setPath(input->getPath());
                            inputSocket->setValue(input->getValue());
                            inputSocket->setUnit(input->getUnit());
                            inputSocket->setColorSpace(input->getColorSpace());
                            if (input->isUniform())
                            {
                                inputSocket->setUniform();
                            }
                        }
                        inputSocket->makeConnection(input);
                        inputSocket->setMetadata(input->getMetadata());
                    }
                }
            }
        }
    }

    // Set variable names for inputs and outputs in the graph.
    setVariableNames(context);
}

void ShaderGraph::disconnect(ShaderNode* node) const
{
    for (ShaderInput* input : node->getInputs())
    {
        input->breakConnection();
    }
    for (ShaderOutput* output : node->getOutputs())
    {
        output->breakConnections();
    }
}

void ShaderGraph::optimize(GenContext& context)
{
    size_t numEdits = 0;
    for (ShaderNode* node : getNodes())
    {
        if (node->hasClassification(ShaderNode::Classification::CONSTANT))
        {
            // Constant nodes can be elided by moving their value downstream.
            bypass(context, node, 0);
            ++numEdits;
        }
        else if (node->hasClassification(ShaderNode::Classification::DOT))
        {
            // Filename dot nodes must be elided so they do not create extra samplers.
            ShaderInput* in = node->getInput("in");
            if (in->getChannels().empty() && *in->getType() == *Type::FILENAME)
            {
                bypass(context, node, 0);
                ++numEdits;
            }
        }
        // Adding more nodes here requires them to have an input that is tagged
        // "uniform" in the NodeDef or to handle very specific cases, like FILENAME.
    }

    if (numEdits > 0)
    {
        std::set<ShaderNode*> usedNodesSet;
        std::vector<ShaderNode*> usedNodesVec;

        // Traverse the graph to find nodes still in use
        for (ShaderGraphOutputSocket* outputSocket : getOutputSockets())
        {
            // Make sure to not include connections to the graph itself.
            ShaderOutput* upstreamPort = outputSocket->getConnection();
            if (upstreamPort && upstreamPort->getNode() != this)
            {
                for (ShaderGraphEdge edge : ShaderGraph::traverseUpstream(upstreamPort))
                {
                    ShaderNode* node = edge.upstream->getNode();
                    if (usedNodesSet.count(node) == 0)
                    {
                        usedNodesSet.insert(node);
                        usedNodesVec.push_back(node);
                    }
                }
            }
        }

        // Remove any unused nodes
        for (ShaderNode* node : _nodeOrder)
        {
            if (usedNodesSet.count(node) == 0)
            {
                // Break all connections
                disconnect(node);

                // Erase from storage
                _nodeMap.erase(node->getName());
            }
        }

        _nodeOrder = usedNodesVec;
    }
}

void ShaderGraph::bypass(GenContext& context, ShaderNode* node, size_t inputIndex, size_t outputIndex)
{
    ShaderInput* input = node->getInput(inputIndex);
    ShaderOutput* output = node->getOutput(outputIndex);

    ShaderOutput* upstream = input->getConnection();
    if (upstream)
    {
        // Re-route the upstream output to the downstream inputs.
        // Iterate a copy of the connection vector since the
        // original vector will change when breaking connections.
        ShaderInputVec downstreamConnections = output->getConnections();
        for (ShaderInput* downstream : downstreamConnections)
        {
            output->breakConnection(downstream);
            downstream->makeConnection(upstream);
        }
    }
    else
    {
        // No node connected upstream to re-route,
        // so push the input's value and element path downstream instead.
        // Iterate a copy of the connection vector since the
        // original vector will change when breaking connections.
        ShaderInputVec downstreamConnections = output->getConnections();
        for (ShaderInput* downstream : downstreamConnections)
        {
            output->breakConnection(downstream);
            downstream->setValue(input->getValue());
            downstream->setPath(input->getPath());
            const string& inputUnit = input->getUnit();
            if (!inputUnit.empty())
            {
                downstream->setUnit(inputUnit);
            }
            const string& inputColorSpace = input->getColorSpace();
            if (!inputColorSpace.empty())
            {
                downstream->setColorSpace(inputColorSpace);
            }

            // Swizzle the input value. Once done clear the channel to indicate
            // no further swizzling is reqiured.
            const string& channels = downstream->getChannels();
            if (!channels.empty())
            {
                downstream->setValue(context.getShaderGenerator().getSyntax().getSwizzledValue(input->getValue(),
                                                                                               input->getType(),
                                                                                               channels,
                                                                                               downstream->getType()));
                downstream->setType(downstream->getType());
                downstream->setChannels(EMPTY_STRING);
            }
        }
    }
}

void ShaderGraph::topologicalSort()
{
    // Calculate a topological order of the children, using Kahn's algorithm
    // to avoid recursion.
    //
    // Running time: O(numNodes + numEdges).

    // Calculate in-degrees for all nodes, and enqueue those with degree 0.
    std::unordered_map<ShaderNode*, int> inDegree(_nodeMap.size());
    std::deque<ShaderNode*> nodeQueue;
    for (ShaderNode* node : _nodeOrder)
    {
        int connectionCount = 0;
        for (const ShaderInput* input : node->getInputs())
        {
            if (input->getConnection() && input->getConnection()->getNode() != this)
            {
                ++connectionCount;
            }
        }

        inDegree[node] = connectionCount;

        if (connectionCount == 0)
        {
            nodeQueue.push_back(node);
        }
    }

    _nodeOrder.resize(_nodeMap.size(), nullptr);
    size_t count = 0;

    while (!nodeQueue.empty())
    {
        // Pop the queue and add to topological order.
        ShaderNode* node = nodeQueue.front();
        nodeQueue.pop_front();
        _nodeOrder[count++] = node;

        // Find connected nodes and decrease their in-degree,
        // adding node to the queue if in-degrees becomes 0.
        for (const ShaderOutput* output : node->getOutputs())
        {
            for (const ShaderInput* input : output->getConnections())
            {
                ShaderNode* downstreamNode = const_cast<ShaderNode*>(input->getNode());
                if (downstreamNode != this)
                {
                    if (--inDegree[downstreamNode] <= 0)
                    {
                        nodeQueue.push_back(downstreamNode);
                    }
                }
            }
        }
    }
}

void ShaderGraph::setVariableNames(GenContext& context)
{
    // Make sure inputs and outputs have variable names valid for the
    // target shading language, and are unique to avoid name conflicts.

    const Syntax& syntax = context.getShaderGenerator().getSyntax();

    for (ShaderGraphInputSocket* inputSocket : getInputSockets())
    {
        const string variable = syntax.getVariableName(inputSocket->getName(), inputSocket->getType(), _identifiers);
        inputSocket->setVariable(variable);
    }
    for (ShaderGraphOutputSocket* outputSocket : getOutputSockets())
    {
        const string variable = syntax.getVariableName(outputSocket->getName(), outputSocket->getType(), _identifiers);
        outputSocket->setVariable(variable);
    }
    for (ShaderNode* node : getNodes())
    {
        for (ShaderInput* input : node->getInputs())
        {
            string variable = input->getFullName();
            variable = syntax.getVariableName(variable, input->getType(), _identifiers);
            input->setVariable(variable);
        }
        for (ShaderOutput* output : node->getOutputs())
        {
            string variable = output->getFullName();
            variable = syntax.getVariableName(variable, output->getType(), _identifiers);
            output->setVariable(variable);
        }
    }
}

void ShaderGraph::populateColorTransformMap(ColorManagementSystemPtr colorManagementSystem, ShaderPort* shaderPort,
                                            const string& sourceColorSpace, const string& targetColorSpace, bool asInput)
{
    if (!shaderPort ||
        sourceColorSpace.empty() ||
        targetColorSpace.empty() ||
        sourceColorSpace == targetColorSpace)
    {
        return;
    }

    if (*(shaderPort->getType()) == *Type::COLOR3 || *(shaderPort->getType()) == *Type::COLOR4)
    {
        // Store the source color space on the shader port.
        shaderPort->setColorSpace(sourceColorSpace);

        // Update the color transform map, if a color management system is provided.
        if (colorManagementSystem)
        {
            ColorSpaceTransform transform(sourceColorSpace, targetColorSpace, shaderPort->getType());
            if (colorManagementSystem->supportsTransform(transform))
            {
                if (asInput)
                {
                    _inputColorTransformMap.emplace(static_cast<ShaderInput*>(shaderPort), transform);
                }
                else
                {
                    _outputColorTransformMap.emplace(static_cast<ShaderOutput*>(shaderPort), transform);
                }
            }
            else
            {
                std::cerr << "Unsupported color space transform from " <<
                              sourceColorSpace << " to " << targetColorSpace << std::endl;
            }
        }
    }
}

void ShaderGraph::populateUnitTransformMap(UnitSystemPtr unitSystem, ShaderPort* shaderPort, ValueElementPtr input, const string& globalTargetUnitSpace, bool asInput)
{
    if (!unitSystem || globalTargetUnitSpace.empty())
    {
        return;
    }

    const string& sourceUnitSpace = input->getUnit();
    if (!shaderPort || sourceUnitSpace.empty())
    {
        return;
    }

    const string& unitType = input->getUnitType();
    if (!input->getDocument()->getUnitTypeDef(unitType))
    {
        return;
    }

    string targetUnitSpace = input->getActiveUnit();
    if (targetUnitSpace.empty())
    {
        targetUnitSpace = globalTargetUnitSpace;
    }

    // TODO: Consider this to be an optimization option as
    // this allows for the source and target unit to be the same value
    // while still allowing target unit updates on a compiled shader as the
    // target is exposed as an input uniform.
    //if (sourceUnitSpace == targetUnitSpace)
    //{
    //    return;
    //}

    // Only support convertion for float and vectors. arrays, matrices are not supported.
    // TODO: This should be provided by the UnitSystem.
    bool supportedType = (*shaderPort->getType() == *Type::FLOAT ||
                          *shaderPort->getType() == *Type::VECTOR2 ||
                          *shaderPort->getType() == *Type::VECTOR3 ||
                          *shaderPort->getType() == *Type::VECTOR4);
    if (supportedType)
    {
        UnitTransform transform(sourceUnitSpace, targetUnitSpace, shaderPort->getType(), unitType);
        if (unitSystem->supportsTransform(transform))
        {
            shaderPort->setUnit(sourceUnitSpace);
            if (asInput)
            {
                _inputUnitTransformMap.emplace(static_cast<ShaderInput*>(shaderPort), transform);
            }
            else
            {
                _outputUnitTransformMap.emplace(static_cast<ShaderOutput*>(shaderPort), transform);
            }
        }
    }
}

namespace
{
static const ShaderGraphEdgeIterator NULL_EDGE_ITERATOR(nullptr);
}

//
// ShaderGraphEdgeIterator methods
//

ShaderGraphEdgeIterator::ShaderGraphEdgeIterator(ShaderOutput* output) :
    _upstream(output),
    _downstream(nullptr)
{
}

ShaderGraphEdgeIterator& ShaderGraphEdgeIterator::operator++()
{
    if (_upstream && _upstream->getNode()->numInputs())
    {
        // Traverse to the first upstream edge of this element.
        _stack.emplace_back(_upstream, 0);

        ShaderInput* input = _upstream->getNode()->getInput(0);
        ShaderOutput* output = input->getConnection();

        if (output && !output->getNode()->isAGraph())
        {
            extendPathUpstream(output, input);
            return *this;
        }
    }

    while (true)
    {
        if (_upstream)
        {
            returnPathDownstream(_upstream);
        }

        if (_stack.empty())
        {
            // Traversal is complete.
            *this = ShaderGraphEdgeIterator::end();
            return *this;
        }

        // Traverse to our siblings.
        StackFrame& parentFrame = _stack.back();
        while (parentFrame.second + 1 < parentFrame.first->getNode()->numInputs())
        {
            ShaderInput* input = parentFrame.first->getNode()->getInput(++parentFrame.second);
            ShaderOutput* output = input->getConnection();

            if (output && !output->getNode()->isAGraph())
            {
                extendPathUpstream(output, input);
                return *this;
            }
        }

        // Traverse to our parent's siblings.
        returnPathDownstream(parentFrame.first);
        _stack.pop_back();
    }

    return *this;
}

const ShaderGraphEdgeIterator& ShaderGraphEdgeIterator::end()
{
    return NULL_EDGE_ITERATOR;
}

void ShaderGraphEdgeIterator::extendPathUpstream(ShaderOutput* upstream, ShaderInput* downstream)
{
    // Check for cycles.
    if (_path.count(upstream))
    {
        throw ExceptionFoundCycle("Encountered cycle at element: " + upstream->getFullName());
    }

    // Extend the current path to the new element.
    _path.insert(upstream);
    _upstream = upstream;
    _downstream = downstream;
}

void ShaderGraphEdgeIterator::returnPathDownstream(ShaderOutput* upstream)
{
    _path.erase(upstream);
    _upstream = nullptr;
    _downstream = nullptr;
}

MATERIALX_NAMESPACE_END
