//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Interface.h>

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Document.h>

#include <stdexcept>

MATERIALX_NAMESPACE_BEGIN

const string PortElement::NODE_NAME_ATTRIBUTE = "nodename";
const string PortElement::NODE_GRAPH_ATTRIBUTE = "nodegraph";
const string PortElement::OUTPUT_ATTRIBUTE = "output";
const string PortElement::CHANNELS_ATTRIBUTE = "channels";
const string InterfaceElement::NODE_DEF_ATTRIBUTE = "nodedef";
const string InterfaceElement::TARGET_ATTRIBUTE = "target";
const string InterfaceElement::VERSION_ATTRIBUTE = "version";
const string InterfaceElement::DEFAULT_VERSION_ATTRIBUTE = "isdefaultversion";
const string Input::DEFAULT_GEOM_PROP_ATTRIBUTE = "defaultgeomprop";
const string Output::DEFAULT_INPUT_ATTRIBUTE = "defaultinput";

// Map from type strings to swizzle pattern character sets.
const std::unordered_map<string, CharSet> PortElement::CHANNELS_CHARACTER_SET = {
    { "float", { '0', '1', 'r', 'x' } },
    { "color3", { '0', '1', 'r', 'g', 'b' } },
    { "color4", { '0', '1', 'r', 'g', 'b', 'a' } },
    { "vector2", { '0', '1', 'x', 'y' } },
    { "vector3", { '0', '1', 'x', 'y', 'z' } },
    { "vector4", { '0', '1', 'x', 'y', 'z', 'w' } }
};

// Map from type strings to swizzle pattern lengths.
const std::unordered_map<string, size_t> PortElement::CHANNELS_PATTERN_LENGTH = {
    { "float", 1 },
    { "color3", 3 },
    { "color4", 4 },
    { "vector2", 2 },
    { "vector3", 3 },
    { "vector4", 4 }
};

//
// PortElement methods
//

void PortElement::setConnectedNode(ConstNodePtr node)
{
    if (node)
    {
        setNodeName(node->getName());
    }
    else
    {
        removeAttribute(NODE_NAME_ATTRIBUTE);
    }
}

NodePtr PortElement::getConnectedNode() const
{
    ConstGraphElementPtr graph = getAncestorOfType<GraphElement>();
    return graph ? graph->getNode(getNodeName()) : nullptr;
}

void PortElement::setConnectedOutput(ConstOutputPtr output)
{
    if (output)
    {
        setOutputString(output->getName());
        ConstElementPtr parent = output->getParent();
        if (parent->isA<NodeGraph>())
        {
            setNodeGraphString(parent->getName());
            removeAttribute(NODE_NAME_ATTRIBUTE);
        }
        else if (parent->isA<Node>())
        {
            setNodeName(parent->getName());
            removeAttribute(NODE_GRAPH_ATTRIBUTE);
        }
    }
    else
    {
        removeAttribute(OUTPUT_ATTRIBUTE);
        removeAttribute(NODE_GRAPH_ATTRIBUTE);
        removeAttribute(NODE_NAME_ATTRIBUTE);
    }
}

OutputPtr PortElement::getConnectedOutput() const
{
    const string& outputString = getOutputString();
    OutputPtr result = nullptr;

    // Determine the scope at which the connected output may be found.
    ConstElementPtr parent = getParent();
    ConstElementPtr scope = parent ? parent->getParent() : nullptr;

    // Look for a nodegraph output.
    if (hasNodeGraphString())
    {
        NodeGraphPtr nodeGraph = resolveNameReference<NodeGraph>(getNodeGraphString(), scope);
        if (!nodeGraph)
        {
            nodeGraph = resolveNameReference<NodeGraph>(getNodeGraphString());
        }
        if (nodeGraph)
        {
            std::vector<OutputPtr> outputs = nodeGraph->getOutputs();
            if (!outputs.empty())
            {
                if (outputString.empty())
                {
                    result = outputs[0];
                }
                else
                {
                    result = nodeGraph->getOutput(outputString);
                }
            }
        }
    }
    // Look for a node output.
    else if (hasNodeName())
    {
        const string& nodeName = getNodeName();
        NodePtr node = resolveNameReference<Node>(nodeName, scope);
        if (!node)
        {
            node = resolveNameReference<Node>(nodeName);
        }
        if (node)
        {
            std::vector<OutputPtr> outputs = node->getOutputs();
            if (!outputs.empty())
            {
                if (outputString.empty())
                {
                    result = outputs[0];
                }
                else
                {
                    result = node->getOutput(outputString);
                }
            }
        }
    }

    // Look for an output at document scope.
    if (!result)
    {
        result = getDocument()->getOutput(outputString);
    }
    return result;
}

bool PortElement::validate(string* message) const
{
    bool res = true;

    NodePtr connectedNode = getConnectedNode();
    if (hasNodeName() || hasOutputString())
    {
        NodeGraphPtr nodeGraph = resolveNameReference<NodeGraph>(getNodeName());
        if (!nodeGraph)
        {
            validateRequire(connectedNode != nullptr, res, message, "Invalid port connection");
        }
    }
    if (connectedNode)
    {
        const string& outputString = getOutputString();
        if (!outputString.empty())
        {
            OutputPtr output;
            if (hasNodeName())
            {
                NodeDefPtr nodeDef = connectedNode->getNodeDef();
                output = nodeDef ? nodeDef->getOutput(outputString) : OutputPtr();
                if (output)
                {
                    validateRequire(connectedNode->getType() == MULTI_OUTPUT_TYPE_STRING, res, message,
                                    "Multi-output type expected in port connection");
                }
            }
            else if (hasNodeGraphString())
            {
                NodeGraphPtr nodeGraph = resolveNameReference<NodeGraph>(getNodeGraphString());
                if (nodeGraph)
                {
                    output = nodeGraph->getOutput(outputString);
                    if (nodeGraph->getNodeDef())
                    {
                        validateRequire(nodeGraph->getOutputCount() > 1, res, message,
                                        "Multi-output type expected in port connection");
                    }
                }
            }
            else
            {
                // Document has no concept of a multioutput type
                output = getDocument()->getOutput(outputString);
            }
            validateRequire(output != nullptr, res, message, "No output found for port connection");

            if (output)
            {
                if (hasChannels())
                {
                    bool valid = validChannelsString(getChannels(), output->getType(), getType());
                    validateRequire(valid, res, message, "Invalid channels string in port connection");
                }
                else
                {
                    validateRequire(getType() == output->getType(), res, message, "Mismatched types in port connection");
                }
            }
        }
        else if (hasChannels())
        {
            bool valid = validChannelsString(getChannels(), connectedNode->getType(), getType());
            validateRequire(valid, res, message, "Invalid channels string in port connection");
        }
        else if (connectedNode->getType() != MULTI_OUTPUT_TYPE_STRING)
        {
            validateRequire(getType() == connectedNode->getType(), res, message, "Mismatched types in port connection");
        }
    }
    return ValueElement::validate(message) && res;
}

bool PortElement::validChannelsCharacters(const string& channels, const string& sourceType)
{
    if (!CHANNELS_CHARACTER_SET.count(sourceType))
    {
        return false;
    }
    const CharSet& validCharSet = CHANNELS_CHARACTER_SET.at(sourceType);
    for (const char& channelChar : channels)
    {
        if (!validCharSet.count(channelChar))
        {
            return false;
        }
    }
    return true;
}

bool PortElement::validChannelsString(const string& channels, const string& sourceType, const string& destinationType)
{
    if (!validChannelsCharacters(channels, sourceType))
    {
        return false;
    }
    if (!CHANNELS_PATTERN_LENGTH.count(destinationType) ||
        CHANNELS_PATTERN_LENGTH.at(destinationType) != channels.size())
    {
        return false;
    }

    return true;
}

//
// Input methods
//

NodePtr Input::getConnectedNode() const
{
    // Handle interface name references.
    InputPtr graphInput = getInterfaceInput();
    if (graphInput && (graphInput->hasNodeName() || graphInput->hasNodeGraphString()))
    {
        return graphInput->getConnectedNode();
    }

    // Handle inputs of compound nodegraphs.
    if (getParent()->isA<NodeGraph>())
    {
        NodePtr rootNode = getDocument()->getNode(getNodeName());
        if (rootNode)
        {
            return rootNode;
        }
    }

    // Handle transitive connections via outputs.
    OutputPtr output = getConnectedOutput();
    if (output)
    {
        NodePtr node = output->getConnectedNode();
        if (node)
        {
            return node;
        }
    }

    return PortElement::getConnectedNode();
}

InputPtr Input::getInterfaceInput() const
{
    if (hasInterfaceName())
    {
        ConstGraphElementPtr graph = getAncestorOfType<GraphElement>();
        if (graph)
        {
            return graph->getInput(getInterfaceName());
        }
    }
    return nullptr;
}

GeomPropDefPtr Input::getDefaultGeomProp() const
{
    const string& defaultGeomProp = getAttribute(DEFAULT_GEOM_PROP_ATTRIBUTE);
    if (!defaultGeomProp.empty())
    {
        ConstDocumentPtr doc = getDocument();
        return doc->getChildOfType<GeomPropDef>(defaultGeomProp);
    }
    return nullptr;
}

bool Input::validate(string* message) const
{
    bool res = true;
    ConstElementPtr parent = getParent();

    if (hasDefaultGeomPropString())
    {
        validateRequire(parent->isA<NodeDef>(), res, message, "Invalid defaultgeomprop on non-definition input");
        validateRequire(getDefaultGeomProp() != nullptr, res, message, "Invalid defaultgeomprop string");
    }
    if (parent->isA<Node>())
    {
        bool hasValueBinding = hasValue();
        bool hasConnection = hasNodeName() || hasNodeGraphString() || hasOutputString() || hasInterfaceName();
        validateRequire(hasValueBinding || hasConnection, res, message, "Node input binds no value or connection");
    }
    else if (parent->isA<NodeGraph>())
    {
        validateRequire(parent->asA<NodeGraph>()->getNodeDef() == nullptr, res, message, "Input element in a functional nodegraph has no effect");
    }
    return PortElement::validate(message) && res;
}

//
// Output methods
//

Edge Output::getUpstreamEdge(size_t index) const
{
    if (index < getUpstreamEdgeCount())
    {
        return Edge(getSelfNonConst(), nullptr, getConnectedNode());
    }

    return NULL_EDGE;
}

bool Output::hasUpstreamCycle() const
{
    try
    {
        for (Edge edge : traverseGraph()) { }
    }
    catch (ExceptionFoundCycle&)
    {
        return true;
    }
    return false;
}

bool Output::validate(string* message) const
{
    bool res = true;
    validateRequire(!hasUpstreamCycle(), res, message, "Cycle in upstream path");
    return PortElement::validate(message) && res;
}

//
// InterfaceElement methods
//

InputPtr InterfaceElement::getActiveInput(const string& name) const
{
    for (ConstElementPtr elem : traverseInheritance())
    {
        InputPtr input = elem->asA<InterfaceElement>()->getInput(name);
        if (input)
        {
            return input;
        }
    }
    return nullptr;
}

vector<InputPtr> InterfaceElement::getActiveInputs() const
{
    vector<InputPtr> activeInputs;
    StringSet activeInputNamesSet;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<InputPtr> inputs = elem->asA<InterfaceElement>()->getInputs();
        for (const InputPtr& input : inputs)
        {
            if (input && activeInputNamesSet.insert(input->getName()).second)
            {
                activeInputs.push_back(input);
            }
        }
    }
    return activeInputs;
}

OutputPtr InterfaceElement::getActiveOutput(const string& name) const
{
    for (ConstElementPtr elem : traverseInheritance())
    {
        OutputPtr output = elem->asA<InterfaceElement>()->getOutput(name);
        if (output)
        {
            return output;
        }
    }
    return nullptr;
}

vector<OutputPtr> InterfaceElement::getActiveOutputs() const
{
    vector<OutputPtr> activeOutputs;
    StringSet activeOutputNamesSet;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<OutputPtr> outputs = elem->asA<InterfaceElement>()->getOutputs();
        for (const OutputPtr& output : outputs)
        {
            if (output && activeOutputNamesSet.insert(output->getName()).second)
            {
                activeOutputs.push_back(output);
            }
        }
    }
    return activeOutputs;
}

void InterfaceElement::setConnectedOutput(const string& inputName, OutputPtr output)
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        input = addInput(inputName);
    }
    if (output)
    {
        input->setType(output->getType());
    }
    input->setConnectedOutput(output);
}

OutputPtr InterfaceElement::getConnectedOutput(const string& inputName) const
{
    InputPtr input = getInput(inputName);
    if (!input)
    {
        return OutputPtr();
    }
    return input->getConnectedOutput();
}

TokenPtr InterfaceElement::getActiveToken(const string& name) const
{
    for (ConstElementPtr elem : traverseInheritance())
    {
        TokenPtr token = elem->asA<InterfaceElement>()->getToken(name);
        if (token)
        {
            return token;
        }
    }
    return nullptr;
}

vector<TokenPtr> InterfaceElement::getActiveTokens() const
{
    vector<TokenPtr> activeTokens;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<TokenPtr> tokens = elem->asA<InterfaceElement>()->getTokens();
        activeTokens.insert(activeTokens.end(), tokens.begin(), tokens.end());
    }
    return activeTokens;
}

ValueElementPtr InterfaceElement::getActiveValueElement(const string& name) const
{
    for (ConstElementPtr interface : traverseInheritance())
    {
        ValueElementPtr valueElem = interface->getChildOfType<ValueElement>(name);
        if (valueElem)
        {
            return valueElem;
        }
    }
    return nullptr;
}

vector<ValueElementPtr> InterfaceElement::getActiveValueElements() const
{
    vector<ValueElementPtr> activeValueElems;
    StringSet activeValueElemNamesSet;
    for (ConstElementPtr interface : traverseInheritance())
    {
        vector<ValueElementPtr> valueElems = interface->getChildrenOfType<ValueElement>();
        for (const ValueElementPtr& valueElem : valueElems)
        {
            if (valueElem && activeValueElemNamesSet.insert(valueElem->getName()).second)
            {
                activeValueElems.push_back(valueElem);
            }
        }
    }
    return activeValueElems;
}

ValuePtr InterfaceElement::getInputValue(const string& name, const string& target) const
{
    InputPtr input = getInput(name);
    if (input)
    {
        return input->getValue();
    }

    // Return the value, if any, stored in our declaration.
    ConstInterfaceElementPtr decl = getDeclaration(target);
    if (decl)
    {
        input = decl->getInput(name);
        if (input)
        {
            return input->getValue();
        }
    }

    return ValuePtr();
}

void InterfaceElement::setVersionIntegers(int majorVersion, int minorVersion)
{
    string versionString = std::to_string(majorVersion) + "." +
                           std::to_string(minorVersion);
    setVersionString(versionString);
}

std::pair<int, int> InterfaceElement::getVersionIntegers() const
{
    const string& versionString = getVersionString();
    StringVec splitVersion = splitString(versionString, ".");
    try
    {
        if (splitVersion.size() == 2)
        {
            return { std::stoi(splitVersion[0]), std::stoi(splitVersion[1]) };
        }
        else if (splitVersion.size() == 1)
        {
            return { std::stoi(splitVersion[0]), 0 };
        }
    }
    catch (std::invalid_argument&)
    {
    }
    catch (std::out_of_range&)
    {
    }
    return { 0, 0 };
}

void InterfaceElement::registerChildElement(ElementPtr child)
{
    TypedElement::registerChildElement(child);
    if (child->isA<Input>())
    {
        _inputCount++;
    }
    else if (child->isA<Output>())
    {
        _outputCount++;
    }
}

void InterfaceElement::unregisterChildElement(ElementPtr child)
{
    TypedElement::unregisterChildElement(child);
    if (child->isA<Input>())
    {
        _inputCount--;
    }
    else if (child->isA<Output>())
    {
        _outputCount--;
    }
}

ConstInterfaceElementPtr InterfaceElement::getDeclaration(const string&) const
{
    return InterfaceElementPtr();
}

void InterfaceElement::clearContent()
{
    _inputCount = 0;
    _outputCount = 0;
    TypedElement::clearContent();
}

bool InterfaceElement::hasExactInputMatch(ConstInterfaceElementPtr declaration, string* message) const
{
    for (InputPtr input : getActiveInputs())
    {
        InputPtr declarationInput = declaration->getActiveInput(input->getName());
        if (!declarationInput ||
            declarationInput->getType() != input->getType())
        {
            if (message)
            {
                *message += "Input '" + input->getName() + "' doesn't match declaration";
            }
            return false;
        }
    }
    return true;
}

MATERIALX_NAMESPACE_END
