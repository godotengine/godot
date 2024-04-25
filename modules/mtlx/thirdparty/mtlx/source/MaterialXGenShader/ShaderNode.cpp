//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/ShaderNode.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

MATERIALX_NAMESPACE_BEGIN

const string ShaderMetadataRegistry::USER_DATA_NAME = "ShaderMetadataRegistry";

//
// ShaderPort methods
//

ShaderPort::ShaderPort(ShaderNode* node, const TypeDesc* type, const string& name, ValuePtr value) :
    _node(node),
    _type(type),
    _name(name),
    _variable(name),
    _value(value),
    _flags(0)
{
}

string ShaderPort::getFullName() const
{
    return (_node->getName() + "_" + _name);
}

string ShaderPort::getValueString() const
{
    return getValue() ? getValue()->getValueString() : EMPTY_STRING;
}

//
// ShaderInput methods
//

ShaderInput::ShaderInput(ShaderNode* node, const TypeDesc* type, const string& name) :
    ShaderPort(node, type, name),
    _connection(nullptr)
{
}

void ShaderInput::makeConnection(ShaderOutput* src)
{
    // Make sure this is a new connection.
    if (src != _connection)
    {
        // Break the old connection.
        breakConnection();
        if (src)
        {
            // Make the new connection.
            _connection = src;
            src->_connections.push_back(this);
        }
    }
}

void ShaderInput::breakConnection()
{
    if (_connection)
    {
        // Find and erase this input from the connected output's connection vector.
        ShaderInputVec& connected = _connection->_connections;
        ShaderInputVec::iterator it = std::find(connected.begin(), connected.end(), this);
        if (it != connected.end())
        {
            connected.erase(it);
        }
        // Clear the connection.
        _connection = nullptr;
    }
}

ShaderNode* ShaderInput::getConnectedSibling() const
{
    if (_connection && _connection->getNode()->getParent() == _node->getParent())
    {
        return _connection->getNode();
    }
    return nullptr;
}

//
// ShaderOutput methods
//

ShaderOutput::ShaderOutput(ShaderNode* node, const TypeDesc* type, const string& name) :
    ShaderPort(node, type, name)
{
}

void ShaderOutput::makeConnection(ShaderInput* dst)
{
    dst->makeConnection(this);
}

void ShaderOutput::breakConnection(ShaderInput* dst)
{
    if (std::find(_connections.begin(), _connections.end(), dst) == _connections.end())
    {
        throw ExceptionShaderGenError(
            "Cannot break non-existent connection from output: " + getFullName() + " to input: " + dst->getFullName());
    }
    dst->breakConnection();
}

void ShaderOutput::breakConnections()
{
    ShaderInputVec inputVec(_connections);
    for (ShaderInput* input : inputVec)
    {
        input->breakConnection();
    }

    if (!_connections.empty())
    {
        throw ExceptionShaderGenError("Number of output connections not broken properly'" + std::to_string(_connections.size()) +
                                      " for output: " + getFullName());
    }
}

namespace
{
ShaderNodePtr createEmptyNode()
{
    return std::make_shared<ShaderNode>(nullptr, "");
}
} // namespace

const ShaderNodePtr ShaderNode::NONE = createEmptyNode();

const string ShaderNode::CONSTANT = "constant";
const string ShaderNode::DOT = "dot";
const string ShaderNode::IMAGE = "image";
const string ShaderNode::SURFACESHADER = "surfaceshader";
const string ShaderNode::SCATTER_MODE = "scatter_mode";
const string ShaderNode::BSDF_R = "R";
const string ShaderNode::BSDF_T = "T";
const string ShaderNode::TEXTURE2D_GROUPNAME = "texture2d";
const string ShaderNode::TEXTURE3D_GROUPNAME = "texture3d";
const string ShaderNode::PROCEDURAL2D_GROUPNAME = "procedural2d";
const string ShaderNode::PROCEDURAL3D_GROUPNAME = "procedural3d";
const string ShaderNode::GEOMETRIC_GROUPNAME = "geometric";

//
// ShaderNode methods
//

ShaderNode::ShaderNode(const ShaderGraph* parent, const string& name) :
    _parent(parent),
    _name(name),
    _classification(0),
    _impl(nullptr)
{
}

ShaderNodePtr ShaderNode::create(const ShaderGraph* parent, const string& name, const NodeDef& nodeDef, GenContext& context)
{
    ShaderNodePtr newNode = std::make_shared<ShaderNode>(parent, name);

    const ShaderGenerator& shadergen = context.getShaderGenerator();

    // Find the implementation for this nodedef
    newNode->_impl = shadergen.getImplementation(nodeDef, context);
    if (!newNode->_impl)
    {
        throw ExceptionShaderGenError("Could not find a matching implementation for node '" + nodeDef.getNodeString() +
                                      "' matching target '" + shadergen.getTarget() + "'");
    }

    // Create interface from nodedef
    for (const ValueElementPtr& port : nodeDef.getActiveValueElements())
    {
        const TypeDesc* portType = TypeDesc::get(port->getType());
        if (port->isA<Output>())
        {
            newNode->addOutput(port->getName(), portType);
        }
        else if (port->isA<Input>())
        {
            ShaderInput* input;
            const string& portValue = port->getResolvedValueString();
            std::pair<const TypeDesc*, ValuePtr> enumResult;
            const string& enumNames = port->getAttribute(ValueElement::ENUM_ATTRIBUTE);
            if (context.getShaderGenerator().getSyntax().remapEnumeration(portValue, portType, enumNames, enumResult))
            {
                input = newNode->addInput(port->getName(), enumResult.first);
                input->setValue(enumResult.second);
            }
            else
            {
                input = newNode->addInput(port->getName(), portType);
                if (!portValue.empty())
                {
                    input->setValue(port->getResolvedValue());
                }
            }
            if (port->getIsUniform())
            {
                input->setUniform();
            }
        }
    }

    // Add any additional inputs required by the implementation
    newNode->getImplementation().addInputs(*newNode, context);

    // Add a default output if needed
    if (newNode->numOutputs() == 0)
    {
        newNode->addOutput("out", TypeDesc::get(nodeDef.getType()));
    }

    const string& nodeDefName = nodeDef.getName();
    const string& groupName = nodeDef.getNodeGroup();

    //
    // Set node classification, defaulting to texture node
    //
    newNode->_classification = Classification::TEXTURE;

    // First, check for specific output types
    const ShaderOutput* primaryOutput = newNode->getOutput();
    if (*primaryOutput->getType() == *Type::MATERIAL)
    {
        newNode->_classification = Classification::MATERIAL;
    }
    else if (*primaryOutput->getType() == *Type::SURFACESHADER)
    {
        if (nodeDefName == "ND_surface_unlit")
        {
            newNode->_classification = Classification::SHADER | Classification::SURFACE | Classification::UNLIT;
        }
        else
        {
            newNode->_classification = Classification::SHADER | Classification::SURFACE | Classification::CLOSURE;
        }
    }
    else if (*primaryOutput->getType() == *Type::VOLUMESHADER)
    {
        newNode->_classification = Classification::SHADER | Classification::VOLUME | Classification::CLOSURE;
    }
    else if (*primaryOutput->getType() == *Type::LIGHTSHADER)
    {
        newNode->_classification = Classification::LIGHT | Classification::SHADER | Classification::CLOSURE;
    }
    else if (*primaryOutput->getType() == *Type::BSDF)
    {
        newNode->_classification = Classification::BSDF | Classification::CLOSURE;

        // Add additional classifications for BSDF reflection and/or transmission.
        const string& bsdfType = nodeDef.getAttribute("bsdf");
        if (bsdfType == BSDF_R)
        {
            newNode->_classification |= Classification::BSDF_R;
        }
        else if (bsdfType == BSDF_T)
        {
            newNode->_classification |= Classification::BSDF_T;
        }
        else
        {
            newNode->_classification |= (Classification::BSDF_R | Classification::BSDF_T);
        }

        // Check specifically for the vertical layering node
        if (nodeDefName == "ND_layer_bsdf" || nodeDefName == "ND_layer_vdf")
        {
            newNode->_classification |= Classification::LAYER;
        }
        // Check specifically for the thin-film node
        else if (nodeDefName == "ND_thin_film_bsdf")
        {
            newNode->_classification |= Classification::THINFILM;
        }
    }
    else if (*primaryOutput->getType() == *Type::EDF)
    {
        newNode->_classification = Classification::EDF | Classification::CLOSURE;
    }
    else if (*primaryOutput->getType() == *Type::VDF)
    {
        newNode->_classification = Classification::VDF | Classification::CLOSURE;
    }
    // Second, check for specific nodes types
    else if (nodeDef.getNodeString() == CONSTANT)
    {
        newNode->_classification = Classification::TEXTURE | Classification::CONSTANT;
    }
    else if (nodeDef.getNodeString() == DOT)
    {
        newNode->_classification = Classification::TEXTURE | Classification::DOT;
    }
    // Third, check for file texture classification by group name
    else if (groupName == TEXTURE2D_GROUPNAME || groupName == TEXTURE3D_GROUPNAME)
    {
        newNode->_classification = Classification::TEXTURE | Classification::FILETEXTURE;
    }

    // Add in classification based on group name
    if (groupName == TEXTURE2D_GROUPNAME || groupName == PROCEDURAL2D_GROUPNAME)
    {
        newNode->_classification |= Classification::SAMPLE2D;
    }
    else if (groupName == TEXTURE3D_GROUPNAME || groupName == PROCEDURAL3D_GROUPNAME)
    {
        newNode->_classification |= Classification::SAMPLE3D;
    }
    else if (groupName == GEOMETRIC_GROUPNAME)
    {
        newNode->_classification |= Classification::GEOMETRIC;
    }

    // Create any metadata.
    newNode->createMetadata(nodeDef, context);

    return newNode;
}

ShaderNodePtr ShaderNode::create(const ShaderGraph* parent, const string& name, ShaderNodeImplPtr impl, unsigned int classification)
{
    ShaderNodePtr newNode = std::make_shared<ShaderNode>(parent, name);
    newNode->_impl = impl;
    newNode->_classification = classification;
    return newNode;
}

void ShaderNode::initialize(const Node& node, const NodeDef& nodeDef, GenContext& context)
{
    // Copy input values from the given node
    for (InputPtr nodeInput : node.getActiveInputs())
    {
        ShaderInput* input = getInput(nodeInput->getName());
        ValueElementPtr nodeDefInput = nodeDef.getActiveValueElement(nodeInput->getName());
        if (input && nodeDefInput)
        {
            ValuePtr portValue = nodeInput->getResolvedValue();
            if (!portValue)
            {
                InputPtr interfaceInput = nodeInput->getInterfaceInput();
                if (interfaceInput)
                {
                    portValue = interfaceInput->getValue();
                }
            }
            const string& valueString = portValue ? portValue->getValueString() : EMPTY_STRING;
            std::pair<const TypeDesc*, ValuePtr> enumResult;
            const string& enumNames = nodeDefInput->getAttribute(ValueElement::ENUM_ATTRIBUTE);
            const TypeDesc* type = TypeDesc::get(nodeDefInput->getType());
            if (context.getShaderGenerator().getSyntax().remapEnumeration(valueString, type, enumNames, enumResult))
            {
                input->setValue(enumResult.second);
            }
            else if (!valueString.empty())
            {
                input->setValue(portValue);
            }

            input->setChannels(nodeInput->getChannels());
        }
    }

    // Set implementation specific values.
    if (_impl)
    {
        _impl->setValues(node, *this, context);
    }

    // Set element paths for children on the node
    for (const ValueElementPtr& nodeValue : node.getActiveValueElements())
    {
        ShaderInput* input = getInput(nodeValue->getName());
        if (input)
        {
            string path = nodeValue->getNamePath();
            InputPtr nodeInput = nodeValue->asA<Input>();
            if (nodeInput)
            {
                InputPtr interfaceInput = nodeInput->getInterfaceInput();
                if (interfaceInput)
                {
                    path = interfaceInput->getNamePath();
                }
            }
            input->setPath(path);
        }
    }

    // Set element paths based on the node definition. Note that these
    // paths don't actually exist at time of shader generation since there
    // are no inputs specified on the node itself
    //
    const string& nodePath = node.getNamePath();
    for (auto nodeInput : nodeDef.getActiveInputs())
    {
        ShaderInput* input = getInput(nodeInput->getName());
        if (input && input->getPath().empty())
        {
            input->setPath(nodePath + NAME_PATH_SEPARATOR + nodeInput->getName());
        }
    }

    // For BSDF nodes see if there is a scatter_mode input,
    // and update the classification accordingly.
    if (hasClassification(Classification::BSDF))
    {
        const InputPtr scatterModeInput = node.getInput(SCATTER_MODE);
        const string& scatterMode = scatterModeInput ? scatterModeInput->getValueString() : EMPTY_STRING;
        // If scatter mode is only T, set classification to only transmission.
        // Note: For only R we must still keep classification at default value (both reflection/transmission)
        // since reflection needs to attenuate the transmission amount in HW shaders when layering is used.
        if (scatterMode == BSDF_T)
        {
            _classification |= Classification::BSDF_T;
            _classification &= ~Classification::BSDF_R;
        }
    }
}

void ShaderNode::createMetadata(const NodeDef& nodeDef, GenContext& context)
{
    ShaderMetadataRegistryPtr registry = context.getUserData<ShaderMetadataRegistry>(ShaderMetadataRegistry::USER_DATA_NAME);
    if (!(registry && registry->getAllMetadata().size()))
    {
        // Early out if no metadata is registered.
        return;
    }

    // Set metadata on the node according to the nodedef attributes.
    ShaderMetadataVecPtr nodeMetadataStorage = getMetadata();
    for (const string& nodedefAttr : nodeDef.getAttributeNames())
    {
        const ShaderMetadata* metadataEntry = registry->findMetadata(nodedefAttr);
        if (metadataEntry)
        {
            const string& attrValue = nodeDef.getAttribute(nodedefAttr);
            if (!attrValue.empty())
            {
                ValuePtr value = Value::createValueFromStrings(attrValue, metadataEntry->type->getName());
                if (!value)
                {
                    value = metadataEntry->value;
                }
                if (value)
                {
                    if (!nodeMetadataStorage)
                    {
                        nodeMetadataStorage = std::make_shared<ShaderMetadataVec>();
                        setMetadata(nodeMetadataStorage);
                    }
                    nodeMetadataStorage->push_back(ShaderMetadata(metadataEntry->name, metadataEntry->type, value));
                }
            }
        }
    }

    // Set metadata on inputs according to attributes on the nodedef's inputs
    for (const ValueElementPtr& nodedefPort : nodeDef.getActiveValueElements())
    {
        ShaderInput* input = getInput(nodedefPort->getName());
        if (input)
        {
            ShaderMetadataVecPtr inputMetadataStorage = input->getMetadata();

            for (const string& nodedefPortAttr : nodedefPort->getAttributeNames())
            {
                const ShaderMetadata* metadataEntry = registry->findMetadata(nodedefPortAttr);
                if (metadataEntry)
                {
                    const string& attrValue = nodedefPort->getAttribute(nodedefPortAttr);
                    if (!attrValue.empty())
                    {
                        const TypeDesc* type = metadataEntry->type ? metadataEntry->type : input->getType();
                        ValuePtr value = Value::createValueFromStrings(attrValue, type->getName());
                        if (!value)
                        {
                            value = metadataEntry->value;
                        }
                        if (value)
                        {
                            if (!inputMetadataStorage)
                            {
                                inputMetadataStorage = std::make_shared<ShaderMetadataVec>();
                                input->setMetadata(inputMetadataStorage);
                            }
                            inputMetadataStorage->push_back(ShaderMetadata(metadataEntry->name, type, value));
                        }
                    }
                }
            }
        }
    }
}

ShaderInput* ShaderNode::getInput(const string& name)
{
    auto it = _inputMap.find(name);
    return it != _inputMap.end() ? it->second.get() : nullptr;
}

ShaderOutput* ShaderNode::getOutput(const string& name)
{
    auto it = _outputMap.find(name);
    return it != _outputMap.end() ? it->second.get() : nullptr;
}

const ShaderInput* ShaderNode::getInput(const string& name) const
{
    auto it = _inputMap.find(name);
    return it != _inputMap.end() ? it->second.get() : nullptr;
}

const ShaderOutput* ShaderNode::getOutput(const string& name) const
{
    auto it = _outputMap.find(name);
    return it != _outputMap.end() ? it->second.get() : nullptr;
}

ShaderInput* ShaderNode::addInput(const string& name, const TypeDesc* type)
{
    if (getInput(name))
    {
        throw ExceptionShaderGenError("An input named '" + name + "' already exists on node '" + _name + "'");
    }

    ShaderInputPtr input = std::make_shared<ShaderInput>(this, type, name);
    _inputMap[name] = input;
    _inputOrder.push_back(input.get());

    return input.get();
}

ShaderOutput* ShaderNode::addOutput(const string& name, const TypeDesc* type)
{
    if (getOutput(name))
    {
        throw ExceptionShaderGenError("An output named '" + name + "' already exists on node '" + _name + "'");
    }

    ShaderOutputPtr output = std::make_shared<ShaderOutput>(this, type, name);
    _outputMap[name] = output;
    _outputOrder.push_back(output.get());

    return output.get();
}

MATERIALX_NAMESPACE_END
