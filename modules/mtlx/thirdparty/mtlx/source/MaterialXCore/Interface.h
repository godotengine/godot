//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_INTERFACE_H
#define MATERIALX_INTERFACE_H

/// @file
/// Interface element subclasses

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Geom.h>

MATERIALX_NAMESPACE_BEGIN

class PortElement;
class Input;
class Output;
class InterfaceElement;
class Node;
class NodeDef;

/// A shared pointer to a PortElement
using PortElementPtr = shared_ptr<PortElement>;
/// A shared pointer to a const PortElement
using ConstPortElementPtr = shared_ptr<const PortElement>;

/// A shared pointer to an Input
using InputPtr = shared_ptr<Input>;
/// A shared pointer to a const Input
using ConstInputPtr = shared_ptr<const Input>;

/// A shared pointer to an Output
using OutputPtr = shared_ptr<Output>;
/// A shared pointer to a const Output
using ConstOutputPtr = shared_ptr<const Output>;

/// A shared pointer to an InterfaceElement
using InterfaceElementPtr = shared_ptr<InterfaceElement>;
/// A shared pointer to a const InterfaceElement
using ConstInterfaceElementPtr = shared_ptr<const InterfaceElement>;

using CharSet = std::set<char>;

/// @class PortElement
/// The base class for port elements such as Input and Output.
///
/// Port elements support spatially-varying upstream connections to nodes.
class MX_CORE_API PortElement : public ValueElement
{
  protected:
    PortElement(ElementPtr parent, const string& category, const string& name) :
        ValueElement(parent, category, name)
    {
    }

  public:
    virtual ~PortElement() { }

  protected:
    using NodePtr = shared_ptr<Node>;
    using ConstNodePtr = shared_ptr<const Node>;

  public:
    /// @name Node Name
    /// @{

    /// Set the node name string of this element, creating a connection to
    /// the Node with the given name within the same NodeGraph.
    void setNodeName(const string& node)
    {
        setAttribute(NODE_NAME_ATTRIBUTE, node);
    }

    /// Return true if this element has a node name string.
    bool hasNodeName() const
    {
        return hasAttribute(NODE_NAME_ATTRIBUTE);
    }

    /// Return the node name string of this element.
    const string& getNodeName() const
    {
        return getAttribute(NODE_NAME_ATTRIBUTE);
    }

    /// @}
    /// @name Node Graph
    /// @{

    /// Set the node graph string of this element.
    void setNodeGraphString(const string& node)
    {
        setAttribute(NODE_GRAPH_ATTRIBUTE, node);
    }

    /// Return true if this element has a node graph string.
    bool hasNodeGraphString() const
    {
        return hasAttribute(NODE_GRAPH_ATTRIBUTE);
    }

    /// Return the node graph string of this element.
    const string& getNodeGraphString() const
    {
        return getAttribute(NODE_GRAPH_ATTRIBUTE);
    }

    /// @}
    /// @name Output
    /// @{

    /// Set the output string of this element.
    void setOutputString(const string& output)
    {
        setAttribute(OUTPUT_ATTRIBUTE, output);
    }

    /// Return true if this element has an output string.
    bool hasOutputString() const
    {
        return hasAttribute(OUTPUT_ATTRIBUTE);
    }

    /// Set the output to which this input is connected.  If the output
    /// argument is null, then any existing output connection will be cleared.
    void setConnectedOutput(ConstOutputPtr output);

    /// Return the output, if any, to which this input is connected.
    virtual OutputPtr getConnectedOutput() const;

    /// Return the output string of this element.
    const string& getOutputString() const
    {
        return getAttribute(OUTPUT_ATTRIBUTE);
    }

    /// @}
    /// @name Channels
    /// @{

    /// Set the channels string of this element, defining a channel swizzle
    /// that will be applied to the upstream result if this port is connected.
    void setChannels(const string& channels)
    {
        setAttribute(CHANNELS_ATTRIBUTE, channels);
    }

    /// Return true if this element has a channels string.
    bool hasChannels() const
    {
        return hasAttribute(CHANNELS_ATTRIBUTE);
    }

    /// Return the channels string of this element.
    const string& getChannels() const
    {
        return getAttribute(CHANNELS_ATTRIBUTE);
    }

    /// Return true if the given channels characters are valid for the given
    /// source type string.
    static bool validChannelsCharacters(const string& channels, const string& sourceType);

    /// Return true if the given channels string is valid for the given source
    /// and destination type strings.
    static bool validChannelsString(const string& channels, const string& sourceType, const string& destinationType);

    /// @}
    /// @name Connections
    /// @{

    /// Set the node to which this element is connected.  The given node must
    /// belong to the same node graph.  If the node argument is null, then
    /// any existing node connection will be cleared.
    void setConnectedNode(ConstNodePtr node);

    /// Return the node, if any, to which this element is connected.
    virtual NodePtr getConnectedNode() const;

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}

  public:
    static const string NODE_NAME_ATTRIBUTE;
    static const string NODE_GRAPH_ATTRIBUTE;
    static const string OUTPUT_ATTRIBUTE;
    static const string CHANNELS_ATTRIBUTE;

  private:
    static const std::unordered_map<string, CharSet> CHANNELS_CHARACTER_SET;
    static const std::unordered_map<string, size_t> CHANNELS_PATTERN_LENGTH;
};

/// @class Input
/// An input element within a Node or NodeDef.
///
/// An Input holds either a uniform value or a connection to a spatially-varying
/// Output, either of which may be modified within the scope of a Material.
class MX_CORE_API Input : public PortElement
{
  public:
    Input(ElementPtr parent, const string& name) :
        PortElement(parent, CATEGORY, name)
    {
    }
    virtual ~Input() { }

  public:
    /// @name Default Geometric Property
    /// @{

    /// Set the defaultgeomprop string for the input.
    void setDefaultGeomPropString(const string& geomprop)
    {
        setAttribute(DEFAULT_GEOM_PROP_ATTRIBUTE, geomprop);
    }

    /// Return true if the given input has a defaultgeomprop string.
    bool hasDefaultGeomPropString() const
    {
        return hasAttribute(DEFAULT_GEOM_PROP_ATTRIBUTE);
    }

    /// Return the defaultgeomprop string for the input.
    const string& getDefaultGeomPropString() const
    {
        return getAttribute(DEFAULT_GEOM_PROP_ATTRIBUTE);
    }

    /// Return the GeomPropDef element to use, if defined for this input.
    GeomPropDefPtr getDefaultGeomProp() const;

    /// @}
    /// @name Connections
    /// @{

    /// Return the node, if any, to which this input is connected.
    NodePtr getConnectedNode() const override;

    /// Return the input on the parent graph corresponding to the interface name
    /// for this input.
    InputPtr getInterfaceInput() const;

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}

  public:
    static const string CATEGORY;
    static const string DEFAULT_GEOM_PROP_ATTRIBUTE;
};

/// @class Output
/// A spatially-varying output element within a NodeGraph or NodeDef.
class MX_CORE_API Output : public PortElement
{
  public:
    Output(ElementPtr parent, const string& name) :
        PortElement(parent, CATEGORY, name)
    {
    }
    virtual ~Output() { }

  public:
    /// @name Traversal
    /// @{

    /// Return the Edge with the given index that lies directly upstream from
    /// this element in the dataflow graph.
    Edge getUpstreamEdge(size_t index = 0) const override;

    /// Return the number of queriable upstream edges for this element.
    size_t getUpstreamEdgeCount() const override
    {
        return 1;
    }

    /// Return true if a cycle exists in any upstream path from this element.
    bool hasUpstreamCycle() const;

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}

  public:
    static const string CATEGORY;
    static const string DEFAULT_INPUT_ATTRIBUTE;
};

/// @class InterfaceElement
/// The base class for interface elements such as Node, NodeDef, and NodeGraph.
///
/// An InterfaceElement supports a set of Input and Output elements, with an API
/// for setting their values.
class MX_CORE_API InterfaceElement : public TypedElement
{
  protected:
    InterfaceElement(ElementPtr parent, const string& category, const string& name) :
        TypedElement(parent, category, name),
        _inputCount(0),
        _outputCount(0)
    {
    }

  public:
    virtual ~InterfaceElement() { }

  protected:
    using NodeDefPtr = shared_ptr<NodeDef>;
    using ConstNodeDefPtr = shared_ptr<const NodeDef>;

  public:
    /// @name NodeDef String
    /// @{

    /// Set the NodeDef string for the interface.
    void setNodeDefString(const string& nodeDef)
    {
        setAttribute(NODE_DEF_ATTRIBUTE, nodeDef);
    }

    /// Return true if the given interface has a NodeDef string.
    bool hasNodeDefString() const
    {
        return hasAttribute(NODE_DEF_ATTRIBUTE);
    }

    /// Return the NodeDef string for the interface.
    const string& getNodeDefString() const
    {
        return getAttribute(NODE_DEF_ATTRIBUTE);
    }

    /// @}
    /// @name Inputs
    /// @{

    /// Add an Input to this interface.
    /// @param name The name of the new Input.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @param type An optional type string.
    /// @return A shared pointer to the new Input.
    InputPtr addInput(const string& name = EMPTY_STRING,
                      const string& type = DEFAULT_TYPE_STRING)
    {
        InputPtr child = addChild<Input>(name);
        child->setType(type);
        return child;
    }

    /// Return the Input, if any, with the given name.
    InputPtr getInput(const string& name) const
    {
        return getChildOfType<Input>(name);
    }

    /// Return a vector of all Input elements.
    vector<InputPtr> getInputs() const
    {
        return getChildrenOfType<Input>();
    }

    /// Return the number of Input elements.
    size_t getInputCount() const
    {
        return _inputCount;
    }

    /// Remove the Input, if any, with the given name.
    void removeInput(const string& name)
    {
        removeChildOfType<Input>(name);
    }

    /// Return the first Input with the given name that belongs to this
    /// interface, taking interface inheritance into account.
    InputPtr getActiveInput(const string& name) const;

    /// Return a vector of all Input elements that belong to this interface,
    /// taking inheritance into account.
    vector<InputPtr> getActiveInputs() const;

    /// @}
    /// @name Outputs
    /// @{

    /// Add an Output to this interface.
    /// @param name The name of the new Output.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @param type An optional type string.
    /// @return A shared pointer to the new Output.
    OutputPtr addOutput(const string& name = EMPTY_STRING,
                        const string& type = DEFAULT_TYPE_STRING)
    {
        OutputPtr output = addChild<Output>(name);
        output->setType(type);
        return output;
    }

    /// Return the Output, if any, with the given name.
    OutputPtr getOutput(const string& name) const
    {
        return getChildOfType<Output>(name);
    }

    /// Return a vector of all Output elements.
    vector<OutputPtr> getOutputs() const
    {
        return getChildrenOfType<Output>();
    }

    /// Return the number of Output elements.
    size_t getOutputCount() const
    {
        return _outputCount;
    }

    /// Remove the Output, if any, with the given name.
    void removeOutput(const string& name)
    {
        removeChildOfType<Output>(name);
    }

    /// Return the first Output with the given name that belongs to this
    /// interface, taking interface inheritance into account.
    OutputPtr getActiveOutput(const string& name) const;

    /// Return a vector of all Output elements that belong to this interface,
    /// taking inheritance into account.
    vector<OutputPtr> getActiveOutputs() const;

    /// Set the output to which the given input is connected, creating a
    /// child input if needed.  If the node argument is null, then any
    /// existing output connection on the input will be cleared.
    void setConnectedOutput(const string& inputName, OutputPtr output);

    /// Return the output connected to the given input.  If the given input is
    /// not present, then an empty OutputPtr is returned.
    OutputPtr getConnectedOutput(const string& inputName) const;

    /// @}
    /// @name Tokens
    /// @{

    /// Add a Token to this interface.
    /// @param name The name of the new Token.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Token.
    TokenPtr addToken(const string& name = EMPTY_STRING)
    {
        return addChild<Token>(name);
    }

    /// Return the Token, if any, with the given name.
    TokenPtr getToken(const string& name) const
    {
        return getChildOfType<Token>(name);
    }

    /// Return a vector of all Token elements.
    vector<TokenPtr> getTokens() const
    {
        return getChildrenOfType<Token>();
    }

    /// Remove the Token, if any, with the given name.
    void removeToken(const string& name)
    {
        removeChildOfType<Token>(name);
    }

    /// Return the first Token with the given name that belongs to this
    /// interface, taking interface inheritance into account.
    TokenPtr getActiveToken(const string& name) const;

    /// Return a vector of all Token elements that belong to this interface,
    /// taking inheritance into account.
    vector<TokenPtr> getActiveTokens() const;

    /// @}
    /// @name Value Elements
    /// @{

    /// Return the ValueElement, if any, with the given name.
    ValueElementPtr getValueElement(const string& name) const
    {
        return getChildOfType<ValueElement>(name);
    }

    /// Return the first value element with the given name that belongs to this
    /// interface, taking interface inheritance into account.
    /// Examples of value elements are Input, Output, and Token.
    ValueElementPtr getActiveValueElement(const string& name) const;

    /// Return a vector of all value elements that belong to this interface,
    /// taking inheritance into account.
    /// Examples of value elements are Input, Output, and Token.
    vector<ValueElementPtr> getActiveValueElements() const;

    /// @}
    /// @name Values
    /// @{

    /// Set the typed value of an input by its name, creating a child element
    /// to hold the input if needed.
    template <class T> InputPtr setInputValue(const string& name,
                                              const T& value,
                                              const string& type = EMPTY_STRING);

    /// Return the typed value of an input by its name, taking both the calling
    /// element and its declaration into account.
    /// @param name The name of the input to be evaluated.
    /// @param target An optional target name, which will be used to filter
    ///    the declarations that are considered.
    /// @return If the given input is found in this interface or its
    ///    declaration, then a shared pointer to its value is returned;
    ///    otherwise, an empty shared pointer is returned.
    ValuePtr getInputValue(const string& name, const string& target = EMPTY_STRING) const;

    /// Set the string value of a Token by its name, creating a child element
    /// to hold the Token if needed.
    TokenPtr setTokenValue(const string& name, const string& value)
    {
        TokenPtr token = getToken(name);
        if (!token)
            token = addToken(name);
        token->setValue<string>(value);
        return token;
    }

    /// Return the string value of a Token by its name, or an empty string if
    /// the given Token is not present.
    string getTokenValue(const string& name)
    {
        TokenPtr token = getToken(name);
        return token ? token->getValueString() : EMPTY_STRING;
    }

    /// @}
    /// @name Target
    /// @{

    /// Set the target string of this interface.
    void setTarget(const string& target)
    {
        setAttribute(TARGET_ATTRIBUTE, target);
    }

    /// Return true if the given interface has a target string.
    bool hasTarget() const
    {
        return hasAttribute(TARGET_ATTRIBUTE);
    }

    /// Return the target string of this interface.
    const string& getTarget() const
    {
        return getAttribute(TARGET_ATTRIBUTE);
    }

    /// @}
    /// @name Version
    /// @{

    /// Set the version string of this interface.
    void setVersionString(const string& version)
    {
        setAttribute(VERSION_ATTRIBUTE, version);
    }

    /// Return true if this interface has a version string.
    bool hasVersionString() const
    {
        return hasAttribute(VERSION_ATTRIBUTE);
    }

    /// Return the version string of this interface.
    const string& getVersionString() const
    {
        return getAttribute(VERSION_ATTRIBUTE);
    }

    /// Set the major and minor versions as an integer pair.
    void setVersionIntegers(int majorVersion, int minorVersion);

    /// Return the major and minor versions as an integer pair.
    virtual std::pair<int, int> getVersionIntegers() const;

    /// @}
    /// @name Default Version
    /// @{

    /// Set the default version flag of this element.
    void setDefaultVersion(bool defaultVersion)
    {
        setTypedAttribute<bool>(DEFAULT_VERSION_ATTRIBUTE, defaultVersion);
    }

    /// Return the default version flag of this element.
    bool getDefaultVersion() const
    {
        return getTypedAttribute<bool>(DEFAULT_VERSION_ATTRIBUTE);
    }

    /// @}
    /// @name Utility
    /// @{

    /// Return the first declaration of this interface, optionally filtered
    ///    by the given target name.
    /// @param target An optional target name, which will be used to filter
    ///    the declarations that are considered.
    /// @return A shared pointer to declaration, or an empty shared pointer if
    ///    no declaration was found.
    virtual ConstInterfaceElementPtr getDeclaration(const string& target = EMPTY_STRING) const;

    /// Clear all attributes and descendants from this element.
    void clearContent() override;

    /// Return true if this instance has an exact input match with the given
    /// declaration, where each input of this the instance corresponds to a
    /// declaration input of the same name and type.
    ///
    /// If an exact input match is not found, and the optional message argument
    /// is provided, then an error message will be appended to the given string.
    bool hasExactInputMatch(ConstInterfaceElementPtr declaration, string* message = nullptr) const;

    /// @}

  public:
    static const string NODE_DEF_ATTRIBUTE;
    static const string TARGET_ATTRIBUTE;
    static const string VERSION_ATTRIBUTE;
    static const string DEFAULT_VERSION_ATTRIBUTE;

  protected:
    void registerChildElement(ElementPtr child) override;
    void unregisterChildElement(ElementPtr child) override;

  private:
    size_t _inputCount;
    size_t _outputCount;
};

template <class T> InputPtr InterfaceElement::setInputValue(const string& name,
                                                            const T& value,
                                                            const string& type)
{
    InputPtr input = getChildOfType<Input>(name);
    if (!input)
        input = addInput(name);
    input->setValue(value, type);
    return input;
}

MATERIALX_NAMESPACE_END

#endif
