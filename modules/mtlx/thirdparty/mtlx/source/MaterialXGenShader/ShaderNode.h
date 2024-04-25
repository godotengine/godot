//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SHADERNODE_H
#define MATERIALX_SHADERNODE_H

/// @file
/// Classes for nodes created during shader generation

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenShader/GenUserData.h>

#include <MaterialXCore/Node.h>

MATERIALX_NAMESPACE_BEGIN

class ShaderNode;
class ShaderPort;
class ShaderInput;
class ShaderOutput;
class ShaderGraph;

/// Shared pointer to a ShaderPort
using ShaderPortPtr = shared_ptr<class ShaderPort>;
/// Shared pointer to a ShaderInput
using ShaderInputPtr = shared_ptr<class ShaderInput>;
/// Shared pointer to a ShaderOutput
using ShaderOutputPtr = shared_ptr<class ShaderOutput>;
/// Shared pointer to a ShaderNode
using ShaderNodePtr = shared_ptr<class ShaderNode>;
/// A vector of ShaderInput pointers
using ShaderInputVec = vector<ShaderInput*>;

/// Metadata to be exported to generated shader.
struct MX_GENSHADER_API ShaderMetadata
{
    string name;
    const TypeDesc* type;
    ValuePtr value;
    ShaderMetadata(const string& n, const TypeDesc* t, ValuePtr v = nullptr) :
        name(n),
        type(t),
        value(v)
    {
    }
};
using ShaderMetadataVec = vector<ShaderMetadata>;
using ShaderMetadataVecPtr = shared_ptr<ShaderMetadataVec>;

/// @class ShaderMetadataRegistry
/// A registry for metadata that will be exported to the generated shader
/// if found on nodes and inputs during shader generation.
class MX_GENSHADER_API ShaderMetadataRegistry : public GenUserData
{
  public:
    static const string USER_DATA_NAME;

    /// Add a new metadata entry to the registry.
    /// The entry contains the name and data type
    /// for the metadata.
    void addMetadata(const string& name, const TypeDesc* type, ValuePtr value = nullptr)
    {
        if (_entryIndex.count(name) == 0)
        {
            _entryIndex[name] = _entries.size();
            _entries.emplace_back(name, type, value);
        }
    }

    /// Return the metadata registered for the given name, or nullptr
    /// if no such entry is found.
    const ShaderMetadata* findMetadata(const string& name) const
    {
        auto it = _entryIndex.find(name);
        return it != _entryIndex.end() ? &_entries[it->second] : nullptr;
    }

    /// Return the metadata registered for the given name, or nullptr
    /// if no such entry is found.
    ShaderMetadata* findMetadata(const string& name)
    {
        auto it = _entryIndex.find(name);
        return it != _entryIndex.end() ? &_entries[it->second] : nullptr;
    }

    /// Return all entries in the registry.
    const ShaderMetadataVec& getAllMetadata() const
    {
        return _entries;
    }

    /// Clear all entries in the registry.
    void clear()
    {
        _entryIndex.clear();
        _entries.clear();
    }

  protected:
    vector<ShaderMetadata> _entries;
    std::unordered_map<string, size_t> _entryIndex;
};

using ShaderMetadataRegistryPtr = shared_ptr<ShaderMetadataRegistry>;

/// Flags set on shader ports.
class MX_GENSHADER_API ShaderPortFlag
{
  public:
    static const uint32_t UNIFORM    = 1u << 0;
    static const uint32_t EMITTED    = 1u << 1;
    static const uint32_t BIND_INPUT = 1u << 2;
};

/// @class ShaderPort
/// An input or output port on a ShaderNode
class MX_GENSHADER_API ShaderPort : public std::enable_shared_from_this<ShaderPort>
{
  public:
    /// Constructor.
    ShaderPort(ShaderNode* node, const TypeDesc* type, const string& name, ValuePtr value = nullptr);

    /// Return a shared pointer instance of this object.
    ShaderPortPtr getSelf()
    {
        return shared_from_this();
    }

    /// Return the node this port belongs to.
    ShaderNode* getNode() { return _node; }

    /// Return the node this port belongs to.
    const ShaderNode* getNode() const { return _node; }

    /// Set the data type for this port.
    void setType(const TypeDesc* type) { _type = type; }

    /// Return the data type for this port.
    const TypeDesc* getType() const { return _type; }

    /// Set the name of this port.
    void setName(const string& name) { _name = name; }

    /// Return the name of this port.
    const string& getName() const { return _name; }

    /// Return the name of this port.
    string getFullName() const;

    /// Set the variable name of this port.
    void setVariable(const string& name) { _variable = name; }

    /// Return the variable name of this port.
    const string& getVariable() const { return _variable; }

    /// Set the variable semantic of this port.
    void setSemantic(const string& semantic) { _semantic = semantic; }

    /// Return the variable semantic of this port.
    const string& getSemantic() const { return _semantic; }

    /// Set a value on this port.
    void setValue(ValuePtr value) { _value = value; }

    /// Return the value set on this port.
    ValuePtr getValue() const { return _value; }

    /// Return the value set on this port as a string, or an empty string if there is no value.
    string getValueString() const;

    /// Set a source color space for the value on this port.
    void setColorSpace(const string& colorspace) { _colorspace = colorspace; }

    /// Return the source color space for the value on this port.
    const string& getColorSpace() const { return _colorspace; }

    /// Set a unit type for the value on this port.
    void setUnit(const string& unit) { _unit = unit; }

    /// Return the unit type for the value on this port.
    const string& getUnit() const { return _unit; }

    /// Set geomprop name if the input has a default
    /// geomprop to be assigned when it is unconnected.
    void setGeomProp(const string& geomprop) { _geomprop = geomprop; }

    /// Get geomprop name.
    const string& getGeomProp() const { return _geomprop; }

    /// Set the path to this port.
    void setPath(const string& path) { _path = path; }

    /// Return the path to this port.
    const string& getPath() const { return _path; }

    /// Set flags on this port.
    void setFlags(uint32_t flags) { _flags = flags; }

    /// Return flags set on this port.
    uint32_t getFlags() const { return _flags; }

    /// Set the on|off state of a given flag.
    void setFlag(uint32_t flag, bool value)
    {
        _flags = value ? (_flags | flag) : (_flags & ~flag);
    }

    /// Return the on|off state of a given flag.
    bool getFlag(uint32_t flag) const
    {
        return ((_flags & flag) != 0);
    }

    /// Set the uniform flag this port to true.
    void setUniform() { _flags |= ShaderPortFlag::UNIFORM; }

    /// Return the uniform flag on this port.
    bool isUniform() const { return (_flags & ShaderPortFlag::UNIFORM) != 0; }

    /// Set the emitted state on this port to true.
    void setEmitted() { _flags |= ShaderPortFlag::EMITTED; }

    /// Return the emitted state of this port.
    bool isEmitted() const { return (_flags & ShaderPortFlag::EMITTED) != 0; }

    /// Set the bind input state on this port to true.
    void setBindInput() { _flags |= ShaderPortFlag::BIND_INPUT; }

    /// Return the emitted state of this port.
    bool isBindInput() const { return (_flags & ShaderPortFlag::BIND_INPUT) != 0; }

    /// Set the metadata vector.
    void setMetadata(ShaderMetadataVecPtr metadata) { _metadata = metadata; }

    /// Get the metadata vector.
    ShaderMetadataVecPtr getMetadata() { return _metadata; }

    /// Get the metadata vector.
    const ShaderMetadataVecPtr& getMetadata() const { return _metadata; }

  protected:
    ShaderNode* _node;
    const TypeDesc* _type;
    string _name;
    string _path;
    string _semantic;
    string _variable;
    ValuePtr _value;
    string _unit;
    string _colorspace;
    string _geomprop;
    ShaderMetadataVecPtr _metadata;
    uint32_t _flags;
};

/// @class ShaderInput
/// An input on a ShaderNode
class MX_GENSHADER_API ShaderInput : public ShaderPort
{
  public:
    ShaderInput(ShaderNode* node, const TypeDesc* type, const string& name);

    /// Return a connection to an upstream node output,
    /// or nullptr if not connected.
    ShaderOutput* getConnection() { return _connection; }

    /// Return a connection to an upstream node output,
    /// or nullptr if not connected.
    const ShaderOutput* getConnection() const { return _connection; }

    /// Make a connection from the given source output to this input.
    void makeConnection(ShaderOutput* src);

    /// Break the connection to this input.
    void breakConnection();

    /// Set optional channels value
    void setChannels(const string& channels) { _channels = channels; }

    /// Get optional channels value
    const string& getChannels() const { return _channels; }

    /// Return the sibling node connected upstream,
    /// or nullptr if there is no sibling upstream.
    ShaderNode* getConnectedSibling() const;

  protected:
    ShaderOutput* _connection;
    string _channels;
    friend class ShaderOutput;
};

/// @class ShaderOutput
/// An output on a ShaderNode
class MX_GENSHADER_API ShaderOutput : public ShaderPort
{
  public:
    ShaderOutput(ShaderNode* node, const TypeDesc* type, const string& name);

    /// Return a set of connections to downstream node inputs,
    /// empty if not connected.
    const ShaderInputVec& getConnections() const { return _connections; }

    /// Make a connection from this output to the given input
    void makeConnection(ShaderInput* dst);

    /// Break a connection from this output to the given input
    void breakConnection(ShaderInput* dst);

    /// Break all connections from this output
    void breakConnections();

  protected:
    ShaderInputVec _connections;
    friend class ShaderInput;
};

/// @class ShaderNode
/// Class representing a node in the shader generation DAG
class MX_GENSHADER_API ShaderNode
{
  public:
    virtual ~ShaderNode() { }

    /// Flags for classifying nodes into different categories.
    class Classification
    {
      public:
        // Node classes
        static const uint32_t TEXTURE       = 1 << 0;  /// Any node that outputs floats, colors, vectors, etc.
        static const uint32_t CLOSURE       = 1 << 1;  /// Any node that represents light integration
        static const uint32_t SHADER        = 1 << 2;  /// Any node that outputs a shader
        static const uint32_t MATERIAL      = 1 << 3;  /// Any node that outputs a material
        // Specific texture node types
        static const uint32_t FILETEXTURE   = 1 << 4;  /// A file texture node
        static const uint32_t CONDITIONAL   = 1 << 5;  /// A conditional node
        static const uint32_t CONSTANT      = 1 << 6;  /// A constant node
        // Specific closure types
        static const uint32_t BSDF          = 1 << 7;  /// A BSDF node
        static const uint32_t BSDF_R        = 1 << 8;  /// A reflection BSDF node
        static const uint32_t BSDF_T        = 1 << 9;  /// A transmission BSDF node
        static const uint32_t EDF           = 1 << 10; /// A EDF node
        static const uint32_t VDF           = 1 << 11; /// A VDF node
        static const uint32_t LAYER         = 1 << 12; /// A node for vertical layering of other closure nodes
        static const uint32_t THINFILM      = 1 << 13; /// A node for adding thin-film over microfacet BSDF nodes
        // Specific shader types
        static const uint32_t SURFACE       = 1 << 14; /// A surface shader node
        static const uint32_t VOLUME        = 1 << 15; /// A volume shader node
        static const uint32_t LIGHT         = 1 << 16; /// A light shader node
        static const uint32_t UNLIT         = 1 << 17; /// An unlit surface shader node
        // Types based on nodegroup
        static const uint32_t SAMPLE2D      = 1 << 18; /// Can be sampled in 2D (uv space)
        static const uint32_t SAMPLE3D      = 1 << 19; /// Can be sampled in 3D (position)
        static const uint32_t GEOMETRIC     = 1 << 20; /// Geometric input
        static const uint32_t DOT           = 1 << 21; /// A dot node
    };

    static const ShaderNodePtr NONE;

    static const string CONSTANT;
    static const string DOT;
    static const string IMAGE;
    static const string SURFACESHADER;
    static const string SCATTER_MODE;
    static const string BSDF_R;
    static const string BSDF_T;
    static const string TRANSFORM_POINT;
    static const string TRANSFORM_VECTOR;
    static const string TRANSFORM_NORMAL;
    static const string TEXTURE2D_GROUPNAME;
    static const string TEXTURE3D_GROUPNAME;
    static const string PROCEDURAL2D_GROUPNAME;
    static const string PROCEDURAL3D_GROUPNAME;
    static const string GEOMETRIC_GROUPNAME;

  public:
    /// Constructor.
    ShaderNode(const ShaderGraph* parent, const string& name);

    /// Create a new node from a nodedef.
    static ShaderNodePtr create(const ShaderGraph* parent, const string& name, const NodeDef& nodeDef,
                                GenContext& context);

    /// Create a new node from a node implementation.
    static ShaderNodePtr create(const ShaderGraph* parent, const string& name, ShaderNodeImplPtr impl,
                                unsigned int classification = Classification::TEXTURE);

    /// Return true if this node is a graph.
    virtual bool isAGraph() const { return false; }

    /// Return the parent graph that owns this node.
    /// If this node is a root graph it has no parent
    /// and nullptr will be returned.
    const ShaderGraph* getParent() const
    {
        return _parent;
    }

    /// Set classification bits for this node,
    /// replacing any previous set bits.
    void setClassification(uint32_t c)
    {
        _classification = c;
    }

    /// Get classification bits set for this node.
    uint32_t getClassification() const
    {
        return _classification;
    }

    /// Add classification bits to this node.
    void addClassification(uint32_t c)
    {
        _classification |= c;
    }

    /// Return true if this node matches the given classification.
    bool hasClassification(uint32_t c) const
    {
        return (_classification & c) == c;
    }

    /// Return the name of this node.
    const string& getName() const
    {
        return _name;
    }

    /// Return the implementation used for this node.
    const ShaderNodeImpl& getImplementation() const
    {
        return *_impl;
    }

    /// Initialize this shader node with all required data
    /// from the given node and nodedef.
    void initialize(const Node& node, const NodeDef& nodeDef, GenContext& context);

    /// Add inputs/outputs
    ShaderInput* addInput(const string& name, const TypeDesc* type);
    ShaderOutput* addOutput(const string& name, const TypeDesc* type);

    /// Get number of inputs/outputs
    size_t numInputs() const { return _inputOrder.size(); }
    size_t numOutputs() const { return _outputOrder.size(); }

    /// Get inputs/outputs by index
    ShaderInput* getInput(size_t index) { return _inputOrder[index]; }
    ShaderOutput* getOutput(size_t index = 0) { return _outputOrder[index]; }
    const ShaderInput* getInput(size_t index) const { return _inputOrder[index]; }
    const ShaderOutput* getOutput(size_t index = 0) const { return _outputOrder[index]; }

    /// Get inputs/outputs by name
    ShaderInput* getInput(const string& name);
    ShaderOutput* getOutput(const string& name);
    const ShaderInput* getInput(const string& name) const;
    const ShaderOutput* getOutput(const string& name) const;

    /// Get vector of inputs/outputs
    const vector<ShaderInput*>& getInputs() const { return _inputOrder; }
    const vector<ShaderOutput*>& getOutputs() const { return _outputOrder; }

    /// Set the metadata vector.
    void setMetadata(ShaderMetadataVecPtr metadata) { _metadata = metadata; }

    /// Get the metadata vector.
    ShaderMetadataVecPtr getMetadata() { return _metadata; }

    /// Get the metadata vector.
    const ShaderMetadataVecPtr& getMetadata() const { return _metadata; }

    /// Returns true if an input is editable by users.
    /// Editable inputs are allowed to be published as shader uniforms
    /// and hence must be presentable in a user interface.
    bool isEditable(const ShaderInput& input) const
    {
        return (!_impl || _impl->isEditable(input));
    }

    /// Returns true if a graph input is accessible by users.
    /// Editable inputs are allowed to be published as shader uniforms
    /// and hence must be presentable in a user interface.
    bool isEditable(const ShaderGraphInputSocket& input) const
    {
        return (!_impl || _impl->isEditable(input));
    }

  protected:
    /// Create metadata from the nodedef according to registered metadata.
    void createMetadata(const NodeDef& nodeDef, GenContext& context);

    const ShaderGraph* _parent;
    string _name;
    uint32_t _classification;

    std::unordered_map<string, ShaderInputPtr> _inputMap;
    vector<ShaderInput*> _inputOrder;

    std::unordered_map<string, ShaderOutputPtr> _outputMap;
    vector<ShaderOutput*> _outputOrder;

    ShaderNodeImplPtr _impl;
    ShaderMetadataVecPtr _metadata;

    friend class ShaderGraph;
};

MATERIALX_NAMESPACE_END

#endif
