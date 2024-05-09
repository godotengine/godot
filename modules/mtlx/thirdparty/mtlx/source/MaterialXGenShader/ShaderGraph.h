//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SHADERGRAPH_H
#define MATERIALX_SHADERGRAPH_H

/// @file
/// Shader graph class

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/ColorManagementSystem.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenShader/UnitSystem.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>

MATERIALX_NAMESPACE_BEGIN

class Syntax;
class ShaderGraphEdge;
class ShaderGraphEdgeIterator;
class GenOptions;

/// An internal input socket in a shader graph,
/// used for connecting internal nodes to the outside
using ShaderGraphInputSocket = ShaderOutput;

/// An internal output socket in a shader graph,
/// used for connecting internal nodes to the outside
using ShaderGraphOutputSocket = ShaderInput;

/// A shared pointer to a shader graph
using ShaderGraphPtr = shared_ptr<class ShaderGraph>;

/// @class ShaderGraph
/// Class representing a graph (DAG) for shader generation
class MX_GENSHADER_API ShaderGraph : public ShaderNode
{
  public:
    /// Constructor.
    ShaderGraph(const ShaderGraph* parent, const string& name, ConstDocumentPtr document, const StringSet& reservedWords);

    /// Desctructor.
    virtual ~ShaderGraph() { }

    /// Create a new shader graph from an element.
    /// Supported elements are outputs and shader nodes.
    static ShaderGraphPtr create(const ShaderGraph* parent, const string& name, ElementPtr element,
                                 GenContext& context);

    /// Create a new shader graph from a nodegraph.
    static ShaderGraphPtr create(const ShaderGraph* parent, const NodeGraph& nodeGraph,
                                 GenContext& context);

    /// Return true if this node is a graph.
    bool isAGraph() const override { return true; }

    /// Get an internal node by name
    ShaderNode* getNode(const string& name);

    /// Get an internal node by name
    const ShaderNode* getNode(const string& name) const;

    /// Get a vector of all nodes in order
    const vector<ShaderNode*>& getNodes() const { return _nodeOrder; }

    /// Get number of input sockets
    size_t numInputSockets() const { return numOutputs(); }

    /// Get number of output sockets
    size_t numOutputSockets() const { return numInputs(); }

    /// Get socket by index
    ShaderGraphInputSocket* getInputSocket(size_t index) { return getOutput(index); }
    ShaderGraphOutputSocket* getOutputSocket(size_t index = 0) { return getInput(index); }
    const ShaderGraphInputSocket* getInputSocket(size_t index) const { return getOutput(index); }
    const ShaderGraphOutputSocket* getOutputSocket(size_t index = 0) const { return getInput(index); }

    /// Get socket by name
    ShaderGraphInputSocket* getInputSocket(const string& name) { return getOutput(name); }
    ShaderGraphOutputSocket* getOutputSocket(const string& name) { return getInput(name); }
    const ShaderGraphInputSocket* getInputSocket(const string& name) const { return getOutput(name); }
    const ShaderGraphOutputSocket* getOutputSocket(const string& name) const { return getInput(name); }

    /// Get vector of sockets
    const vector<ShaderGraphInputSocket*>& getInputSockets() const { return _outputOrder; }
    const vector<ShaderGraphOutputSocket*>& getOutputSockets() const { return _inputOrder; }

    /// Apply color and unit transforms to each input of a node.
    void applyInputTransforms(ConstNodePtr node, ShaderNodePtr shaderNode, GenContext& context);

    /// Create a new node in the graph
    ShaderNode* createNode(ConstNodePtr node, GenContext& context);

    /// Add input/output sockets
    ShaderGraphInputSocket* addInputSocket(const string& name, const TypeDesc* type);
    ShaderGraphOutputSocket* addOutputSocket(const string& name, const TypeDesc* type);

    /// Add a default geometric node and connect to the given input.
    void addDefaultGeomNode(ShaderInput* input, const GeomPropDef& geomprop, GenContext& context);

    /// Sort the nodes in topological order.
    void topologicalSort();

    /// Return an iterator for traversal upstream from the given output
    static ShaderGraphEdgeIterator traverseUpstream(ShaderOutput* output);

    /// Return the map of unique identifiers used in the scope of this graph.
    IdentifierMap& getIdentifierMap() { return _identifiers; }

  protected:
    /// Create node connections corresponding to the connection between a pair of elements.
    /// @param downstreamElement Element representing the node to connect to.
    /// @param upstreamElement Element representing the node to connect from
    /// @param connectingElement If non-null, specifies the element on on the downstream node to connect to.
    /// @param context Context for generation.
    void createConnectedNodes(const ElementPtr& downstreamElement,
                              const ElementPtr& upstreamElement,
                              ElementPtr connectingElement,
                              GenContext& context);

    /// Add a node to the graph
    void addNode(ShaderNodePtr node);

    /// Add input sockets from an interface element (nodedef, nodegraph or node)
    void addInputSockets(const InterfaceElement& elem, GenContext& context);

    /// Add output sockets from an interface element (nodedef, nodegraph or node)
    void addOutputSockets(const InterfaceElement& elem);

    /// Traverse from the given root element and add all dependencies upstream.
    /// The traversal is done in the context of a material, if given, to include
    /// bind input elements in the traversal.
    void addUpstreamDependencies(const Element& root, GenContext& context);

    /// Add a color transform node and connect to the given input.
    void addColorTransformNode(ShaderInput* input, const ColorSpaceTransform& transform, GenContext& context);

    /// Add a color transform node and connect to the given output.
    void addColorTransformNode(ShaderOutput* output, const ColorSpaceTransform& transform, GenContext& context);

    /// Add a unit transform node and connect to the given input.
    void addUnitTransformNode(ShaderInput* input, const UnitTransform& transform, GenContext& context);

    /// Add a unit transform node and connect to the given output.
    void addUnitTransformNode(ShaderOutput* output, const UnitTransform& transform, GenContext& context);

    /// Perform all post-build operations on the graph.
    void finalize(GenContext& context);

    /// Optimize the graph, removing redundant paths.
    void optimize(GenContext& context);

    /// Bypass a node for a particular input and output,
    /// effectively connecting the input's upstream connection
    /// with the output's downstream connections.
    void bypass(GenContext& context, ShaderNode* node, size_t inputIndex, size_t outputIndex = 0);

    /// For inputs and outputs in the graph set the variable names to be used
    /// in generated code. Making sure variable names are valid and unique
    /// to avoid name conflicts during shader generation.
    void setVariableNames(GenContext& context);

    /// Populate the color transform map for the given shader port, if the provided combination of
    /// source and target color spaces are supported for its data type.
    void populateColorTransformMap(ColorManagementSystemPtr colorManagementSystem, ShaderPort* shaderPort,
                                   const string& sourceColorSpace, const string& targetColorSpace, bool asInput);

    /// Populates the appropriate unit transform map if the provided input/parameter or output
    /// has a unit attribute and is of the supported type
    void populateUnitTransformMap(UnitSystemPtr unitSystem, ShaderPort* shaderPort, ValueElementPtr element, const string& targetUnitSpace, bool asInput);

    /// Break all connections on a node
    void disconnect(ShaderNode* node) const;

    ConstDocumentPtr _document;
    std::unordered_map<string, ShaderNodePtr> _nodeMap;
    std::vector<ShaderNode*> _nodeOrder;
    IdentifierMap _identifiers;

    // Temporary storage for inputs that require color transformations
    std::unordered_map<ShaderInput*, ColorSpaceTransform> _inputColorTransformMap;
    // Temporary storage for inputs that require unit transformations
    std::unordered_map<ShaderInput*, UnitTransform> _inputUnitTransformMap;

    // Temporary storage for outputs that require color transformations
    std::unordered_map<ShaderOutput*, ColorSpaceTransform> _outputColorTransformMap;
    // Temporary storage for outputs that require unit transformations
    std::unordered_map<ShaderOutput*, UnitTransform> _outputUnitTransformMap;
};

/// @class ShaderGraphEdge
/// An edge returned during shader graph traversal.
class MX_GENSHADER_API ShaderGraphEdge
{
  public:
    ShaderGraphEdge(ShaderOutput* up, ShaderInput* down) :
        upstream(up),
        downstream(down)
    {
    }
    ShaderOutput* upstream;
    ShaderInput* downstream;
};

/// @class ShaderGraphEdgeIterator
/// Iterator class for traversing edges between nodes in a shader graph.
class MX_GENSHADER_API ShaderGraphEdgeIterator
{
  public:
    ShaderGraphEdgeIterator(ShaderOutput* output);
    ~ShaderGraphEdgeIterator() { }

    bool operator==(const ShaderGraphEdgeIterator& rhs) const
    {
        return _upstream == rhs._upstream &&
               _downstream == rhs._downstream &&
               _stack == rhs._stack;
    }
    bool operator!=(const ShaderGraphEdgeIterator& rhs) const
    {
        return !(*this == rhs);
    }

    /// Dereference this iterator, returning the current output in the traversal.
    ShaderGraphEdge operator*() const
    {
        return ShaderGraphEdge(_upstream, _downstream);
    }

    /// Iterate to the next edge in the traversal.
    /// @throws ExceptionFoundCycle if a cycle is encountered.
    ShaderGraphEdgeIterator& operator++();

    /// Return a reference to this iterator to begin traversal
    ShaderGraphEdgeIterator& begin()
    {
        return *this;
    }

    /// Return the end iterator.
    static const ShaderGraphEdgeIterator& end();

  private:
    void extendPathUpstream(ShaderOutput* upstream, ShaderInput* downstream);
    void returnPathDownstream(ShaderOutput* upstream);

    ShaderOutput* _upstream;
    ShaderInput* _downstream;
    using StackFrame = std::pair<ShaderOutput*, size_t>;
    std::vector<StackFrame> _stack;
    std::set<ShaderOutput*> _path;
};

MATERIALX_NAMESPACE_END

#endif
