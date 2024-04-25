//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SHADERNODEIMPL_H
#define MATERIALX_SHADERNODEIMPL_H

/// @file
/// Base class for shader node implementations

#include <MaterialXGenShader/Library.h>

#include <MaterialXGenShader/Export.h>
#include <MaterialXCore/Util.h>

MATERIALX_NAMESPACE_BEGIN

class InterfaceElement;
class Node;
using ShaderGraphInputSocket = ShaderOutput;

/// Shared pointer to a ShaderNodeImpl
using ShaderNodeImplPtr = shared_ptr<class ShaderNodeImpl>;

/// @class ShaderNodeImpl
/// Class handling the shader generation implementation for a node.
/// Responsible for emitting the function definition and function call
/// that is the node implementation.
class MX_GENSHADER_API ShaderNodeImpl
{
  public:
    virtual ~ShaderNodeImpl() { }

    /// Return an identifier for the target used by this implementation.
    /// By default an empty string is returned, representing all targets.
    /// Only override this method if your derived node implementation class
    /// is for a specific target.
    virtual const string& getTarget() const { return EMPTY_STRING; }

    /// Initialize with the given implementation element.
    /// Initialization must set the name and hash for the implementation,
    /// as well as any other data needed to emit code for the node.
    virtual void initialize(const InterfaceElement& element, GenContext& context);

    /// Return the name of this implementation.
    const string& getName() const
    {
        return _name;
    }

    /// Return a hash for this implementation.
    /// The hash should correspond to the function signature generated for the node,
    /// and can be used to compare implementations, e.g. to query if an identical
    /// function has already been emitted during shader generation.
    size_t getHash() const
    {
        return _hash;
    }

    /// Add additional inputs on a node.
    virtual void addInputs(ShaderNode& node, GenContext& context) const;

    /// Set values for additional inputs on a node.
    virtual void setValues(const Node& node, ShaderNode& shaderNode, GenContext& context) const;

    /// Add additional classifications on a node.
    virtual void addClassification(ShaderNode& node) const;

    /// Create shader variables needed for the implementation of this node (e.g. uniforms, inputs and outputs).
    /// Used if the node requires input data from the application.
    virtual void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const;

    /// Emit function definition for the given node instance.
    virtual void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;

    /// Emit the function call or inline source code for given node instance in the given context.
    virtual void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;

    /// Emit declaration and initialization of output variables to use in a function call.
    virtual void emitOutputVariables(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;

    /// Return a pointer to the graph if this implementation is using a graph,
    /// or returns nullptr otherwise.
    virtual ShaderGraph* getGraph() const;

    /// Returns true if an input is editable by users.
    /// Editable inputs are allowed to be published as shader uniforms
    /// and hence must be presentable in a user interface.
    /// By default all inputs are considered to be editable.
    virtual bool isEditable(const ShaderInput& /*input*/) const
    {
        return true;
    }

    /// Returns true if a graph input is accessible by users.
    /// Accessible inputs are allowed to be published as shader uniforms
    /// and hence must be presentable in a user interface.
    /// By default all graph inputs are considered to be acessible.
    virtual bool isEditable(const ShaderGraphInputSocket& /*input*/) const
    {
        return true;
    }

  protected:
    /// Protected constructor
    ShaderNodeImpl();

  protected:
    string _name;
    size_t _hash;
};

/// A no operation node, to be used for organizational nodes that has no code to execute.
class MX_GENSHADER_API NopNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();
};

MATERIALX_NAMESPACE_END

#endif
