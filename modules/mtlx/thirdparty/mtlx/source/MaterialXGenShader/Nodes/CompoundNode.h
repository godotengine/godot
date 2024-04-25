//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_COMPOUNDNODE_H
#define MATERIALX_COMPOUNDNODE_H

#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/Shader.h>

MATERIALX_NAMESPACE_BEGIN

/// Compound node implementation
class MX_GENSHADER_API CompoundNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void initialize(const InterfaceElement& element, GenContext& context) override;

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    ShaderGraph* getGraph() const override { return _rootGraph.get(); }

  protected:
    ShaderGraphPtr _rootGraph;
    string _functionName;
};

MATERIALX_NAMESPACE_END

#endif
