//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTSHADERNODEMSL_H
#define MATERIALX_LIGHTSHADERNODEMSL_H

#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

/// LightShader node implementation for MSL
/// Used for all light shaders implemented in source code.
class MX_GENMSL_API LightShaderNodeMsl : public SourceCodeNode
{
  public:
    LightShaderNodeMsl();

    static ShaderNodeImplPtr create();

    const string& getTarget() const override;

    void initialize(const InterfaceElement& element, GenContext& context) override;

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    VariableBlock _lightUniforms;
};

MATERIALX_NAMESPACE_END

#endif
