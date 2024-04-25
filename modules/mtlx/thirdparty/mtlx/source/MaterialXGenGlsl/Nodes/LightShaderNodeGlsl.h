//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTSHADERNODEGLSL_H
#define MATERIALX_LIGHTSHADERNODEGLSL_H

#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

/// LightShader node implementation for GLSL
/// Used for all light shaders implemented in source code.
class MX_GENGLSL_API LightShaderNodeGlsl : public SourceCodeNode
{
  public:
    LightShaderNodeGlsl();

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
