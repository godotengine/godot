//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTCOMPOUNDNODEGLSL_H
#define MATERIALX_LIGHTCOMPOUNDNODEGLSL_H

#include <MaterialXGenGlsl/Export.h>

#include <MaterialXGenShader/Nodes/CompoundNode.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

class GlslShaderGenerator;

/// LightCompound node implementation for GLSL
class MX_GENGLSL_API LightCompoundNodeGlsl : public CompoundNode
{
  public:
    LightCompoundNodeGlsl();

    static ShaderNodeImplPtr create();

    const string& getTarget() const override;

    void initialize(const InterfaceElement& element, GenContext& context) override;

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    void emitFunctionDefinition(ClosureContext* cct, GenContext& context, ShaderStage& stage) const;

    VariableBlock _lightUniforms;
};

MATERIALX_NAMESPACE_END

#endif
