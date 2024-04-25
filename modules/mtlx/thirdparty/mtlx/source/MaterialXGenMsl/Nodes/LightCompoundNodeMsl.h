//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTCOMPOUNDNODEMSL_H
#define MATERIALX_LIGHTCOMPOUNDNODEMSL_H

#include <MaterialXGenMsl/Export.h>

#include <MaterialXGenShader/Nodes/CompoundNode.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

class MslShaderGenerator;

/// LightCompound node implementation for MSL
class MX_GENMSL_API LightCompoundNodeMsl : public CompoundNode
{
  public:
    LightCompoundNodeMsl();

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
