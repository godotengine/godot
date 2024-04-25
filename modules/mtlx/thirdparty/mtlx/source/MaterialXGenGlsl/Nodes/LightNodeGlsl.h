//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTNODEGLSL_H
#define MATERIALX_LIGHTNODEGLSL_H

#include <MaterialXGenGlsl/GlslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Light node implementation for GLSL
class MX_GENGLSL_API LightNodeGlsl : public GlslImplementation
{
  public:
    LightNodeGlsl();

    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  private:
    mutable ClosureContext _callEmission;
};

MATERIALX_NAMESPACE_END

#endif
