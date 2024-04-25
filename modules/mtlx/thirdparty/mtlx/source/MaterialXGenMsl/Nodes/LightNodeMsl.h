//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTNODEMSL_H
#define MATERIALX_LIGHTNODEMSL_H

#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Light node implementation for MSL
class MX_GENMSL_API LightNodeMsl : public MslImplementation
{
  public:
    LightNodeMsl();

    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  private:
    mutable ClosureContext _callEmission;
};

MATERIALX_NAMESPACE_END

#endif
