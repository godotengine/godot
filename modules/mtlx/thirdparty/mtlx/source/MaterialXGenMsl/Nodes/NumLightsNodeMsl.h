//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_NUMLIGHTSNODEMSL_H
#define MATERIALX_NUMLIGHTSNODEMSL_H

#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Utility node for getting number of active lights for MSL.
class MX_GENMSL_API NumLightsNodeMsl : public MslImplementation
{
  public:
    NumLightsNodeMsl();

    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
