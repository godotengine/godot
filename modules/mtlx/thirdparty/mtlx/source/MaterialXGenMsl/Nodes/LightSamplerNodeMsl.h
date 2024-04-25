//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIGHTSAMPLERNODEMSL_H
#define MATERIALX_LIGHTSAMPLERNODEMSL_H

#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Utility node for sampling lights for MSL.
class MX_GENMSL_API LightSamplerNodeMsl : public MslImplementation
{
  public:
    LightSamplerNodeMsl();

    static ShaderNodeImplPtr create();

    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
