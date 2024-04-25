//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GEOMCOLORNODEMSL_H
#define MATERIALX_GEOMCOLORNODEMSL_H

#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// GeomColor node implementation for MSL
class MX_GENMSL_API GeomColorNodeMsl : public MslImplementation
{
  public:
    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
