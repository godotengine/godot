//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GEOMPROPVALUENODEMSL_H
#define MATERIALX_GEOMPROPVALUENODEMSL_H

#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// GeomPropValue node implementation for MSL
class MX_GENMSL_API GeomPropValueNodeMsl : public MslImplementation
{
  public:
    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    bool isEditable(const ShaderInput& /*input*/) const override { return false; }
};

/// GeomPropValue node non-implementation for MSL
class MX_GENMSL_API GeomPropValueNodeMslAsUniform : public MslImplementation
{
  public:
    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
