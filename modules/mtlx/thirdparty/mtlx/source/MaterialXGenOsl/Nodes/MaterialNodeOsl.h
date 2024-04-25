//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MATERIALNODEOSL_H
#define MATERIALX_MATERIALNODEOSL_H

#include <MaterialXGenOsl/Export.h>

#include <MaterialXGenShader/Nodes/MaterialNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Material node implementation for OSL
class MX_GENOSL_API MaterialNodeOsl : public MaterialNode
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
