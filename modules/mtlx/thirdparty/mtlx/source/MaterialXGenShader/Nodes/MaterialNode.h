//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MATERIALNODE_H
#define MATERIALX_MATERIALNODE_H

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/ShaderNodeImpl.h>

MATERIALX_NAMESPACE_BEGIN

/// Material node implementation
class MX_GENSHADER_API MaterialNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void addClassification(ShaderNode& node) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
