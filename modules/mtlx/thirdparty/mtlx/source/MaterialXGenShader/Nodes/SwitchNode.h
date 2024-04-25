//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SWITCHNODE_H
#define MATERIALX_SWITCHNODE_H

#include <MaterialXGenShader/ShaderNodeImpl.h>

MATERIALX_NAMESPACE_BEGIN

/// Switch node implementation
class MX_GENSHADER_API SwitchNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  public:
    static const StringVec INPUT_NAMES;
};

MATERIALX_NAMESPACE_END

#endif
