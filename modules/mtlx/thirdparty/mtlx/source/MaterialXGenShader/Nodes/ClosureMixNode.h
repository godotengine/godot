//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSUREMIXNODE_H
#define MATERIALX_CLOSUREMIXNODE_H

#include <MaterialXGenShader/ShaderNodeImpl.h>

MATERIALX_NAMESPACE_BEGIN

/// Closure mix node implementation.
class MX_GENSHADER_API ClosureMixNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    /// String constants
    static const string FG;
    static const string BG;
    static const string MIX;
};

MATERIALX_NAMESPACE_END

#endif
