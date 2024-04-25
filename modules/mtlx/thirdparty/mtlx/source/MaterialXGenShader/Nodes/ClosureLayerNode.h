//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSURELAYERNODE_H
#define MATERIALX_CLOSURELAYERNODE_H

#include <MaterialXGenShader/ShaderNodeImpl.h>

MATERIALX_NAMESPACE_BEGIN

/// Closure layer node implementation.
class MX_GENSHADER_API ClosureLayerNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    /// String constants
    static const string TOP;
    static const string BASE;
    static const string THICKNESS;
    static const string IOR;
};

MATERIALX_NAMESPACE_END

#endif
