//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSURESOURCECODENODE_H
#define MATERIALX_CLOSURESOURCECODENODE_H

#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

/// @class ClosureSourceCodeNode
/// Implemention for a closure node using data-driven static source code.
class MX_GENSHADER_API ClosureSourceCodeNode : public SourceCodeNode
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
