//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSURECOMPOUNDNODEMDL_H
#define MATERIALX_CLOSURECOMPOUNDNODEMDL_H

#include <MaterialXGenMdl/Export.h>
#include <MaterialXGenMdl/Nodes/CompoundNodeMdl.h>

MATERIALX_NAMESPACE_BEGIN

/// Compound node implementation
class MX_GENMDL_API ClosureCompoundNodeMdl : public CompoundNodeMdl
{
  public:
    static ShaderNodeImplPtr create();

    void addClassification(ShaderNode& node) const override;
    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
