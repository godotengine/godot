//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MATERIALNODEMDL_H
#define MATERIALX_MATERIALNODEMDL_H

#include <MaterialXGenMdl/Export.h>

#include <MaterialXGenShader/Nodes/MaterialNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Material node implementation for MDL
class MX_GENMDL_API MaterialNodeMdl : public MaterialNode
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
