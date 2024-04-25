//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSURESOURCECODEMDL_H
#define MATERIALX_CLOSURESOURCECODEMDL_H

#include <MaterialXGenMdl/Nodes/SourceCodeNodeMdl.h>

MATERIALX_NAMESPACE_BEGIN

class MX_GENMDL_API ClosureSourceCodeNodeMdl : public SourceCodeNodeMdl
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
