//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SURFACENODEOSL_H
#define MATERIALX_SURFACENODEOSL_H

#include <MaterialXGenOsl/Export.h>

#include <MaterialXGenShader/Nodes/ClosureSourceCodeNode.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

/// Surface node implementation for OSL
class MX_GENOSL_API SurfaceNodeOsl : public ClosureSourceCodeNode
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  private:
    mutable ClosureContext _cct;
};

MATERIALX_NAMESPACE_END

#endif
