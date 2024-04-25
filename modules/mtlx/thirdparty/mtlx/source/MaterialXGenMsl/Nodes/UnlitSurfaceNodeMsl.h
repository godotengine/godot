//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_UNLITSURFACENODEMSL_H
#define MATERIALX_UNLITSURFACENODEMSL_H

#include <MaterialXGenMsl/Export.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Unlit surface node implementation for MSL
class MX_GENMSL_API UnlitSurfaceNodeMsl : public MslImplementation
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
