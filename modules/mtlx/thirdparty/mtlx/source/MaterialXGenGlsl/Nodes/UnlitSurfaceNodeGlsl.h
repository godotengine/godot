//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_UNLITSURFACENODEGLSL_H
#define MATERIALX_UNLITSURFACENODEGLSL_H

#include <MaterialXGenGlsl/Export.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Unlit surface node implementation for GLSL
class MX_GENGLSL_API UnlitSurfaceNodeGlsl : public GlslImplementation
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
