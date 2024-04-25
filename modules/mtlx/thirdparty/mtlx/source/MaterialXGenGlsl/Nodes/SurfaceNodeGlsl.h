//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SURFACENODEGLSL_H
#define MATERIALX_SURFACENODEGLSL_H

#include <MaterialXGenGlsl/Export.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Surface node implementation for GLSL
class MX_GENGLSL_API SurfaceNodeGlsl : public GlslImplementation
{
  public:
    SurfaceNodeGlsl();

    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    virtual void emitLightLoop(const ShaderNode& node, GenContext& context, ShaderStage& stage, const string& outColor) const;

  protected:
    /// Closure contexts for calling closure functions.
    mutable ClosureContext _callReflection;
    mutable ClosureContext _callTransmission;
    mutable ClosureContext _callIndirect;
    mutable ClosureContext _callEmission;
};

MATERIALX_NAMESPACE_END

#endif
