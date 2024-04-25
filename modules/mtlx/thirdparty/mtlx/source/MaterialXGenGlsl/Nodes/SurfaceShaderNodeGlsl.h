//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SURFACESHADERNODEGLSL_H
#define MATERIALX_SURFACESHADERNODEGLSL_H

#include <MaterialXGenGlsl/Export.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

/// SurfaceShader node implementation for GLSL
/// Used for all surface shaders implemented in source code.
class MX_GENGLSL_API SurfaceShaderNodeGlsl : public SourceCodeNode
{
  public:
    static ShaderNodeImplPtr create();

    const string& getTarget() const override;

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
