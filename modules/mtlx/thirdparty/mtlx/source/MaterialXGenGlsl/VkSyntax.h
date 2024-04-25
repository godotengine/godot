//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_VKSYNTAX_H
#define MATERIALX_VKSYNTAX_H

/// @file
/// Vulkan GLSL syntax class

#include <MaterialXGenGlsl/GlslSyntax.h>

MATERIALX_NAMESPACE_BEGIN

/// Syntax class for Vulkan GLSL
class MX_GENGLSL_API VkSyntax : public GlslSyntax
{
  public:
    VkSyntax();

    static SyntaxPtr create() { return std::make_shared<VkSyntax>(); }

    const string& getInputQualifier() const override { return INPUT_QUALIFIER; }
};

MATERIALX_NAMESPACE_END

#endif
