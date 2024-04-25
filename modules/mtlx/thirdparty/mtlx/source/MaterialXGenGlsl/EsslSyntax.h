//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_ESSLSYNTAX_H
#define MATERIALX_ESSLSYNTAX_H

/// @file
/// ESSL syntax class

#include <MaterialXGenGlsl/GlslSyntax.h>

MATERIALX_NAMESPACE_BEGIN

/// Syntax class for ESSL (OpenGL ES Shading Language)
class MX_GENGLSL_API EsslSyntax : public GlslSyntax
{
  public:
    EsslSyntax();

    static SyntaxPtr create() { return std::make_shared<EsslSyntax>(); }
};

MATERIALX_NAMESPACE_END

#endif
