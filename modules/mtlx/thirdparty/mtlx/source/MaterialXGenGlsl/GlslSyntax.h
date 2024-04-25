//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GLSL_SYNTAX_H
#define MATERIALX_GLSL_SYNTAX_H

/// @file
/// GLSL syntax class

#include <MaterialXGenGlsl/Export.h>

#include <MaterialXGenShader/Syntax.h>

MATERIALX_NAMESPACE_BEGIN

/// Syntax class for GLSL (OpenGL Shading Language)
class MX_GENGLSL_API GlslSyntax : public Syntax
{
  public:
    GlslSyntax();

    static SyntaxPtr create() { return std::make_shared<GlslSyntax>(); }

    const string& getInputQualifier() const override { return INPUT_QUALIFIER; }
    const string& getOutputQualifier() const override { return OUTPUT_QUALIFIER; }
    const string& getConstantQualifier() const override { return CONSTANT_QUALIFIER; };
    const string& getUniformQualifier() const override { return UNIFORM_QUALIFIER; };
    const string& getSourceFileExtension() const override { return SOURCE_FILE_EXTENSION; };

    bool typeSupported(const TypeDesc* type) const override;

    /// Given an input specification attempt to remap this to an enumeration which is accepted by
    /// the shader generator. The enumeration may be converted to a different type than the input.
    bool remapEnumeration(const string& value, const TypeDesc* type, const string& enumNames, std::pair<const TypeDesc*, ValuePtr>& result) const override;

    static const string INPUT_QUALIFIER;
    static const string OUTPUT_QUALIFIER;
    static const string UNIFORM_QUALIFIER;
    static const string CONSTANT_QUALIFIER;
    static const string FLAT_QUALIFIER;
    static const string SOURCE_FILE_EXTENSION;

    static const StringVec VEC2_MEMBERS;
    static const StringVec VEC3_MEMBERS;
    static const StringVec VEC4_MEMBERS;
};

MATERIALX_NAMESPACE_END

#endif
