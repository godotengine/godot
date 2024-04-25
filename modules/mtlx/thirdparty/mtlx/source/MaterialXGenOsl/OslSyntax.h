//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_OSLSYNTAX_H
#define MATERIALX_OSLSYNTAX_H

/// @file
/// OSL syntax class

#include <MaterialXGenOsl/Export.h>

#include <MaterialXGenShader/Syntax.h>

MATERIALX_NAMESPACE_BEGIN

/// @class OslSyntax
/// Syntax class for OSL (Open Shading Language)
class MX_GENOSL_API OslSyntax : public Syntax
{
  public:
    OslSyntax();

    static SyntaxPtr create() { return std::make_shared<OslSyntax>(); }

    const string& getOutputQualifier() const override;
    const string& getConstantQualifier() const override { return EMPTY_STRING; };
    const string& getSourceFileExtension() const override { return SOURCE_FILE_EXTENSION; };

    static const string OUTPUT_QUALIFIER;
    static const string SOURCE_FILE_EXTENSION;
    static const StringVec VECTOR_MEMBERS;
    static const StringVec VECTOR2_MEMBERS;
    static const StringVec VECTOR4_MEMBERS;
    static const StringVec COLOR4_MEMBERS;
};

MATERIALX_NAMESPACE_END

#endif
