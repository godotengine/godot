//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MDLSHADERGENERATOR_H
#define MATERIALX_MDLSHADERGENERATOR_H

/// @file
/// MDL shading language generator

#include <MaterialXGenMdl/Export.h>

#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Generator context data class to pass strings.
class MX_GENMDL_API GenMdlOptions : public GenUserData
{
  public:
    /// MDL Versions supported by the Code Generator
    enum class MdlVersion
    {
        MDL_1_6,
        MDL_1_7,
        MDL_1_8,
        MDL_LATEST = MDL_1_8
    };

    /// Create MDL code generator options with default values.
    GenMdlOptions() :
        targetVersion(MdlVersion::MDL_LATEST) { }

    /// Unique identifier for the MDL options on the GenContext object.
    static const string GEN_CONTEXT_USER_DATA_KEY;

    /// The MDL version number the generated module will have.
    /// Allows to generate MDL for older applications by limiting support according
    /// to the corresponding specification. By default this option is MDL_LATEST.
    MdlVersion targetVersion;
};

/// Shared pointer to GenMdlOptions
using GenMdlOptionsPtr = shared_ptr<class GenMdlOptions>;

/// Shared pointer to an MdlShaderGenerator
using MdlShaderGeneratorPtr = shared_ptr<class MdlShaderGenerator>;

/// @class MdlShaderGenerator
/// Shader generator for MDL (Material Definition Language).
class MX_GENMDL_API MdlShaderGenerator : public ShaderGenerator
{
  public:
    MdlShaderGenerator();

    static ShaderGeneratorPtr create() { return std::make_shared<MdlShaderGenerator>(); }

    /// Return a unique identifier for the target this generator is for
    const string& getTarget() const override { return TARGET; }

    /// Generate a shader starting from the given element, translating
    /// the element and all dependencies upstream into shader code.
    ShaderPtr generate(const string& name, ElementPtr element, GenContext& context) const override;

    /// Return a registered shader node implementation given an implementation element.
    /// The element must be an Implementation or a NodeGraph acting as implementation.
    ShaderNodeImplPtr getImplementation(const NodeDef& nodedef, GenContext& context) const override;

    /// Return the result of an upstream connection or value for an input.
    string getUpstreamResult(const ShaderInput* input, GenContext& context) const override;

    /// Unique identifier for this generator target
    static const string TARGET;

    /// Map of code snippets for geomprops in MDL.
    static const std::unordered_map<string, string> GEOMPROP_DEFINITIONS;

    /// Add the MDL file header containing the version number of the generated module..
    void emitMdlVersionNumber(GenContext& context, ShaderStage& stage) const;

    /// Add the version number suffix appended to MDL modules that use versions.
    void emitMdlVersionFilenameSuffix(GenContext& context, ShaderStage& stage) const;

    /// Get the version number suffix appended to MDL modules that use versions.
    const string& getMdlVersionFilenameSuffix(GenContext& context) const;

  protected:
    // Create and initialize a new MDL shader for shader generation.
    ShaderPtr createShader(const string& name, ElementPtr element, GenContext& context) const;

    // Emit a block of shader inputs.
    void emitShaderInputs(const DocumentPtr doc, const VariableBlock& inputs, ShaderStage& stage) const;
};

namespace MDL
{

// Identifiers for MDL variable blocks
extern MX_GENMDL_API const string INPUTS;
extern MX_GENMDL_API const string OUTPUTS;

} // namespace MDL

MATERIALX_NAMESPACE_END

#endif
