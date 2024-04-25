//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MSLSHADERGENERATOR_H
#define MATERIALX_MSLSHADERGENERATOR_H

/// @file
/// MSL shader generator

#include <MaterialXGenMsl/Export.h>

#include <MaterialXGenShader/HwShaderGenerator.h>

#define TEXTURE_NAME(t) ((t) + "_tex")
#define SAMPLER_NAME(t) ((t) + "_sampler")

MATERIALX_NAMESPACE_BEGIN

using MslShaderGeneratorPtr = shared_ptr<class MslShaderGenerator>;

/// Base class for MSL (OpenGL Shading Language) code generation.
/// A generator for a specific MSL target should be derived from this class.
class MX_GENMSL_API MslShaderGenerator : public HwShaderGenerator
{
  public:
    MslShaderGenerator();

    static ShaderGeneratorPtr create() { return std::make_shared<MslShaderGenerator>(); }

    /// Generate a shader starting from the given element, translating
    /// the element and all dependencies upstream into shader code.
    ShaderPtr generate(const string& name, ElementPtr element, GenContext& context) const override;

    /// Return a unique identifier for the target this generator is for
    const string& getTarget() const override { return TARGET; }

    /// Return the version string for the MSL version this generator is for
    virtual const string& getVersion() const { return VERSION; }

    /// Emit a shader variable.
    void emitVariableDeclaration(const ShaderPort* variable, const string& qualifier, GenContext& context, ShaderStage& stage,
                                 bool assignValue = true) const override;

    /// Return a registered shader node implementation given an implementation element.
    /// The element must be an Implementation or a NodeGraph acting as implementation.
    ShaderNodeImplPtr getImplementation(const NodeDef& nodedef, GenContext& context) const override;

    /// Determine the prefix of vertex data variables.
    string getVertexDataPrefix(const VariableBlock& vertexData) const override;

  public:
    /// Unique identifier for this generator target
    static const string TARGET;

    /// Version string for the generator target
    static const string VERSION;

  protected:
    virtual void emitVertexStage(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const;
    virtual void emitPixelStage(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const;

    virtual void emitMetalTextureClass(GenContext& context, ShaderStage& stage) const;
    virtual void emitDirectives(GenContext& context, ShaderStage& stage) const;
    virtual void emitConstants(GenContext& context, ShaderStage& stage) const;
    virtual void emitLightData(GenContext& context, ShaderStage& stage) const;
    virtual void emitInputs(GenContext& context, ShaderStage& stage) const;
    virtual void emitOutputs(GenContext& context, ShaderStage& stage) const;

    virtual void emitMathMatrixScalarMathOperators(GenContext& context, ShaderStage& stage) const;
    virtual void MetalizeGeneratedShader(ShaderStage& shaderStage) const;

    void emitConstantBufferDeclarations(GenContext& context,
                                        HwResourceBindingContextPtr resourceBindingCtx,
                                        ShaderStage& stage) const;

    enum EmitGlobalScopeContext
    {
        EMIT_GLOBAL_SCOPE_CONTEXT_ENTRY_FUNCTION_RESOURCES = 0,
        EMIT_GLOBAL_SCOPE_CONTEXT_MEMBER_INIT = 1,
        EMIT_GLOBAL_SCOPE_CONTEXT_MEMBER_DECL = 2,
        EMIT_GLOBAL_SCOPE_CONTEXT_CONSTRUCTOR_ARGS = 3,
        EMIT_GLOBAL_SCOPE_CONTEXT_CONSTRUCTOR_INIT = 4
    };

    void emitGlobalVariables(GenContext& context, ShaderStage& stage,
                             EmitGlobalScopeContext situation,
                             bool isVertexShader,
                             bool needsLightData) const;

    void emitInputs(GenContext& context, ShaderStage& stage,
                    const VariableBlock& inputs) const;

    virtual HwResourceBindingContextPtr getResourceBindingContext(GenContext& context) const;

    /// Logic to indicate whether code to support direct lighting should be emitted.
    /// By default if the graph is classified as a shader, or BSDF node then lighting is assumed to be required.
    /// Derived classes can override this logic.
    virtual bool requiresLighting(const ShaderGraph& graph) const;

    /// Emit specular environment lookup code
    virtual void emitSpecularEnvironment(GenContext& context, ShaderStage& stage) const;

    /// Emit transmission rendering code
    virtual void emitTransmissionRender(GenContext& context, ShaderStage& stage) const;

    /// Emit function definitions for lighting code
    virtual void emitLightFunctionDefinitions(const ShaderGraph& graph, GenContext& context, ShaderStage& stage) const;

    static void toVec4(const TypeDesc* type, string& variable);

    /// Nodes used internally for light sampling.
    vector<ShaderNodePtr> _lightSamplingNodes;
};

/// Base class for common MSL node implementations
class MX_GENMSL_API MslImplementation : public HwImplementation
{
  public:
    const string& getTarget() const override;

  protected:
    MslImplementation() { }
};

MATERIALX_NAMESPACE_END

#endif
