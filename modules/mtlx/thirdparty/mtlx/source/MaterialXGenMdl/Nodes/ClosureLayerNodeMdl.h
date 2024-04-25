//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSURELAYERNODEMDL_H
#define MATERIALX_CLOSURELAYERNODEMDL_H

#include <MaterialXGenMdl/Export.h>
#include <MaterialXGenMdl/Nodes/SourceCodeNodeMdl.h>

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>

MATERIALX_NAMESPACE_BEGIN

/// Holds all constants required by the layering and its transformations.
class MX_GENMDL_API StringConstantsMdl
{
    StringConstantsMdl() = delete;

  public:
    /// String constants
    static const string TOP;  ///< layer parameter name of the top component
    static const string BASE; ///< layer parameter name of the base component
    static const string FG;   ///< parameter of the mix node
    static const string BG;   ///< parameter of the mix node
    static const string IN1;  ///< parameter of the add and multiply nodes
    static const string IN2;  ///< parameter of the add and multiply nodes

    static const string THICKNESS; ///< thickness parameter name of the thin_film_bsdf
    static const string IOR;       ///< ior parameter name of the thin_film_bsdf

    static const string THIN_FILM_THICKNESS; ///< helper parameter name for transporting thickness
    static const string THIN_FILM_IOR;       ///< helper parameter name for transporting ior

    static const string EMPTY; ///< the empty string ""
};

/// Helper class to be injected into nodes that need to carry thin film parameters from the
/// thin_film_bsdf through layers and mixers, etc., to the elemental bsdfs that support thin film.
/// Because thin-film can not be layered on any BSDF in MDL, we try to push down the parameters to
/// the nodes that support thin-film.
template <typename TBase> class CarryThinFilmParameters : public TBase
{
  public:
    /// Add the thin film inputs for transporting the parameter.
    /// `addInputs` for the injected base class is called first.
    void addInputs(ShaderNode& node, GenContext& context) const override
    {
        TBase::addInputs(node, context);
        node.addInput(StringConstantsMdl::THIN_FILM_THICKNESS, Type::FLOAT);
        node.addInput(StringConstantsMdl::THIN_FILM_IOR, Type::FLOAT);
    }

    /// Mark the thin film parameters as not editable because connections are
    /// created while emitting the MDL code and the default values should not
    /// get exposed to the public material interface.
    bool isEditable(const ShaderInput& input) const override
    {
        if (input.getName() == StringConstantsMdl::THIN_FILM_THICKNESS ||
            input.getName() == StringConstantsMdl::THIN_FILM_IOR)
        {
            return false;
        }
        return TBase::isEditable(input);
    }
};

/// Closure layer node implementation for MDL.
class MX_GENMDL_API ClosureLayerNodeMdl : public CarryThinFilmParameters<ShaderNodeImpl>
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    void emitBsdfOverBsdfFunctionCalls(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage,
        const ShaderGenerator& shadergen,
        ShaderNode* top,
        ShaderNode* base,
        ShaderOutput* output) const;

    void emitBsdfOverBsdfFunctionCalls_thinFilm(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage,
        const ShaderGenerator& shadergen,
        ShaderNode* top,
        ShaderNode* base,
        ShaderOutput* output) const;
};

/// Layerable BSDF node.
/// Because MDL does not support vertical layering the nodes are transformed in a way that
/// the base node is passed as parameter to the top layer node.
/// Note, not all elemental bsdfs support this kind of transformation.
class MX_GENMDL_API LayerableNodeMdl : public SourceCodeNodeMdl
{
  public:
    virtual ~LayerableNodeMdl() = default;
    static ShaderNodeImplPtr create();

    void addInputs(ShaderNode& node, GenContext&) const override;
};

/// Used for elemental nodes that can consume thin film.
class MX_GENMDL_API ThinFilmReceiverNodeMdl : public CarryThinFilmParameters<LayerableNodeMdl>
{
    using Base = CarryThinFilmParameters<LayerableNodeMdl>;

  public:
    static ShaderNodeImplPtr create();
};

/// Base class for operators that on bsdfs that need to transport the thin film parameters
class ThinFilmCombineNodeMdl : public CarryThinFilmParameters<SourceCodeNodeMdl>
{
    using Base = CarryThinFilmParameters<SourceCodeNodeMdl>;

  public:
    virtual ~ThinFilmCombineNodeMdl() = default;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    virtual const string& getOperatorName(size_t index) const = 0;
};

/// Used for mix_bsdf nodes.
class MX_GENMDL_API MixBsdfNodeMdl : public ThinFilmCombineNodeMdl
{
  public:
    static ShaderNodeImplPtr create();

  protected:
    virtual const string& getOperatorName(size_t index) const final;
};

/// Used for add_bsdf and multpli_bsdf nodes.
class MX_GENMDL_API AddOrMultiplyBsdfNodeMdl : public ThinFilmCombineNodeMdl
{
  public:
    static ShaderNodeImplPtr create();

  protected:
    virtual const string& getOperatorName(size_t index) const final;
};

MATERIALX_NAMESPACE_END

#endif
