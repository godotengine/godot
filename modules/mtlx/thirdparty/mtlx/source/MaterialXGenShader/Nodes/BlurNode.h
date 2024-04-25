//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_BLURNODE_H
#define MATERIALX_BLURNODE_H

#include <MaterialXGenShader/Nodes/ConvolutionNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Blur node implementation
class MX_GENSHADER_API BlurNode : public ConvolutionNode
{
  public:
    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    /// Constructor
    BlurNode();

    /// Emit function definitions for sampling functions used by this node.
    virtual void emitSamplingFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const = 0;

    /// Return if given type is an acceptible input
    bool acceptsInputType(const TypeDesc* type) const override;

    /// Compute offset strings for sampling
    void computeSampleOffsetStrings(const string& sampleSizeName, const string& offsetTypeString,
                                    unsigned int filterWidth, StringVec& offsetStrings) const override;

    /// Output sample array
    virtual void outputSampleArray(const ShaderGenerator& shadergen, ShaderStage& stage, const TypeDesc* inputType,
                                   const string& sampleName, const StringVec& sampleStrings) const;

    static const string _sampleSizeFunctionUV;
    static const float _filterSize;
    static const float _filterOffset;

    /// Box filter option on blur
    static const string BOX_FILTER;
    /// Box filter weights variable name
    static const string BOX_WEIGHTS_VARIABLE;

    /// Gaussian filter option on blur
    static const string GAUSSIAN_FILTER;
    /// Gaussian filter weights variable name
    static const string GAUSSIAN_WEIGHTS_VARIABLE;

    /// List of filters
    static const string FILTER_LIST;

    /// String constants
    static const string IN_STRING;
    static const string FILTER_TYPE_STRING;
    static const string FILTER_SIZE_STRING;
};

MATERIALX_NAMESPACE_END

#endif
