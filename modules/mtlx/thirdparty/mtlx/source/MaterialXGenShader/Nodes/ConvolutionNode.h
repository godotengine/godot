//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CONVOLUTIONNODE_H
#define MATERIALX_CONVOLUTIONNODE_H

#include <MaterialXGenShader/ShaderNodeImpl.h>

#include <array>

MATERIALX_NAMESPACE_BEGIN

extern MX_GENSHADER_API const std::array<float, 3> GAUSSIAN_KERNEL_3;
extern MX_GENSHADER_API const std::array<float, 5> GAUSSIAN_KERNEL_5;
extern MX_GENSHADER_API const std::array<float, 7> GAUSSIAN_KERNEL_7;

/// Utility class for implementations of nodes which perform convolutions
///
class MX_GENSHADER_API ConvolutionNode : public ShaderNodeImpl
{
  public:
    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

  protected:
    /// Constructor
    ConvolutionNode();

    /// Derived classes are responsible for returning if a given type is an acceptable input.
    virtual bool acceptsInputType(const TypeDesc* type) const = 0;

    // Derived classes are responsible for computing offset strings relative to the center sample
    // The sample size and offset type are passed in as arguments.
    virtual void computeSampleOffsetStrings(const string& sampleSizeName, const string& offsetTypeString,
                                            unsigned int filterWidth, StringVec& offsetStrings) const = 0;

    /// Get input which is used for sampling. If there is none
    /// then a null pointer is returned.
    virtual const ShaderInput* getSamplingInput(const ShaderNode& node) const;

    /// Generate upstream / input sampling code in uv space and cache the output variable names which
    /// will hold the sample values after execution.
    void emitInputSamplesUV(const ShaderNode& node,
                            unsigned int sampleCount, unsigned int filterWidth,
                            float filterSize, float filterOffset,
                            const string& sampleSizeFunctionUV,
                            GenContext& context, ShaderStage& stage,
                            StringVec& sampleStrings) const;
};

MATERIALX_NAMESPACE_END

#endif
