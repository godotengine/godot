//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_BLURNODEMDL_H
#define MATERIALX_BLURNODEMDL_H

#include <MaterialXGenMdl/Export.h>

#include <MaterialXGenShader/Nodes/BlurNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Blur node MDL implementation
class MX_GENMDL_API BlurNodeMdl : public BlurNode
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    void emitSamplingFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    /// Output sample array
    void outputSampleArray(const ShaderGenerator& shadergen, ShaderStage& stage, const TypeDesc* inputType,
                           const string& sampleName, const StringVec& sampleStrings) const override;
};

MATERIALX_NAMESPACE_END

#endif
