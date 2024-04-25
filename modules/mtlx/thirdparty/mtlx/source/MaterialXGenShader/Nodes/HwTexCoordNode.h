//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_HWTEXCOORDNODE_H
#define MATERIALX_HWTEXCOORDNODE_H

#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Generic texture coordinate node for hardware languages
class MX_GENSHADER_API HwTexCoordNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    virtual string getIndex(const ShaderNode& node) const;

    static string INDEX;
};

MATERIALX_NAMESPACE_END

#endif
