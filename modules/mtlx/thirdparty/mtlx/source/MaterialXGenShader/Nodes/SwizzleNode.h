//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SWIZZLENODE_H
#define MATERIALX_SWIZZLENODE_H

#include <MaterialXGenShader/ShaderNodeImpl.h>

MATERIALX_NAMESPACE_BEGIN

/// Swizzle node implementation
class MX_GENSHADER_API SwizzleNode : public ShaderNodeImpl
{
  public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    /// Returns true if the input is editable by users.
    /// Editable inputs are allowed to be published as shader uniforms
    /// and hence must be presentable in a user interface.
    bool isEditable(const ShaderInput& input) const override;

  protected:
    /// Get the implementation name of the variable that supports swizzles.
    /// Allows to override that name in generated code based on the target language.
    virtual string getVariableName(const ShaderInput* input) const;
};

MATERIALX_NAMESPACE_END

#endif
