//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SOURCECODEMDL_H
#define MATERIALX_SOURCECODEMDL_H

#include <MaterialXGenMdl/Export.h>

#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Node implementation using data-driven static source code.
/// This is the default implementation used for all nodes that
/// do not have a custom ShaderNodeImpl class.
class MX_GENMDL_API SourceCodeNodeMdl : public SourceCodeNode
{
  public:
    static ShaderNodeImplPtr create();

    void initialize(const InterfaceElement& element, GenContext& context) override;
    void emitFunctionDefinition(const ShaderNode&, GenContext&, ShaderStage&) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    string _returnStruct;
};

MATERIALX_NAMESPACE_END

#endif
