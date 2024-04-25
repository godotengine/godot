//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_COMPOUNDNODEMDL_H
#define MATERIALX_COMPOUNDNODEMDL_H

#include <MaterialXGenMdl/Export.h>

#include <MaterialXGenShader/Nodes/CompoundNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Generator context data class to pass strings.
class GenUserDataString : public GenUserData
{
  public:
    GenUserDataString(const std::string& value) : _value(value) {}
    const string& getValue() const { return _value; }

  private:
    string _value;
};

/// Shared pointer to a GenUserDataString
using GenUserDataStringPtr = std::shared_ptr<GenUserDataString>;

/// Compound node implementation
class MX_GENMDL_API CompoundNodeMdl : public CompoundNode
{
  public:
    static ShaderNodeImplPtr create();

    void initialize(const InterfaceElement& element, GenContext& context) override;
    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

    bool isReturnStruct() const { return !_returnStruct.empty(); }
    bool unrollReturnStructMembers() const { return _unrollReturnStructMembers; }

  protected:
    void emitFunctionSignature(const ShaderNode& node, GenContext& context, ShaderStage& stage) const;

    string _returnStruct;
    bool _unrollReturnStructMembers = false;

    static const string GEN_USER_DATA_RETURN_STRUCT_FIELD_NAME;
};

MATERIALX_NAMESPACE_END

#endif
