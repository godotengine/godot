//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_HWTRANSFORMNODE_H
#define MATERIALX_HWTRANSFORMNODE_H

#include <MaterialXGenShader/HwShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

/// Generic transformation node for hardware languages
class MX_GENSHADER_API HwTransformNode : public ShaderNodeImpl
{
  public:
    void createVariables(const ShaderNode& node, GenContext& context, Shader& shader) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    virtual const string& getMatrix(const string& fromSpace, const string& toSpace) const;
    virtual const string& getModelToWorldMatrix() const = 0;
    virtual const string& getWorldToModelMatrix() const = 0;
    virtual string getHomogeneousCoordinate() const = 0;
    virtual bool shouldNormalize() const { return false; }

    virtual string getFromSpace(const ShaderNode&) const;
    virtual string getToSpace(const ShaderNode&) const;

    static const string FROM_SPACE;
    static const string TO_SPACE;
    static const string MODEL;
    static const string OBJECT;
    static const string WORLD;
};

class MX_GENSHADER_API HwTransformVectorNode : public HwTransformNode
{
  public:
    static ShaderNodeImplPtr create();

  protected:
    const string& getModelToWorldMatrix() const override { return HW::T_WORLD_MATRIX; }
    const string& getWorldToModelMatrix() const override { return HW::T_WORLD_INVERSE_MATRIX; }
    string getHomogeneousCoordinate() const override { return "0.0"; }
};

class MX_GENSHADER_API HwTransformPointNode : public HwTransformVectorNode
{
  public:
    static ShaderNodeImplPtr create();

  protected:
    string getHomogeneousCoordinate() const override { return "1.0"; }
};

class MX_GENSHADER_API HwTransformNormalNode : public HwTransformNode
{
  public:
    static ShaderNodeImplPtr create();

  protected:
    const string& getModelToWorldMatrix() const override { return HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX; }
    const string& getWorldToModelMatrix() const override { return HW::T_WORLD_TRANSPOSE_MATRIX; }
    string getHomogeneousCoordinate() const override { return "0.0"; }
    bool shouldNormalize() const override { return true; }
};

MATERIALX_NAMESPACE_END

#endif
