//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// blocklayout.h:
//   Methods and classes related to uniform layout and packing in GLSL and HLSL.
//

#ifndef COMMON_BLOCKLAYOUT_H_
#define COMMON_BLOCKLAYOUT_H_

#include <cstddef>
#include <map>
#include <vector>

#include <GLSLANG/ShaderLang.h>
#include "angle_gl.h"

namespace sh
{
struct ShaderVariable;
struct InterfaceBlock;

struct BlockMemberInfo
{
    constexpr BlockMemberInfo() = default;

    constexpr BlockMemberInfo(int offset, int arrayStride, int matrixStride, bool isRowMajorMatrix)
        : offset(offset),
          arrayStride(arrayStride),
          matrixStride(matrixStride),
          isRowMajorMatrix(isRowMajorMatrix)
    {}

    constexpr BlockMemberInfo(int offset,
                              int arrayStride,
                              int matrixStride,
                              bool isRowMajorMatrix,
                              int topLevelArrayStride)
        : offset(offset),
          arrayStride(arrayStride),
          matrixStride(matrixStride),
          isRowMajorMatrix(isRowMajorMatrix),
          topLevelArrayStride(topLevelArrayStride)
    {}

    // A single integer identifying the offset of an active variable.
    int offset = -1;

    // A single integer identifying the stride between array elements in an active variable.
    int arrayStride = -1;

    // A single integer identifying the stride between columns of a column-major matrix or rows of a
    // row-major matrix.
    int matrixStride = -1;

    // A single integer identifying whether an active variable is a row-major matrix.
    bool isRowMajorMatrix = false;

    // A single integer identifying the number of active array elements of the top-level shader
    // storage block member containing the active variable.
    int topLevelArrayStride = -1;
};

constexpr size_t ComponentAlignment(size_t numComponents)
{
    return (numComponents == 3u ? 4u : numComponents);
}

constexpr BlockMemberInfo kDefaultBlockMemberInfo;

class BlockLayoutEncoder
{
  public:
    BlockLayoutEncoder();
    virtual ~BlockLayoutEncoder() {}

    BlockMemberInfo encodeType(GLenum type,
                               const std::vector<unsigned int> &arraySizes,
                               bool isRowMajorMatrix);

    size_t getCurrentOffset() const { return mCurrentOffset * kBytesPerComponent; }
    size_t getShaderVariableSize(const ShaderVariable &structVar, bool isRowMajor);

    // Called when entering/exiting a structure variable.
    virtual void enterAggregateType(const ShaderVariable &structVar) = 0;
    virtual void exitAggregateType(const ShaderVariable &structVar)  = 0;

    static constexpr size_t kBytesPerComponent           = 4u;
    static constexpr unsigned int kComponentsPerRegister = 4u;

    static size_t GetBlockRegister(const BlockMemberInfo &info);
    static size_t GetBlockRegisterElement(const BlockMemberInfo &info);

  protected:
    void align(size_t baseAlignment);

    virtual void getBlockLayoutInfo(GLenum type,
                                    const std::vector<unsigned int> &arraySizes,
                                    bool isRowMajorMatrix,
                                    int *arrayStrideOut,
                                    int *matrixStrideOut) = 0;
    virtual void advanceOffset(GLenum type,
                               const std::vector<unsigned int> &arraySizes,
                               bool isRowMajorMatrix,
                               int arrayStride,
                               int matrixStride)          = 0;

    size_t mCurrentOffset;
};

// Will return default values for everything.
class DummyBlockEncoder : public BlockLayoutEncoder
{
  public:
    DummyBlockEncoder() = default;

    void enterAggregateType(const ShaderVariable &structVar) override {}
    void exitAggregateType(const ShaderVariable &structVar) override {}

  protected:
    void getBlockLayoutInfo(GLenum type,
                            const std::vector<unsigned int> &arraySizes,
                            bool isRowMajorMatrix,
                            int *arrayStrideOut,
                            int *matrixStrideOut) override;

    void advanceOffset(GLenum type,
                       const std::vector<unsigned int> &arraySizes,
                       bool isRowMajorMatrix,
                       int arrayStride,
                       int matrixStride) override
    {}
};

// Block layout according to the std140 block layout
// See "Standard Uniform Block Layout" in Section 2.11.6 of the OpenGL ES 3.0 specification

class Std140BlockEncoder : public BlockLayoutEncoder
{
  public:
    Std140BlockEncoder();

    void enterAggregateType(const ShaderVariable &structVar) override;
    void exitAggregateType(const ShaderVariable &structVar) override;

  protected:
    void getBlockLayoutInfo(GLenum type,
                            const std::vector<unsigned int> &arraySizes,
                            bool isRowMajorMatrix,
                            int *arrayStrideOut,
                            int *matrixStrideOut) override;
    void advanceOffset(GLenum type,
                       const std::vector<unsigned int> &arraySizes,
                       bool isRowMajorMatrix,
                       int arrayStride,
                       int matrixStride) override;

    virtual size_t getBaseAlignment(const ShaderVariable &variable) const;
    virtual size_t getTypeBaseAlignment(GLenum type, bool isRowMajorMatrix) const;
};

class Std430BlockEncoder : public Std140BlockEncoder
{
  public:
    Std430BlockEncoder();

  protected:
    size_t getBaseAlignment(const ShaderVariable &variable) const override;
    size_t getTypeBaseAlignment(GLenum type, bool isRowMajorMatrix) const override;
};

using BlockLayoutMap = std::map<std::string, BlockMemberInfo>;

void GetInterfaceBlockInfo(const std::vector<ShaderVariable> &fields,
                           const std::string &prefix,
                           BlockLayoutEncoder *encoder,
                           BlockLayoutMap *blockInfoOut);

// Used for laying out the default uniform block on the Vulkan backend.
void GetUniformBlockInfo(const std::vector<ShaderVariable> &uniforms,
                         const std::string &prefix,
                         BlockLayoutEncoder *encoder,
                         BlockLayoutMap *blockInfoOut);

class ShaderVariableVisitor
{
  public:
    virtual ~ShaderVariableVisitor() {}

    virtual void enterStruct(const ShaderVariable &structVar) {}
    virtual void exitStruct(const ShaderVariable &structVar) {}

    virtual void enterStructAccess(const ShaderVariable &structVar, bool isRowMajor) {}
    virtual void exitStructAccess(const ShaderVariable &structVar, bool isRowMajor) {}

    virtual void enterArray(const ShaderVariable &arrayVar) {}
    virtual void exitArray(const ShaderVariable &arrayVar) {}

    virtual void enterArrayElement(const ShaderVariable &arrayVar, unsigned int arrayElement) {}
    virtual void exitArrayElement(const ShaderVariable &arrayVar, unsigned int arrayElement) {}

    virtual void visitSampler(const sh::ShaderVariable &sampler) {}

    virtual void visitVariable(const ShaderVariable &variable, bool isRowMajor) = 0;

  protected:
    ShaderVariableVisitor() {}
};

class VariableNameVisitor : public ShaderVariableVisitor
{
  public:
    VariableNameVisitor(const std::string &namePrefix, const std::string &mappedNamePrefix);
    ~VariableNameVisitor() override;

    void enterStruct(const ShaderVariable &structVar) override;
    void exitStruct(const ShaderVariable &structVar) override;
    void enterStructAccess(const ShaderVariable &structVar, bool isRowMajor) override;
    void exitStructAccess(const ShaderVariable &structVar, bool isRowMajor) override;
    void enterArray(const ShaderVariable &arrayVar) override;
    void exitArray(const ShaderVariable &arrayVar) override;
    void enterArrayElement(const ShaderVariable &arrayVar, unsigned int arrayElement) override;
    void exitArrayElement(const ShaderVariable &arrayVar, unsigned int arrayElement) override;

  protected:
    virtual void visitNamedSampler(const sh::ShaderVariable &sampler,
                                   const std::string &name,
                                   const std::string &mappedName,
                                   const std::vector<unsigned int> &arraySizes)
    {}
    virtual void visitNamedVariable(const ShaderVariable &variable,
                                    bool isRowMajor,
                                    const std::string &name,
                                    const std::string &mappedName,
                                    const std::vector<unsigned int> &arraySizes) = 0;

    std::string collapseNameStack() const;
    std::string collapseMappedNameStack() const;

  private:
    void visitSampler(const sh::ShaderVariable &sampler) final;
    void visitVariable(const ShaderVariable &variable, bool isRowMajor) final;

    std::vector<std::string> mNameStack;
    std::vector<std::string> mMappedNameStack;
    std::vector<unsigned int> mArraySizeStack;
};

class BlockEncoderVisitor : public VariableNameVisitor
{
  public:
    BlockEncoderVisitor(const std::string &namePrefix,
                        const std::string &mappedNamePrefix,
                        BlockLayoutEncoder *encoder);
    ~BlockEncoderVisitor() override;

    void enterStructAccess(const ShaderVariable &structVar, bool isRowMajor) override;
    void exitStructAccess(const ShaderVariable &structVar, bool isRowMajor) override;
    void enterArrayElement(const ShaderVariable &arrayVar, unsigned int arrayElement) override;
    void exitArrayElement(const ShaderVariable &arrayVar, unsigned int arrayElement) override;

    void visitNamedVariable(const ShaderVariable &variable,
                            bool isRowMajor,
                            const std::string &name,
                            const std::string &mappedName,
                            const std::vector<unsigned int> &arraySizes) override;

    virtual void encodeVariable(const ShaderVariable &variable,
                                const BlockMemberInfo &variableInfo,
                                const std::string &name,
                                const std::string &mappedName)
    {}

  protected:
    int mTopLevelArraySize           = 1;
    int mTopLevelArrayStride         = 0;
    bool mIsTopLevelArrayStrideReady = true;
    bool mSkipEnabled                = false;

  private:
    BlockLayoutEncoder *mEncoder;
    unsigned int mStructStackSize = 0;
};

void TraverseShaderVariable(const ShaderVariable &variable,
                            bool isRowMajorLayout,
                            ShaderVariableVisitor *visitor);

template <typename T>
void TraverseShaderVariables(const std::vector<T> &vars,
                             bool isRowMajorLayout,
                             ShaderVariableVisitor *visitor)
{
    for (const T &var : vars)
    {
        TraverseShaderVariable(var, isRowMajorLayout, visitor);
    }
}
}  // namespace sh

#endif  // COMMON_BLOCKLAYOUT_H_
