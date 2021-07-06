//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ProgramLinkedResources.h: implements link-time checks for default block uniforms, and generates
// uniform locations. Populates data structures related to uniforms so that they can be stored in
// program state.

#ifndef LIBANGLE_UNIFORMLINKER_H_
#define LIBANGLE_UNIFORMLINKER_H_

#include "angle_gl.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/VaryingPacking.h"

#include <functional>

namespace sh
{
class BlockLayoutEncoder;
struct BlockMemberInfo;
struct InterfaceBlock;
struct ShaderVariable;
class BlockEncoderVisitor;
class ShaderVariableVisitor;
struct ShaderVariable;
}  // namespace sh

namespace gl
{
struct BufferVariable;
struct Caps;
class Context;
class InfoLog;
struct InterfaceBlock;
enum class LinkMismatchError;
struct LinkedUniform;
class ProgramState;
class ProgramBindings;
class ProgramAliasedBindings;
class Shader;
struct ShaderVariableBuffer;
struct UnusedUniform;
struct VariableLocation;

using AtomicCounterBuffer = ShaderVariableBuffer;

class UniformLinker final : angle::NonCopyable
{
  public:
    UniformLinker(const ProgramState &state);
    ~UniformLinker();

    bool link(const Caps &caps,
              InfoLog &infoLog,
              const ProgramAliasedBindings &uniformLocationBindings);

    void getResults(std::vector<LinkedUniform> *uniforms,
                    std::vector<UnusedUniform> *unusedUniforms,
                    std::vector<VariableLocation> *uniformLocations);

  private:
    bool validateGraphicsUniforms(InfoLog &infoLog) const;

    bool flattenUniformsAndCheckCapsForShader(Shader *shader,
                                              const Caps &caps,
                                              std::vector<LinkedUniform> &samplerUniforms,
                                              std::vector<LinkedUniform> &imageUniforms,
                                              std::vector<LinkedUniform> &atomicCounterUniforms,
                                              std::vector<UnusedUniform> &unusedUniforms,
                                              InfoLog &infoLog);

    bool flattenUniformsAndCheckCaps(const Caps &caps, InfoLog &infoLog);
    bool checkMaxCombinedAtomicCounters(const Caps &caps, InfoLog &infoLog);

    bool indexUniforms(InfoLog &infoLog, const ProgramAliasedBindings &uniformLocationBindings);
    bool gatherUniformLocationsAndCheckConflicts(
        InfoLog &infoLog,
        const ProgramAliasedBindings &uniformLocationBindings,
        std::set<GLuint> *ignoredLocations,
        int *maxUniformLocation);
    void pruneUnusedUniforms();

    const ProgramState &mState;
    std::vector<LinkedUniform> mUniforms;
    std::vector<UnusedUniform> mUnusedUniforms;
    std::vector<VariableLocation> mUniformLocations;
};

using GetBlockSizeFunc = std::function<
    bool(const std::string &blockName, const std::string &blockMappedName, size_t *sizeOut)>;
using GetBlockMemberInfoFunc = std::function<
    bool(const std::string &name, const std::string &mappedName, sh::BlockMemberInfo *infoOut)>;

// This class is intended to be used during the link step to store interface block information.
// It is called by the Impl class during ProgramImpl::link so that it has access to the
// real block size and layout.
class InterfaceBlockLinker : angle::NonCopyable
{
  public:
    virtual ~InterfaceBlockLinker();

    // This is called once per shader stage. It stores a pointer to the block vector, so it's
    // important that this class does not persist longer than the duration of Program::link.
    void addShaderBlocks(ShaderType shader, const std::vector<sh::InterfaceBlock> *blocks);

    // This is called once during a link operation, after all shader blocks are added.
    void linkBlocks(const GetBlockSizeFunc &getBlockSize,
                    const GetBlockMemberInfoFunc &getMemberInfo) const;

  protected:
    InterfaceBlockLinker(std::vector<InterfaceBlock> *blocksOut,
                         std::vector<std::string> *unusedInterfaceBlocksOut);
    void defineInterfaceBlock(const GetBlockSizeFunc &getBlockSize,
                              const GetBlockMemberInfoFunc &getMemberInfo,
                              const sh::InterfaceBlock &interfaceBlock,
                              ShaderType shaderType) const;

    virtual size_t getCurrentBlockMemberIndex() const = 0;

    ShaderMap<const std::vector<sh::InterfaceBlock> *> mShaderBlocks;

    std::vector<InterfaceBlock> *mBlocksOut;
    std::vector<std::string> *mUnusedInterfaceBlocksOut;

    virtual sh::ShaderVariableVisitor *getVisitor(const GetBlockMemberInfoFunc &getMemberInfo,
                                                  const std::string &namePrefix,
                                                  const std::string &mappedNamePrefix,
                                                  ShaderType shaderType,
                                                  int blockIndex) const = 0;
};

class UniformBlockLinker final : public InterfaceBlockLinker
{
  public:
    UniformBlockLinker(std::vector<InterfaceBlock> *blocksOut,
                       std::vector<LinkedUniform> *uniformsOut,
                       std::vector<std::string> *unusedInterfaceBlocksOut);
    ~UniformBlockLinker() override;

  private:
    size_t getCurrentBlockMemberIndex() const override;

    sh::ShaderVariableVisitor *getVisitor(const GetBlockMemberInfoFunc &getMemberInfo,
                                          const std::string &namePrefix,
                                          const std::string &mappedNamePrefix,
                                          ShaderType shaderType,
                                          int blockIndex) const override;

    std::vector<LinkedUniform> *mUniformsOut;
};

class ShaderStorageBlockLinker final : public InterfaceBlockLinker
{
  public:
    ShaderStorageBlockLinker(std::vector<InterfaceBlock> *blocksOut,
                             std::vector<BufferVariable> *bufferVariablesOut,
                             std::vector<std::string> *unusedInterfaceBlocksOut);
    ~ShaderStorageBlockLinker() override;

  private:
    size_t getCurrentBlockMemberIndex() const override;

    sh::ShaderVariableVisitor *getVisitor(const GetBlockMemberInfoFunc &getMemberInfo,
                                          const std::string &namePrefix,
                                          const std::string &mappedNamePrefix,
                                          ShaderType shaderType,
                                          int blockIndex) const override;

    std::vector<BufferVariable> *mBufferVariablesOut;
};

class AtomicCounterBufferLinker final : angle::NonCopyable
{
  public:
    AtomicCounterBufferLinker(std::vector<AtomicCounterBuffer> *atomicCounterBuffersOut);
    ~AtomicCounterBufferLinker();

    void link(const std::map<int, unsigned int> &sizeMap) const;

  private:
    std::vector<AtomicCounterBuffer> *mAtomicCounterBuffersOut;
};

// The link operation is responsible for finishing the link of uniform and interface blocks.
// This way it can filter out unreferenced resources and still have access to the info.
// TODO(jmadill): Integrate uniform linking/filtering as well as interface blocks.
struct UnusedUniform
{
    UnusedUniform(std::string name, bool isSampler)
    {
        this->name      = name;
        this->isSampler = isSampler;
    }

    std::string name;
    bool isSampler;
};

struct ProgramLinkedResources
{
    ProgramLinkedResources(GLuint maxVaryingVectors,
                           PackMode packMode,
                           std::vector<InterfaceBlock> *uniformBlocksOut,
                           std::vector<LinkedUniform> *uniformsOut,
                           std::vector<InterfaceBlock> *shaderStorageBlocksOut,
                           std::vector<BufferVariable> *bufferVariablesOut,
                           std::vector<AtomicCounterBuffer> *atomicCounterBuffersOut);
    ~ProgramLinkedResources();

    VaryingPacking varyingPacking;
    UniformBlockLinker uniformBlockLinker;
    ShaderStorageBlockLinker shaderStorageBlockLinker;
    AtomicCounterBufferLinker atomicCounterBufferLinker;
    std::vector<UnusedUniform> unusedUniforms;
    std::vector<std::string> unusedInterfaceBlocks;
};

class CustomBlockLayoutEncoderFactory : angle::NonCopyable
{
  public:
    virtual ~CustomBlockLayoutEncoderFactory() {}

    virtual sh::BlockLayoutEncoder *makeEncoder() = 0;
};

// Used by the backends in Program*::linkResources to parse interface blocks and provide
// information to ProgramLinkedResources' linkers.
class ProgramLinkedResourcesLinker final : angle::NonCopyable
{
  public:
    ProgramLinkedResourcesLinker(CustomBlockLayoutEncoderFactory *customEncoderFactory)
        : mCustomEncoderFactory(customEncoderFactory)
    {}

    void linkResources(const ProgramState &programState,
                       const ProgramLinkedResources &resources) const;

  private:
    void getAtomicCounterBufferSizeMap(const ProgramState &programState,
                                       std::map<int, unsigned int> &sizeMapOut) const;

    CustomBlockLayoutEncoderFactory *mCustomEncoderFactory;
};

}  // namespace gl

#endif  // LIBANGLE_UNIFORMLINKER_H_
