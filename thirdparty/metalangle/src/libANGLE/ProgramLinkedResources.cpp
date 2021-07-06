//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ProgramLinkedResources.cpp: implements link-time checks for default block uniforms, and generates
// uniform locations. Populates data structures related to uniforms so that they can be stored in
// program state.

#include "libANGLE/ProgramLinkedResources.h"

#include "common/string_utils.h"
#include "common/utilities.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Context.h"
#include "libANGLE/Shader.h"
#include "libANGLE/features.h"

namespace gl
{
namespace
{
LinkedUniform *FindUniform(std::vector<LinkedUniform> &list, const std::string &name)
{
    for (LinkedUniform &uniform : list)
    {
        if (uniform.name == name)
            return &uniform;
    }

    return nullptr;
}

template <typename VarT>
void SetActive(std::vector<VarT> *list, const std::string &name, ShaderType shaderType, bool active)
{
    for (auto &variable : *list)
    {
        if (variable.name == name)
        {
            variable.setActive(shaderType, active);
            return;
        }
    }
}

// GLSL ES Spec 3.00.3, section 4.3.5.
LinkMismatchError LinkValidateUniforms(const sh::ShaderVariable &uniform1,
                                       const sh::ShaderVariable &uniform2,
                                       std::string *mismatchedStructFieldName)
{
#if ANGLE_PROGRAM_LINK_VALIDATE_UNIFORM_PRECISION == ANGLE_ENABLED
    const bool validatePrecision = true;
#else
    const bool validatePrecision = false;
#endif

    LinkMismatchError linkError = Program::LinkValidateVariablesBase(
        uniform1, uniform2, validatePrecision, true, mismatchedStructFieldName);
    if (linkError != LinkMismatchError::NO_MISMATCH)
    {
        return linkError;
    }

    // GLSL ES Spec 3.10.4, section 4.4.5.
    if (uniform1.binding != -1 && uniform2.binding != -1 && uniform1.binding != uniform2.binding)
    {
        return LinkMismatchError::BINDING_MISMATCH;
    }

    // GLSL ES Spec 3.10.4, section 9.2.1.
    if (uniform1.location != -1 && uniform2.location != -1 &&
        uniform1.location != uniform2.location)
    {
        return LinkMismatchError::LOCATION_MISMATCH;
    }
    if (uniform1.offset != uniform2.offset)
    {
        return LinkMismatchError::OFFSET_MISMATCH;
    }

    return LinkMismatchError::NO_MISMATCH;
}

using ShaderUniform = std::pair<ShaderType, const sh::ShaderVariable *>;

bool ValidateGraphicsUniformsPerShader(Shader *shaderToLink,
                                       bool extendLinkedUniforms,
                                       std::map<std::string, ShaderUniform> *linkedUniforms,
                                       InfoLog &infoLog)
{
    ASSERT(shaderToLink && linkedUniforms);

    for (const sh::ShaderVariable &uniform : shaderToLink->getUniforms())
    {
        const auto &entry = linkedUniforms->find(uniform.name);
        if (entry != linkedUniforms->end())
        {
            const sh::ShaderVariable &linkedUniform = *(entry->second.second);
            std::string mismatchedStructFieldName;
            LinkMismatchError linkError =
                LinkValidateUniforms(uniform, linkedUniform, &mismatchedStructFieldName);
            if (linkError != LinkMismatchError::NO_MISMATCH)
            {
                LogLinkMismatch(infoLog, uniform.name, "uniform", linkError,
                                mismatchedStructFieldName, entry->second.first,
                                shaderToLink->getType());
                return false;
            }
        }
        else if (extendLinkedUniforms)
        {
            (*linkedUniforms)[uniform.name] = std::make_pair(shaderToLink->getType(), &uniform);
        }
    }

    return true;
}

GLuint GetMaximumShaderUniformVectors(ShaderType shaderType, const Caps &caps)
{
    switch (shaderType)
    {
        case ShaderType::Vertex:
            return caps.maxVertexUniformVectors;
        case ShaderType::Fragment:
            return caps.maxFragmentUniformVectors;

        case ShaderType::Compute:
        case ShaderType::Geometry:
            return caps.maxShaderUniformComponents[shaderType] / 4;

        default:
            UNREACHABLE();
            return 0u;
    }
}

enum class UniformType : uint8_t
{
    Variable      = 0,
    Sampler       = 1,
    Image         = 2,
    AtomicCounter = 3,

    InvalidEnum = 4,
    EnumCount   = 4,
};

const char *GetUniformResourceNameString(UniformType uniformType)
{
    switch (uniformType)
    {
        case UniformType::Variable:
            return "uniform";
        case UniformType::Sampler:
            return "texture image unit";
        case UniformType::Image:
            return "image uniform";
        case UniformType::AtomicCounter:
            return "atomic counter";
        default:
            UNREACHABLE();
            return "";
    }
}

std::string GetUniformResourceLimitName(ShaderType shaderType, UniformType uniformType)
{
    // Special case: MAX_TEXTURE_IMAGE_UNITS (no "MAX_FRAGMENT_TEXTURE_IMAGE_UNITS")
    if (shaderType == ShaderType::Fragment && uniformType == UniformType::Sampler)
    {
        return "MAX_TEXTURE_IMAGE_UNITS";
    }

    std::ostringstream ostream;
    ostream << "MAX_" << GetShaderTypeString(shaderType) << "_";

    switch (uniformType)
    {
        case UniformType::Variable:
            // For vertex and fragment shaders, ES 2.0 only defines MAX_VERTEX_UNIFORM_VECTORS and
            // MAX_FRAGMENT_UNIFORM_VECTORS ([OpenGL ES 2.0] Table 6.20).
            if (shaderType == ShaderType::Vertex || shaderType == ShaderType::Fragment)
            {
                ostream << "UNIFORM_VECTORS";
                break;
            }
            // For compute and geometry shaders, there are no definitions on
            // "MAX_COMPUTE_UNIFORM_VECTORS" or "MAX_GEOMETRY_UNIFORM_VECTORS_EXT"
            // ([OpenGL ES 3.1] Table 20.45, [EXT_geometry_shader] Table 20.43gs).
            else
            {
                ostream << "UNIFORM_COMPONENTS";
            }
            break;
        case UniformType::Sampler:
            ostream << "TEXTURE_IMAGE_UNITS";
            break;
        case UniformType::Image:
            ostream << "IMAGE_UNIFORMS";
            break;
        case UniformType::AtomicCounter:
            ostream << "ATOMIC_COUNTERS";
            break;
        default:
            UNREACHABLE();
            return "";
    }

    if (shaderType == ShaderType::Geometry)
    {
        ostream << "_EXT";
    }

    return ostream.str();
}

void LogUniformsExceedLimit(ShaderType shaderType,
                            UniformType uniformType,
                            GLuint limit,
                            InfoLog &infoLog)
{
    infoLog << GetShaderTypeString(shaderType) << " shader "
            << GetUniformResourceNameString(uniformType) << "s count exceeds "
            << GetUniformResourceLimitName(shaderType, uniformType) << "(" << limit << ")";
}

// The purpose of this visitor is to capture the uniforms in a uniform block. Each new uniform is
// added to "uniformsOut".
class UniformBlockEncodingVisitor : public sh::VariableNameVisitor
{
  public:
    UniformBlockEncodingVisitor(const GetBlockMemberInfoFunc &getMemberInfo,
                                const std::string &namePrefix,
                                const std::string &mappedNamePrefix,
                                std::vector<LinkedUniform> *uniformsOut,
                                ShaderType shaderType,
                                int blockIndex)
        : sh::VariableNameVisitor(namePrefix, mappedNamePrefix),
          mGetMemberInfo(getMemberInfo),
          mUniformsOut(uniformsOut),
          mShaderType(shaderType),
          mBlockIndex(blockIndex)
    {}

    void visitNamedVariable(const sh::ShaderVariable &variable,
                            bool isRowMajor,
                            const std::string &name,
                            const std::string &mappedName,
                            const std::vector<unsigned int> &arraySizes) override
    {
        // If getBlockMemberInfo returns false, the variable is optimized out.
        sh::BlockMemberInfo variableInfo;
        if (!mGetMemberInfo(name, mappedName, &variableInfo))
            return;

        std::string nameWithArrayIndex       = name;
        std::string mappedNameWithArrayIndex = mappedName;

        if (variable.isArray())
        {
            nameWithArrayIndex += "[0]";
            mappedNameWithArrayIndex += "[0]";
        }

        if (mBlockIndex == -1)
        {
            SetActive(mUniformsOut, nameWithArrayIndex, mShaderType, variable.active);
            return;
        }

        LinkedUniform newUniform(variable.type, variable.precision, nameWithArrayIndex,
                                 variable.arraySizes, -1, -1, -1, mBlockIndex, variableInfo);
        newUniform.mappedName = mappedNameWithArrayIndex;
        newUniform.setActive(mShaderType, variable.active);

        // Since block uniforms have no location, we don't need to store them in the uniform
        // locations list.
        mUniformsOut->push_back(newUniform);
    }

  private:
    const GetBlockMemberInfoFunc &mGetMemberInfo;
    std::vector<LinkedUniform> *mUniformsOut;
    const ShaderType mShaderType;
    const int mBlockIndex;
};

// The purpose of this visitor is to capture the buffer variables in a shader storage block. Each
// new buffer variable is stored in "bufferVariablesOut".
class ShaderStorageBlockVisitor : public sh::BlockEncoderVisitor
{
  public:
    ShaderStorageBlockVisitor(const GetBlockMemberInfoFunc &getMemberInfo,
                              const std::string &namePrefix,
                              const std::string &mappedNamePrefix,
                              std::vector<BufferVariable> *bufferVariablesOut,
                              ShaderType shaderType,
                              int blockIndex)
        : sh::BlockEncoderVisitor(namePrefix, mappedNamePrefix, &mDummyEncoder),
          mGetMemberInfo(getMemberInfo),
          mBufferVariablesOut(bufferVariablesOut),
          mShaderType(shaderType),
          mBlockIndex(blockIndex)
    {}

    void visitNamedVariable(const sh::ShaderVariable &variable,
                            bool isRowMajor,
                            const std::string &name,
                            const std::string &mappedName,
                            const std::vector<unsigned int> &arraySizes) override
    {
        if (mSkipEnabled)
            return;

        // If getBlockMemberInfo returns false, the variable is optimized out.
        sh::BlockMemberInfo variableInfo;
        if (!mGetMemberInfo(name, mappedName, &variableInfo))
            return;

        std::string nameWithArrayIndex       = name;
        std::string mappedNameWithArrayIndex = mappedName;

        if (variable.isArray())
        {
            nameWithArrayIndex += "[0]";
            mappedNameWithArrayIndex += "[0]";
        }

        if (mBlockIndex == -1)
        {
            SetActive(mBufferVariablesOut, nameWithArrayIndex, mShaderType, variable.active);
            return;
        }

        BufferVariable newBufferVariable(variable.type, variable.precision, nameWithArrayIndex,
                                         variable.arraySizes, mBlockIndex, variableInfo);
        newBufferVariable.mappedName = mappedNameWithArrayIndex;
        newBufferVariable.setActive(mShaderType, variable.active);

        newBufferVariable.topLevelArraySize = mTopLevelArraySize;

        mBufferVariablesOut->push_back(newBufferVariable);
    }

  private:
    const GetBlockMemberInfoFunc &mGetMemberInfo;
    std::vector<BufferVariable> *mBufferVariablesOut;
    const ShaderType mShaderType;
    const int mBlockIndex;
    sh::DummyBlockEncoder mDummyEncoder;
};

struct ShaderUniformCount
{
    unsigned int vectorCount        = 0;
    unsigned int samplerCount       = 0;
    unsigned int imageCount         = 0;
    unsigned int atomicCounterCount = 0;
};

ShaderUniformCount &operator+=(ShaderUniformCount &lhs, const ShaderUniformCount &rhs)
{
    lhs.vectorCount += rhs.vectorCount;
    lhs.samplerCount += rhs.samplerCount;
    lhs.imageCount += rhs.imageCount;
    lhs.atomicCounterCount += rhs.atomicCounterCount;
    return lhs;
}

// The purpose of this visitor is to flatten struct and array uniforms into a list of singleton
// uniforms. They are stored in separate lists by uniform type so they can be sorted in order.
// Counts for each uniform category are stored and can be queried with "getCounts".
class FlattenUniformVisitor : public sh::VariableNameVisitor
{
  public:
    FlattenUniformVisitor(ShaderType shaderType,
                          const sh::ShaderVariable &uniform,
                          std::vector<LinkedUniform> *uniforms,
                          std::vector<LinkedUniform> *samplerUniforms,
                          std::vector<LinkedUniform> *imageUniforms,
                          std::vector<LinkedUniform> *atomicCounterUniforms,
                          std::vector<UnusedUniform> *unusedUniforms)
        : sh::VariableNameVisitor("", ""),
          mShaderType(shaderType),
          mMarkActive(uniform.active),
          mMarkStaticUse(uniform.staticUse),
          mBinding(uniform.binding),
          mOffset(uniform.offset),
          mLocation(uniform.location),
          mUniforms(uniforms),
          mSamplerUniforms(samplerUniforms),
          mImageUniforms(imageUniforms),
          mAtomicCounterUniforms(atomicCounterUniforms),
          mUnusedUniforms(unusedUniforms)
    {}

    void visitNamedSampler(const sh::ShaderVariable &sampler,
                           const std::string &name,
                           const std::string &mappedName,
                           const std::vector<unsigned int> &arraySizes) override
    {
        visitNamedVariable(sampler, false, name, mappedName, arraySizes);
    }

    void visitNamedVariable(const sh::ShaderVariable &variable,
                            bool isRowMajor,
                            const std::string &name,
                            const std::string &mappedName,
                            const std::vector<unsigned int> &arraySizes) override
    {
        bool isSampler                          = IsSamplerType(variable.type);
        bool isImage                            = IsImageType(variable.type);
        bool isAtomicCounter                    = IsAtomicCounterType(variable.type);
        std::vector<LinkedUniform> *uniformList = mUniforms;
        if (isSampler)
        {
            uniformList = mSamplerUniforms;
        }
        else if (isImage)
        {
            uniformList = mImageUniforms;
        }
        else if (isAtomicCounter)
        {
            uniformList = mAtomicCounterUniforms;
        }

        std::string fullNameWithArrayIndex(name);
        std::string fullMappedNameWithArrayIndex(mappedName);

        if (variable.isArray())
        {
            // We're following the GLES 3.1 November 2016 spec section 7.3.1.1 Naming Active
            // Resources and including [0] at the end of array variable names.
            fullNameWithArrayIndex += "[0]";
            fullMappedNameWithArrayIndex += "[0]";
        }

        LinkedUniform *existingUniform = FindUniform(*uniformList, fullNameWithArrayIndex);
        if (existingUniform)
        {
            if (getBinding() != -1)
            {
                existingUniform->binding = getBinding();
            }
            if (getOffset() != -1)
            {
                existingUniform->offset = getOffset();
            }
            if (mLocation != -1)
            {
                existingUniform->location = mLocation;
            }
            if (mMarkActive)
            {
                existingUniform->active = true;
                existingUniform->setActive(mShaderType, true);
            }
            if (mMarkStaticUse)
            {
                existingUniform->staticUse = true;
            }
        }
        else
        {
            LinkedUniform linkedUniform(variable.type, variable.precision, fullNameWithArrayIndex,
                                        variable.arraySizes, getBinding(), getOffset(), mLocation,
                                        -1, sh::kDefaultBlockMemberInfo);
            linkedUniform.mappedName = fullMappedNameWithArrayIndex;
            linkedUniform.active     = mMarkActive;
            linkedUniform.staticUse  = mMarkStaticUse;
            linkedUniform.outerArraySizes = arraySizes;
            if (variable.hasParentArrayIndex())
            {
                linkedUniform.setParentArrayIndex(variable.parentArrayIndex());
            }
            if (mMarkActive)
            {
                linkedUniform.setActive(mShaderType, true);
            }
            else
            {
                mUnusedUniforms->emplace_back(linkedUniform.name, linkedUniform.isSampler());
            }

            uniformList->push_back(linkedUniform);
        }

        unsigned int elementCount = variable.getBasicTypeElementCount();

        // Samplers and images aren't "real" uniforms, so they don't count towards register usage.
        // Likewise, don't count "real" uniforms towards opaque count.

        if (!IsOpaqueType(variable.type))
        {
            mUniformCount.vectorCount += VariableRegisterCount(variable.type) * elementCount;
        }

        mUniformCount.samplerCount += (isSampler ? elementCount : 0);
        mUniformCount.imageCount += (isImage ? elementCount : 0);
        mUniformCount.atomicCounterCount += (isAtomicCounter ? elementCount : 0);

        if (mLocation != -1)
        {
            mLocation += elementCount;
        }
    }

    void enterStructAccess(const sh::ShaderVariable &structVar, bool isRowMajor) override
    {
        mStructStackSize++;
        sh::VariableNameVisitor::enterStructAccess(structVar, isRowMajor);
    }

    void exitStructAccess(const sh::ShaderVariable &structVar, bool isRowMajor) override
    {
        mStructStackSize--;
        sh::VariableNameVisitor::exitStructAccess(structVar, isRowMajor);
    }

    ShaderUniformCount getCounts() const { return mUniformCount; }

  private:
    int getBinding() const { return mStructStackSize == 0 ? mBinding : -1; }
    int getOffset() const { return mStructStackSize == 0 ? mOffset : -1; }

    ShaderType mShaderType;

    // Active and StaticUse are given separately because they are tracked at struct granularity.
    bool mMarkActive;
    bool mMarkStaticUse;
    int mBinding;
    int mOffset;
    int mLocation;
    std::vector<LinkedUniform> *mUniforms;
    std::vector<LinkedUniform> *mSamplerUniforms;
    std::vector<LinkedUniform> *mImageUniforms;
    std::vector<LinkedUniform> *mAtomicCounterUniforms;
    std::vector<UnusedUniform> *mUnusedUniforms;
    ShaderUniformCount mUniformCount;
    unsigned int mStructStackSize = 0;
};

class InterfaceBlockInfo final : angle::NonCopyable
{
  public:
    InterfaceBlockInfo(CustomBlockLayoutEncoderFactory *customEncoderFactory)
        : mCustomEncoderFactory(customEncoderFactory)
    {}

    void getShaderBlockInfo(const std::vector<sh::InterfaceBlock> &interfaceBlocks);

    bool getBlockSize(const std::string &name, const std::string &mappedName, size_t *sizeOut);
    bool getBlockMemberInfo(const std::string &name,
                            const std::string &mappedName,
                            sh::BlockMemberInfo *infoOut);

  private:
    size_t getBlockInfo(const sh::InterfaceBlock &interfaceBlock);

    std::map<std::string, size_t> mBlockSizes;
    sh::BlockLayoutMap mBlockLayout;
    // Based on the interface block layout, the std140 or std430 encoders are used.  On some
    // platforms (currently only D3D), there could be another non-standard encoder used.
    CustomBlockLayoutEncoderFactory *mCustomEncoderFactory;
};

void InterfaceBlockInfo::getShaderBlockInfo(const std::vector<sh::InterfaceBlock> &interfaceBlocks)
{
    for (const sh::InterfaceBlock &interfaceBlock : interfaceBlocks)
    {
        if (!IsActiveInterfaceBlock(interfaceBlock))
            continue;

        if (mBlockSizes.count(interfaceBlock.name) > 0)
            continue;

        size_t dataSize                  = getBlockInfo(interfaceBlock);
        mBlockSizes[interfaceBlock.name] = dataSize;
    }
}

size_t InterfaceBlockInfo::getBlockInfo(const sh::InterfaceBlock &interfaceBlock)
{
    ASSERT(IsActiveInterfaceBlock(interfaceBlock));

    // define member uniforms
    sh::Std140BlockEncoder std140Encoder;
    sh::Std430BlockEncoder std430Encoder;
    sh::BlockLayoutEncoder *customEncoder = nullptr;
    sh::BlockLayoutEncoder *encoder       = nullptr;

    if (interfaceBlock.layout == sh::BLOCKLAYOUT_STD140)
    {
        encoder = &std140Encoder;
    }
    else if (interfaceBlock.layout == sh::BLOCKLAYOUT_STD430)
    {
        encoder = &std430Encoder;
    }
    else if (mCustomEncoderFactory)
    {
        encoder = customEncoder = mCustomEncoderFactory->makeEncoder();
    }
    else
    {
        UNREACHABLE();
        return 0;
    }

    sh::GetInterfaceBlockInfo(interfaceBlock.fields, interfaceBlock.fieldPrefix(), encoder,
                              &mBlockLayout);

    size_t offset = encoder->getCurrentOffset();

    SafeDelete(customEncoder);

    return offset;
}

bool InterfaceBlockInfo::getBlockSize(const std::string &name,
                                      const std::string &mappedName,
                                      size_t *sizeOut)
{
    size_t nameLengthWithoutArrayIndex;
    ParseArrayIndex(name, &nameLengthWithoutArrayIndex);
    std::string baseName = name.substr(0u, nameLengthWithoutArrayIndex);
    auto sizeIter        = mBlockSizes.find(baseName);
    if (sizeIter == mBlockSizes.end())
    {
        *sizeOut = 0;
        return false;
    }

    *sizeOut = sizeIter->second;
    return true;
}

bool InterfaceBlockInfo::getBlockMemberInfo(const std::string &name,
                                            const std::string &mappedName,
                                            sh::BlockMemberInfo *infoOut)
{
    auto infoIter = mBlockLayout.find(name);
    if (infoIter == mBlockLayout.end())
    {
        *infoOut = sh::kDefaultBlockMemberInfo;
        return false;
    }

    *infoOut = infoIter->second;
    return true;
}
}  // anonymous namespace

UniformLinker::UniformLinker(const ProgramState &state) : mState(state) {}

UniformLinker::~UniformLinker() = default;

void UniformLinker::getResults(std::vector<LinkedUniform> *uniforms,
                               std::vector<UnusedUniform> *unusedUniforms,
                               std::vector<VariableLocation> *uniformLocations)
{
    uniforms->swap(mUniforms);
    unusedUniforms->swap(mUnusedUniforms);
    uniformLocations->swap(mUniformLocations);
}

bool UniformLinker::link(const Caps &caps,
                         InfoLog &infoLog,
                         const ProgramAliasedBindings &uniformLocationBindings)
{
    if (mState.getAttachedShader(ShaderType::Vertex) &&
        mState.getAttachedShader(ShaderType::Fragment))
    {
        ASSERT(mState.getAttachedShader(ShaderType::Compute) == nullptr);
        if (!validateGraphicsUniforms(infoLog))
        {
            return false;
        }
    }

    // Flatten the uniforms list (nested fields) into a simple list (no nesting).
    // Also check the maximum uniform vector and sampler counts.
    if (!flattenUniformsAndCheckCaps(caps, infoLog))
    {
        return false;
    }

    if (!checkMaxCombinedAtomicCounters(caps, infoLog))
    {
        return false;
    }

    if (!indexUniforms(infoLog, uniformLocationBindings))
    {
        return false;
    }

    return true;
}

bool UniformLinker::validateGraphicsUniforms(InfoLog &infoLog) const
{
    // Check that uniforms defined in the graphics shaders are identical
    std::map<std::string, ShaderUniform> linkedUniforms;

    for (const ShaderType shaderType : kAllGraphicsShaderTypes)
    {
        Shader *currentShader = mState.getAttachedShader(shaderType);
        if (currentShader)
        {
            if (shaderType == ShaderType::Vertex)
            {
                for (const sh::ShaderVariable &vertexUniform : currentShader->getUniforms())
                {
                    linkedUniforms[vertexUniform.name] =
                        std::make_pair(ShaderType::Vertex, &vertexUniform);
                }
            }
            else
            {
                bool isLastShader = (shaderType == ShaderType::Fragment);
                if (!ValidateGraphicsUniformsPerShader(currentShader, !isLastShader,
                                                       &linkedUniforms, infoLog))
                {
                    return false;
                }
            }
        }
    }

    return true;
}

bool UniformLinker::indexUniforms(InfoLog &infoLog,
                                  const ProgramAliasedBindings &uniformLocationBindings)
{
    // Locations which have been allocated for an unused uniform.
    std::set<GLuint> ignoredLocations;

    int maxUniformLocation = -1;

    // Gather uniform locations that have been set either using the bindUniformLocation API or by
    // using a location layout qualifier and check conflicts between them.
    if (!gatherUniformLocationsAndCheckConflicts(infoLog, uniformLocationBindings,
                                                 &ignoredLocations, &maxUniformLocation))
    {
        return false;
    }

    // Conflicts have been checked, now we can prune non-statically used uniforms. Code further down
    // the line relies on only having statically used uniforms in mUniforms.
    pruneUnusedUniforms();

    // Gather uniforms that have their location pre-set and uniforms that don't yet have a location.
    std::vector<VariableLocation> unlocatedUniforms;
    std::map<GLuint, VariableLocation> preLocatedUniforms;

    for (size_t uniformIndex = 0; uniformIndex < mUniforms.size(); uniformIndex++)
    {
        const LinkedUniform &uniform = mUniforms[uniformIndex];

        if ((uniform.isBuiltIn() && !uniform.isEmulatedBuiltIn()) ||
            IsAtomicCounterType(uniform.type))
        {
            continue;
        }

        int preSetLocation = uniformLocationBindings.getBinding(uniform);
        int shaderLocation = uniform.location;

        if (shaderLocation != -1)
        {
            preSetLocation = shaderLocation;
        }

        unsigned int elementCount = uniform.getBasicTypeElementCount();
        for (unsigned int arrayIndex = 0; arrayIndex < elementCount; arrayIndex++)
        {
            VariableLocation location(arrayIndex, static_cast<unsigned int>(uniformIndex));

            if ((arrayIndex == 0 && preSetLocation != -1) || shaderLocation != -1)
            {
                int elementLocation                 = preSetLocation + arrayIndex;
                preLocatedUniforms[elementLocation] = location;
            }
            else
            {
                unlocatedUniforms.push_back(location);
            }
        }
    }

    // Make enough space for all uniforms, with pre-set locations or not.
    mUniformLocations.resize(
        std::max(unlocatedUniforms.size() + preLocatedUniforms.size() + ignoredLocations.size(),
                 static_cast<size_t>(maxUniformLocation + 1)));

    // Assign uniforms with pre-set locations
    for (const auto &uniform : preLocatedUniforms)
    {
        mUniformLocations[uniform.first] = uniform.second;
    }

    // Assign ignored uniforms
    for (const auto &ignoredLocation : ignoredLocations)
    {
        mUniformLocations[ignoredLocation].markIgnored();
    }

    // Automatically assign locations for the rest of the uniforms
    size_t nextUniformLocation = 0;
    for (const auto &unlocatedUniform : unlocatedUniforms)
    {
        while (mUniformLocations[nextUniformLocation].used() ||
               mUniformLocations[nextUniformLocation].ignored)
        {
            nextUniformLocation++;
        }

        ASSERT(nextUniformLocation < mUniformLocations.size());
        mUniformLocations[nextUniformLocation] = unlocatedUniform;
        nextUniformLocation++;
    }

    return true;
}

bool UniformLinker::gatherUniformLocationsAndCheckConflicts(
    InfoLog &infoLog,
    const ProgramAliasedBindings &uniformLocationBindings,
    std::set<GLuint> *ignoredLocations,
    int *maxUniformLocation)
{
    // All the locations where another uniform can't be located.
    std::set<GLuint> reservedLocations;

    for (const LinkedUniform &uniform : mUniforms)
    {
        if (uniform.isBuiltIn() && !uniform.isEmulatedBuiltIn())
        {
            continue;
        }

        int apiBoundLocation = uniformLocationBindings.getBinding(uniform);
        int shaderLocation   = uniform.location;

        if (shaderLocation != -1)
        {
            unsigned int elementCount = uniform.getBasicTypeElementCount();

            for (unsigned int arrayIndex = 0; arrayIndex < elementCount; arrayIndex++)
            {
                // GLSL ES 3.10 section 4.4.3
                int elementLocation = shaderLocation + arrayIndex;
                *maxUniformLocation = std::max(*maxUniformLocation, elementLocation);
                if (reservedLocations.find(elementLocation) != reservedLocations.end())
                {
                    infoLog << "Multiple uniforms bound to location " << elementLocation << ".";
                    return false;
                }
                reservedLocations.insert(elementLocation);
                if (!uniform.active)
                {
                    ignoredLocations->insert(elementLocation);
                }
            }
        }
        else if (apiBoundLocation != -1 && uniform.staticUse)
        {
            // Only the first location is reserved even if the uniform is an array.
            *maxUniformLocation = std::max(*maxUniformLocation, apiBoundLocation);
            if (reservedLocations.find(apiBoundLocation) != reservedLocations.end())
            {
                infoLog << "Multiple uniforms bound to location " << apiBoundLocation << ".";
                return false;
            }
            reservedLocations.insert(apiBoundLocation);
            if (!uniform.active)
            {
                ignoredLocations->insert(apiBoundLocation);
            }
        }
    }

    // Record the uniform locations that were bound using the API for uniforms that were not found
    // from the shader. Other uniforms should not be assigned to those locations.
    for (const auto &locationBinding : uniformLocationBindings)
    {
        GLuint location = locationBinding.second.location;
        if (reservedLocations.find(location) == reservedLocations.end())
        {
            ignoredLocations->insert(location);
            *maxUniformLocation = std::max(*maxUniformLocation, static_cast<int>(location));
        }
    }

    return true;
}

void UniformLinker::pruneUnusedUniforms()
{
    auto uniformIter = mUniforms.begin();
    while (uniformIter != mUniforms.end())
    {
        if (uniformIter->active)
        {
            ++uniformIter;
        }
        else
        {
            mUnusedUniforms.emplace_back(uniformIter->name, uniformIter->isSampler());
            uniformIter = mUniforms.erase(uniformIter);
        }
    }
}

bool UniformLinker::flattenUniformsAndCheckCapsForShader(
    Shader *shader,
    const Caps &caps,
    std::vector<LinkedUniform> &samplerUniforms,
    std::vector<LinkedUniform> &imageUniforms,
    std::vector<LinkedUniform> &atomicCounterUniforms,
    std::vector<UnusedUniform> &unusedUniforms,
    InfoLog &infoLog)
{
    ShaderUniformCount shaderUniformCount;
    for (const sh::ShaderVariable &uniform : shader->getUniforms())
    {
        FlattenUniformVisitor flattener(shader->getType(), uniform, &mUniforms, &samplerUniforms,
                                        &imageUniforms, &atomicCounterUniforms, &unusedUniforms);
        sh::TraverseShaderVariable(uniform, false, &flattener);

        if (uniform.active)
        {
            shaderUniformCount += flattener.getCounts();
        }
        else
        {
            unusedUniforms.emplace_back(uniform.name, IsSamplerType(uniform.type));
        }
    }

    ShaderType shaderType = shader->getType();

    // TODO (jiawei.shao@intel.com): check whether we need finer-grained component counting
    GLuint maxUniformVectorsCount = GetMaximumShaderUniformVectors(shaderType, caps);
    if (shaderUniformCount.vectorCount > maxUniformVectorsCount)
    {
        GLuint maxUniforms = 0u;

        // See comments in GetUniformResourceLimitName()
        if (shaderType == ShaderType::Vertex || shaderType == ShaderType::Fragment)
        {
            maxUniforms = maxUniformVectorsCount;
        }
        else
        {
            maxUniforms = maxUniformVectorsCount * 4;
        }

        LogUniformsExceedLimit(shaderType, UniformType::Variable, maxUniforms, infoLog);
        return false;
    }

    if (shaderUniformCount.samplerCount > caps.maxShaderTextureImageUnits[shaderType])
    {
        LogUniformsExceedLimit(shaderType, UniformType::Sampler,
                               caps.maxShaderTextureImageUnits[shaderType], infoLog);
        return false;
    }

    if (shaderUniformCount.imageCount > caps.maxShaderImageUniforms[shaderType])
    {
        LogUniformsExceedLimit(shaderType, UniformType::Image,
                               caps.maxShaderImageUniforms[shaderType], infoLog);
        return false;
    }

    if (shaderUniformCount.atomicCounterCount > caps.maxShaderAtomicCounters[shaderType])
    {
        LogUniformsExceedLimit(shaderType, UniformType::AtomicCounter,
                               caps.maxShaderAtomicCounters[shaderType], infoLog);
        return false;
    }

    return true;
}

bool UniformLinker::flattenUniformsAndCheckCaps(const Caps &caps, InfoLog &infoLog)
{
    std::vector<LinkedUniform> samplerUniforms;
    std::vector<LinkedUniform> imageUniforms;
    std::vector<LinkedUniform> atomicCounterUniforms;
    std::vector<UnusedUniform> unusedUniforms;

    for (const ShaderType shaderType : AllShaderTypes())
    {
        Shader *shader = mState.getAttachedShader(shaderType);
        if (!shader)
        {
            continue;
        }

        if (!flattenUniformsAndCheckCapsForShader(shader, caps, samplerUniforms, imageUniforms,
                                                  atomicCounterUniforms, unusedUniforms, infoLog))
        {
            return false;
        }
    }

    mUniforms.insert(mUniforms.end(), samplerUniforms.begin(), samplerUniforms.end());
    mUniforms.insert(mUniforms.end(), imageUniforms.begin(), imageUniforms.end());
    mUniforms.insert(mUniforms.end(), atomicCounterUniforms.begin(), atomicCounterUniforms.end());
    mUnusedUniforms.insert(mUnusedUniforms.end(), unusedUniforms.begin(), unusedUniforms.end());
    return true;
}

bool UniformLinker::checkMaxCombinedAtomicCounters(const Caps &caps, InfoLog &infoLog)
{
    unsigned int atomicCounterCount = 0;
    for (const auto &uniform : mUniforms)
    {
        if (IsAtomicCounterType(uniform.type) && uniform.active)
        {
            atomicCounterCount += uniform.getBasicTypeElementCount();
            if (atomicCounterCount > caps.maxCombinedAtomicCounters)
            {
                infoLog << "atomic counter count exceeds MAX_COMBINED_ATOMIC_COUNTERS"
                        << caps.maxCombinedAtomicCounters << ").";
                return false;
            }
        }
    }
    return true;
}

// InterfaceBlockLinker implementation.
InterfaceBlockLinker::InterfaceBlockLinker(std::vector<InterfaceBlock> *blocksOut,
                                           std::vector<std::string> *unusedInterfaceBlocksOut)
    : mShaderBlocks({}), mBlocksOut(blocksOut), mUnusedInterfaceBlocksOut(unusedInterfaceBlocksOut)
{}

InterfaceBlockLinker::~InterfaceBlockLinker() {}

void InterfaceBlockLinker::addShaderBlocks(ShaderType shaderType,
                                           const std::vector<sh::InterfaceBlock> *blocks)
{
    mShaderBlocks[shaderType] = blocks;
}

void InterfaceBlockLinker::linkBlocks(const GetBlockSizeFunc &getBlockSize,
                                      const GetBlockMemberInfoFunc &getMemberInfo) const
{
    ASSERT(mBlocksOut->empty());

    std::set<std::string> visitedList;

    for (const ShaderType shaderType : AllShaderTypes())
    {
        if (!mShaderBlocks[shaderType])
        {
            continue;
        }

        for (const sh::InterfaceBlock &block : *mShaderBlocks[shaderType])
        {
            if (!IsActiveInterfaceBlock(block))
            {
                mUnusedInterfaceBlocksOut->push_back(block.name);
                continue;
            }

            if (visitedList.count(block.name) == 0)
            {
                defineInterfaceBlock(getBlockSize, getMemberInfo, block, shaderType);
                visitedList.insert(block.name);
                continue;
            }

            if (!block.active)
            {
                mUnusedInterfaceBlocksOut->push_back(block.name);
                continue;
            }

            for (InterfaceBlock &priorBlock : *mBlocksOut)
            {
                if (block.name == priorBlock.name)
                {
                    priorBlock.setActive(shaderType, true);

                    std::unique_ptr<sh::ShaderVariableVisitor> visitor(
                        getVisitor(getMemberInfo, block.fieldPrefix(), block.fieldMappedPrefix(),
                                   shaderType, -1));

                    sh::TraverseShaderVariables(block.fields, false, visitor.get());
                }
            }
        }
    }
}

void InterfaceBlockLinker::defineInterfaceBlock(const GetBlockSizeFunc &getBlockSize,
                                                const GetBlockMemberInfoFunc &getMemberInfo,
                                                const sh::InterfaceBlock &interfaceBlock,
                                                ShaderType shaderType) const
{
    size_t blockSize = 0;
    std::vector<unsigned int> blockIndexes;

    int blockIndex = static_cast<int>(mBlocksOut->size());
    // Track the first and last block member index to determine the range of active block members in
    // the block.
    size_t firstBlockMemberIndex = getCurrentBlockMemberIndex();

    std::unique_ptr<sh::ShaderVariableVisitor> visitor(
        getVisitor(getMemberInfo, interfaceBlock.fieldPrefix(), interfaceBlock.fieldMappedPrefix(),
                   shaderType, blockIndex));
    sh::TraverseShaderVariables(interfaceBlock.fields, false, visitor.get());

    size_t lastBlockMemberIndex = getCurrentBlockMemberIndex();

    for (size_t blockMemberIndex = firstBlockMemberIndex; blockMemberIndex < lastBlockMemberIndex;
         ++blockMemberIndex)
    {
        blockIndexes.push_back(static_cast<unsigned int>(blockMemberIndex));
    }

    for (unsigned int arrayElement = 0; arrayElement < interfaceBlock.elementCount();
         ++arrayElement)
    {
        std::string blockArrayName       = interfaceBlock.name;
        std::string blockMappedArrayName = interfaceBlock.mappedName;
        if (interfaceBlock.isArray())
        {
            blockArrayName += ArrayString(arrayElement);
            blockMappedArrayName += ArrayString(arrayElement);
        }

        // Don't define this block at all if it's not active in the implementation.
        if (!getBlockSize(blockArrayName, blockMappedArrayName, &blockSize))
        {
            continue;
        }

        // ESSL 3.10 section 4.4.4 page 58:
        // Any uniform or shader storage block declared without a binding qualifier is initially
        // assigned to block binding point zero.
        int blockBinding =
            (interfaceBlock.binding == -1 ? 0 : interfaceBlock.binding + arrayElement);
        InterfaceBlock block(interfaceBlock.name, interfaceBlock.mappedName,
                             interfaceBlock.isArray(), arrayElement, blockBinding);
        block.memberIndexes = blockIndexes;
        block.setActive(shaderType, interfaceBlock.active);

        // Since all block elements in an array share the same active interface blocks, they
        // will all be active once any block member is used. So, since interfaceBlock.name[0]
        // was active, here we will add every block element in the array.
        block.dataSize = static_cast<unsigned int>(blockSize);
        mBlocksOut->push_back(block);
    }
}

// UniformBlockLinker implementation.
UniformBlockLinker::UniformBlockLinker(std::vector<InterfaceBlock> *blocksOut,
                                       std::vector<LinkedUniform> *uniformsOut,
                                       std::vector<std::string> *unusedInterfaceBlocksOut)
    : InterfaceBlockLinker(blocksOut, unusedInterfaceBlocksOut), mUniformsOut(uniformsOut)
{}

UniformBlockLinker::~UniformBlockLinker() {}

size_t UniformBlockLinker::getCurrentBlockMemberIndex() const
{
    return mUniformsOut->size();
}

sh::ShaderVariableVisitor *UniformBlockLinker::getVisitor(
    const GetBlockMemberInfoFunc &getMemberInfo,
    const std::string &namePrefix,
    const std::string &mappedNamePrefix,
    ShaderType shaderType,
    int blockIndex) const
{
    return new UniformBlockEncodingVisitor(getMemberInfo, namePrefix, mappedNamePrefix,
                                           mUniformsOut, shaderType, blockIndex);
}

// ShaderStorageBlockLinker implementation.
ShaderStorageBlockLinker::ShaderStorageBlockLinker(
    std::vector<InterfaceBlock> *blocksOut,
    std::vector<BufferVariable> *bufferVariablesOut,
    std::vector<std::string> *unusedInterfaceBlocksOut)
    : InterfaceBlockLinker(blocksOut, unusedInterfaceBlocksOut),
      mBufferVariablesOut(bufferVariablesOut)
{}

ShaderStorageBlockLinker::~ShaderStorageBlockLinker() {}

size_t ShaderStorageBlockLinker::getCurrentBlockMemberIndex() const
{
    return mBufferVariablesOut->size();
}

sh::ShaderVariableVisitor *ShaderStorageBlockLinker::getVisitor(
    const GetBlockMemberInfoFunc &getMemberInfo,
    const std::string &namePrefix,
    const std::string &mappedNamePrefix,
    ShaderType shaderType,
    int blockIndex) const
{
    return new ShaderStorageBlockVisitor(getMemberInfo, namePrefix, mappedNamePrefix,
                                         mBufferVariablesOut, shaderType, blockIndex);
}

// AtomicCounterBufferLinker implementation.
AtomicCounterBufferLinker::AtomicCounterBufferLinker(
    std::vector<AtomicCounterBuffer> *atomicCounterBuffersOut)
    : mAtomicCounterBuffersOut(atomicCounterBuffersOut)
{}

AtomicCounterBufferLinker::~AtomicCounterBufferLinker() {}

void AtomicCounterBufferLinker::link(const std::map<int, unsigned int> &sizeMap) const
{
    for (auto &atomicCounterBuffer : *mAtomicCounterBuffersOut)
    {
        auto bufferSize = sizeMap.find(atomicCounterBuffer.binding);
        ASSERT(bufferSize != sizeMap.end());
        atomicCounterBuffer.dataSize = bufferSize->second;
    }
}

ProgramLinkedResources::ProgramLinkedResources(
    GLuint maxVaryingVectors,
    PackMode packMode,
    std::vector<InterfaceBlock> *uniformBlocksOut,
    std::vector<LinkedUniform> *uniformsOut,
    std::vector<InterfaceBlock> *shaderStorageBlocksOut,
    std::vector<BufferVariable> *bufferVariablesOut,
    std::vector<AtomicCounterBuffer> *atomicCounterBuffersOut)
    : varyingPacking(maxVaryingVectors, packMode),
      uniformBlockLinker(uniformBlocksOut, uniformsOut, &unusedInterfaceBlocks),
      shaderStorageBlockLinker(shaderStorageBlocksOut, bufferVariablesOut, &unusedInterfaceBlocks),
      atomicCounterBufferLinker(atomicCounterBuffersOut)
{}

ProgramLinkedResources::~ProgramLinkedResources() = default;

void ProgramLinkedResourcesLinker::linkResources(const ProgramState &programState,
                                                 const ProgramLinkedResources &resources) const
{
    // Gather uniform interface block info.
    InterfaceBlockInfo uniformBlockInfo(mCustomEncoderFactory);
    for (const ShaderType shaderType : AllShaderTypes())
    {
        Shader *shader = programState.getAttachedShader(shaderType);
        if (shader)
        {
            uniformBlockInfo.getShaderBlockInfo(shader->getUniformBlocks());
        }
    }

    auto getUniformBlockSize = [&uniformBlockInfo](const std::string &name,
                                                   const std::string &mappedName, size_t *sizeOut) {
        return uniformBlockInfo.getBlockSize(name, mappedName, sizeOut);
    };

    auto getUniformBlockMemberInfo = [&uniformBlockInfo](const std::string &name,
                                                         const std::string &mappedName,
                                                         sh::BlockMemberInfo *infoOut) {
        return uniformBlockInfo.getBlockMemberInfo(name, mappedName, infoOut);
    };

    // Link uniform interface blocks.
    resources.uniformBlockLinker.linkBlocks(getUniformBlockSize, getUniformBlockMemberInfo);

    // Gather storage bufer interface block info.
    InterfaceBlockInfo shaderStorageBlockInfo(mCustomEncoderFactory);
    for (const ShaderType shaderType : AllShaderTypes())
    {
        Shader *shader = programState.getAttachedShader(shaderType);
        if (shader)
        {
            shaderStorageBlockInfo.getShaderBlockInfo(shader->getShaderStorageBlocks());
        }
    }
    auto getShaderStorageBlockSize = [&shaderStorageBlockInfo](const std::string &name,
                                                               const std::string &mappedName,
                                                               size_t *sizeOut) {
        return shaderStorageBlockInfo.getBlockSize(name, mappedName, sizeOut);
    };

    auto getShaderStorageBlockMemberInfo = [&shaderStorageBlockInfo](const std::string &name,
                                                                     const std::string &mappedName,
                                                                     sh::BlockMemberInfo *infoOut) {
        return shaderStorageBlockInfo.getBlockMemberInfo(name, mappedName, infoOut);
    };

    // Link storage buffer interface blocks.
    resources.shaderStorageBlockLinker.linkBlocks(getShaderStorageBlockSize,
                                                  getShaderStorageBlockMemberInfo);

    // Gather and link atomic counter buffer interface blocks.
    std::map<int, unsigned int> sizeMap;
    getAtomicCounterBufferSizeMap(programState, sizeMap);
    resources.atomicCounterBufferLinker.link(sizeMap);
}

void ProgramLinkedResourcesLinker::getAtomicCounterBufferSizeMap(
    const ProgramState &programState,
    std::map<int, unsigned int> &sizeMapOut) const
{
    for (unsigned int index : programState.getAtomicCounterUniformRange())
    {
        const LinkedUniform &glUniform = programState.getUniforms()[index];

        auto &bufferDataSize = sizeMapOut[glUniform.binding];

        // Calculate the size of the buffer by finding the end of the last uniform with the same
        // binding. The end of the uniform is calculated by finding the initial offset of the
        // uniform and adding size of the uniform. For arrays, the size is the number of elements
        // times the element size (should always by 4 for atomic_units).
        unsigned dataOffset =
            glUniform.offset + static_cast<unsigned int>(glUniform.getBasicTypeElementCount() *
                                                         glUniform.getElementSize());
        if (dataOffset > bufferDataSize)
        {
            bufferDataSize = dataOffset;
        }
    }
}

}  // namespace gl
