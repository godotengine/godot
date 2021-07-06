//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Program.h: Defines the gl::Program class. Implements GL program objects
// and related functionality. [OpenGL ES 2.0.24] section 2.10.3 page 28.

#ifndef LIBANGLE_PROGRAM_H_
#define LIBANGLE_PROGRAM_H_

#include <GLES2/gl2.h>
#include <GLSLANG/ShaderVars.h>

#include <array>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/Optional.h"
#include "common/angleutils.h"
#include "common/mathutil.h"
#include "common/utilities.h"

#include "libANGLE/Constants.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/Uniform.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class GLImplFactory;
class ProgramImpl;
struct TranslatedAttribute;
}  // namespace rx

namespace gl
{
class Buffer;
class BinaryInputStream;
class BinaryOutputStream;
struct Caps;
class Context;
struct Extensions;
class Framebuffer;
class InfoLog;
class Shader;
class ShaderProgramManager;
class State;
struct UnusedUniform;
struct Version;

extern const char *const g_fakepath;

enum class LinkMismatchError
{
    // Shared
    NO_MISMATCH,
    TYPE_MISMATCH,
    ARRAY_SIZE_MISMATCH,
    PRECISION_MISMATCH,
    STRUCT_NAME_MISMATCH,
    FIELD_NUMBER_MISMATCH,
    FIELD_NAME_MISMATCH,

    // Varying specific
    INTERPOLATION_TYPE_MISMATCH,
    INVARIANCE_MISMATCH,

    // Uniform specific
    BINDING_MISMATCH,
    LOCATION_MISMATCH,
    OFFSET_MISMATCH,
    INSTANCE_NAME_MISMATCH,

    // Interface block specific
    LAYOUT_QUALIFIER_MISMATCH,
    MATRIX_PACKING_MISMATCH
};

class InfoLog : angle::NonCopyable
{
  public:
    InfoLog();
    ~InfoLog();

    size_t getLength() const;
    void getLog(GLsizei bufSize, GLsizei *length, char *infoLog) const;

    void appendSanitized(const char *message);
    void reset();

    // This helper class ensures we append a newline after writing a line.
    class StreamHelper : angle::NonCopyable
    {
      public:
        StreamHelper(StreamHelper &&rhs) : mStream(rhs.mStream) { rhs.mStream = nullptr; }

        StreamHelper &operator=(StreamHelper &&rhs)
        {
            std::swap(mStream, rhs.mStream);
            return *this;
        }

        ~StreamHelper()
        {
            // Write newline when destroyed on the stack
            if (mStream)
            {
                (*mStream) << std::endl;
            }
        }

        template <typename T>
        StreamHelper &operator<<(const T &value)
        {
            (*mStream) << value;
            return *this;
        }

      private:
        friend class InfoLog;

        StreamHelper(std::stringstream *stream) : mStream(stream) { ASSERT(stream); }

        std::stringstream *mStream;
    };

    template <typename T>
    StreamHelper operator<<(const T &value)
    {
        ensureInitialized();
        StreamHelper helper(mLazyStream.get());
        helper << value;
        return helper;
    }

    std::string str() const { return mLazyStream ? mLazyStream->str() : ""; }

    bool empty() const;

  private:
    void ensureInitialized()
    {
        if (!mLazyStream)
        {
            mLazyStream.reset(new std::stringstream());
        }
    }

    std::unique_ptr<std::stringstream> mLazyStream;
};

void LogLinkMismatch(InfoLog &infoLog,
                     const std::string &variableName,
                     const char *variableType,
                     LinkMismatchError linkError,
                     const std::string &mismatchedStructOrBlockFieldName,
                     ShaderType shaderType1,
                     ShaderType shaderType2);

bool IsActiveInterfaceBlock(const sh::InterfaceBlock &interfaceBlock);

void WriteBlockMemberInfo(BinaryOutputStream *stream, const sh::BlockMemberInfo &var);
void LoadBlockMemberInfo(BinaryInputStream *stream, sh::BlockMemberInfo *var);

// Struct used for correlating uniforms/elements of uniform arrays to handles
struct VariableLocation
{
    static constexpr unsigned int kUnused = GL_INVALID_INDEX;

    VariableLocation();
    VariableLocation(unsigned int arrayIndex, unsigned int index);

    // If used is false, it means this location is only used to fill an empty space in an array,
    // and there is no corresponding uniform variable for this location. It can also mean the
    // uniform was optimized out by the implementation.
    bool used() const { return (index != kUnused); }
    void markUnused() { index = kUnused; }
    void markIgnored() { ignored = true; }

    bool operator==(const VariableLocation &other) const
    {
        return arrayIndex == other.arrayIndex && index == other.index;
    }

    // "arrayIndex" stores the index of the innermost GLSL array. It's zero for non-arrays.
    unsigned int arrayIndex;
    // "index" is an index of the variable. The variable contains the indices for other than the
    // innermost GLSL arrays.
    unsigned int index;

    // If this location was bound to an unreferenced uniform.  Setting data on this uniform is a
    // no-op.
    bool ignored;
};

// Information about a variable binding.
// Currently used by CHROMIUM_path_rendering
struct BindingInfo
{
    // The type of binding, for example GL_FLOAT_VEC3.
    // This can be GL_NONE if the variable is optimized away.
    GLenum type;

    // This is the name of the variable in
    // the translated shader program. Note that
    // this can be empty in the case where the
    // variable has been optimized away.
    std::string name;

    // True if the binding is valid, otherwise false.
    bool valid;
};

// This small structure encapsulates binding sampler uniforms to active GL textures.
struct SamplerBinding
{
    SamplerBinding(TextureType textureTypeIn,
                   SamplerFormat formatIn,
                   size_t elementCount,
                   bool unreferenced);
    SamplerBinding(const SamplerBinding &other);
    ~SamplerBinding();

    // Necessary for retrieving active textures from the GL state.
    TextureType textureType;

    SamplerFormat format;

    // List of all textures bound to this sampler, of type textureType.
    std::vector<GLuint> boundTextureUnits;

    // A note if this sampler is an unreferenced uniform.
    bool unreferenced;
};

// A varying with tranform feedback enabled. If it's an array, either the whole array or one of its
// elements specified by 'arrayIndex' can set to be enabled.
struct TransformFeedbackVarying : public sh::ShaderVariable
{
    TransformFeedbackVarying(const sh::ShaderVariable &varyingIn, GLuint index)
        : sh::ShaderVariable(varyingIn), arrayIndex(index)
    {
        ASSERT(!isArrayOfArrays());
    }

    TransformFeedbackVarying(const sh::ShaderVariable &field, const sh::ShaderVariable &parent)
        : arrayIndex(GL_INVALID_INDEX)
    {
        sh::ShaderVariable *thisVar = this;
        *thisVar                    = field;
        interpolation               = parent.interpolation;
        isInvariant                 = parent.isInvariant;
        name                        = parent.name + "." + name;
        mappedName                  = parent.mappedName + "." + mappedName;
    }

    std::string nameWithArrayIndex() const
    {
        std::stringstream fullNameStr;
        fullNameStr << name;
        if (arrayIndex != GL_INVALID_INDEX)
        {
            fullNameStr << "[" << arrayIndex << "]";
        }
        return fullNameStr.str();
    }
    GLsizei size() const
    {
        return (isArray() && arrayIndex == GL_INVALID_INDEX ? getOutermostArraySize() : 1);
    }

    GLuint arrayIndex;
};

struct ImageBinding
{
    ImageBinding(size_t count);
    ImageBinding(GLuint imageUnit, size_t count, bool unreferenced);
    ImageBinding(const ImageBinding &other);
    ~ImageBinding();

    std::vector<GLuint> boundImageUnits;

    // A note if this image unit is an unreferenced uniform.
    bool unreferenced;
};

class ProgramState final : angle::NonCopyable
{
  public:
    ProgramState();
    ~ProgramState();

    const std::string &getLabel();

    Shader *getAttachedShader(ShaderType shaderType) const;
    const gl::ShaderMap<Shader *> &getAttachedShaders() const { return mAttachedShaders; }
    const std::vector<std::string> &getTransformFeedbackVaryingNames() const
    {
        return mTransformFeedbackVaryingNames;
    }
    GLint getTransformFeedbackBufferMode() const { return mTransformFeedbackBufferMode; }
    GLuint getUniformBlockBinding(GLuint uniformBlockIndex) const
    {
        ASSERT(uniformBlockIndex < mUniformBlocks.size());
        return mUniformBlocks[uniformBlockIndex].binding;
    }
    GLuint getShaderStorageBlockBinding(GLuint blockIndex) const
    {
        ASSERT(blockIndex < mShaderStorageBlocks.size());
        return mShaderStorageBlocks[blockIndex].binding;
    }
    const UniformBlockBindingMask &getActiveUniformBlockBindingsMask() const
    {
        return mActiveUniformBlockBindings;
    }
    const std::vector<sh::ShaderVariable> &getProgramInputs() const { return mProgramInputs; }
    const AttributesMask &getActiveAttribLocationsMask() const
    {
        return mActiveAttribLocationsMask;
    }
    const AttributesMask &getNonBuiltinAttribLocationsMask() const { return mAttributesMask; }
    unsigned int getMaxActiveAttribLocation() const { return mMaxActiveAttribLocation; }
    DrawBufferMask getActiveOutputVariables() const { return mActiveOutputVariables; }
    const std::vector<sh::ShaderVariable> &getOutputVariables() const { return mOutputVariables; }
    const std::vector<VariableLocation> &getOutputLocations() const { return mOutputLocations; }
    const std::vector<VariableLocation> &getSecondaryOutputLocations() const
    {
        return mSecondaryOutputLocations;
    }
    const std::vector<LinkedUniform> &getUniforms() const { return mUniforms; }
    const std::vector<VariableLocation> &getUniformLocations() const { return mUniformLocations; }
    const std::vector<InterfaceBlock> &getUniformBlocks() const { return mUniformBlocks; }
    const std::vector<InterfaceBlock> &getShaderStorageBlocks() const
    {
        return mShaderStorageBlocks;
    }
    const std::vector<BufferVariable> &getBufferVariables() const { return mBufferVariables; }
    const std::vector<SamplerBinding> &getSamplerBindings() const { return mSamplerBindings; }
    const std::vector<ImageBinding> &getImageBindings() const { return mImageBindings; }
    const sh::WorkGroupSize &getComputeShaderLocalSize() const { return mComputeShaderLocalSize; }
    const RangeUI &getSamplerUniformRange() const { return mSamplerUniformRange; }
    const RangeUI &getImageUniformRange() const { return mImageUniformRange; }
    const RangeUI &getAtomicCounterUniformRange() const { return mAtomicCounterUniformRange; }
    const ComponentTypeMask &getAttributesTypeMask() const { return mAttributesTypeMask; }

    const std::vector<TransformFeedbackVarying> &getLinkedTransformFeedbackVaryings() const
    {
        return mLinkedTransformFeedbackVaryings;
    }
    const std::vector<GLsizei> &getTransformFeedbackStrides() const
    {
        return mTransformFeedbackStrides;
    }
    size_t getTransformFeedbackBufferCount() const { return mTransformFeedbackStrides.size(); }
    const std::vector<AtomicCounterBuffer> &getAtomicCounterBuffers() const
    {
        return mAtomicCounterBuffers;
    }

    // Count the number of uniform and storage buffer declarations, counting arrays as one.
    size_t getUniqueUniformBlockCount() const;
    size_t getUniqueStorageBlockCount() const;

    GLuint getUniformIndexFromName(const std::string &name) const;
    GLuint getUniformIndexFromLocation(GLint location) const;
    Optional<GLuint> getSamplerIndex(GLint location) const;
    bool isSamplerUniformIndex(GLuint index) const;
    GLuint getSamplerIndexFromUniformIndex(GLuint uniformIndex) const;
    GLuint getUniformIndexFromSamplerIndex(GLuint samplerIndex) const;
    bool isImageUniformIndex(GLuint index) const;
    GLuint getImageIndexFromUniformIndex(GLuint uniformIndex) const;
    GLuint getUniformIndexFromImageIndex(GLuint imageIndex) const;
    GLuint getAttributeLocation(const std::string &name) const;

    GLuint getBufferVariableIndexFromName(const std::string &name) const;

    int getNumViews() const { return mNumViews; }
    bool usesMultiview() const { return mNumViews != -1; }

    const ShaderBitSet &getLinkedShaderStages() const { return mLinkedShaderStages; }
    bool hasLinkedShaderStage(ShaderType shaderType) const
    {
        return mLinkedShaderStages[shaderType];
    }
    size_t getLinkedShaderStageCount() const { return mLinkedShaderStages.count(); }
    bool isCompute() const { return hasLinkedShaderStage(ShaderType::Compute); }

    bool hasAttachedShader() const;

    const ActiveTextureMask &getActiveSamplersMask() const { return mActiveSamplersMask; }
    SamplerFormat getSamplerFormatForTextureUnitIndex(size_t textureUnitIndex) const
    {
        return mActiveSamplerFormats[textureUnitIndex];
    }
    ShaderType getFirstAttachedShaderStageType() const;
    ShaderType getLastAttachedShaderStageType() const;

  private:
    friend class MemoryProgramCache;
    friend class Program;

    void updateTransformFeedbackStrides();
    void updateActiveSamplers();
    void updateActiveImages();
    void updateProgramInterfaceInputs();
    void updateProgramInterfaceOutputs();

    // Scans the sampler bindings for type conflicts with sampler 'textureUnitIndex'.
    void setSamplerUniformTextureTypeAndFormat(size_t textureUnitIndex);

    std::string mLabel;

    sh::WorkGroupSize mComputeShaderLocalSize;

    ShaderMap<Shader *> mAttachedShaders;

    std::vector<std::string> mTransformFeedbackVaryingNames;
    std::vector<TransformFeedbackVarying> mLinkedTransformFeedbackVaryings;
    GLenum mTransformFeedbackBufferMode;

    // For faster iteration on the blocks currently being bound.
    UniformBlockBindingMask mActiveUniformBlockBindings;

    // Vertex attributes, Fragment input varyings, etc.
    std::vector<sh::ShaderVariable> mProgramInputs;
    angle::BitSet<MAX_VERTEX_ATTRIBS> mActiveAttribLocationsMask;
    unsigned int mMaxActiveAttribLocation;
    ComponentTypeMask mAttributesTypeMask;
    // mAttributesMask is identical to mActiveAttribLocationsMask with built-in attributes removed.
    AttributesMask mAttributesMask;

    // Uniforms are sorted in order:
    //  1. Non-opaque uniforms
    //  2. Sampler uniforms
    //  3. Image uniforms
    //  4. Atomic counter uniforms
    //  5. Uniform block uniforms
    // This makes opaque uniform validation easier, since we don't need a separate list.
    // For generating the entries and naming them we follow the spec: GLES 3.1 November 2016 section
    // 7.3.1.1 Naming Active Resources. There's a separate entry for each struct member and each
    // inner array of an array of arrays. Names and mapped names of uniforms that are arrays include
    // [0] in the end. This makes implementation of queries simpler.
    std::vector<LinkedUniform> mUniforms;

    std::vector<VariableLocation> mUniformLocations;
    std::vector<InterfaceBlock> mUniformBlocks;
    std::vector<BufferVariable> mBufferVariables;
    std::vector<InterfaceBlock> mShaderStorageBlocks;
    std::vector<AtomicCounterBuffer> mAtomicCounterBuffers;
    RangeUI mSamplerUniformRange;
    RangeUI mImageUniformRange;
    RangeUI mAtomicCounterUniformRange;

    // An array of the samplers that are used by the program
    std::vector<SamplerBinding> mSamplerBindings;

    // An array of the images that are used by the program
    std::vector<ImageBinding> mImageBindings;

    // Names and mapped names of output variables that are arrays include [0] in the end, similarly
    // to uniforms.
    std::vector<sh::ShaderVariable> mOutputVariables;
    std::vector<VariableLocation> mOutputLocations;

    // EXT_blend_func_extended secondary outputs (ones with index 1) in ESSL 3.00 shaders.
    std::vector<VariableLocation> mSecondaryOutputLocations;

    DrawBufferMask mActiveOutputVariables;

    // Fragment output variable base types: FLOAT, INT, or UINT.  Ordered by location.
    std::vector<GLenum> mOutputVariableTypes;
    ComponentTypeMask mDrawBufferTypeMask;

    bool mBinaryRetrieveableHint;
    bool mSeparable;
    ShaderBitSet mLinkedShaderStages;

    // ANGLE_multiview.
    int mNumViews;

    // GL_EXT_geometry_shader.
    PrimitiveMode mGeometryShaderInputPrimitiveType;
    PrimitiveMode mGeometryShaderOutputPrimitiveType;
    int mGeometryShaderInvocations;
    int mGeometryShaderMaxVertices;

    // GL_ANGLE_multi_draw
    int mDrawIDLocation;

    // GL_ANGLE_base_vertex_base_instance
    int mBaseVertexLocation;
    int mBaseInstanceLocation;
    // Cached value of base vertex and base instance
    // need to reset them to zero if using non base vertex or base instance draw calls.
    GLint mCachedBaseVertex;
    GLuint mCachedBaseInstance;

    // The size of the data written to each transform feedback buffer per vertex.
    std::vector<GLsizei> mTransformFeedbackStrides;

    // Cached mask of active samplers and sampler types.
    ActiveTextureMask mActiveSamplersMask;
    ActiveTextureArray<uint32_t> mActiveSamplerRefCounts;
    ActiveTextureArray<TextureType> mActiveSamplerTypes;
    ActiveTextureArray<SamplerFormat> mActiveSamplerFormats;

    // Cached mask of active images.
    ActiveTextureMask mActiveImagesMask;
};

struct ProgramBinding
{
    ProgramBinding() : location(GL_INVALID_INDEX), aliased(false) {}
    ProgramBinding(GLuint index) : location(index), aliased(false) {}

    GLuint location;
    // Whether another binding was set that may potentially alias this.
    bool aliased;
};

class ProgramBindings final : angle::NonCopyable
{
  public:
    ProgramBindings();
    ~ProgramBindings();

    void bindLocation(GLuint index, const std::string &name);
    int getBindingByName(const std::string &name) const;
    int getBinding(const sh::ShaderVariable &variable) const;

    using const_iterator = std::unordered_map<std::string, GLuint>::const_iterator;
    const_iterator begin() const;
    const_iterator end() const;

  private:
    std::unordered_map<std::string, GLuint> mBindings;
};

// Uniforms and Fragment Outputs require special treatment due to array notation (e.g., "[0]")
class ProgramAliasedBindings final : angle::NonCopyable
{
  public:
    ProgramAliasedBindings();
    ~ProgramAliasedBindings();

    void bindLocation(GLuint index, const std::string &name);
    int getBindingByName(const std::string &name) const;
    int getBinding(const sh::ShaderVariable &variable) const;

    using const_iterator = std::unordered_map<std::string, ProgramBinding>::const_iterator;
    const_iterator begin() const;
    const_iterator end() const;

  private:
    std::unordered_map<std::string, ProgramBinding> mBindings;
};

struct ProgramVaryingRef
{
    const sh::ShaderVariable *get() const { return frontShader ? frontShader : backShader; }

    const sh::ShaderVariable *frontShader = nullptr;
    const sh::ShaderVariable *backShader  = nullptr;
};

using ProgramMergedVaryings = std::map<std::string, ProgramVaryingRef>;

class Program final : angle::NonCopyable, public LabeledObject
{
  public:
    Program(rx::GLImplFactory *factory, ShaderProgramManager *manager, ShaderProgramID handle);
    void onDestroy(const Context *context);

    ShaderProgramID id() const;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    ANGLE_INLINE rx::ProgramImpl *getImplementation() const
    {
        ASSERT(mLinkResolved);
        return mProgram;
    }

    void attachShader(Shader *shader);
    void detachShader(const Context *context, Shader *shader);
    int getAttachedShadersCount() const;

    const Shader *getAttachedShader(ShaderType shaderType) const;

    void bindAttributeLocation(GLuint index, const char *name);
    void bindUniformLocation(GLuint index, const char *name);

    // CHROMIUM_path_rendering
    BindingInfo getFragmentInputBindingInfo(GLint index) const;
    void bindFragmentInputLocation(GLint index, const char *name);
    void pathFragmentInputGen(GLint index, GLenum genMode, GLint components, const GLfloat *coeffs);

    // EXT_blend_func_extended
    void bindFragmentOutputLocation(GLuint index, const char *name);
    void bindFragmentOutputIndex(GLuint index, const char *name);

    // KHR_parallel_shader_compile
    // Try to link the program asynchrously. As a result, background threads may be launched to
    // execute the linking tasks concurrently.
    angle::Result link(const Context *context);

    // Peek whether there is any running linking tasks.
    bool isLinking() const;

    bool isLinked() const
    {
        ASSERT(mLinkResolved);
        return mLinked;
    }

    bool hasLinkedShaderStage(ShaderType shaderType) const
    {
        ASSERT(shaderType != ShaderType::InvalidEnum);
        return mState.hasLinkedShaderStage(shaderType);
    }
    bool isCompute() const { return mState.isCompute(); }

    angle::Result loadBinary(const Context *context,
                             GLenum binaryFormat,
                             const void *binary,
                             GLsizei length);
    angle::Result saveBinary(Context *context,
                             GLenum *binaryFormat,
                             void *binary,
                             GLsizei bufSize,
                             GLsizei *length) const;
    GLint getBinaryLength(Context *context) const;
    void setBinaryRetrievableHint(bool retrievable);
    bool getBinaryRetrievableHint() const;

    void setSeparable(bool separable);
    bool isSeparable() const;

    int getInfoLogLength() const;
    void getInfoLog(GLsizei bufSize, GLsizei *length, char *infoLog) const;
    void getAttachedShaders(GLsizei maxCount, GLsizei *count, ShaderProgramID *shaders) const;

    GLuint getAttributeLocation(const std::string &name) const;
    bool isAttribLocationActive(size_t attribLocation) const;

    void getActiveAttribute(GLuint index,
                            GLsizei bufsize,
                            GLsizei *length,
                            GLint *size,
                            GLenum *type,
                            GLchar *name) const;
    GLint getActiveAttributeCount() const;
    GLint getActiveAttributeMaxLength() const;
    const std::vector<sh::ShaderVariable> &getAttributes() const;

    GLint getFragDataLocation(const std::string &name) const;
    size_t getOutputResourceCount() const;
    const std::vector<GLenum> &getOutputVariableTypes() const;
    DrawBufferMask getActiveOutputVariables() const
    {
        ASSERT(mLinkResolved);
        return mState.mActiveOutputVariables;
    }

    // EXT_blend_func_extended
    GLint getFragDataIndex(const std::string &name) const;

    void getActiveUniform(GLuint index,
                          GLsizei bufsize,
                          GLsizei *length,
                          GLint *size,
                          GLenum *type,
                          GLchar *name) const;
    GLint getActiveUniformCount() const;
    size_t getActiveBufferVariableCount() const;
    GLint getActiveUniformMaxLength() const;
    bool isValidUniformLocation(GLint location) const;
    const LinkedUniform &getUniformByLocation(GLint location) const;
    const VariableLocation &getUniformLocation(GLint location) const;

    const std::vector<VariableLocation> &getUniformLocations() const
    {
        ASSERT(mLinkResolved);
        return mState.mUniformLocations;
    }

    const LinkedUniform &getUniformByIndex(GLuint index) const
    {
        ASSERT(mLinkResolved);
        ASSERT(index < static_cast<size_t>(mState.mUniforms.size()));
        return mState.mUniforms[index];
    }

    const BufferVariable &getBufferVariableByIndex(GLuint index) const;

    enum SetUniformResult
    {
        SamplerChanged,
        NoSamplerChange,
    };

    GLint getUniformLocation(const std::string &name) const;
    GLuint getUniformIndex(const std::string &name) const;
    void setUniform1fv(GLint location, GLsizei count, const GLfloat *v);
    void setUniform2fv(GLint location, GLsizei count, const GLfloat *v);
    void setUniform3fv(GLint location, GLsizei count, const GLfloat *v);
    void setUniform4fv(GLint location, GLsizei count, const GLfloat *v);
    void setUniform1iv(Context *context, GLint location, GLsizei count, const GLint *v);
    void setUniform2iv(GLint location, GLsizei count, const GLint *v);
    void setUniform3iv(GLint location, GLsizei count, const GLint *v);
    void setUniform4iv(GLint location, GLsizei count, const GLint *v);
    void setUniform1uiv(GLint location, GLsizei count, const GLuint *v);
    void setUniform2uiv(GLint location, GLsizei count, const GLuint *v);
    void setUniform3uiv(GLint location, GLsizei count, const GLuint *v);
    void setUniform4uiv(GLint location, GLsizei count, const GLuint *v);
    void setUniformMatrix2fv(GLint location,
                             GLsizei count,
                             GLboolean transpose,
                             const GLfloat *value);
    void setUniformMatrix3fv(GLint location,
                             GLsizei count,
                             GLboolean transpose,
                             const GLfloat *value);
    void setUniformMatrix4fv(GLint location,
                             GLsizei count,
                             GLboolean transpose,
                             const GLfloat *value);
    void setUniformMatrix2x3fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value);
    void setUniformMatrix3x2fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value);
    void setUniformMatrix2x4fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value);
    void setUniformMatrix4x2fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value);
    void setUniformMatrix3x4fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value);
    void setUniformMatrix4x3fv(GLint location,
                               GLsizei count,
                               GLboolean transpose,
                               const GLfloat *value);

    void getUniformfv(const Context *context, GLint location, GLfloat *params) const;
    void getUniformiv(const Context *context, GLint location, GLint *params) const;
    void getUniformuiv(const Context *context, GLint location, GLuint *params) const;

    void getActiveUniformBlockName(const GLuint blockIndex,
                                   GLsizei bufSize,
                                   GLsizei *length,
                                   GLchar *blockName) const;
    void getActiveShaderStorageBlockName(const GLuint blockIndex,
                                         GLsizei bufSize,
                                         GLsizei *length,
                                         GLchar *blockName) const;

    ANGLE_INLINE GLuint getActiveUniformBlockCount() const
    {
        ASSERT(mLinkResolved);
        return static_cast<GLuint>(mState.mUniformBlocks.size());
    }

    ANGLE_INLINE GLuint getActiveAtomicCounterBufferCount() const
    {
        ASSERT(mLinkResolved);
        return static_cast<GLuint>(mState.mAtomicCounterBuffers.size());
    }

    ANGLE_INLINE GLuint getActiveShaderStorageBlockCount() const
    {
        ASSERT(mLinkResolved);
        return static_cast<GLuint>(mState.mShaderStorageBlocks.size());
    }

    GLint getActiveUniformBlockMaxNameLength() const;
    GLint getActiveShaderStorageBlockMaxNameLength() const;

    GLuint getUniformBlockIndex(const std::string &name) const;
    GLuint getShaderStorageBlockIndex(const std::string &name) const;

    void bindUniformBlock(GLuint uniformBlockIndex, GLuint uniformBlockBinding);
    GLuint getUniformBlockBinding(GLuint uniformBlockIndex) const;
    GLuint getShaderStorageBlockBinding(GLuint shaderStorageBlockIndex) const;

    const InterfaceBlock &getUniformBlockByIndex(GLuint index) const;
    const InterfaceBlock &getShaderStorageBlockByIndex(GLuint index) const;

    void setTransformFeedbackVaryings(GLsizei count,
                                      const GLchar *const *varyings,
                                      GLenum bufferMode);
    void getTransformFeedbackVarying(GLuint index,
                                     GLsizei bufSize,
                                     GLsizei *length,
                                     GLsizei *size,
                                     GLenum *type,
                                     GLchar *name) const;
    GLsizei getTransformFeedbackVaryingCount() const;
    GLsizei getTransformFeedbackVaryingMaxLength() const;
    GLenum getTransformFeedbackBufferMode() const;
    GLuint getTransformFeedbackVaryingResourceIndex(const GLchar *name) const;
    const TransformFeedbackVarying &getTransformFeedbackVaryingResource(GLuint index) const;

    bool hasDrawIDUniform() const;
    void setDrawIDUniform(GLint drawid);

    bool hasBaseVertexUniform() const;
    void setBaseVertexUniform(GLint baseVertex);
    bool hasBaseInstanceUniform() const;
    void setBaseInstanceUniform(GLuint baseInstance);

    ANGLE_INLINE void addRef()
    {
        ASSERT(mLinkResolved);
        mRefCount++;
    }

    ANGLE_INLINE void release(const Context *context)
    {
        ASSERT(mLinkResolved);
        mRefCount--;

        if (mRefCount == 0 && mDeleteStatus)
        {
            deleteSelf(context);
        }
    }

    unsigned int getRefCount() const;
    bool isInUse() const { return getRefCount() != 0; }
    void flagForDeletion();
    bool isFlaggedForDeletion() const;

    void validate(const Caps &caps);
    bool validateSamplers(InfoLog *infoLog, const Caps &caps)
    {
        // Skip cache if we're using an infolog, so we get the full error.
        // Also skip the cache if the sample mapping has changed, or if we haven't ever validated.
        if (infoLog == nullptr && mCachedValidateSamplersResult.valid())
        {
            return mCachedValidateSamplersResult.value();
        }

        return validateSamplersImpl(infoLog, caps);
    }

    bool isValidated() const;

    const AttributesMask &getActiveAttribLocationsMask() const
    {
        ASSERT(mLinkResolved);
        return mState.mActiveAttribLocationsMask;
    }

    const std::vector<SamplerBinding> &getSamplerBindings() const;
    const std::vector<ImageBinding> &getImageBindings() const
    {
        ASSERT(mLinkResolved);
        return mState.mImageBindings;
    }
    const sh::WorkGroupSize &getComputeShaderLocalSize() const;
    PrimitiveMode getGeometryShaderInputPrimitiveType() const;
    PrimitiveMode getGeometryShaderOutputPrimitiveType() const;
    GLint getGeometryShaderInvocations() const;
    GLint getGeometryShaderMaxVertices() const;

    const ProgramState &getState() const
    {
        ASSERT(mLinkResolved);
        return mState;
    }

    static LinkMismatchError LinkValidateVariablesBase(
        const sh::ShaderVariable &variable1,
        const sh::ShaderVariable &variable2,
        bool validatePrecision,
        bool validateArraySize,
        std::string *mismatchedStructOrBlockMemberName);

    GLuint getInputResourceIndex(const GLchar *name) const;
    GLuint getOutputResourceIndex(const GLchar *name) const;
    void getInputResourceName(GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name) const;
    void getOutputResourceName(GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name) const;
    void getUniformResourceName(GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name) const;
    void getBufferVariableResourceName(GLuint index,
                                       GLsizei bufSize,
                                       GLsizei *length,
                                       GLchar *name) const;
    const sh::ShaderVariable &getInputResource(size_t index) const;
    GLuint getResourceMaxNameSize(const sh::ShaderVariable &resource, GLint max) const;
    GLuint getInputResourceMaxNameSize() const;
    GLuint getOutputResourceMaxNameSize() const;
    GLuint getResourceLocation(const GLchar *name, const sh::ShaderVariable &variable) const;
    GLuint getInputResourceLocation(const GLchar *name) const;
    GLuint getOutputResourceLocation(const GLchar *name) const;
    const std::string getResourceName(const sh::ShaderVariable &resource) const;
    const std::string getInputResourceName(GLuint index) const;
    const std::string getOutputResourceName(GLuint index) const;
    const sh::ShaderVariable &getOutputResource(size_t index) const;

    const ProgramBindings &getAttributeBindings() const;
    const ProgramBindings &getFragmentInputBindings() const;
    const ProgramAliasedBindings &getUniformLocationBindings() const;

    int getNumViews() const
    {
        ASSERT(mLinkResolved);
        return mState.getNumViews();
    }

    bool usesMultiview() const { return mState.usesMultiview(); }

    ComponentTypeMask getDrawBufferTypeMask() const;
    ComponentTypeMask getAttributesTypeMask() const;
    AttributesMask getAttributesMask() const;

    const std::vector<GLsizei> &getTransformFeedbackStrides() const;

    const ActiveTextureMask &getActiveSamplersMask() const { return mState.mActiveSamplersMask; }
    const ActiveTextureMask &getActiveImagesMask() const { return mState.mActiveImagesMask; }

    const ActiveTextureArray<TextureType> &getActiveSamplerTypes() const
    {
        return mState.mActiveSamplerTypes;
    }

    // Program dirty bits.
    enum DirtyBitType
    {
        DIRTY_BIT_UNIFORM_BLOCK_BINDING_0,
        DIRTY_BIT_UNIFORM_BLOCK_BINDING_MAX =
            DIRTY_BIT_UNIFORM_BLOCK_BINDING_0 + IMPLEMENTATION_MAX_COMBINED_SHADER_UNIFORM_BUFFERS,

        DIRTY_BIT_COUNT = DIRTY_BIT_UNIFORM_BLOCK_BINDING_MAX,
    };

    using DirtyBits = angle::BitSet<DIRTY_BIT_COUNT>;

    angle::Result syncState(const Context *context);

    // Try to resolve linking. Inlined to make sure its overhead is as low as possible.
    void resolveLink(const Context *context)
    {
        if (!mLinkResolved)
        {
            resolveLinkImpl(context);
        }
    }

    ANGLE_INLINE bool hasAnyDirtyBit() const { return mDirtyBits.any(); }

    // Writes a program's binary to the output memory buffer.
    void serialize(const Context *context, angle::MemoryBuffer *binaryOut) const;

  private:
    struct LinkingState;

    ~Program() override;

    // Loads program state according to the specified binary blob.
    angle::Result deserialize(const Context *context, BinaryInputStream &stream, InfoLog &infoLog);

    void unlink();
    void deleteSelf(const Context *context);

    bool linkValidateShaders(InfoLog &infoLog);
    bool linkAttributes(const Context *context, InfoLog &infoLog);
    bool linkInterfaceBlocks(const Caps &caps,
                             const Version &version,
                             bool webglCompatibility,
                             InfoLog &infoLog,
                             GLuint *combinedShaderStorageBlocksCount);
    bool linkVaryings(InfoLog &infoLog) const;

    bool linkUniforms(const Caps &caps,
                      InfoLog &infoLog,
                      const ProgramAliasedBindings &uniformLocationBindings,
                      GLuint *combinedImageUniformsCount,
                      std::vector<UnusedUniform> *unusedUniforms);
    void linkSamplerAndImageBindings(GLuint *combinedImageUniformsCount);
    bool linkAtomicCounterBuffers();

    void updateLinkedShaderStages();

    static LinkMismatchError LinkValidateVaryings(const sh::ShaderVariable &outputVarying,
                                                  const sh::ShaderVariable &inputVarying,
                                                  int shaderVersion,
                                                  bool validateGeometryShaderInputVarying,
                                                  std::string *mismatchedStructFieldName);

    bool linkValidateShaderInterfaceMatching(Shader *generatingShader,
                                             Shader *consumingShader,
                                             InfoLog &infoLog) const;

    // Check for aliased path rendering input bindings (if any).
    // If more than one binding refer statically to the same location the link must fail.
    bool linkValidateFragmentInputBindings(InfoLog &infoLog) const;

    bool linkValidateBuiltInVaryings(InfoLog &infoLog) const;
    bool linkValidateTransformFeedback(const Version &version,
                                       InfoLog &infoLog,
                                       const ProgramMergedVaryings &linkedVaryings,
                                       const Caps &caps) const;
    bool linkValidateGlobalNames(InfoLog &infoLog) const;

    void gatherTransformFeedbackVaryings(const ProgramMergedVaryings &varyings);

    ProgramMergedVaryings getMergedVaryings() const;
    int getOutputLocationForLink(const sh::ShaderVariable &outputVariable) const;
    bool isOutputSecondaryForLink(const sh::ShaderVariable &outputVariable) const;
    bool linkOutputVariables(const Caps &caps,
                             const Extensions &extensions,
                             const Version &version,
                             GLuint combinedImageUniformsCount,
                             GLuint combinedShaderStorageBlocksCount);

    void setUniformValuesFromBindingQualifiers();

    void initInterfaceBlockBindings();

    // Both these function update the cached uniform values and return a modified "count"
    // so that the uniform update doesn't overflow the uniform.
    template <typename T>
    GLsizei clampUniformCount(const VariableLocation &locationInfo,
                              GLsizei count,
                              int vectorSize,
                              const T *v);
    template <size_t cols, size_t rows, typename T>
    GLsizei clampMatrixUniformCount(GLint location, GLsizei count, GLboolean transpose, const T *v);

    void updateSamplerUniform(Context *context,
                              const VariableLocation &locationInfo,
                              GLsizei clampedCount,
                              const GLint *v);

    template <typename DestT>
    void getUniformInternal(const Context *context,
                            DestT *dataOut,
                            GLint location,
                            GLenum nativeType,
                            int components) const;

    void getResourceName(const std::string name,
                         GLsizei bufSize,
                         GLsizei *length,
                         GLchar *dest) const;

    template <typename T>
    GLint getActiveInterfaceBlockMaxNameLength(const std::vector<T> &resources) const;

    GLuint getSamplerUniformBinding(const VariableLocation &uniformLocation) const;
    GLuint getImageUniformBinding(const VariableLocation &uniformLocation) const;

    bool validateSamplersImpl(InfoLog *infoLog, const Caps &caps);

    // Block until linking is finished and resolve it.
    void resolveLinkImpl(const gl::Context *context);

    void postResolveLink(const gl::Context *context);

    ProgramState mState;
    rx::ProgramImpl *mProgram;

    bool mValidated;

    ProgramBindings mAttributeBindings;

    // Note that this has nothing to do with binding layout qualifiers that can be set for some
    // uniforms in GLES3.1+. It is used to pre-set the location of uniforms.
    ProgramAliasedBindings mUniformLocationBindings;

    // CHROMIUM_path_rendering
    ProgramBindings mFragmentInputBindings;

    // EXT_blend_func_extended
    ProgramAliasedBindings mFragmentOutputLocations;
    ProgramAliasedBindings mFragmentOutputIndexes;

    bool mLinked;
    bool mLinkResolved;
    std::unique_ptr<LinkingState> mLinkingState;
    bool mDeleteStatus;  // Flag to indicate that the program can be deleted when no longer in use

    unsigned int mRefCount;

    ShaderProgramManager *mResourceManager;
    const ShaderProgramID mHandle;

    InfoLog mInfoLog;

    // Cache for sampler validation
    Optional<bool> mCachedValidateSamplersResult;

    DirtyBits mDirtyBits;
};
}  // namespace gl

#endif  // LIBANGLE_PROGRAM_H_
