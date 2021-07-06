//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Shader.h: Defines the abstract gl::Shader class and its concrete derived
// classes VertexShader and FragmentShader. Implements GL shader objects and
// related functionality. [OpenGL ES 2.0.24] section 2.10 page 24 and section
// 3.8 page 84.

#ifndef LIBANGLE_SHADER_H_
#define LIBANGLE_SHADER_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#include <GLSLANG/ShaderLang.h>
#include "angle_gl.h"

#include "common/Optional.h"
#include "common/angleutils.h"
#include "libANGLE/Compiler.h"
#include "libANGLE/Debug.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class GLImplFactory;
class ShaderImpl;
class ShaderSh;
class WaitableCompileEvent;
}  // namespace rx

namespace angle
{
class WaitableEvent;
class WorkerThreadPool;
}  // namespace angle

namespace gl
{
class CompileTask;
class Context;
struct Limitations;
class ShaderProgramManager;
class State;

// We defer the compile until link time, or until properties are queried.
enum class CompileStatus
{
    NOT_COMPILED,
    COMPILE_REQUESTED,
    COMPILED,
};

class ShaderState final : angle::NonCopyable
{
  public:
    ShaderState(ShaderType shaderType);
    ~ShaderState();

    const std::string &getLabel() const { return mLabel; }

    const std::string &getSource() const { return mSource; }
    const std::string &getTranslatedSource() const { return mTranslatedSource; }

    ShaderType getShaderType() const { return mShaderType; }
    int getShaderVersion() const { return mShaderVersion; }

    const std::vector<sh::ShaderVariable> &getInputVaryings() const { return mInputVaryings; }
    const std::vector<sh::ShaderVariable> &getOutputVaryings() const { return mOutputVaryings; }
    const std::vector<sh::ShaderVariable> &getUniforms() const { return mUniforms; }
    const std::vector<sh::InterfaceBlock> &getUniformBlocks() const { return mUniformBlocks; }
    const std::vector<sh::InterfaceBlock> &getShaderStorageBlocks() const
    {
        return mShaderStorageBlocks;
    }
    const std::vector<sh::ShaderVariable> &getActiveAttributes() const { return mActiveAttributes; }
    const std::vector<sh::ShaderVariable> &getAllAttributes() const { return mAllAttributes; }
    const std::vector<sh::ShaderVariable> &getActiveOutputVariables() const
    {
        return mActiveOutputVariables;
    }

    bool compilePending() const { return mCompileStatus == CompileStatus::COMPILE_REQUESTED; }

  private:
    friend class Shader;

    std::string mLabel;

    ShaderType mShaderType;
    int mShaderVersion;
    std::string mTranslatedSource;
    std::string mSource;

    sh::WorkGroupSize mLocalSize;

    std::vector<sh::ShaderVariable> mInputVaryings;
    std::vector<sh::ShaderVariable> mOutputVaryings;
    std::vector<sh::ShaderVariable> mUniforms;
    std::vector<sh::InterfaceBlock> mUniformBlocks;
    std::vector<sh::InterfaceBlock> mShaderStorageBlocks;
    std::vector<sh::ShaderVariable> mAllAttributes;
    std::vector<sh::ShaderVariable> mActiveAttributes;
    std::vector<sh::ShaderVariable> mActiveOutputVariables;

    // ANGLE_multiview.
    int mNumViews;

    // Geometry Shader.
    Optional<PrimitiveMode> mGeometryShaderInputPrimitiveType;
    Optional<PrimitiveMode> mGeometryShaderOutputPrimitiveType;
    Optional<GLint> mGeometryShaderMaxVertices;
    int mGeometryShaderInvocations;

    // Indicates if this shader has been successfully compiled
    CompileStatus mCompileStatus;
};

class Shader final : angle::NonCopyable, public LabeledObject
{
  public:
    Shader(ShaderProgramManager *manager,
           rx::GLImplFactory *implFactory,
           const gl::Limitations &rendererLimitations,
           ShaderType type,
           ShaderProgramID handle);

    void onDestroy(const Context *context);

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    ShaderType getType() const { return mType; }
    ShaderProgramID getHandle() const;

    rx::ShaderImpl *getImplementation() const { return mImplementation.get(); }

    void setSource(GLsizei count, const char *const *string, const GLint *length);
    int getInfoLogLength();
    void getInfoLog(GLsizei bufSize, GLsizei *length, char *infoLog);
    int getSourceLength() const;
    const std::string &getSourceString() const { return mState.getSource(); }
    void getSource(GLsizei bufSize, GLsizei *length, char *buffer) const;
    int getTranslatedSourceLength();
    int getTranslatedSourceWithDebugInfoLength();
    const std::string &getTranslatedSource();
    void getTranslatedSource(GLsizei bufSize, GLsizei *length, char *buffer);
    void getTranslatedSourceWithDebugInfo(GLsizei bufSize, GLsizei *length, char *buffer);

    void compile(const Context *context);
    bool isCompiled();
    bool isCompleted();

    void addRef();
    void release(const Context *context);
    unsigned int getRefCount() const;
    bool isFlaggedForDeletion() const;
    void flagForDeletion();

    int getShaderVersion();

    const std::vector<sh::ShaderVariable> &getInputVaryings();
    const std::vector<sh::ShaderVariable> &getOutputVaryings();
    const std::vector<sh::ShaderVariable> &getUniforms();
    const std::vector<sh::InterfaceBlock> &getUniformBlocks();
    const std::vector<sh::InterfaceBlock> &getShaderStorageBlocks();
    const std::vector<sh::ShaderVariable> &getActiveAttributes();
    const std::vector<sh::ShaderVariable> &getAllAttributes();
    const std::vector<sh::ShaderVariable> &getActiveOutputVariables();

    // Returns mapped name of a transform feedback varying. The original name may contain array
    // brackets with an index inside, which will get copied to the mapped name. The varying must be
    // known to be declared in the shader.
    std::string getTransformFeedbackVaryingMappedName(const std::string &tfVaryingName);

    const sh::WorkGroupSize &getWorkGroupSize();

    int getNumViews();

    Optional<PrimitiveMode> getGeometryShaderInputPrimitiveType();
    Optional<PrimitiveMode> getGeometryShaderOutputPrimitiveType();
    int getGeometryShaderInvocations();
    Optional<GLint> getGeometryShaderMaxVertices();

    const std::string &getCompilerResourcesString() const;

  private:
    struct CompilingState;

    ~Shader() override;
    static void GetSourceImpl(const std::string &source,
                              GLsizei bufSize,
                              GLsizei *length,
                              char *buffer);

    void resolveCompile();

    ShaderState mState;
    std::unique_ptr<rx::ShaderImpl> mImplementation;
    const gl::Limitations &mRendererLimitations;
    const ShaderProgramID mHandle;
    const ShaderType mType;
    unsigned int mRefCount;  // Number of program objects this shader is attached to
    bool mDeleteStatus;  // Flag to indicate that the shader can be deleted when no longer in use
    std::string mInfoLog;

    // We keep a reference to the translator in order to defer compiles while preserving settings.
    BindingPointer<Compiler> mBoundCompiler;
    std::unique_ptr<CompilingState> mCompilingState;
    std::string mCompilerResourcesString;

    ShaderProgramManager *mResourceManager;

    GLuint mCurrentMaxComputeWorkGroupInvocations;
};

bool CompareShaderVar(const sh::ShaderVariable &x, const sh::ShaderVariable &y);

const char *GetShaderTypeString(ShaderType type);
}  // namespace gl

#endif  // LIBANGLE_SHADER_H_
