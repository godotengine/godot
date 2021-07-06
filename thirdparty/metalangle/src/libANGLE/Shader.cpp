//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Shader.cpp: Implements the gl::Shader class and its  derived classes
// VertexShader and FragmentShader. Implements GL shader objects and related
// functionality. [OpenGL ES 2.0.24] section 2.10 page 24 and section 3.8 page 84.

#include "libANGLE/Shader.h"

#include <functional>
#include <sstream>

#include "GLSLANG/ShaderLang.h"
#include "common/utilities.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Compiler.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Context.h"
#include "libANGLE/ResourceManager.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/ShaderImpl.h"
#include "platform/FrontendFeatures.h"

namespace gl
{

namespace
{
template <typename VarT>
std::vector<VarT> GetActiveShaderVariables(const std::vector<VarT> *variableList)
{
    ASSERT(variableList);
    std::vector<VarT> result;
    for (size_t varIndex = 0; varIndex < variableList->size(); varIndex++)
    {
        const VarT &var = variableList->at(varIndex);
        if (var.active)
        {
            result.push_back(var);
        }
    }
    return result;
}

template <typename VarT>
const std::vector<VarT> &GetShaderVariables(const std::vector<VarT> *variableList)
{
    ASSERT(variableList);
    return *variableList;
}

}  // anonymous namespace

// true if varying x has a higher priority in packing than y
bool CompareShaderVar(const sh::ShaderVariable &x, const sh::ShaderVariable &y)
{
    if (x.type == y.type)
    {
        return x.getArraySizeProduct() > y.getArraySizeProduct();
    }

    // Special case for handling structs: we sort these to the end of the list
    if (x.type == GL_NONE)
    {
        return false;
    }

    if (y.type == GL_NONE)
    {
        return true;
    }

    return gl::VariableSortOrder(x.type) < gl::VariableSortOrder(y.type);
}

const char *GetShaderTypeString(ShaderType type)
{
    switch (type)
    {
        case ShaderType::Vertex:
            return "VERTEX";

        case ShaderType::Fragment:
            return "FRAGMENT";

        case ShaderType::Compute:
            return "COMPUTE";

        case ShaderType::Geometry:
            return "GEOMETRY";

        default:
            UNREACHABLE();
            return "";
    }
}

class ScopedExit final : angle::NonCopyable
{
  public:
    ScopedExit(std::function<void()> exit) : mExit(exit) {}
    ~ScopedExit() { mExit(); }

  private:
    std::function<void()> mExit;
};

struct Shader::CompilingState
{
    std::shared_ptr<rx::WaitableCompileEvent> compileEvent;
    ShCompilerInstance shCompilerInstance;
};

ShaderState::ShaderState(ShaderType shaderType)
    : mLabel(),
      mShaderType(shaderType),
      mShaderVersion(100),
      mNumViews(-1),
      mGeometryShaderInvocations(1),
      mCompileStatus(CompileStatus::NOT_COMPILED)
{
    mLocalSize.fill(-1);
}

ShaderState::~ShaderState() {}

Shader::Shader(ShaderProgramManager *manager,
               rx::GLImplFactory *implFactory,
               const gl::Limitations &rendererLimitations,
               ShaderType type,
               ShaderProgramID handle)
    : mState(type),
      mImplementation(implFactory->createShader(mState)),
      mRendererLimitations(rendererLimitations),
      mHandle(handle),
      mType(type),
      mRefCount(0),
      mDeleteStatus(false),
      mResourceManager(manager),
      mCurrentMaxComputeWorkGroupInvocations(0u)
{
    ASSERT(mImplementation);
}

void Shader::onDestroy(const gl::Context *context)
{
    resolveCompile();
    mImplementation->destroy();
    mBoundCompiler.set(context, nullptr);
    mImplementation.reset(nullptr);
    delete this;
}

Shader::~Shader()
{
    ASSERT(!mImplementation);
}

void Shader::setLabel(const Context *context, const std::string &label)
{
    mState.mLabel = label;
}

const std::string &Shader::getLabel() const
{
    return mState.mLabel;
}

ShaderProgramID Shader::getHandle() const
{
    return mHandle;
}

void Shader::setSource(GLsizei count, const char *const *string, const GLint *length)
{
    std::ostringstream stream;

    for (int i = 0; i < count; i++)
    {
        if (length == nullptr || length[i] < 0)
        {
            stream.write(string[i], strlen(string[i]));
        }
        else
        {
            stream.write(string[i], length[i]);
        }
    }

    mState.mSource = stream.str();
}

int Shader::getInfoLogLength()
{
    resolveCompile();
    if (mInfoLog.empty())
    {
        return 0;
    }

    return (static_cast<int>(mInfoLog.length()) + 1);
}

void Shader::getInfoLog(GLsizei bufSize, GLsizei *length, char *infoLog)
{
    resolveCompile();

    int index = 0;

    if (bufSize > 0)
    {
        index = std::min(bufSize - 1, static_cast<GLsizei>(mInfoLog.length()));
        memcpy(infoLog, mInfoLog.c_str(), index);

        infoLog[index] = '\0';
    }

    if (length)
    {
        *length = index;
    }
}

int Shader::getSourceLength() const
{
    return mState.mSource.empty() ? 0 : (static_cast<int>(mState.mSource.length()) + 1);
}

int Shader::getTranslatedSourceLength()
{
    resolveCompile();

    if (mState.mTranslatedSource.empty())
    {
        return 0;
    }

    return (static_cast<int>(mState.mTranslatedSource.length()) + 1);
}

int Shader::getTranslatedSourceWithDebugInfoLength()
{
    resolveCompile();

    const std::string &debugInfo = mImplementation->getDebugInfo();
    if (debugInfo.empty())
    {
        return 0;
    }

    return (static_cast<int>(debugInfo.length()) + 1);
}

// static
void Shader::GetSourceImpl(const std::string &source,
                           GLsizei bufSize,
                           GLsizei *length,
                           char *buffer)
{
    int index = 0;

    if (bufSize > 0)
    {
        index = std::min(bufSize - 1, static_cast<GLsizei>(source.length()));
        memcpy(buffer, source.c_str(), index);

        buffer[index] = '\0';
    }

    if (length)
    {
        *length = index;
    }
}

void Shader::getSource(GLsizei bufSize, GLsizei *length, char *buffer) const
{
    GetSourceImpl(mState.mSource, bufSize, length, buffer);
}

void Shader::getTranslatedSource(GLsizei bufSize, GLsizei *length, char *buffer)
{
    GetSourceImpl(getTranslatedSource(), bufSize, length, buffer);
}

const std::string &Shader::getTranslatedSource()
{
    resolveCompile();
    return mState.mTranslatedSource;
}

void Shader::getTranslatedSourceWithDebugInfo(GLsizei bufSize, GLsizei *length, char *buffer)
{
    resolveCompile();
    const std::string &debugInfo = mImplementation->getDebugInfo();
    GetSourceImpl(debugInfo, bufSize, length, buffer);
}

void Shader::compile(const Context *context)
{
    resolveCompile();

    mState.mTranslatedSource.clear();
    mInfoLog.clear();
    mState.mShaderVersion = 100;
    mState.mInputVaryings.clear();
    mState.mOutputVaryings.clear();
    mState.mUniforms.clear();
    mState.mUniformBlocks.clear();
    mState.mShaderStorageBlocks.clear();
    mState.mActiveAttributes.clear();
    mState.mActiveOutputVariables.clear();
    mState.mNumViews = -1;
    mState.mGeometryShaderInputPrimitiveType.reset();
    mState.mGeometryShaderOutputPrimitiveType.reset();
    mState.mGeometryShaderMaxVertices.reset();
    mState.mGeometryShaderInvocations = 1;

    mState.mCompileStatus = CompileStatus::COMPILE_REQUESTED;
    mBoundCompiler.set(context, context->getCompiler());

    ShCompileOptions options = (SH_OBJECT_CODE | SH_VARIABLES | SH_EMULATE_GL_DRAW_ID |
                                SH_EMULATE_GL_BASE_VERTEX_BASE_INSTANCE);

    // Add default options to WebGL shaders to prevent unexpected behavior during
    // compilation.
    if (context->getExtensions().webglCompatibility)
    {
        options |= SH_INIT_GL_POSITION;
        options |= SH_LIMIT_CALL_STACK_DEPTH;
        options |= SH_LIMIT_EXPRESSION_COMPLEXITY;
        options |= SH_ENFORCE_PACKING_RESTRICTIONS;
        options |= SH_INIT_SHARED_VARIABLES;
    }

    // Some targets (eg D3D11 Feature Level 9_3 and below) do not support non-constant loop
    // indexes in fragment shaders. Shader compilation will fail. To provide a better error
    // message we can instruct the compiler to pre-validate.
    if (mRendererLimitations.shadersRequireIndexedLoopValidation)
    {
        options |= SH_VALIDATE_LOOP_INDEXING;
    }

    if (context->getFrontendFeatures().scalarizeVecAndMatConstructorArgs.enabled)
    {
        options |= SH_SCALARIZE_VEC_AND_MAT_CONSTRUCTOR_ARGS;
    }

    mCurrentMaxComputeWorkGroupInvocations = context->getCaps().maxComputeWorkGroupInvocations;

    ASSERT(mBoundCompiler.get());
    ShCompilerInstance compilerInstance = mBoundCompiler->getInstance(mState.mShaderType);
    ShHandle compilerHandle             = compilerInstance.getHandle();
    ASSERT(compilerHandle);
    mCompilerResourcesString = compilerInstance.getBuiltinResourcesString();

    mCompilingState.reset(new CompilingState());
    mCompilingState->shCompilerInstance = std::move(compilerInstance);
    mCompilingState->compileEvent =
        mImplementation->compile(context, &(mCompilingState->shCompilerInstance), options);
}

void Shader::resolveCompile()
{
    if (!mState.compilePending())
    {
        return;
    }

    ASSERT(mCompilingState.get());

    mCompilingState->compileEvent->wait();

    mInfoLog += mCompilingState->compileEvent->getInfoLog();

    ScopedExit exit([this]() {
        mBoundCompiler->putInstance(std::move(mCompilingState->shCompilerInstance));
        mCompilingState->compileEvent.reset();
        mCompilingState.reset();
    });

    ShHandle compilerHandle = mCompilingState->shCompilerInstance.getHandle();
    if (!mCompilingState->compileEvent->getResult())
    {
        mInfoLog += sh::GetInfoLog(compilerHandle);
        WARN() << std::endl << mInfoLog;
        mState.mCompileStatus = CompileStatus::NOT_COMPILED;
        return;
    }

    mState.mTranslatedSource = sh::GetObjectCode(compilerHandle);

#if !defined(NDEBUG)
    // Prefix translated shader with commented out un-translated shader.
    // Useful in diagnostics tools which capture the shader source.
    std::ostringstream shaderStream;
    shaderStream << "// GLSL\n";
    shaderStream << "//\n";

    std::istringstream inputSourceStream(mState.mSource);
    std::string line;
    while (std::getline(inputSourceStream, line))
    {
        // Remove null characters from the source line
        line.erase(std::remove(line.begin(), line.end(), '\0'), line.end());

        shaderStream << "// " << line;

        // glslang complains if a comment ends with backslash
        if (!line.empty() && line.back() == '\\')
        {
            shaderStream << "\\";
        }

        shaderStream << std::endl;
    }
    shaderStream << "\n\n";
    shaderStream << mState.mTranslatedSource;
    mState.mTranslatedSource = shaderStream.str();
#endif  // !defined(NDEBUG)

    // Gather the shader information
    mState.mShaderVersion = sh::GetShaderVersion(compilerHandle);

    mState.mUniforms            = GetShaderVariables(sh::GetUniforms(compilerHandle));
    mState.mUniformBlocks       = GetShaderVariables(sh::GetUniformBlocks(compilerHandle));
    mState.mShaderStorageBlocks = GetShaderVariables(sh::GetShaderStorageBlocks(compilerHandle));

    switch (mState.mShaderType)
    {
        case ShaderType::Compute:
        {
            mState.mLocalSize = sh::GetComputeShaderLocalGroupSize(compilerHandle);
            if (mState.mLocalSize.isDeclared())
            {
                angle::CheckedNumeric<uint32_t> checked_local_size_product(mState.mLocalSize[0]);
                checked_local_size_product *= mState.mLocalSize[1];
                checked_local_size_product *= mState.mLocalSize[2];

                if (!checked_local_size_product.IsValid())
                {
                    WARN() << std::endl
                           << "Integer overflow when computing the product of local_size_x, "
                           << "local_size_y and local_size_z.";
                    mState.mCompileStatus = CompileStatus::NOT_COMPILED;
                    return;
                }
                if (checked_local_size_product.ValueOrDie() >
                    mCurrentMaxComputeWorkGroupInvocations)
                {
                    WARN() << std::endl
                           << "The total number of invocations within a work group exceeds "
                           << "MAX_COMPUTE_WORK_GROUP_INVOCATIONS.";
                    mState.mCompileStatus = CompileStatus::NOT_COMPILED;
                    return;
                }
            }
            break;
        }
        case ShaderType::Vertex:
        {
            mState.mOutputVaryings   = GetShaderVariables(sh::GetOutputVaryings(compilerHandle));
            mState.mAllAttributes    = GetShaderVariables(sh::GetAttributes(compilerHandle));
            mState.mActiveAttributes = GetActiveShaderVariables(&mState.mAllAttributes);
            mState.mNumViews         = sh::GetVertexShaderNumViews(compilerHandle);
            break;
        }
        case ShaderType::Fragment:
        {
            mState.mAllAttributes    = GetShaderVariables(sh::GetAttributes(compilerHandle));
            mState.mActiveAttributes = GetActiveShaderVariables(&mState.mAllAttributes);
            mState.mInputVaryings    = GetShaderVariables(sh::GetInputVaryings(compilerHandle));
            // TODO(jmadill): Figure out why we only sort in the FS, and if we need to.
            std::sort(mState.mInputVaryings.begin(), mState.mInputVaryings.end(), CompareShaderVar);
            mState.mActiveOutputVariables =
                GetActiveShaderVariables(sh::GetOutputVariables(compilerHandle));
            break;
        }
        case ShaderType::Geometry:
        {
            mState.mInputVaryings  = GetShaderVariables(sh::GetInputVaryings(compilerHandle));
            mState.mOutputVaryings = GetShaderVariables(sh::GetOutputVaryings(compilerHandle));

            if (sh::HasValidGeometryShaderInputPrimitiveType(compilerHandle))
            {
                mState.mGeometryShaderInputPrimitiveType = FromGLenum<PrimitiveMode>(
                    sh::GetGeometryShaderInputPrimitiveType(compilerHandle));
            }
            if (sh::HasValidGeometryShaderOutputPrimitiveType(compilerHandle))
            {
                mState.mGeometryShaderOutputPrimitiveType = FromGLenum<PrimitiveMode>(
                    sh::GetGeometryShaderOutputPrimitiveType(compilerHandle));
            }
            if (sh::HasValidGeometryShaderMaxVertices(compilerHandle))
            {
                mState.mGeometryShaderMaxVertices =
                    sh::GetGeometryShaderMaxVertices(compilerHandle);
            }
            mState.mGeometryShaderInvocations = sh::GetGeometryShaderInvocations(compilerHandle);
            break;
        }
        default:
            UNREACHABLE();
    }

    ASSERT(!mState.mTranslatedSource.empty());

    bool success          = mCompilingState->compileEvent->postTranslate(&mInfoLog);
    mState.mCompileStatus = success ? CompileStatus::COMPILED : CompileStatus::NOT_COMPILED;
}

void Shader::addRef()
{
    mRefCount++;
}

void Shader::release(const Context *context)
{
    mRefCount--;

    if (mRefCount == 0 && mDeleteStatus)
    {
        mResourceManager->deleteShader(context, mHandle);
    }
}

unsigned int Shader::getRefCount() const
{
    return mRefCount;
}

bool Shader::isFlaggedForDeletion() const
{
    return mDeleteStatus;
}

void Shader::flagForDeletion()
{
    mDeleteStatus = true;
}

bool Shader::isCompiled()
{
    resolveCompile();
    return mState.mCompileStatus == CompileStatus::COMPILED;
}

bool Shader::isCompleted()
{
    return (!mState.compilePending() || mCompilingState->compileEvent->isReady());
}

int Shader::getShaderVersion()
{
    resolveCompile();
    return mState.mShaderVersion;
}

const std::vector<sh::ShaderVariable> &Shader::getInputVaryings()
{
    resolveCompile();
    return mState.getInputVaryings();
}

const std::vector<sh::ShaderVariable> &Shader::getOutputVaryings()
{
    resolveCompile();
    return mState.getOutputVaryings();
}

const std::vector<sh::ShaderVariable> &Shader::getUniforms()
{
    resolveCompile();
    return mState.getUniforms();
}

const std::vector<sh::InterfaceBlock> &Shader::getUniformBlocks()
{
    resolveCompile();
    return mState.getUniformBlocks();
}

const std::vector<sh::InterfaceBlock> &Shader::getShaderStorageBlocks()
{
    resolveCompile();
    return mState.getShaderStorageBlocks();
}

const std::vector<sh::ShaderVariable> &Shader::getActiveAttributes()
{
    resolveCompile();
    return mState.getActiveAttributes();
}

const std::vector<sh::ShaderVariable> &Shader::getAllAttributes()
{
    resolveCompile();
    return mState.getAllAttributes();
}

const std::vector<sh::ShaderVariable> &Shader::getActiveOutputVariables()
{
    resolveCompile();
    return mState.getActiveOutputVariables();
}

std::string Shader::getTransformFeedbackVaryingMappedName(const std::string &tfVaryingName)
{
    // TODO(jiawei.shao@intel.com): support transform feedback on geometry shader.
    ASSERT(mState.getShaderType() == ShaderType::Vertex ||
           mState.getShaderType() == ShaderType::Geometry);
    const auto &varyings = getOutputVaryings();
    auto bracketPos      = tfVaryingName.find("[");
    if (bracketPos != std::string::npos)
    {
        auto tfVaryingBaseName = tfVaryingName.substr(0, bracketPos);
        for (const auto &varying : varyings)
        {
            if (varying.name == tfVaryingBaseName)
            {
                std::string mappedNameWithArrayIndex =
                    varying.mappedName + tfVaryingName.substr(bracketPos);
                return mappedNameWithArrayIndex;
            }
        }
    }
    else
    {
        for (const auto &varying : varyings)
        {
            if (varying.name == tfVaryingName)
            {
                return varying.mappedName;
            }
            else if (varying.isStruct())
            {
                GLuint fieldIndex = 0;
                const auto *field = FindShaderVarField(varying, tfVaryingName, &fieldIndex);
                ASSERT(field != nullptr && !field->isStruct() && !field->isArray());
                return varying.mappedName + "." + field->mappedName;
            }
        }
    }
    UNREACHABLE();
    return std::string();
}

const sh::WorkGroupSize &Shader::getWorkGroupSize()
{
    resolveCompile();
    return mState.mLocalSize;
}

int Shader::getNumViews()
{
    resolveCompile();
    return mState.mNumViews;
}

Optional<PrimitiveMode> Shader::getGeometryShaderInputPrimitiveType()
{
    resolveCompile();
    return mState.mGeometryShaderInputPrimitiveType;
}

Optional<PrimitiveMode> Shader::getGeometryShaderOutputPrimitiveType()
{
    resolveCompile();
    return mState.mGeometryShaderOutputPrimitiveType;
}

int Shader::getGeometryShaderInvocations()
{
    resolveCompile();
    return mState.mGeometryShaderInvocations;
}

Optional<GLint> Shader::getGeometryShaderMaxVertices()
{
    resolveCompile();
    return mState.mGeometryShaderMaxVertices;
}

const std::string &Shader::getCompilerResourcesString() const
{
    return mCompilerResourcesString;
}

}  // namespace gl
