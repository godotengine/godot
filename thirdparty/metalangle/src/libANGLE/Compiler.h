//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Compiler.h: Defines the gl::Compiler class, abstracting the ESSL compiler
// that a GL context holds.

#ifndef LIBANGLE_COMPILER_H_
#define LIBANGLE_COMPILER_H_

#include <vector>

#include "GLSLANG/ShaderLang.h"
#include "common/PackedEnums.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

namespace rx
{
class CompilerImpl;
class GLImplFactory;
}  // namespace rx

namespace gl
{
class ShCompilerInstance;
class State;

class Compiler final : public RefCountObjectNoID
{
  public:
    Compiler(rx::GLImplFactory *implFactory, const State &data);

    ShCompilerInstance getInstance(ShaderType shaderType);
    void putInstance(ShCompilerInstance &&instance);
    ShShaderOutput getShaderOutputType() const { return mOutputType; }

  private:
    ~Compiler() override;
    std::unique_ptr<rx::CompilerImpl> mImplementation;
    ShShaderSpec mSpec;
    ShShaderOutput mOutputType;
    ShBuiltInResources mResources;
    ShaderMap<std::vector<ShCompilerInstance>> mPools;
};

class ShCompilerInstance final : public angle::NonCopyable
{
  public:
    ShCompilerInstance();
    ShCompilerInstance(ShHandle handle, ShShaderOutput outputType, ShaderType shaderType);
    ~ShCompilerInstance();
    void destroy();

    ShCompilerInstance(ShCompilerInstance &&other);
    ShCompilerInstance &operator=(ShCompilerInstance &&other);

    ShHandle getHandle();
    ShaderType getShaderType() const;
    const std::string &getBuiltinResourcesString();
    ShShaderOutput getShaderOutputType() const;

  private:
    ShHandle mHandle;
    ShShaderOutput mOutputType;
    ShaderType mShaderType;
};

}  // namespace gl

#endif  // LIBANGLE_COMPILER_H_
