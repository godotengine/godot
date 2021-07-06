//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShaderImpl.h: Defines the abstract rx::ShaderImpl class.

#ifndef LIBANGLE_RENDERER_SHADERIMPL_H_
#define LIBANGLE_RENDERER_SHADERIMPL_H_

#include <functional>

#include "common/angleutils.h"
#include "libANGLE/Shader.h"
#include "libANGLE/WorkerThread.h"

namespace gl
{
class ShCompilerInstance;
}  // namespace gl

namespace rx
{

using UpdateShaderStateFunctor = std::function<void(bool compiled, ShHandle handle)>;
class WaitableCompileEvent : public angle::WaitableEvent
{
  public:
    WaitableCompileEvent(std::shared_ptr<angle::WaitableEvent> waitableEvent);
    ~WaitableCompileEvent() override;

    void wait() override;

    bool isReady() override;

    virtual bool getResult() = 0;

    virtual bool postTranslate(std::string *infoLog) = 0;

    const std::string &getInfoLog();

  protected:
    std::shared_ptr<angle::WaitableEvent> mWaitableEvent;
    std::string mInfoLog;
};

class ShaderImpl : angle::NonCopyable
{
  public:
    ShaderImpl(const gl::ShaderState &data) : mData(data) {}
    virtual ~ShaderImpl() {}

    virtual void destroy() {}

    virtual std::shared_ptr<WaitableCompileEvent> compile(const gl::Context *context,
                                                          gl::ShCompilerInstance *compilerInstance,
                                                          ShCompileOptions options) = 0;

    virtual std::string getDebugInfo() const = 0;

    const gl::ShaderState &getData() const { return mData; }

  protected:
    std::shared_ptr<WaitableCompileEvent> compileImpl(const gl::Context *context,
                                                      gl::ShCompilerInstance *compilerInstance,
                                                      const std::string &source,
                                                      ShCompileOptions compileOptions);

    const gl::ShaderState &mData;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_SHADERIMPL_H_
