//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShaderImpl.cpp: Implementation methods of ShaderImpl

#include "libANGLE/renderer/ShaderImpl.h"

#include "libANGLE/Context.h"

namespace rx
{

WaitableCompileEvent::WaitableCompileEvent(std::shared_ptr<angle::WaitableEvent> waitableEvent)
    : mWaitableEvent(waitableEvent)
{}

WaitableCompileEvent::~WaitableCompileEvent()
{
    mWaitableEvent.reset();
}

void WaitableCompileEvent::wait()
{
    mWaitableEvent->wait();
}

bool WaitableCompileEvent::isReady()
{
    return mWaitableEvent->isReady();
}

const std::string &WaitableCompileEvent::getInfoLog()
{
    return mInfoLog;
}

class TranslateTask : public angle::Closure
{
  public:
    TranslateTask(ShHandle handle, ShCompileOptions options, const std::string &source)
        : mHandle(handle), mOptions(options), mSource(source), mResult(false)
    {}

    void operator()() override
    {
        const char *source = mSource.c_str();
        mResult            = sh::Compile(mHandle, &source, 1, mOptions);
    }

    bool getResult() { return mResult; }

    ShHandle getHandle() { return mHandle; }

  private:
    ShHandle mHandle;
    ShCompileOptions mOptions;
    std::string mSource;
    bool mResult;
};

class WaitableCompileEventImpl final : public WaitableCompileEvent
{
  public:
    WaitableCompileEventImpl(std::shared_ptr<angle::WaitableEvent> waitableEvent,
                             std::shared_ptr<TranslateTask> translateTask)
        : WaitableCompileEvent(waitableEvent), mTranslateTask(translateTask)
    {}

    bool getResult() override { return mTranslateTask->getResult(); }

    bool postTranslate(std::string *infoLog) override { return true; }

  private:
    std::shared_ptr<TranslateTask> mTranslateTask;
};

std::shared_ptr<WaitableCompileEvent> ShaderImpl::compileImpl(
    const gl::Context *context,
    gl::ShCompilerInstance *compilerInstance,
    const std::string &source,
    ShCompileOptions compileOptions)
{
#if defined(ANGLE_ENABLE_ASSERTS)
    compileOptions |= SH_VALIDATE_AST;
#endif

    auto workerThreadPool = context->getWorkerThreadPool();
    auto translateTask =
        std::make_shared<TranslateTask>(compilerInstance->getHandle(), compileOptions, source);

    return std::make_shared<WaitableCompileEventImpl>(
        angle::WorkerThreadPool::PostWorkerTask(workerThreadPool, translateTask), translateTask);
}

}  // namespace rx
