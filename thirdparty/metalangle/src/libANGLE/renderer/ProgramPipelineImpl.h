//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ProgramPipelineImpl.h: Defines the abstract rx::ProgramPipelineImpl class.

#ifndef LIBANGLE_RENDERER_PROGRAMPIPELINEIMPL_H_
#define LIBANGLE_RENDERER_PROGRAMPIPELINEIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/ProgramPipeline.h"

namespace rx
{
class ContextImpl;

class ProgramPipelineImpl : public angle::NonCopyable
{
  public:
    ProgramPipelineImpl(const gl::ProgramPipelineState &state) : mState(state) {}
    virtual ~ProgramPipelineImpl() {}
    virtual void destroy(const ContextImpl *contextImpl) {}

  protected:
    const gl::ProgramPipelineState &mState;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_PROGRAMPIPELINEIMPL_H_
