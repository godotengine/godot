//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ProgramPipelineGL.cpp: Implements the class methods for ProgramPipelineGL.

#include "libANGLE/renderer/gl/ProgramPipelineGL.h"

#include "common/debug.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace rx
{

ProgramPipelineGL::ProgramPipelineGL(const gl::ProgramPipelineState &data,
                                     const FunctionsGL *functions)
    : ProgramPipelineImpl(data), mFunctions(functions), mProgramPipelineID(0)
{
    ASSERT(mFunctions);
    mFunctions->genProgramPipelines(1, &mProgramPipelineID);
}

ProgramPipelineGL::~ProgramPipelineGL()
{
    if (mProgramPipelineID != 0)
    {
        mFunctions->deleteProgramPipelines(1, &mProgramPipelineID);
        mProgramPipelineID = 0;
    }
}

}  // namespace rx
