//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CompilerGL:
//   Implementation of the GL compiler methods.
//

#include "libANGLE/renderer/gl/CompilerGL.h"

#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace rx
{

namespace
{

ShShaderOutput GetShaderOutputType(const FunctionsGL *functions)
{
    ASSERT(functions);

    if (functions->standard == STANDARD_GL_DESKTOP)
    {
        // GLSL outputs
        if (functions->isAtLeastGL(gl::Version(4, 5)))
        {
            return SH_GLSL_450_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(4, 4)))
        {
            return SH_GLSL_440_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(4, 3)))
        {
            return SH_GLSL_430_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(4, 2)))
        {
            return SH_GLSL_420_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(4, 1)))
        {
            return SH_GLSL_410_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(4, 0)))
        {
            return SH_GLSL_400_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(3, 3)))
        {
            return SH_GLSL_330_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(3, 2)))
        {
            return SH_GLSL_150_CORE_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(3, 1)))
        {
            return SH_GLSL_140_OUTPUT;
        }
        else if (functions->isAtLeastGL(gl::Version(3, 0)))
        {
            return SH_GLSL_130_OUTPUT;
        }
        else
        {
            return SH_GLSL_COMPATIBILITY_OUTPUT;
        }
    }
    else if (functions->standard == STANDARD_GL_ES)
    {
        // ESSL outputs
        return SH_ESSL_OUTPUT;
    }
    else
    {
        UNREACHABLE();
        return ShShaderOutput(0);
    }
}

}  // anonymous namespace

CompilerGL::CompilerGL(const FunctionsGL *functions)
    : mTranslatorOutputType(GetShaderOutputType(functions))
{}

ShShaderOutput CompilerGL::getTranslatorOutputType() const
{
    return mTranslatorOutputType;
}

}  // namespace rx
