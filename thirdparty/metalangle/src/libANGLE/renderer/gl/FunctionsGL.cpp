//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FunctionsGL.cpp: Implements the FuntionsGL class to contain loaded GL functions

#include "libANGLE/renderer/gl/FunctionsGL.h"

#include <algorithm>

#include "common/string_utils.h"
#include "libANGLE/AttributeMap.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"

namespace rx
{

static void GetGLVersion(PFNGLGETSTRINGPROC getStringFunction,
                         gl::Version *outVersion,
                         StandardGL *outStandard)
{
    const std::string version = reinterpret_cast<const char *>(getStringFunction(GL_VERSION));
    if (version.find("OpenGL ES") == std::string::npos)
    {
        // OpenGL spec states the GL_VERSION string will be in the following format:
        // <version number><space><vendor-specific information>
        // The version number is either of the form major number.minor number or major
        // number.minor number.release number, where the numbers all have one or more
        // digits
        *outStandard = STANDARD_GL_DESKTOP;
        *outVersion  = gl::Version(version[0] - '0', version[2] - '0');
    }
    else
    {
        // ES spec states that the GL_VERSION string will be in the following format:
        // "OpenGL ES N.M vendor-specific information"
        *outStandard = STANDARD_GL_ES;
        *outVersion  = gl::Version(version[10] - '0', version[12] - '0');
    }
}

static std::vector<std::string> GetIndexedExtensions(PFNGLGETINTEGERVPROC getIntegerFunction,
                                                     PFNGLGETSTRINGIPROC getStringIFunction)
{
    std::vector<std::string> result;

    GLint numExtensions;
    getIntegerFunction(GL_NUM_EXTENSIONS, &numExtensions);

    result.reserve(numExtensions);

    for (GLint i = 0; i < numExtensions; i++)
    {
        result.push_back(reinterpret_cast<const char *>(getStringIFunction(GL_EXTENSIONS, i)));
    }

    return result;
}

#if defined(ANGLE_ENABLE_OPENGL_NULL)
static GLenum INTERNAL_GL_APIENTRY DummyCheckFramebufferStatus(GLenum)
{
    return GL_FRAMEBUFFER_COMPLETE;
}

static void INTERNAL_GL_APIENTRY DummyGetProgramiv(GLuint program, GLenum pname, GLint *params)
{
    switch (pname)
    {
        case GL_LINK_STATUS:
            *params = GL_TRUE;
            break;
        case GL_VALIDATE_STATUS:
            *params = GL_TRUE;
            break;
        default:
            break;
    }
}

static void INTERNAL_GL_APIENTRY DummyGetShaderiv(GLuint program, GLenum pname, GLint *params)
{
    switch (pname)
    {
        case GL_COMPILE_STATUS:
            *params = GL_TRUE;
            break;
        default:
            break;
    }
}
#endif  // defined(ANGLE_ENABLE_OPENGL_NULL)

#define ASSIGN(NAME, FP) *reinterpret_cast<void **>(&FP) = loadProcAddress(NAME)

FunctionsGL::FunctionsGL() : version(), standard(), extensions() {}

FunctionsGL::~FunctionsGL() {}

void FunctionsGL::initialize(const egl::AttributeMap &displayAttributes)
{
    // Grab the version number
    ASSIGN("glGetString", getString);
    ASSIGN("glGetIntegerv", getIntegerv);
    GetGLVersion(getString, &version, &standard);

    // Grab the GL extensions
    if (isAtLeastGL(gl::Version(3, 0)) || isAtLeastGLES(gl::Version(3, 0)))
    {
        ASSIGN("glGetStringi", getStringi);
        extensions = GetIndexedExtensions(getIntegerv, getStringi);
    }
    else
    {
        const char *exts = reinterpret_cast<const char *>(getString(GL_EXTENSIONS));
        angle::SplitStringAlongWhitespace(std::string(exts), &extensions);
    }

    std::set<std::string> extensionSet;
    for (const auto &extension : extensions)
    {
        extensionSet.insert(extension);
    }

    // Note:
    // Even though extensions are written against specific versions of GL, many drivers expose the
    // extensions in even older versions.  Always try loading the extensions regardless of GL
    // version.

    // Load the entry points

#if defined(ANGLE_ENABLE_OPENGL_NULL)
    EGLint deviceType =
        static_cast<EGLint>(displayAttributes.get(EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE, EGL_NONE));
#endif  // defined(ANGLE_ENABLE_GL_NULL)

    switch (standard)
    {
        case STANDARD_GL_DESKTOP:
        {
            // Check the context profile
            profile = 0;
            if (isAtLeastGL(gl::Version(3, 2)))
            {
                getIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);
            }

#if defined(ANGLE_ENABLE_OPENGL_NULL)
            if (deviceType == EGL_PLATFORM_ANGLE_DEVICE_TYPE_NULL_ANGLE)
            {
                initProcsDesktopGLNULL(version, extensionSet);
            }
            else
#endif  // defined(ANGLE_ENABLE_GL_NULL)
            {
                initProcsDesktopGL(version, extensionSet);
            }
            break;
        }

        case STANDARD_GL_ES:
        {
            // No profiles in GLES
            profile = 0;

#if defined(ANGLE_ENABLE_OPENGL_NULL)
            if (deviceType == EGL_PLATFORM_ANGLE_DEVICE_TYPE_NULL_ANGLE)
            {
                initProcsGLESNULL(version, extensionSet);
            }
            else
#endif  // defined(ANGLE_ENABLE_GL_NULL)
            {
                initProcsGLES(version, extensionSet);
            }
            break;
        }

        default:
            UNREACHABLE();
            break;
    }

#if defined(ANGLE_ENABLE_OPENGL_NULL)
    if (deviceType == EGL_PLATFORM_ANGLE_DEVICE_TYPE_NULL_ANGLE)
    {
        initProcsSharedExtensionsNULL(extensionSet);
        initializeDummyFunctionsForNULLDriver(extensionSet);
    }
    else
#endif  // defined(ANGLE_ENABLE_OPENGL_NULL)
    {
        initProcsSharedExtensions(extensionSet);
    }
}

bool FunctionsGL::isAtLeastGL(const gl::Version &glVersion) const
{
    return standard == STANDARD_GL_DESKTOP && version >= glVersion;
}

bool FunctionsGL::isAtMostGL(const gl::Version &glVersion) const
{
    return standard == STANDARD_GL_DESKTOP && glVersion >= version;
}

bool FunctionsGL::isAtLeastGLES(const gl::Version &glesVersion) const
{
    return standard == STANDARD_GL_ES && version >= glesVersion;
}

bool FunctionsGL::isAtMostGLES(const gl::Version &glesVersion) const
{
    return standard == STANDARD_GL_ES && glesVersion >= version;
}

bool FunctionsGL::hasExtension(const std::string &ext) const
{
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

bool FunctionsGL::hasGLExtension(const std::string &ext) const
{
    return standard == STANDARD_GL_DESKTOP && hasExtension(ext);
}

bool FunctionsGL::hasGLESExtension(const std::string &ext) const
{
    return standard == STANDARD_GL_ES && hasExtension(ext);
}

#if defined(ANGLE_ENABLE_OPENGL_NULL)
void FunctionsGL::initializeDummyFunctionsForNULLDriver(const std::set<std::string> &extensionSet)
{
    // This is a quick hack to get the NULL driver working, but we might want to implement a true
    // NULL/stub driver that never calls into the OS. See Chromium's implementation in
    // ui/gl/gl_stub_api.cc. This might be useful for testing things like perf scaling due to
    // the caps returned by the drivers (i.e. number of texture units) or a true NULL back-end
    // that could be used in a VM for things like fuzzing.
    // TODO(jmadill): Implement true no-op/stub back-end.
    ASSIGN("glGetString", getString);
    ASSIGN("glGetStringi", getStringi);
    ASSIGN("glGetIntegerv", getIntegerv);
    ASSIGN("glGetIntegeri_v", getIntegeri_v);

    getProgramiv           = &DummyGetProgramiv;
    getShaderiv            = &DummyGetShaderiv;
    checkFramebufferStatus = &DummyCheckFramebufferStatus;

    if (isAtLeastGLES(gl::Version(3, 0)) || isAtLeastGL(gl::Version(4, 2)) ||
        extensionSet.count("GL_ARB_internalformat_query") > 0)
    {
        ASSIGN("glGetInternalformativ", getInternalformativ);
    }

    if (isAtLeastGL(gl::Version(4, 3)))
    {
        ASSIGN("glGetInternalformati64v", getInternalformati64v);
    }

    if (extensionSet.count("GL_NV_internalformat_sample_query") > 0)
    {
        ASSIGN("glGetInternalformatSampleivNV", getInternalformatSampleivNV);
    }
}
#endif  // defined(ANGLE_ENABLE_OPENGL_NULL)

}  // namespace rx
