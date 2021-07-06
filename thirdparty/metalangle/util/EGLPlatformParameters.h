//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EGLPlatformParameters: Basic description of an EGL device.

#ifndef UTIL_EGLPLATFORMPARAMETERS_H_
#define UTIL_EGLPLATFORMPARAMETERS_H_

#include "util/util_gl.h"

#include <tuple>

namespace angle
{
struct PlatformMethods;
}  // namespace angle

struct EGLPlatformParameters
{
    EGLPlatformParameters() = default;

    explicit EGLPlatformParameters(EGLint renderer) : renderer(renderer) {}

    EGLPlatformParameters(EGLint renderer,
                          EGLint majorVersion,
                          EGLint minorVersion,
                          EGLint deviceType)
        : renderer(renderer),
          majorVersion(majorVersion),
          minorVersion(minorVersion),
          deviceType(deviceType)
    {}

    EGLPlatformParameters(EGLint renderer,
                          EGLint majorVersion,
                          EGLint minorVersion,
                          EGLint deviceType,
                          EGLint presentPath)
        : renderer(renderer),
          majorVersion(majorVersion),
          minorVersion(minorVersion),
          deviceType(deviceType),
          presentPath(presentPath)
    {}

    auto tie() const
    {
        return std::tie(renderer, majorVersion, minorVersion, deviceType, presentPath,
                        debugLayersEnabled, contextVirtualization, platformMethods);
    }

    EGLint renderer                         = EGL_PLATFORM_ANGLE_TYPE_DEFAULT_ANGLE;
    EGLint majorVersion                     = EGL_DONT_CARE;
    EGLint minorVersion                     = EGL_DONT_CARE;
    EGLint deviceType                       = EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE;
    EGLint presentPath                      = EGL_DONT_CARE;
    EGLint debugLayersEnabled               = EGL_DONT_CARE;
    EGLint contextVirtualization            = EGL_DONT_CARE;
    angle::PlatformMethods *platformMethods = nullptr;
};

inline bool operator<(const EGLPlatformParameters &a, const EGLPlatformParameters &b)
{
    return a.tie() < b.tie();
}

inline bool operator==(const EGLPlatformParameters &a, const EGLPlatformParameters &b)
{
    return a.tie() == b.tie();
}

inline bool operator!=(const EGLPlatformParameters &a, const EGLPlatformParameters &b)
{
    return a.tie() != b.tie();
}

#endif  // UTIL_EGLPLATFORMPARAMETERS_H_
