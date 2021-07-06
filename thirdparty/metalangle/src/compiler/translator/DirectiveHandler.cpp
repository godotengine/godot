//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/DirectiveHandler.h"

#include <sstream>

#include "angle_gl.h"
#include "common/debug.h"
#include "compiler/translator/Common.h"
#include "compiler/translator/Diagnostics.h"

namespace sh
{

static TBehavior getBehavior(const std::string &str)
{
    const char kRequire[] = "require";
    const char kEnable[]  = "enable";
    const char kDisable[] = "disable";
    const char kWarn[]    = "warn";

    if (str == kRequire)
        return EBhRequire;
    else if (str == kEnable)
        return EBhEnable;
    else if (str == kDisable)
        return EBhDisable;
    else if (str == kWarn)
        return EBhWarn;
    return EBhUndefined;
}

TDirectiveHandler::TDirectiveHandler(TExtensionBehavior &extBehavior,
                                     TDiagnostics &diagnostics,
                                     int &shaderVersion,
                                     sh::GLenum shaderType,
                                     bool debugShaderPrecisionSupported)
    : mExtensionBehavior(extBehavior),
      mDiagnostics(diagnostics),
      mShaderVersion(shaderVersion),
      mShaderType(shaderType),
      mDebugShaderPrecisionSupported(debugShaderPrecisionSupported)
{}

TDirectiveHandler::~TDirectiveHandler() {}

void TDirectiveHandler::handleError(const angle::pp::SourceLocation &loc, const std::string &msg)
{
    mDiagnostics.error(loc, msg.c_str(), "");
}

void TDirectiveHandler::handlePragma(const angle::pp::SourceLocation &loc,
                                     const std::string &name,
                                     const std::string &value,
                                     bool stdgl)
{
    if (stdgl)
    {
        const char kInvariant[] = "invariant";
        const char kAll[]       = "all";

        if (name == kInvariant && value == kAll)
        {
            if (mShaderVersion == 300 && mShaderType == GL_FRAGMENT_SHADER)
            {
                // ESSL 3.00.4 section 4.6.1
                mDiagnostics.error(
                    loc, "#pragma STDGL invariant(all) can not be used in fragment shader",
                    name.c_str());
            }
            mPragma.stdgl.invariantAll = true;
        }
        // The STDGL pragma is used to reserve pragmas for use by future
        // revisions of GLSL.  Do not generate an error on unexpected
        // name and value.
        return;
    }
    else
    {
        const char kOptimize[]             = "optimize";
        const char kDebug[]                = "debug";
        const char kDebugShaderPrecision[] = "webgl_debug_shader_precision";
        const char kOn[]                   = "on";
        const char kOff[]                  = "off";

        bool invalidValue = false;
        if (name == kOptimize)
        {
            if (value == kOn)
                mPragma.optimize = true;
            else if (value == kOff)
                mPragma.optimize = false;
            else
                invalidValue = true;
        }
        else if (name == kDebug)
        {
            if (value == kOn)
                mPragma.debug = true;
            else if (value == kOff)
                mPragma.debug = false;
            else
                invalidValue = true;
        }
        else if (name == kDebugShaderPrecision && mDebugShaderPrecisionSupported)
        {
            if (value == kOn)
                mPragma.debugShaderPrecision = true;
            else if (value == kOff)
                mPragma.debugShaderPrecision = false;
            else
                invalidValue = true;
        }
        else
        {
            mDiagnostics.report(angle::pp::Diagnostics::PP_UNRECOGNIZED_PRAGMA, loc, name);
            return;
        }

        if (invalidValue)
        {
            mDiagnostics.error(loc, "invalid pragma value - 'on' or 'off' expected", value.c_str());
        }
    }
}

void TDirectiveHandler::handleExtension(const angle::pp::SourceLocation &loc,
                                        const std::string &name,
                                        const std::string &behavior)
{
    const char kExtAll[] = "all";

    TBehavior behaviorVal = getBehavior(behavior);
    if (behaviorVal == EBhUndefined)
    {
        mDiagnostics.error(loc, "behavior invalid", name.c_str());
        return;
    }

    if (name == kExtAll)
    {
        if (behaviorVal == EBhRequire)
        {
            mDiagnostics.error(loc, "extension cannot have 'require' behavior", name.c_str());
        }
        else if (behaviorVal == EBhEnable)
        {
            mDiagnostics.error(loc, "extension cannot have 'enable' behavior", name.c_str());
        }
        else
        {
            for (TExtensionBehavior::iterator iter = mExtensionBehavior.begin();
                 iter != mExtensionBehavior.end(); ++iter)
                iter->second = behaviorVal;
        }
        return;
    }

    TExtensionBehavior::iterator iter = mExtensionBehavior.find(GetExtensionByName(name.c_str()));
    if (iter != mExtensionBehavior.end())
    {
        iter->second = behaviorVal;
        // OVR_multiview is implicitly enabled when OVR_multiview2 is enabled
        if (name == "GL_OVR_multiview2")
        {
            const std::string multiview = "GL_OVR_multiview";
            TExtensionBehavior::iterator iterMultiview =
                mExtensionBehavior.find(GetExtensionByName(multiview.c_str()));
            if (iterMultiview != mExtensionBehavior.end())
            {
                iterMultiview->second = behaviorVal;
            }
        }
        return;
    }

    switch (behaviorVal)
    {
        case EBhRequire:
            mDiagnostics.error(loc, "extension is not supported", name.c_str());
            break;
        case EBhEnable:
        case EBhWarn:
        case EBhDisable:
            mDiagnostics.warning(loc, "extension is not supported", name.c_str());
            break;
        default:
            UNREACHABLE();
            break;
    }
}

void TDirectiveHandler::handleVersion(const angle::pp::SourceLocation &loc,
                                      int version,
                                      ShShaderSpec spec)
{
    if (((version == 100 || version == 300 || version == 310) && !IsDesktopGLSpec(spec)) ||
        IsDesktopGLSpec(spec))
    {
        mShaderVersion = version;
    }
    else
    {
        std::stringstream stream = sh::InitializeStream<std::stringstream>();
        stream << version;
        std::string str = stream.str();
        mDiagnostics.error(loc, "client/version number not supported", str.c_str());
    }
}

}  // namespace sh
