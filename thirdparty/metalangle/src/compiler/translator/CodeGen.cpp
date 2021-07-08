//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifdef ANGLE_ENABLE_ESSL
#    include "compiler/translator/TranslatorESSL.h"
#endif  // ANGLE_ENABLE_ESSL

#ifdef ANGLE_ENABLE_GLSL
#    include "compiler/translator/TranslatorGLSL.h"
#endif  // ANGLE_ENABLE_GLSL

#ifdef ANGLE_ENABLE_HLSL
#    include "compiler/translator/TranslatorHLSL.h"
#endif  // ANGLE_ENABLE_HLSL

#ifdef ANGLE_ENABLE_VULKAN
#    include "compiler/translator/TranslatorVulkan.h"
#endif  // ANGLE_ENABLE_VULKAN

#ifdef ANGLE_ENABLE_METAL
#    include "compiler/translator/TranslatorMetal.h"
#endif  // ANGLE_ENABLE_METAL

#include "compiler/translator/util.h"

namespace sh
{

//
// This function must be provided to create the actual
// compile object used by higher level code.  It returns
// a subclass of TCompiler.
//
TCompiler *ConstructCompiler(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output)
{
#ifdef ANGLE_ENABLE_ESSL
    if (IsOutputESSL(output))
    {
        return new TranslatorESSL(type, spec);
    }
#endif  // ANGLE_ENABLE_ESSL

#ifdef ANGLE_ENABLE_GLSL
    if (IsOutputGLSL(output))
    {
        return new TranslatorGLSL(type, spec, output);
    }
#endif  // ANGLE_ENABLE_GLSL

#ifdef ANGLE_ENABLE_HLSL
    if (IsOutputHLSL(output))
    {
        return new TranslatorHLSL(type, spec, output);
    }
#endif  // ANGLE_ENABLE_HLSL

#ifdef ANGLE_ENABLE_VULKAN
    if (IsOutputVulkan(output))
    {
        return new TranslatorVulkan(type, spec);
    }
#endif  // ANGLE_ENABLE_VULKAN

#ifdef ANGLE_ENABLE_METAL
    if (IsOutputMetal(output))
    {
        return new TranslatorMetal(type, spec);
    }
#endif  // ANGLE_ENABLE_METAL

    // Unsupported compiler or unknown format. Return nullptr per the sh::ConstructCompiler API.
    return nullptr;
}

//
// Delete the compiler made by ConstructCompiler
//
void DeleteCompiler(TCompiler *compiler)
{
    SafeDelete(compiler);
}

}  // namespace sh
