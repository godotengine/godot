//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// TranslatorVulkan:
//   A GLSL-based translator that outputs shaders that fit GL_KHR_vulkan_glsl.
//   The shaders are then fed into glslang to spit out SPIR-V (libANGLE-side).
//   See: https://www.khronos.org/registry/vulkan/specs/misc/GL_KHR_vulkan_glsl.txt
//

#ifndef COMPILER_TRANSLATOR_TRANSLATORVULKAN_H_
#define COMPILER_TRANSLATOR_TRANSLATORVULKAN_H_

#include "compiler/translator/Compiler.h"

namespace sh
{

class TOutputVulkanGLSL;

class TranslatorVulkan : public TCompiler
{
  public:
    TranslatorVulkan(sh::GLenum type, ShShaderSpec spec);

  protected:
    ANGLE_NO_DISCARD bool translate(TIntermBlock *root,
                                    ShCompileOptions compileOptions,
                                    PerformanceDiagnostics *perfDiagnostics) override;
    bool shouldFlattenPragmaStdglInvariantAll() override;

    TIntermBinary *getDriverUniformNegViewportYScaleRef(const TVariable *driverUniforms) const;
    TIntermBinary *getDriverUniformDepthRangeReservedFieldRef(
        const TVariable *driverUniforms) const;
    // Subclass can call this method to transform the AST before writing the final output.
    // See TranslatorMetal.mm.
    ANGLE_NO_DISCARD bool translateImpl(TIntermBlock *root,
                                        ShCompileOptions compileOptions,
                                        PerformanceDiagnostics *perfDiagnostics,
                                        const TVariable **driverUniformsOut,
                                        TOutputVulkanGLSL *outputGLSL);

    // Give subclass such as TranslatorMetal a chance to do depth transform before
    // TranslatorVulkan apply its own transform.
    ANGLE_NO_DISCARD virtual bool transformDepthBeforeCorrection(TIntermBlock *root,
                                                                 const TVariable *driverUniforms)
    {
        return true;
    }

    // Back-end specific fields to be added to driver uniform. See TranslatorMetal.mm.
    virtual void createGraphicsDriverUniformAdditionFields(std::vector<TField *> *fieldsOut) {}
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TRANSLATORVULKAN_H_
