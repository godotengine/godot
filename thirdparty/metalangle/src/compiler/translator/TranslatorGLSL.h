//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_TRANSLATORGLSL_H_
#define COMPILER_TRANSLATOR_TRANSLATORGLSL_H_

#include "compiler/translator/Compiler.h"

namespace sh
{

class TranslatorGLSL : public TCompiler
{
  public:
    TranslatorGLSL(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output);

  protected:
    void initBuiltInFunctionEmulator(BuiltInFunctionEmulator *emu,
                                     ShCompileOptions compileOptions) override;

    ANGLE_NO_DISCARD bool translate(TIntermBlock *root,
                                    ShCompileOptions compileOptions,
                                    PerformanceDiagnostics *perfDiagnostics) override;
    bool shouldFlattenPragmaStdglInvariantAll() override;
    bool shouldCollectVariables(ShCompileOptions compileOptions) override;

  private:
    void writeVersion(TIntermNode *root);
    void writeExtensionBehavior(TIntermNode *root, ShCompileOptions compileOptions);
    void conditionallyOutputInvariantDeclaration(const char *builtinVaryingName);
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TRANSLATORGLSL_H_
