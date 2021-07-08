//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_OUTPUTGLSL_H_
#define COMPILER_TRANSLATOR_OUTPUTGLSL_H_

#include "compiler/translator/OutputGLSLBase.h"

namespace sh
{

class TOutputGLSL : public TOutputGLSLBase
{
  public:
    TOutputGLSL(TInfoSinkBase &objSink,
                ShArrayIndexClampingStrategy clampingStrategy,
                ShHashFunction64 hashFunction,
                NameMap &nameMap,
                TSymbolTable *symbolTable,
                sh::GLenum shaderType,
                int shaderVersion,
                ShShaderOutput output,
                ShCompileOptions compileOptions);

  protected:
    bool writeVariablePrecision(TPrecision) override;
    void visitSymbol(TIntermSymbol *node) override;
    ImmutableString translateTextureFunction(const ImmutableString &name) override;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_OUTPUTGLSL_H_
