//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_OUTPUTESSL_H_
#define COMPILER_TRANSLATOR_OUTPUTESSL_H_

#include "compiler/translator/OutputGLSLBase.h"

namespace sh
{

class TOutputESSL : public TOutputGLSLBase
{
  public:
    TOutputESSL(TInfoSinkBase &objSink,
                ShArrayIndexClampingStrategy clampingStrategy,
                ShHashFunction64 hashFunction,
                NameMap &nameMap,
                TSymbolTable *symbolTable,
                sh::GLenum shaderType,
                int shaderVersion,
                bool forceHighp,
                ShCompileOptions compileOptions);

  protected:
    bool writeVariablePrecision(TPrecision precision) override;

  private:
    bool mForceHighp;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_OUTPUTESSL_H_
