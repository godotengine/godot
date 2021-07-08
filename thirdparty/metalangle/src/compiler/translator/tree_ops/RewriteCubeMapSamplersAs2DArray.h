//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteCubeMapSamplersAs2DArray: Change samplerCube samplers to sampler2DArray, and the
// textureCube* function calls to calls to helper functions that select the cube map face and sample
// from the face as a 2D texture.  This emulates seamful cube map sampling in ES2 (or desktop GL 2)
// and therefore only looks at samplerCube (i.e. not integer variants or cube arrays) and sampling
// functions that are defined in ES GLSL 1.0 (i.e. not the cube overload of texture()).

#ifndef COMPILER_TRANSLATOR_TREEOPS_REWRITECUBEMAPSAMPLERSAS2DARRAY_H_
#define COMPILER_TRANSLATOR_TREEOPS_REWRITECUBEMAPSAMPLERSAS2DARRAY_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

ANGLE_NO_DISCARD bool RewriteCubeMapSamplersAs2DArray(TCompiler *compiler,
                                                      TIntermBlock *root,
                                                      TSymbolTable *symbolTable,
                                                      bool isFragmentShader);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REWRITECUBEMAPSAMPLERSAS2DARRAY_H_
