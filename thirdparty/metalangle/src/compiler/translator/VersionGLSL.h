//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_VERSIONGLSL_H_
#define COMPILER_TRANSLATOR_VERSIONGLSL_H_

#include "compiler/translator/tree_util/IntermTraverse.h"

#include "compiler/translator/Pragma.h"

namespace sh
{

static const int GLSL_VERSION_110 = 110;
static const int GLSL_VERSION_120 = 120;
static const int GLSL_VERSION_130 = 130;
static const int GLSL_VERSION_140 = 140;
static const int GLSL_VERSION_150 = 150;
static const int GLSL_VERSION_330 = 330;
static const int GLSL_VERSION_400 = 400;
static const int GLSL_VERSION_410 = 410;
static const int GLSL_VERSION_420 = 420;
static const int GLSL_VERSION_430 = 430;
static const int GLSL_VERSION_440 = 440;
static const int GLSL_VERSION_450 = 450;

int ShaderOutputTypeToGLSLVersion(ShShaderOutput output);

// Traverses the intermediate tree to return the minimum GLSL version
// required to legally access all built-in features used in the shader.
// GLSL 1.1 which is mandated by OpenGL 2.0 provides:
//   - #version and #extension to declare version and extensions.
//   - built-in functions refract, exp, and log.
//   - updated step() to compare x < edge instead of x <= edge.
// GLSL 1.2 which is mandated by OpenGL 2.1 provides:
//   - many changes to reduce differences when compared to the ES specification.
//   - invariant keyword and its support.
//   - c++ style name hiding rules.
//   - built-in variable gl_PointCoord for fragment shaders.
//   - matrix constructors taking matrix as argument.
//   - array as "out" function parameters
//
// TODO: ES3 equivalent versions of GLSL
class TVersionGLSL : public TIntermTraverser
{
  public:
    TVersionGLSL(sh::GLenum type, const TPragma &pragma, ShShaderOutput output);

    // If output is core profile, returns 150.
    // If output is legacy profile,
    //   Returns 120 if the following is used the shader:
    //   - "invariant",
    //   - "gl_PointCoord",
    //   - matrix/matrix constructors
    //   - array "out" parameters
    //   Else 110 is returned.
    int getVersion() const { return mVersion; }

    void visitSymbol(TIntermSymbol *node) override;
    bool visitAggregate(Visit, TIntermAggregate *node) override;
    bool visitInvariantDeclaration(Visit, TIntermInvariantDeclaration *node) override;
    void visitFunctionPrototype(TIntermFunctionPrototype *node) override;
    bool visitDeclaration(Visit, TIntermDeclaration *node) override;

  private:
    void ensureVersionIsAtLeast(int version);

    int mVersion;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VERSIONGLSL_H_
