//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ExtensionGLSL.h: Defines the TExtensionGLSL class that tracks GLSL extension requirements of
// shaders.

#ifndef COMPILER_TRANSLATOR_EXTENSIONGLSL_H_
#define COMPILER_TRANSLATOR_EXTENSIONGLSL_H_

#include <set>
#include <string>

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

// Traverses the intermediate tree to determine which GLSL extensions are required
// to support the shader.
class TExtensionGLSL : public TIntermTraverser
{
  public:
    TExtensionGLSL(ShShaderOutput output);

    const std::set<std::string> &getEnabledExtensions() const;
    const std::set<std::string> &getRequiredExtensions() const;

    bool visitUnary(Visit visit, TIntermUnary *node) override;
    bool visitAggregate(Visit visit, TIntermAggregate *node) override;

  private:
    void checkOperator(TIntermOperator *node);

    int mTargetVersion;

    std::set<std::string> mEnabledExtensions;
    std::set<std::string> mRequiredExtensions;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_EXTENSIONGLSL_H_
