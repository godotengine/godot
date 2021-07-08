//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// CompilerImpl.h: Defines the rx::CompilerImpl class, an implementation interface
//                 for the gl::Compiler object.

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"

#ifndef LIBANGLE_RENDERER_COMPILERIMPL_H_
#    define LIBANGLE_RENDERER_COMPILERIMPL_H_

namespace rx
{

class CompilerImpl : angle::NonCopyable
{
  public:
    CompilerImpl() {}
    virtual ~CompilerImpl() {}

    // TODO(jmadill): Expose translator built-in resources init method.
    virtual ShShaderOutput getTranslatorOutputType() const = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_COMPILERIMPL_H_
