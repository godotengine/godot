//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CompilerMtl.mm:
//    Implements the class methods for CompilerMtl.
//

#include "libANGLE/renderer/metal/CompilerMtl.h"

#include "common/debug.h"

namespace rx
{

CompilerMtl::CompilerMtl() : CompilerImpl() {}

CompilerMtl::~CompilerMtl() {}

ShShaderOutput CompilerMtl::getTranslatorOutputType() const
{
    return SH_GLSL_METAL_OUTPUT;
}

}  // namespace rx