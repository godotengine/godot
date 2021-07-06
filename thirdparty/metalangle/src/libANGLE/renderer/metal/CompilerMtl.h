//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CompilerMtl.h:
//    Defines the class interface for CompilerMtl, implementing CompilerImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_COMPILERMTL_H_
#define LIBANGLE_RENDERER_METAL_COMPILERMTL_H_

#include "libANGLE/renderer/CompilerImpl.h"

namespace rx
{

class CompilerMtl : public CompilerImpl
{
  public:
    CompilerMtl();
    ~CompilerMtl() override;

    ShShaderOutput getTranslatorOutputType() const override;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_COMPILERMTL_H_ */
