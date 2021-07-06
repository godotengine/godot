//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderMtl.h:
//    Defines the class interface for ShaderMtl, implementing ShaderImpl.
//
#ifndef LIBANGLE_RENDERER_METAL_SHADERMTL_H_
#define LIBANGLE_RENDERER_METAL_SHADERMTL_H_

#include <map>

#include "libANGLE/renderer/ShaderImpl.h"

namespace rx
{

class ShaderMtl : public ShaderImpl
{
  public:
    ShaderMtl(const gl::ShaderState &data);
    ~ShaderMtl() override;

    std::shared_ptr<WaitableCompileEvent> compile(const gl::Context *context,
                                                  gl::ShCompilerInstance *compilerInstance,
                                                  ShCompileOptions options) override;

    std::string getDebugInfo() const override;
};

}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_SHADERMTL_H_ */
