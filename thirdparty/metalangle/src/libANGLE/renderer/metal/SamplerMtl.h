//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SamplerMtl.h:
//    Defines the class interface for SamplerMtl, implementing SamplerImpl.
//

#ifndef LIBANGLE_RENDERER_METAL_SAMPLERMTL_H_
#define LIBANGLE_RENDERER_METAL_SAMPLERMTL_H_

#include "libANGLE/renderer/SamplerImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"

namespace rx
{

class ContextMtl;

class SamplerMtl : public SamplerImpl
{
  public:
    SamplerMtl(const gl::SamplerState &state);
    ~SamplerMtl() override;

    void onDestroy(const gl::Context *context) override;
    angle::Result syncState(const gl::Context *context, const bool dirty) override;
    const mtl::AutoObjCPtr<id<MTLSamplerState>> &getSampler(ContextMtl *contextMtl);

  private:
    mtl::AutoObjCPtr<id<MTLSamplerState>> mSamplerState;
    GLenum mCompareMode = 0;
    GLenum mCompareFunc = 0;
};

}  // namespace rx

#endif