//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SamplerMtl.mm:
//    Defines the class interface for SamplerMtl, implementing SamplerImpl.
//

#include "libANGLE/renderer/metal/SamplerMtl.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"

namespace rx
{

SamplerMtl::SamplerMtl(const gl::SamplerState &state) : SamplerImpl(state) {}

SamplerMtl::~SamplerMtl() = default;

void SamplerMtl::onDestroy(const gl::Context *context)
{
    mSamplerState = nil;
}

const mtl::AutoObjCPtr<id<MTLSamplerState>> &SamplerMtl::getSampler(ContextMtl *contextMtl)
{
    if (!mSamplerState)
    {
        DisplayMtl *displayMtl = contextMtl->getDisplay();

        mtl::SamplerDesc samplerDesc(mState);

        mSamplerState =
            displayMtl->getStateCache().getSamplerState(displayMtl->getMetalDevice(), samplerDesc);
    }

    return mSamplerState;
}

angle::Result SamplerMtl::syncState(const gl::Context *context, const bool dirty)
{
    if (dirty)
    {
        // Recreate sampler
        mSamplerState = nil;

        if (mCompareMode != mState.getCompareMode() || mCompareFunc != mState.getCompareFunc())
        {
            ContextMtl *contextMtl = mtl::GetImpl(context);

            mCompareMode = mState.getCompareMode();
            mCompareFunc = mState.getCompareFunc();

            // Tell context to rebind textures so that ProgramMtl has a chance to verify
            // depth texture compare mode.
            contextMtl->invalidateCurrentTextures();
        }
    }
    return angle::Result::Continue;
}

}  // namespace rx
