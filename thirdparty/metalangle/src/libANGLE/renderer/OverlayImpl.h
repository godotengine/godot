//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// OverlayImpl.h: Defines the abstract rx::OverlayImpl class.

#ifndef LIBANGLE_RENDERER_OVERLAYIMPL_H_
#define LIBANGLE_RENDERER_OVERLAYIMPL_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "common/mathutil.h"
#include "libANGLE/Error.h"
#include "libANGLE/Observer.h"

#include <stdint.h>

namespace gl
{
class Context;
class OverlayState;
}  // namespace gl

namespace rx
{
class OverlayImpl : angle::NonCopyable
{
  public:
    OverlayImpl(const gl::OverlayState &state) : mState(state) {}
    virtual ~OverlayImpl() {}

    virtual void onDestroy(const gl::Context *context) {}

    virtual angle::Result init(const gl::Context *context);

  protected:
    const gl::OverlayState &mState;
};

inline angle::Result OverlayImpl::init(const gl::Context *context)
{
    return angle::Result::Continue;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_OVERLAYIMPL_H_
