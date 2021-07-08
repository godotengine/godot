//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_common.mm:
//      Implementation of mtl::Context, the MTLDevice container & error handler class.
//

#include "libANGLE/renderer/metal/mtl_common.h"

#include <dispatch/dispatch.h>

#include <cstring>

#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"

namespace rx
{
namespace mtl
{

Context::Context(DisplayMtl *display) : mDisplay(display) {}

id<MTLDevice> Context::getMetalDevice() const
{
    return mDisplay->getMetalDevice();
}

mtl::CommandQueue &Context::cmdQueue()
{
    return mDisplay->cmdQueue();
}

}  // namespace mtl
}  // namespace rx