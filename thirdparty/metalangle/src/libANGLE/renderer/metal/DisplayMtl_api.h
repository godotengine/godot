//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayMtl_api.h:
//    Defines the Metal Display APIs to be used by the code outside metal module.
//

#ifndef LIBANGLE_RENDERER_METAL_DISPLAYMTL_API_H_
#define LIBANGLE_RENDERER_METAL_DISPLAYMTL_API_H_

#include "libANGLE/renderer/DisplayImpl.h"

namespace rx
{
// Check whether minimum required Metal version is available on the host platform.
bool IsMetalDisplayAvailable();
DisplayImpl *CreateMetalDisplay(const egl::DisplayState &state);
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_DISPLAYMTL_API_H_ */
