// Copyright 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_DEBUG_ALIAS_H_
#define MINI_CHROMIUM_BASE_DEBUG_ALIAS_H_

namespace base {
namespace debug {

// Make the optimizer think that var is aliased. This is to prevent it from
// optimizing out variables that that would not otherwise be live at the point
// of a potential crash.
void Alias(const void* var);

}  // namespace debug
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_DEBUG_ALIAS_H_
