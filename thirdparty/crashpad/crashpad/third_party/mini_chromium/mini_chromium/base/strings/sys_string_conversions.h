// Copyright 2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_STRINGS_SYS_STRING_CONVERSIONS_H_
#define MINI_CHROMIUM_BASE_STRINGS_SYS_STRING_CONVERSIONS_H_

#include "build/build_config.h"

#if defined(OS_MACOSX)

#include <CoreFoundation/CoreFoundation.h>

#include <string>

#if defined(__OBJC__)
#import <Foundation/Foundation.h>
#else
class NSString;
#endif

namespace base {

std::string SysCFStringRefToUTF8(CFStringRef ref);
std::string SysNSStringToUTF8(NSString* nsstring);
CFStringRef SysUTF8ToCFStringRef(const std::string& utf8);
NSString* SysUTF8ToNSString(const std::string& utf8);

}  // namespace base

#endif  // defined(OS_MACOSX)

#endif  // MINI_CHROMIUM_BASE_STRINGS_SYS_STRING_CONVERSIONS_H_
