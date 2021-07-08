//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// driver_utils_mac.mm : provides mac-specific information about current driver.

#include "libANGLE/renderer/driver_utils.h"

#import <Foundation/Foundation.h>

namespace rx
{

OSVersion GetMacOSVersion()
{
    OSVersion result;

    NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
    result.majorVersion              = static_cast<int>(version.majorVersion);
    result.minorVersion              = static_cast<int>(version.minorVersion);
    result.patchVersion              = static_cast<int>(version.patchVersion);

    return result;
}

}
