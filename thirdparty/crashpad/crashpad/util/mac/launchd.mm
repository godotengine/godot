// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/mac/launchd.h"

#import <Foundation/Foundation.h>

#include "base/mac/foundation_util.h"
#include "base/mac/scoped_launch_data.h"
#include "base/mac/scoped_cftyperef.h"
#include "base/strings/sys_string_conversions.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {

launch_data_t CFPropertyToLaunchData(CFPropertyListRef property_cf) {
  @autoreleasepool {
    // This function mixes Foundation and Core Foundation access to property
    // list elements according to which is more convenient and correct for any
    // specific task.

    launch_data_t data_launch = nullptr;
    CFTypeID type_id_cf = CFGetTypeID(property_cf);

    if (type_id_cf == CFDictionaryGetTypeID()) {
      NSDictionary* dictionary_ns = base::mac::CFToNSCast(
          base::mac::CFCastStrict<CFDictionaryRef>(property_cf));
      base::mac::ScopedLaunchData dictionary_launch(
          LaunchDataAlloc(LAUNCH_DATA_DICTIONARY));

      for (NSString* key in dictionary_ns) {
        if (![key isKindOfClass:[NSString class]]) {
          return nullptr;
        }

        CFPropertyListRef value_cf =
            implicit_cast<CFPropertyListRef>([dictionary_ns objectForKey:key]);
        launch_data_t value_launch = CFPropertyToLaunchData(value_cf);
        if (!value_launch) {
          return nullptr;
        }

        LaunchDataDictInsert(
            dictionary_launch.get(), value_launch, [key UTF8String]);
      }

      data_launch = dictionary_launch.release();

    } else if (type_id_cf == CFArrayGetTypeID()) {
      NSArray* array_ns = base::mac::CFToNSCast(
          base::mac::CFCastStrict<CFArrayRef>(property_cf));
      base::mac::ScopedLaunchData array_launch(
          LaunchDataAlloc(LAUNCH_DATA_ARRAY));
      size_t index = 0;

      for (id element_ns in array_ns) {
        CFPropertyListRef element_cf =
            implicit_cast<CFPropertyListRef>(element_ns);
        launch_data_t element_launch = CFPropertyToLaunchData(element_cf);
        if (!element_launch) {
          return nullptr;
        }

        LaunchDataArraySetIndex(array_launch.get(), element_launch, index++);
      }

      data_launch = array_launch.release();

    } else if (type_id_cf == CFNumberGetTypeID()) {
      CFNumberRef number_cf = base::mac::CFCastStrict<CFNumberRef>(property_cf);
      NSNumber* number_ns = base::mac::CFToNSCast(number_cf);
      switch (CFNumberGetType(number_cf)) {
        case kCFNumberSInt8Type:
        case kCFNumberSInt16Type:
        case kCFNumberSInt32Type:
        case kCFNumberSInt64Type:
        case kCFNumberCharType:
        case kCFNumberShortType:
        case kCFNumberIntType:
        case kCFNumberLongType:
        case kCFNumberLongLongType:
        case kCFNumberCFIndexType:
        case kCFNumberNSIntegerType: {
          data_launch = LaunchDataNewInteger([number_ns longLongValue]);
          break;
        }

        case kCFNumberFloat32Type:
        case kCFNumberFloat64Type:
        case kCFNumberFloatType:
        case kCFNumberDoubleType: {
          data_launch = LaunchDataNewReal([number_ns doubleValue]);
          break;
        }

        default: { return nullptr; }
      }

    } else if (type_id_cf == CFBooleanGetTypeID()) {
      CFBooleanRef boolean_cf =
          base::mac::CFCastStrict<CFBooleanRef>(property_cf);
      data_launch = LaunchDataNewBool(CFBooleanGetValue(boolean_cf));

    } else if (type_id_cf == CFStringGetTypeID()) {
      NSString* string_ns = base::mac::CFToNSCast(
          base::mac::CFCastStrict<CFStringRef>(property_cf));

      // -fileSystemRepresentation might be more correct than -UTF8String,
      // because these strings can hold paths. The analogous function in
      // launchctl, CF2launch_data() (10.9.4
      // launchd-842.92.1/support/launchctl.c), uses UTF-8 instead of filesystem
      // encoding, so do the same here. Note that there’s another occurrence of
      // -UTF8String above, used for dictionary keys.
      data_launch = LaunchDataNewString([string_ns UTF8String]);

    } else if (type_id_cf == CFDataGetTypeID()) {
      NSData* data_ns = base::mac::CFToNSCast(
          base::mac::CFCastStrict<CFDataRef>(property_cf));
      data_launch = LaunchDataNewOpaque([data_ns bytes], [data_ns length]);
    } else {
      base::ScopedCFTypeRef<CFStringRef> type_name_cf(
          CFCopyTypeIDDescription(type_id_cf));
      DLOG(ERROR) << "unable to convert CFTypeID " << type_id_cf << " ("
                  << base::SysCFStringRefToUTF8(type_name_cf) << ")";
    }

    return data_launch;
  }
}

}  // namespace crashpad
