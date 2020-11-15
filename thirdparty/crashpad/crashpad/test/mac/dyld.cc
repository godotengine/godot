// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "test/mac/dyld.h"

#include <AvailabilityMacros.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <stdint.h>

#include "base/logging.h"
#include "snapshot/mac/process_reader_mac.h"
#include "test/scoped_module_handle.h"
#include "util/numeric/safe_assignment.h"

#if MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_13
extern "C" {

// A non-public dyld API, declared in 10.12.4
// dyld-433.5/include/mach-o/dyld_priv.h. The code still exists in 10.13, but
// its symbol is no longer public, so it can’t be used there.
const dyld_all_image_infos* _dyld_get_all_image_infos()
    __attribute__((weak_import));

}  // extern "C"
#endif

namespace crashpad {
namespace test {

const dyld_all_image_infos* DyldGetAllImageInfos() {
#if MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_13
  // When building with the pre-10.13 SDK, the weak_import declaration above is
  // available and a symbol will be present in the SDK to link against. If the
  // old interface is also available at run time (running on pre-10.13), use it.
  if (_dyld_get_all_image_infos) {
    return _dyld_get_all_image_infos();
  }
#elif MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_13
  // When building with the 10.13 SDK or later, but able to run on pre-10.13,
  // look for _dyld_get_all_image_infos in the same module that provides
  // _dyld_image_count. There’s no symbol in the SDK to link against, so this is
  // a little more involved than the pre-10.13 SDK case above.
  Dl_info dli;
  if (!dladdr(reinterpret_cast<void*>(_dyld_image_count), &dli)) {
    LOG(WARNING) << "dladdr: failed";
  } else {
    ScopedModuleHandle module(
        dlopen(dli.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD));
    if (!module.valid()) {
      LOG(WARNING) << "dlopen: " << dlerror();
    } else {
      using DyldGetAllImageInfosType = const dyld_all_image_infos*(*)();
      const auto _dyld_get_all_image_infos =
          module.LookUpSymbol<DyldGetAllImageInfosType>(
              "_dyld_get_all_image_infos");
      if (_dyld_get_all_image_infos) {
        return _dyld_get_all_image_infos();
      }
    }
  }
#endif

  // On 10.13 and later, do it the hard way.
  ProcessReaderMac process_reader;
  if (!process_reader.Initialize(mach_task_self())) {
    return nullptr;
  }

  mach_vm_address_t all_image_info_addr_m =
      process_reader.DyldAllImageInfo(nullptr);
  if (!all_image_info_addr_m) {
    return nullptr;
  }

  uintptr_t all_image_info_addr_u;
  if (!AssignIfInRange(&all_image_info_addr_u, all_image_info_addr_m)) {
    LOG(ERROR) << "all_image_info_addr_m " << all_image_info_addr_m
               << " out of range";
    return nullptr;
  }

  return reinterpret_cast<const dyld_all_image_infos*>(all_image_info_addr_u);
}

}  // namespace test
}  // namespace crashpad
