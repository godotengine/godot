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

#include "test/linux/get_tls.h"

#include "build/build_config.h"
#include "util/misc/from_pointer_cast.h"

namespace crashpad {
namespace test {

LinuxVMAddress GetTLS() {
  LinuxVMAddress tls;
#if defined(ARCH_CPU_ARMEL)
  // 0xffff0fe0 is the address of the kernel user helper __kuser_get_tls().
  auto kuser_get_tls = reinterpret_cast<void* (*)()>(0xffff0fe0);
  tls = FromPointerCast<LinuxVMAddress>(kuser_get_tls());
#elif defined(ARCH_CPU_ARM64)
  // Linux/aarch64 places the tls address in system register tpidr_el0.
  asm("mrs %0, tpidr_el0" : "=r"(tls));
#elif defined(ARCH_CPU_X86)
  uint32_t tls_32;
  asm("movl %%gs:0x0, %0" : "=r"(tls_32));
  tls = tls_32;
#elif defined(ARCH_CPU_X86_64)
  asm("movq %%fs:0x0, %0" : "=r"(tls));
#elif defined(ARCH_CPU_MIPSEL)
  uint32_t tls_32;
  asm("rdhwr   $3,$29\n\t"
      "move    %0,$3\n\t"
      : "=r"(tls_32)
      :
      : "$3");
  tls = tls_32;
#elif defined(ARCH_CPU_MIPS64EL)
  asm("rdhwr   $3,$29\n\t"
      "move    %0,$3\n\t"
      : "=r"(tls)
      :
      : "$3");
#else
#error Port.
#endif  // ARCH_CPU_ARMEL
  return tls;
}

}  // namespace test
}  // namespace crashpad
