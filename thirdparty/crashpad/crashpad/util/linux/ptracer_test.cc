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

#include "util/linux/ptracer.h"

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/linux/get_tls.h"
#include "test/multiprocess.h"
#include "util/file/file_io.h"
#include "util/linux/scoped_ptrace_attach.h"

namespace crashpad {
namespace test {
namespace {

class SameBitnessTest : public Multiprocess {
 public:
  SameBitnessTest() : Multiprocess() {}
  ~SameBitnessTest() {}

 private:
  void MultiprocessParent() override {
    LinuxVMAddress expected_tls;
    CheckedReadFileExactly(
        ReadPipeHandle(), &expected_tls, sizeof(expected_tls));

#if defined(ARCH_CPU_64_BITS)
    constexpr bool am_64_bit = true;
#else
    constexpr bool am_64_bit = false;
#endif  // ARCH_CPU_64_BITS

    ScopedPtraceAttach attach;
    ASSERT_TRUE(attach.ResetAttach(ChildPID()));

    Ptracer ptracer(am_64_bit, /* can_log= */ true);

    EXPECT_EQ(ptracer.Is64Bit(), am_64_bit);

    ThreadInfo thread_info;
    ASSERT_TRUE(ptracer.GetThreadInfo(ChildPID(), &thread_info));

#if defined(ARCH_CPU_X86_64)
    EXPECT_EQ(thread_info.thread_context.t64.fs_base, expected_tls);
#endif  // ARCH_CPU_X86_64

    EXPECT_EQ(thread_info.thread_specific_data_address, expected_tls);
  }

  void MultiprocessChild() override {
    LinuxVMAddress expected_tls = GetTLS();
    CheckedWriteFile(WritePipeHandle(), &expected_tls, sizeof(expected_tls));

    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  DISALLOW_COPY_AND_ASSIGN(SameBitnessTest);
};

TEST(Ptracer, SameBitness) {
  SameBitnessTest test;
  test.Run();
}

// TODO(jperaza): Test against a process with different bitness.

}  // namespace
}  // namespace test
}  // namespace crashpad
