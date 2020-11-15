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

#include "minidump/minidump_context_writer.h"

#include <stdint.h>

#include "gtest/gtest.h"
#include "minidump/minidump_context.h"
#include "minidump/test/minidump_context_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/cpu_context.h"
#include "snapshot/test/test_cpu_context.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

template <typename Writer, typename Context>
void EmptyContextTest(void (*expect_context)(uint32_t, const Context*, bool)) {
  Writer context_writer;
  StringFile string_file;
  EXPECT_TRUE(context_writer.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(), sizeof(Context));

  const Context* observed =
      MinidumpWritableAtRVA<Context>(string_file.string(), 0);
  ASSERT_TRUE(observed);

  expect_context(0, observed, false);
}

TEST(MinidumpContextWriter, MinidumpContextX86Writer) {
  StringFile string_file;

  {
    // Make sure that a context writer that’s untouched writes a zeroed-out
    // context.
    SCOPED_TRACE("zero");

    EmptyContextTest<MinidumpContextX86Writer, MinidumpContextX86>(
        ExpectMinidumpContextX86);
  }

  {
    SCOPED_TRACE("nonzero");

    string_file.Reset();
    constexpr uint32_t kSeed = 0x8086;

    MinidumpContextX86Writer context_writer;
    InitializeMinidumpContextX86(context_writer.context(), kSeed);

    EXPECT_TRUE(context_writer.WriteEverything(&string_file));
    ASSERT_EQ(string_file.string().size(), sizeof(MinidumpContextX86));

    const MinidumpContextX86* observed =
        MinidumpWritableAtRVA<MinidumpContextX86>(string_file.string(), 0);
    ASSERT_TRUE(observed);

    ExpectMinidumpContextX86(kSeed, observed, false);
  }
}

TEST(MinidumpContextWriter, MinidumpContextAMD64Writer) {
  {
    // Make sure that a heap-allocated context writer has the proper alignment,
    // because it may be nonstandard.
    auto context_writer = std::make_unique<MinidumpContextAMD64Writer>();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(context_writer.get()) &
                  (alignof(MinidumpContextAMD64Writer) - 1),
              0u);
  }

  StringFile string_file;

  {
    // Make sure that a context writer that’s untouched writes a zeroed-out
    // context.
    SCOPED_TRACE("zero");

    EmptyContextTest<MinidumpContextAMD64Writer, MinidumpContextAMD64>(
        ExpectMinidumpContextAMD64);
  }

  {
    SCOPED_TRACE("nonzero");

    string_file.Reset();
    constexpr uint32_t kSeed = 0x808664;

    MinidumpContextAMD64Writer context_writer;
    InitializeMinidumpContextAMD64(context_writer.context(), kSeed);

    EXPECT_TRUE(context_writer.WriteEverything(&string_file));
    ASSERT_EQ(string_file.string().size(), sizeof(MinidumpContextAMD64));

    const MinidumpContextAMD64* observed =
        MinidumpWritableAtRVA<MinidumpContextAMD64>(string_file.string(), 0);
    ASSERT_TRUE(observed);

    ExpectMinidumpContextAMD64(kSeed, observed, false);
  }
}

template <typename Writer, typename Context>
void FromSnapshotTest(const CPUContext& snapshot_context,
                      void (*expect_context)(uint32_t, const Context*, bool),
                      uint32_t seed) {
  std::unique_ptr<MinidumpContextWriter> context_writer =
      MinidumpContextWriter::CreateFromSnapshot(&snapshot_context);
  ASSERT_TRUE(context_writer);

  StringFile string_file;
  ASSERT_TRUE(context_writer->WriteEverything(&string_file));

  const Context* observed =
      MinidumpWritableAtRVA<Context>(string_file.string(), 0);
  ASSERT_TRUE(observed);

  expect_context(seed, observed, true);
}

TEST(MinidumpContextWriter, X86_FromSnapshot) {
  constexpr uint32_t kSeed = 32;
  CPUContextX86 context_x86;
  CPUContext context;
  context.x86 = &context_x86;
  InitializeCPUContextX86(&context, kSeed);
  FromSnapshotTest<MinidumpContextX86Writer, MinidumpContextX86>(
      context, ExpectMinidumpContextX86, kSeed);
}

TEST(MinidumpContextWriter, AMD64_FromSnapshot) {
  constexpr uint32_t kSeed = 64;
  CPUContextX86_64 context_x86_64;
  CPUContext context;
  context.x86_64 = &context_x86_64;
  InitializeCPUContextX86_64(&context, kSeed);
  FromSnapshotTest<MinidumpContextAMD64Writer, MinidumpContextAMD64>(
      context, ExpectMinidumpContextAMD64, kSeed);
}

TEST(MinidumpContextWriter, ARM_Zeros) {
  EmptyContextTest<MinidumpContextARMWriter, MinidumpContextARM>(
      ExpectMinidumpContextARM);
}

TEST(MinidumpContextWRiter, ARM64_Zeros) {
  EmptyContextTest<MinidumpContextARM64Writer, MinidumpContextARM64>(
      ExpectMinidumpContextARM64);
}

TEST(MinidumpContextWriter, ARM_FromSnapshot) {
  constexpr uint32_t kSeed = 32;
  CPUContextARM context_arm;
  CPUContext context;
  context.arm = &context_arm;
  InitializeCPUContextARM(&context, kSeed);
  FromSnapshotTest<MinidumpContextARMWriter, MinidumpContextARM>(
      context, ExpectMinidumpContextARM, kSeed);
}

TEST(MinidumpContextWriter, ARM64_FromSnapshot) {
  constexpr uint32_t kSeed = 64;
  CPUContextARM64 context_arm64;
  CPUContext context;
  context.arm64 = &context_arm64;
  InitializeCPUContextARM64(&context, kSeed);
  FromSnapshotTest<MinidumpContextARM64Writer, MinidumpContextARM64>(
      context, ExpectMinidumpContextARM64, kSeed);
}

TEST(MinidumpContextWriter, MIPS_Zeros) {
  EmptyContextTest<MinidumpContextMIPSWriter, MinidumpContextMIPS>(
      ExpectMinidumpContextMIPS);
}

TEST(MinidumpContextWriter, MIPS64_Zeros) {
  EmptyContextTest<MinidumpContextMIPS64Writer, MinidumpContextMIPS64>(
      ExpectMinidumpContextMIPS64);
}

TEST(MinidumpContextWriter, MIPS_FromSnapshot) {
  constexpr uint32_t kSeed = 32;
  CPUContextMIPS context_mips;
  CPUContext context;
  context.mipsel = &context_mips;
  InitializeCPUContextMIPS(&context, kSeed);
  FromSnapshotTest<MinidumpContextMIPSWriter, MinidumpContextMIPS>(
      context, ExpectMinidumpContextMIPS, kSeed);
}

TEST(MinidumpContextWriter, MIPS64_FromSnapshot) {
  constexpr uint32_t kSeed = 64;
  CPUContextMIPS64 context_mips;
  CPUContext context;
  context.mips64 = &context_mips;
  InitializeCPUContextMIPS64(&context, kSeed);
  FromSnapshotTest<MinidumpContextMIPS64Writer, MinidumpContextMIPS64>(
      context, ExpectMinidumpContextMIPS64, kSeed);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
