// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "snapshot/win/process_reader_win.h"

#include <windows.h>
#include <string.h>

#include "gtest/gtest.h"
#include "test/win/win_multiprocess.h"
#include "util/misc/from_pointer_cast.h"
#include "util/synchronization/semaphore.h"
#include "util/thread/thread.h"
#include "util/win/scoped_process_suspend.h"

namespace crashpad {
namespace test {
namespace {

TEST(ProcessReaderWin, SelfBasic) {
  ProcessReaderWin process_reader;
  ASSERT_TRUE(process_reader.Initialize(GetCurrentProcess(),
                                        ProcessSuspensionState::kRunning));

#if !defined(ARCH_CPU_64_BITS)
  EXPECT_FALSE(process_reader.Is64Bit());
#else
  EXPECT_TRUE(process_reader.Is64Bit());
#endif

  EXPECT_EQ(process_reader.GetProcessInfo().ProcessID(), GetCurrentProcessId());

  static constexpr char kTestMemory[] = "Some test memory";
  char buffer[arraysize(kTestMemory)];
  ASSERT_TRUE(process_reader.ReadMemory(
      reinterpret_cast<uintptr_t>(kTestMemory), sizeof(kTestMemory), &buffer));
  EXPECT_STREQ(kTestMemory, buffer);
}

constexpr char kTestMemory[] = "Read me from another process";

class ProcessReaderChild final : public WinMultiprocess {
 public:
  ProcessReaderChild() : WinMultiprocess() {}
  ~ProcessReaderChild() {}

 private:
  void WinMultiprocessParent() override {
    ProcessReaderWin process_reader;
    ASSERT_TRUE(process_reader.Initialize(ChildProcess(),
                                          ProcessSuspensionState::kRunning));

#if !defined(ARCH_CPU_64_BITS)
    EXPECT_FALSE(process_reader.Is64Bit());
#else
    EXPECT_TRUE(process_reader.Is64Bit());
#endif

    WinVMAddress address;
    CheckedReadFileExactly(ReadPipeHandle(), &address, sizeof(address));

    char buffer[sizeof(kTestMemory)];
    ASSERT_TRUE(
        process_reader.ReadMemory(address, sizeof(kTestMemory), &buffer));
    EXPECT_EQ(strcmp(kTestMemory, buffer), 0);
  }

  void WinMultiprocessChild() override {
    WinVMAddress address = FromPointerCast<WinVMAddress>(kTestMemory);
    CheckedWriteFile(WritePipeHandle(), &address, sizeof(address));

    // Wait for the parent to signal that it's OK to exit by closing its end of
    // the pipe.
    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderChild);
};

TEST(ProcessReaderWin, ChildBasic) {
  WinMultiprocess::Run<ProcessReaderChild>();
}

TEST(ProcessReaderWin, SelfOneThread) {
  ProcessReaderWin process_reader;
  ASSERT_TRUE(process_reader.Initialize(GetCurrentProcess(),
                                        ProcessSuspensionState::kRunning));

  const std::vector<ProcessReaderWin::Thread>& threads =
      process_reader.Threads();

  // If other tests ran in this process previously, threads may have been
  // created and may still be running. This check must look for at least one
  // thread, not exactly one thread.
  ASSERT_GE(threads.size(), 1u);

  EXPECT_EQ(threads[0].id, GetCurrentThreadId());
#if defined(ARCH_CPU_64_BITS)
  EXPECT_NE(threads[0].context.native.Rip, 0u);
#else
  EXPECT_NE(threads[0].context.native.Eip, 0u);
#endif

  EXPECT_EQ(threads[0].suspend_count, 0u);
}

class ProcessReaderChildThreadSuspendCount final : public WinMultiprocess {
 public:
  ProcessReaderChildThreadSuspendCount() : WinMultiprocess() {}
  ~ProcessReaderChildThreadSuspendCount() {}

 private:
  enum : unsigned int { kCreatedThreads = 3 };

  class SleepingThread : public Thread {
   public:
    SleepingThread() : done_(nullptr) {}

    void SetHandle(Semaphore* done) {
      done_= done;
    }

    void ThreadMain() override {
      done_->Wait();
    };

   private:
    Semaphore* done_;
  };

  void WinMultiprocessParent() override {
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));
    ASSERT_EQ(c, ' ');

    {
      ProcessReaderWin process_reader;
      ASSERT_TRUE(process_reader.Initialize(ChildProcess(),
                                            ProcessSuspensionState::kRunning));

      const auto& threads = process_reader.Threads();
      ASSERT_GE(threads.size(), kCreatedThreads + 1);
      for (const auto& thread : threads)
        EXPECT_EQ(thread.suspend_count, 0u);
    }

    {
      ScopedProcessSuspend suspend(ChildProcess());

      ProcessReaderWin process_reader;
      ASSERT_TRUE(process_reader.Initialize(
          ChildProcess(), ProcessSuspensionState::kSuspended));

      // Confirm that thread counts are adjusted correctly for the process being
      // suspended.
      const auto& threads = process_reader.Threads();
      ASSERT_GE(threads.size(), kCreatedThreads + 1);
      for (const auto& thread : threads)
        EXPECT_EQ(thread.suspend_count, 0u);
    }
  }

  void WinMultiprocessChild() override {
    // Create three dummy threads so we can confirm we read successfully read
    // more than just the main thread.
    SleepingThread threads[kCreatedThreads];
    Semaphore done(0);
    for (auto& thread : threads)
      thread.SetHandle(&done);
    for (auto& thread : threads)
      thread.Start();

    char c = ' ';
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));

    // Wait for the parent to signal that it's OK to exit by closing its end of
    // the pipe.
    CheckedReadFileAtEOF(ReadPipeHandle());

    for (size_t i = 0; i < arraysize(threads); ++i)
      done.Signal();
    for (auto& thread : threads)
      thread.Join();
  }

  DISALLOW_COPY_AND_ASSIGN(ProcessReaderChildThreadSuspendCount);
};

TEST(ProcessReaderWin, ChildThreadSuspendCounts) {
  WinMultiprocess::Run<ProcessReaderChildThreadSuspendCount>();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
