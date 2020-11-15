// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "snapshot/fuchsia/process_reader_fuchsia.h"

#include <pthread.h>
#include <zircon/process.h>
#include <zircon/syscalls.h>
#include <zircon/syscalls/port.h>
#include <zircon/types.h>

#include "gtest/gtest.h"
#include "test/multiprocess_exec.h"
#include "test/test_paths.h"
#include "util/fuchsia/scoped_task_suspend.h"

namespace crashpad {
namespace test {
namespace {

TEST(ProcessReaderFuchsia, SelfBasic) {
  ProcessReaderFuchsia process_reader;
  ASSERT_TRUE(process_reader.Initialize(*zx::process::self()));

  static constexpr char kTestMemory[] = "Some test memory";
  char buffer[arraysize(kTestMemory)];
  ASSERT_TRUE(process_reader.Memory()->Read(
      reinterpret_cast<zx_vaddr_t>(kTestMemory), sizeof(kTestMemory), &buffer));
  EXPECT_STREQ(kTestMemory, buffer);

  const auto& modules = process_reader.Modules();
  EXPECT_GT(modules.size(), 0u);
  for (const auto& module : modules) {
    EXPECT_FALSE(module.name.empty());
    EXPECT_NE(module.type, ModuleSnapshot::kModuleTypeUnknown);
  }

  const auto& threads = process_reader.Threads();
  EXPECT_GT(threads.size(), 0u);

  zx_info_handle_basic_t info;
  ASSERT_EQ(zx_object_get_info(zx_thread_self(),
                               ZX_INFO_HANDLE_BASIC,
                               &info,
                               sizeof(info),
                               nullptr,
                               nullptr),
            ZX_OK);
  EXPECT_EQ(threads[0].id, info.koid);
  EXPECT_EQ(threads[0].state, ZX_THREAD_STATE_RUNNING);
  EXPECT_EQ(threads[0].name, "initial-thread");
}

constexpr char kTestMemory[] = "Read me from another process";

CRASHPAD_CHILD_TEST_MAIN(ProcessReaderBasicChildTestMain) {
  zx_vaddr_t addr = reinterpret_cast<zx_vaddr_t>(kTestMemory);
  CheckedWriteFile(
      StdioFileHandle(StdioStream::kStandardOutput), &addr, sizeof(addr));
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class BasicChildTest : public MultiprocessExec {
 public:
  BasicChildTest() : MultiprocessExec() {
    SetChildTestMainFunction("ProcessReaderBasicChildTestMain");
  }
  ~BasicChildTest() {}

 private:
  void MultiprocessParent() override {
    ProcessReaderFuchsia process_reader;
    ASSERT_TRUE(process_reader.Initialize(*ChildProcess()));

    zx_vaddr_t addr;
    ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(), &addr, sizeof(addr)));

    std::string read_string;
    ASSERT_TRUE(process_reader.Memory()->ReadCString(addr, &read_string));
    EXPECT_EQ(read_string, kTestMemory);
  }

  DISALLOW_COPY_AND_ASSIGN(BasicChildTest);
};

TEST(ProcessReaderFuchsia, ChildBasic) {
  BasicChildTest test;
  test.Run();
}

void* SignalAndSleep(void* arg) {
  zx_port_packet_t packet = {};
  packet.type = ZX_PKT_TYPE_USER;
  zx_port_queue(*reinterpret_cast<zx_handle_t*>(arg), &packet);
  zx_nanosleep(ZX_TIME_INFINITE);
  return nullptr;
}

CRASHPAD_CHILD_TEST_MAIN(ProcessReaderChildThreadsTestMain) {
  // Create 5 threads with stack sizes of 4096, 8192, ...
  zx_handle_t port;
  zx_status_t status = zx_port_create(0, &port);
  EXPECT_EQ(status, ZX_OK);

  constexpr size_t kNumThreads = 5;
  for (size_t i = 0; i < kNumThreads; ++i) {
    pthread_attr_t attr;
    EXPECT_EQ(pthread_attr_init(&attr), 0);
    EXPECT_EQ(pthread_attr_setstacksize(&attr, (i + 1) * 4096), 0);
    pthread_t thread;
    EXPECT_EQ(pthread_create(&thread, &attr, &SignalAndSleep, &port), 0);
  }

  // Wait until all threads are ready.
  for (size_t i = 0; i < kNumThreads; ++i) {
    zx_port_packet_t packet;
    zx_port_wait(port, ZX_TIME_INFINITE, &packet);
  }

  char c = ' ';
  CheckedWriteFile(
      StdioFileHandle(StdioStream::kStandardOutput), &c, sizeof(c));
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ThreadsChildTest : public MultiprocessExec {
 public:
  ThreadsChildTest() : MultiprocessExec() {
    SetChildTestMainFunction("ProcessReaderChildThreadsTestMain");
  }
  ~ThreadsChildTest() {}

 private:
  void MultiprocessParent() override {
    char c;
    ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(), &c, 1));
    ASSERT_EQ(c, ' ');

    ScopedTaskSuspend suspend(*ChildProcess());

    ProcessReaderFuchsia process_reader;
    ASSERT_TRUE(process_reader.Initialize(*ChildProcess()));

    const auto& threads = process_reader.Threads();
    EXPECT_EQ(threads.size(), 6u);

    for (size_t i = 1; i < 6; ++i) {
      ASSERT_GT(threads[i].stack_regions.size(), 0u);
      EXPECT_EQ(threads[i].stack_regions[0].size(), i * 4096u);
    }
  }

  DISALLOW_COPY_AND_ASSIGN(ThreadsChildTest);
};

TEST(ProcessReaderFuchsia, ChildThreads) {
  ThreadsChildTest test;
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
