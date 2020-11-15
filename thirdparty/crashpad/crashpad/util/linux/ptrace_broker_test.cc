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

#include "util/linux/ptrace_broker.h"

#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <utility>

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/filesystem.h"
#include "test/linux/get_tls.h"
#include "test/multiprocess.h"
#include "test/scoped_temp_dir.h"
#include "util/file/file_io.h"
#include "util/linux/ptrace_client.h"
#include "util/posix/scoped_mmap.h"
#include "util/synchronization/semaphore.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

class ScopedTimeoutThread : public Thread {
 public:
  ScopedTimeoutThread() : join_sem_(0) {}
  ~ScopedTimeoutThread() { EXPECT_TRUE(JoinWithTimeout(5.0)); }

 protected:
  void ThreadMain() override { join_sem_.Signal(); }

 private:
  bool JoinWithTimeout(double timeout) {
    if (!join_sem_.TimedWait(timeout)) {
      return false;
    }
    Join();
    return true;
  }

  Semaphore join_sem_;

  DISALLOW_COPY_AND_ASSIGN(ScopedTimeoutThread);
};

class RunBrokerThread : public ScopedTimeoutThread {
 public:
  RunBrokerThread(PtraceBroker* broker)
      : ScopedTimeoutThread(), broker_(broker) {}

  ~RunBrokerThread() {}

 private:
  void ThreadMain() override {
    EXPECT_EQ(broker_->Run(), 0);
    ScopedTimeoutThread::ThreadMain();
  }

  PtraceBroker* broker_;

  DISALLOW_COPY_AND_ASSIGN(RunBrokerThread);
};

class BlockOnReadThread : public ScopedTimeoutThread {
 public:
  BlockOnReadThread(int readfd, int writefd)
      : ScopedTimeoutThread(), readfd_(readfd), writefd_(writefd) {}

  ~BlockOnReadThread() {}

 private:
  void ThreadMain() override {
    pid_t pid = syscall(SYS_gettid);
    LoggingWriteFile(writefd_, &pid, sizeof(pid));

    LinuxVMAddress tls = GetTLS();
    LoggingWriteFile(writefd_, &tls, sizeof(tls));

    CheckedReadFileAtEOF(readfd_);
    ScopedTimeoutThread::ThreadMain();
  }

  int readfd_;
  int writefd_;

  DISALLOW_COPY_AND_ASSIGN(BlockOnReadThread);
};

class SameBitnessTest : public Multiprocess {
 public:
  SameBitnessTest() : Multiprocess(), mapping_() {}
  ~SameBitnessTest() {}

 protected:
  void PreFork() override {
    ASSERT_NO_FATAL_FAILURE(Multiprocess::PreFork());

    size_t page_size = getpagesize();
    ASSERT_TRUE(mapping_.ResetMmap(nullptr,
                                   page_size * 3,
                                   PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANON,
                                   -1,
                                   0));
    ASSERT_TRUE(mapping_.ResetAddrLen(mapping_.addr(), page_size * 2));

    auto buffer = mapping_.addr_as<char*>();
    for (size_t index = 0; index < mapping_.len(); ++index) {
      buffer[index] = index % 256;
    }
  }

 private:
  void BrokerTests(bool set_broker_pid,
                   LinuxVMAddress child1_tls,
                   LinuxVMAddress child2_tls,
                   pid_t child2_tid,
                   const base::FilePath& file_dir,
                   const base::FilePath& test_file,
                   const std::string& expected_file_contents) {
    int socks[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, socks), 0);
    ScopedFileHandle broker_sock(socks[0]);
    ScopedFileHandle client_sock(socks[1]);

#if defined(ARCH_CPU_64_BITS)
    constexpr bool am_64_bit = true;
#else
    constexpr bool am_64_bit = false;
#endif  // ARCH_CPU_64_BITS

    PtraceBroker broker(
        broker_sock.get(), set_broker_pid ? ChildPID() : -1, am_64_bit);
    RunBrokerThread broker_thread(&broker);
    broker_thread.Start();

    PtraceClient client;
    ASSERT_TRUE(client.Initialize(
        client_sock.get(), ChildPID(), /* try_direct_memory= */ false));

    EXPECT_EQ(client.GetProcessID(), ChildPID());

    std::vector<pid_t> threads;
    ASSERT_TRUE(client.Threads(&threads));
    EXPECT_EQ(threads.size(), 2u);
    if (threads[0] == ChildPID()) {
      EXPECT_EQ(threads[1], child2_tid);
    } else {
      EXPECT_EQ(threads[0], child2_tid);
      EXPECT_EQ(threads[1], ChildPID());
    }

    EXPECT_TRUE(client.Attach(child2_tid));
    EXPECT_EQ(client.Is64Bit(), am_64_bit);

    ThreadInfo info1;
    ASSERT_TRUE(client.GetThreadInfo(ChildPID(), &info1));
    EXPECT_EQ(info1.thread_specific_data_address, child1_tls);

    ThreadInfo info2;
    ASSERT_TRUE(client.GetThreadInfo(child2_tid, &info2));
    EXPECT_EQ(info2.thread_specific_data_address, child2_tls);

    ProcessMemory* memory = client.Memory();
    ASSERT_TRUE(memory);

    auto buffer = std::make_unique<char[]>(mapping_.len());
    ASSERT_TRUE(memory->Read(
        mapping_.addr_as<VMAddress>(), mapping_.len(), buffer.get()));
    auto expected_buffer = mapping_.addr_as<char*>();
    for (size_t index = 0; index < mapping_.len(); ++index) {
      EXPECT_EQ(buffer[index], expected_buffer[index]);
    }

    char first;
    ASSERT_TRUE(
        memory->Read(mapping_.addr_as<VMAddress>(), sizeof(first), &first));
    EXPECT_EQ(first, expected_buffer[0]);

    char last;
    ASSERT_TRUE(memory->Read(mapping_.addr_as<VMAddress>() + mapping_.len() - 1,
                             sizeof(last),
                             &last));
    EXPECT_EQ(last, expected_buffer[mapping_.len() - 1]);

    char unmapped;
    EXPECT_FALSE(memory->Read(mapping_.addr_as<VMAddress>() + mapping_.len(),
                              sizeof(unmapped),
                              &unmapped));

    std::string file_root = file_dir.value() + '/';
    broker.SetFileRoot(file_root.c_str());

    std::string file_contents;
    ASSERT_TRUE(client.ReadFileContents(test_file, &file_contents));
    EXPECT_EQ(file_contents, expected_file_contents);

    ScopedTempDir temp_dir2;
    base::FilePath test_file2(temp_dir2.path().Append("test_file2"));
    ASSERT_TRUE(CreateFile(test_file2));
    EXPECT_FALSE(client.ReadFileContents(test_file2, &file_contents));
  }

  void MultiprocessParent() override {
    LinuxVMAddress child1_tls;
    ASSERT_TRUE(LoggingReadFileExactly(
        ReadPipeHandle(), &child1_tls, sizeof(child1_tls)));

    pid_t child2_tid;
    ASSERT_TRUE(LoggingReadFileExactly(
        ReadPipeHandle(), &child2_tid, sizeof(child2_tid)));

    LinuxVMAddress child2_tls;
    ASSERT_TRUE(LoggingReadFileExactly(
        ReadPipeHandle(), &child2_tls, sizeof(child2_tls)));

    ScopedTempDir temp_dir;
    base::FilePath file_path(temp_dir.path().Append("test_file"));
    std::string expected_file_contents;
    {
      expected_file_contents.resize(4097);
      for (size_t i = 0; i < expected_file_contents.size(); ++i) {
        expected_file_contents[i] = static_cast<char>(i % 256);
      }
      ScopedFileHandle handle(
          LoggingOpenFileForWrite(file_path,
                                  FileWriteMode::kCreateOrFail,
                                  FilePermissions::kWorldReadable));
      ASSERT_TRUE(LoggingWriteFile(handle.get(),
                                   expected_file_contents.data(),
                                   expected_file_contents.size()));
    }

    BrokerTests(true,
                child1_tls,
                child2_tls,
                child2_tid,
                temp_dir.path(),
                file_path,
                expected_file_contents);
    BrokerTests(false,
                child1_tls,
                child2_tls,
                child2_tid,
                temp_dir.path(),
                file_path,
                expected_file_contents);
  }

  void MultiprocessChild() override {
    LinuxVMAddress tls = GetTLS();
    ASSERT_TRUE(LoggingWriteFile(WritePipeHandle(), &tls, sizeof(tls)));

    BlockOnReadThread thread(ReadPipeHandle(), WritePipeHandle());
    thread.Start();

    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  ScopedMmap mapping_;

  DISALLOW_COPY_AND_ASSIGN(SameBitnessTest);
};

TEST(PtraceBroker, SameBitness) {
  SameBitnessTest test;
  test.Run();
}

// TODO(jperaza): Test against a process with different bitness.

}  // namespace
}  // namespace test
}  // namespace crashpad
