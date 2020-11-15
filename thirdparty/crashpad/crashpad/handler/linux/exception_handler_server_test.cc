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

#include "handler/linux/exception_handler_server.h"

#include <sys/types.h>
#include <unistd.h>

#include "base/logging.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/multiprocess.h"
#include "util/linux/direct_ptrace_connection.h"
#include "util/linux/exception_handler_client.h"
#include "util/linux/ptrace_client.h"
#include "util/linux/scoped_pr_set_ptracer.h"
#include "util/synchronization/semaphore.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

// Runs the ExceptionHandlerServer on a background thread.
class RunServerThread : public Thread {
 public:
  RunServerThread(ExceptionHandlerServer* server,
                  ExceptionHandlerServer::Delegate* delegate)
      : server_(server), delegate_(delegate), join_sem_(0) {}

  ~RunServerThread() override {}

  bool JoinWithTimeout(double timeout) {
    if (!join_sem_.TimedWait(timeout)) {
      return false;
    }
    Join();
    return true;
  }

 private:
  // Thread:
  void ThreadMain() override {
    server_->Run(delegate_);
    join_sem_.Signal();
  }

  ExceptionHandlerServer* server_;
  ExceptionHandlerServer::Delegate* delegate_;
  Semaphore join_sem_;

  DISALLOW_COPY_AND_ASSIGN(RunServerThread);
};

class ScopedStopServerAndJoinThread {
 public:
  ScopedStopServerAndJoinThread(ExceptionHandlerServer* server,
                                RunServerThread* thread)
      : server_(server), thread_(thread) {}

  ~ScopedStopServerAndJoinThread() {
    server_->Stop();
    EXPECT_TRUE(thread_->JoinWithTimeout(5.0));
  }

 private:
  ExceptionHandlerServer* server_;
  RunServerThread* thread_;

  DISALLOW_COPY_AND_ASSIGN(ScopedStopServerAndJoinThread);
};

class TestDelegate : public ExceptionHandlerServer::Delegate {
 public:
  TestDelegate()
      : Delegate(), last_exception_address_(0), last_client_(-1), sem_(0) {}

  ~TestDelegate() {}

  bool WaitForException(double timeout_seconds,
                        pid_t* last_client,
                        VMAddress* last_address) {
    if (sem_.TimedWait(timeout_seconds)) {
      *last_client = last_client_;
      *last_address = last_exception_address_;
      return true;
    }

    return false;
  }

  bool HandleException(pid_t client_process_id,
                       const ClientInformation& info) override {
    DirectPtraceConnection connection;
    bool connected = connection.Initialize(client_process_id);
    EXPECT_TRUE(connected);

    last_exception_address_ = info.exception_information_address;
    last_client_ = client_process_id;
    sem_.Signal();
    return connected;
  }

  bool HandleExceptionWithBroker(pid_t client_process_id,
                                 const ClientInformation& info,
                                 int broker_sock) override {
    PtraceClient client;
    bool connected = client.Initialize(broker_sock, client_process_id);
    EXPECT_TRUE(connected);

    last_exception_address_ = info.exception_information_address,
    last_client_ = client_process_id;
    sem_.Signal();
    return connected;
  }

 private:
  VMAddress last_exception_address_;
  pid_t last_client_;
  Semaphore sem_;

  DISALLOW_COPY_AND_ASSIGN(TestDelegate);
};

class MockPtraceStrategyDecider : public PtraceStrategyDecider {
 public:
  MockPtraceStrategyDecider(PtraceStrategyDecider::Strategy strategy)
      : PtraceStrategyDecider(), strategy_(strategy) {}

  ~MockPtraceStrategyDecider() {}

  Strategy ChooseStrategy(int sock, const ucred& client_credentials) override {
    if (strategy_ == Strategy::kUseBroker) {
      ServerToClientMessage message = {};
      message.type = ServerToClientMessage::kTypeForkBroker;

      Errno status;
      bool result = LoggingWriteFile(sock, &message, sizeof(message)) &&
                    LoggingReadFileExactly(sock, &status, sizeof(status));
      EXPECT_TRUE(result);

      if (!result) {
        return Strategy::kError;
      }

      if (status != 0) {
        errno = status;
        ADD_FAILURE() << ErrnoMessage("Handler Client ForkBroker");
        return Strategy::kNoPtrace;
      }
    }
    return strategy_;
  }

 private:
  Strategy strategy_;

  DISALLOW_COPY_AND_ASSIGN(MockPtraceStrategyDecider);
};

class ExceptionHandlerServerTest : public testing::Test {
 public:
  ExceptionHandlerServerTest()
      : server_(),
        delegate_(),
        server_thread_(&server_, &delegate_),
        sock_to_handler_() {}

  ~ExceptionHandlerServerTest() = default;

  int SockToHandler() { return sock_to_handler_.get(); }

  TestDelegate* Delegate() { return &delegate_; }

  void Hangup() { sock_to_handler_.reset(); }

  RunServerThread* ServerThread() { return &server_thread_; }

  ExceptionHandlerServer* Server() { return &server_; }

  class CrashDumpTest : public Multiprocess {
   public:
    CrashDumpTest(ExceptionHandlerServerTest* server_test, bool succeeds)
        : Multiprocess(), server_test_(server_test), succeeds_(succeeds) {}

    ~CrashDumpTest() = default;

    void MultiprocessParent() override {
      ClientInformation info;
      ASSERT_TRUE(
          LoggingReadFileExactly(ReadPipeHandle(), &info, sizeof(info)));

      if (succeeds_) {
        VMAddress last_address;
        pid_t last_client;
        ASSERT_TRUE(server_test_->Delegate()->WaitForException(
            5.0, &last_client, &last_address));
        EXPECT_EQ(last_address, info.exception_information_address);
        EXPECT_EQ(last_client, ChildPID());
      } else {
        CheckedReadFileAtEOF(ReadPipeHandle());
      }
    }

    void MultiprocessChild() override {
      ASSERT_EQ(close(server_test_->sock_to_client_), 0);

      ClientInformation info;
      info.exception_information_address = 42;

      ASSERT_TRUE(LoggingWriteFile(WritePipeHandle(), &info, sizeof(info)));

      // If the current ptrace_scope is restricted, the broker needs to be set
      // as the ptracer for this process. Setting this process as its own
      // ptracer allows the broker to inherit this condition.
      ScopedPrSetPtracer set_ptracer(getpid(), /* may_log= */ true);

      ExceptionHandlerClient client(server_test_->SockToHandler());
      ASSERT_EQ(client.RequestCrashDump(info), 0);
    }

   private:
    ExceptionHandlerServerTest* server_test_;
    bool succeeds_;

    DISALLOW_COPY_AND_ASSIGN(CrashDumpTest);
  };

  void ExpectCrashDumpUsingStrategy(PtraceStrategyDecider::Strategy strategy,
                                    bool succeeds) {
    ScopedStopServerAndJoinThread stop_server(Server(), ServerThread());
    ServerThread()->Start();

    Server()->SetPtraceStrategyDecider(
        std::make_unique<MockPtraceStrategyDecider>(strategy));

    CrashDumpTest test(this, succeeds);
    test.Run();
  }

 protected:
  void SetUp() override {
    int socks[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, socks), 0);
    sock_to_handler_.reset(socks[0]);
    sock_to_client_ = socks[1];

    ASSERT_TRUE(server_.InitializeWithClient(ScopedFileHandle(socks[1])));
  }

 private:
  ExceptionHandlerServer server_;
  TestDelegate delegate_;
  RunServerThread server_thread_;
  ScopedFileHandle sock_to_handler_;
  int sock_to_client_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionHandlerServerTest);
};

TEST_F(ExceptionHandlerServerTest, ShutdownWithNoClients) {
  ServerThread()->Start();
  Hangup();
  ASSERT_TRUE(ServerThread()->JoinWithTimeout(5.0));
}

TEST_F(ExceptionHandlerServerTest, StopWithClients) {
  ServerThread()->Start();
  Server()->Stop();
  ASSERT_TRUE(ServerThread()->JoinWithTimeout(5.0));
}

TEST_F(ExceptionHandlerServerTest, StopBeforeRun) {
  Server()->Stop();
  ServerThread()->Start();
  ASSERT_TRUE(ServerThread()->JoinWithTimeout(5.0));
}

TEST_F(ExceptionHandlerServerTest, MultipleStops) {
  ServerThread()->Start();
  Server()->Stop();
  Server()->Stop();
  ASSERT_TRUE(ServerThread()->JoinWithTimeout(5.0));
}

TEST_F(ExceptionHandlerServerTest, RequestCrashDumpDefault) {
  ScopedStopServerAndJoinThread stop_server(Server(), ServerThread());
  ServerThread()->Start();

  CrashDumpTest test(this, true);
  test.Run();
}

TEST_F(ExceptionHandlerServerTest, RequestCrashDumpNoPtrace) {
  ExpectCrashDumpUsingStrategy(PtraceStrategyDecider::Strategy::kNoPtrace,
                               false);
}

TEST_F(ExceptionHandlerServerTest, RequestCrashDumpForkBroker) {
  ExpectCrashDumpUsingStrategy(PtraceStrategyDecider::Strategy::kUseBroker,
                               true);
}

TEST_F(ExceptionHandlerServerTest, RequestCrashDumpDirectPtrace) {
  ExpectCrashDumpUsingStrategy(PtraceStrategyDecider::Strategy::kDirectPtrace,
                               true);
}

TEST_F(ExceptionHandlerServerTest, RequestCrashDumpError) {
  ExpectCrashDumpUsingStrategy(PtraceStrategyDecider::Strategy::kError, false);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
