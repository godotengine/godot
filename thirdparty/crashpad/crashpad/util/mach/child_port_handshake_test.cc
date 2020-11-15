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

#include "util/mach/child_port_handshake.h"

#include "base/mac/scoped_mach_port.h"
#include "gtest/gtest.h"
#include "test/multiprocess.h"
#include "util/mach/child_port_types.h"
#include "util/mach/mach_extensions.h"

namespace crashpad {
namespace test {
namespace {

class ChildPortHandshakeTest : public Multiprocess {
 public:
  enum class ClientProcess {
    // The child runs the client and the parent runs the server.
    kChildClient = 0,

    // The parent runs the client and the child runs the server.
    kParentClient,
  };

  enum class TestType {
    // The client checks in with the server, transferring a receive right.
    kClientChecksIn_ReceiveRight = 0,

    // In this test, the client checks in with the server normally. It sends a
    // copy of its bootstrap port to the server, because both parent and child
    // should have the same bootstrap port, allowing for verification.
    kClientChecksIn_SendRight,

    // The client checks in with the server, transferring a send-once right.
    kClientChecksIn_SendOnceRight,

    // In this test, the client reads from its pipe, and subsequently exits
    // without checking in. This tests that the server properly detects that it
    // has lost its client after sending instructions to it via the pipe, while
    // waiting for a check-in message.
    kClientDoesNotCheckIn,

    // In this test, the client exits without checking in. This tests that the
    // server properly detects that it has lost a client. Whether or not the
    // client closes the pipe before the server writes to it is a race, and the
    // server needs to be able to detect client loss in both cases, so the
    // ClientDoesNotCheckIn_ReadsPipe and NoClient tests also exist to test
    // these individual cases more deterministically.
    kClientDoesNotCheckIn_ReadsPipe,

    // In this test, the client checks in with the server with an incorrect
    // token value and a copy of its own task port. The server should reject the
    // message because of the invalid token, and return MACH_PORT_NULL to its
    // caller.
    kTokenIncorrect,

    // In this test, the client checks in with the server with an incorrect
    // token value and a copy of its own task port, and subsequently, the
    // correct token value and a copy of its bootstrap port. The server should
    // reject the first because of the invalid token, but it should continue
    // waiting for a message with a valid token as long as the pipe remains
    // open. It should wind wind up returning the bootstrap port, allowing for
    // verification.
    kTokenIncorrectThenCorrect,

    // The server dies. The failure should be reported in the client. This test
    // type is only compatible with ClientProcess::kParentClient.
    kServerDies,
  };

  ChildPortHandshakeTest(ClientProcess client_process, TestType test_type)
      : Multiprocess(),
        child_port_handshake_(),
        client_process_(client_process),
        test_type_(test_type) {
  }

  ~ChildPortHandshakeTest() {
  }

 private:
  void RunServer() {
    if (test_type_ == TestType::kServerDies) {
      return;
    }

    base::mac::ScopedMachReceiveRight receive_right;
    base::mac::ScopedMachSendRight send_right;
    if (test_type_ == TestType::kClientChecksIn_ReceiveRight) {
      receive_right.reset(child_port_handshake_.RunServer(
          ChildPortHandshake::PortRightType::kReceiveRight));
    } else {
      send_right.reset(child_port_handshake_.RunServer(
          ChildPortHandshake::PortRightType::kSendRight));
    }

    switch (test_type_) {
      case TestType::kClientChecksIn_ReceiveRight:
        EXPECT_TRUE(receive_right.is_valid());
        break;

      case TestType::kClientChecksIn_SendRight:
      case TestType::kTokenIncorrectThenCorrect:
        EXPECT_EQ(send_right, bootstrap_port);
        break;

      case TestType::kClientChecksIn_SendOnceRight:
        EXPECT_TRUE(send_right.is_valid());
        EXPECT_NE(send_right, bootstrap_port);
        break;

      case TestType::kClientDoesNotCheckIn:
      case TestType::kClientDoesNotCheckIn_ReadsPipe:
      case TestType::kTokenIncorrect:
        EXPECT_FALSE(send_right.is_valid());
        break;

      case TestType::kServerDies:
        // This was special-cased as an early return above.
        FAIL();
        break;
    }
  }

  void RunClient() {
    switch (test_type_) {
      case TestType::kClientChecksIn_SendRight: {
        ASSERT_TRUE(child_port_handshake_.RunClient(bootstrap_port,
                                                    MACH_MSG_TYPE_COPY_SEND));
        break;
      }

      case TestType::kClientChecksIn_ReceiveRight: {
        mach_port_t receive_right = NewMachPort(MACH_PORT_RIGHT_RECEIVE);
        ASSERT_TRUE(child_port_handshake_.RunClient(
              receive_right, MACH_MSG_TYPE_MOVE_RECEIVE));
        break;
      }

      case TestType::kClientChecksIn_SendOnceRight: {
        base::mac::ScopedMachReceiveRight receive_right(
            NewMachPort(MACH_PORT_RIGHT_RECEIVE));
        ASSERT_TRUE(child_port_handshake_.RunClient(
            receive_right.get(), MACH_MSG_TYPE_MAKE_SEND_ONCE));
        break;
      }

      case TestType::kClientDoesNotCheckIn: {
        child_port_handshake_.ServerWriteFD().reset();
        child_port_handshake_.ClientReadFD().reset();
        break;
      }

      case TestType::kClientDoesNotCheckIn_ReadsPipe: {
        // Don’t run the standard client routine. Instead, drain the pipe, which
        // will get the parent to the point that it begins waiting for a
        // check-in message. Then, exit. The pipe is drained using the same
        // implementation that the real client would use.
        child_port_handshake_.ServerWriteFD().reset();
        base::ScopedFD client_read_fd = child_port_handshake_.ClientReadFD();
        child_port_token_t token;
        std::string service_name;
        ASSERT_TRUE(ChildPortHandshake::RunClientInternal_ReadPipe(
            client_read_fd.get(), &token, &service_name));
        break;
      }

      case TestType::kTokenIncorrect: {
        // Don’t run the standard client routine. Instead, read the token and
        // service name, mutate the token, and then check in with the bad token.
        // The parent should reject the message.
        child_port_handshake_.ServerWriteFD().reset();
        base::ScopedFD client_read_fd = child_port_handshake_.ClientReadFD();
        child_port_token_t token;
        std::string service_name;
        ASSERT_TRUE(ChildPortHandshake::RunClientInternal_ReadPipe(
            client_read_fd.get(), &token, &service_name));
        child_port_token_t bad_token = ~token;
        ASSERT_TRUE(ChildPortHandshake::RunClientInternal_SendCheckIn(
            service_name,
            bad_token,
            mach_task_self(),
            MACH_MSG_TYPE_COPY_SEND));
        break;
      }

      case TestType::kTokenIncorrectThenCorrect: {
        // Don’t run the standard client routine. Instead, read the token and
        // service name. Mutate the token, and check in with the bad token,
        // expecting the parent to reject the message. Then, check in with the
        // correct token, expecting the parent to accept it.
        child_port_handshake_.ServerWriteFD().reset();
        base::ScopedFD client_read_fd = child_port_handshake_.ClientReadFD();
        child_port_token_t token;
        std::string service_name;
        ASSERT_TRUE(ChildPortHandshake::RunClientInternal_ReadPipe(
            client_read_fd.release(), &token, &service_name));
        child_port_token_t bad_token = ~token;
        ASSERT_TRUE(ChildPortHandshake::RunClientInternal_SendCheckIn(
            service_name,
            bad_token,
            mach_task_self(),
            MACH_MSG_TYPE_COPY_SEND));
        ASSERT_TRUE(ChildPortHandshake::RunClientInternal_SendCheckIn(
            service_name, token, bootstrap_port, MACH_MSG_TYPE_COPY_SEND));
        break;
      }

      case TestType::kServerDies: {
        ASSERT_EQ(client_process_, ClientProcess::kParentClient);
        ASSERT_FALSE(child_port_handshake_.RunClient(bootstrap_port,
                                                     MACH_MSG_TYPE_COPY_SEND));
        break;
      }
    }
  }

  // Multiprocess:

  void MultiprocessParent() override {
    switch (client_process_) {
      case ClientProcess::kChildClient:
        RunServer();
        break;
      case ClientProcess::kParentClient:
        RunClient();
        break;
    }
  }

  void MultiprocessChild() override {
    switch (client_process_) {
      case ClientProcess::kChildClient:
        RunClient();
        break;
      case ClientProcess::kParentClient:
        RunServer();
        break;
    }
  }

 private:
  ChildPortHandshake child_port_handshake_;
  ClientProcess client_process_;
  TestType test_type_;

  DISALLOW_COPY_AND_ASSIGN(ChildPortHandshakeTest);
};

TEST(ChildPortHandshake, ChildClientChecksIn_ReceiveRight) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kClientChecksIn_ReceiveRight);
  test.Run();
}

TEST(ChildPortHandshake, ChildClientChecksIn_SendRight) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kClientChecksIn_SendRight);
  test.Run();
}

TEST(ChildPortHandshake, ChildClientChecksIn_SendOnceRight) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kClientChecksIn_SendOnceRight);
  test.Run();
}

TEST(ChildPortHandshake, ChildClientDoesNotCheckIn) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kClientDoesNotCheckIn);
  test.Run();
}

TEST(ChildPortHandshake, ChildClientDoesNotCheckIn_ReadsPipe) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kClientDoesNotCheckIn_ReadsPipe);
  test.Run();
}

TEST(ChildPortHandshake, ChildClientTokenIncorrect) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kTokenIncorrect);
  test.Run();
}

TEST(ChildPortHandshake, ChildClientTokenIncorrectThenCorrect) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kChildClient,
      ChildPortHandshakeTest::TestType::kTokenIncorrectThenCorrect);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientChecksIn_ReceiveRight) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kClientChecksIn_ReceiveRight);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientChecksIn_SendRight) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kClientChecksIn_SendRight);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientChecksIn_SendOnceRight) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kClientChecksIn_SendOnceRight);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientDoesNotCheckIn) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kClientDoesNotCheckIn);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientDoesNotCheckIn_ReadsPipe) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kClientDoesNotCheckIn_ReadsPipe);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientTokenIncorrect) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kTokenIncorrect);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientTokenIncorrectThenCorrect) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kTokenIncorrectThenCorrect);
  test.Run();
}

TEST(ChildPortHandshake, ParentClientServerDies) {
  ChildPortHandshakeTest test(
      ChildPortHandshakeTest::ClientProcess::kParentClient,
      ChildPortHandshakeTest::TestType::kServerDies);
  test.Run();
}

TEST(ChildPortHandshake, NoClient) {
  // In this test, the client never checks in with the server because it never
  // even runs. This tests that the server properly detects that it has no
  // client at all, and does not terminate execution with an error such as
  // “broken pipe” when attempting to send instructions to the client. This test
  // is similar to kClientDoesNotCheckIn, but because there’s no client at all,
  // the server is guaranteed to see that its pipe partner is gone.
  ChildPortHandshake child_port_handshake;
  base::mac::ScopedMachSendRight child_port(child_port_handshake.RunServer(
      ChildPortHandshake::PortRightType::kSendRight));
  EXPECT_FALSE(child_port.is_valid());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
