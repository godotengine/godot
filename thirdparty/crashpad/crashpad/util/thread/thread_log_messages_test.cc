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

#include "util/thread/thread_log_messages.h"

#include <string.h>
#include <sys/types.h>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

TEST(ThreadLogMessages, Empty) {
  ThreadLogMessages thread_log_messages;

  const std::vector<std::string>& log_messages =
      thread_log_messages.log_messages();

  EXPECT_TRUE(log_messages.empty());
}

// For a message formatted like "[preamble] message\n", returns just "message".
// If the message is not formatted as expected, a gtest expectation failure will
// be recorded and this function will return an empty string.
std::string MessageString(const std::string& log_message) {
  if (log_message.size() < 1) {
    EXPECT_GE(log_message.size(), 1u);
    return std::string();
  }

  constexpr char kStartChar = '[';
  if (log_message[0] != kStartChar) {
    EXPECT_EQ(log_message[0], kStartChar);
    return std::string();
  }

  static constexpr char kFindString[] = "] ";
  size_t pos = log_message.find(kFindString);
  if (pos == std::string::npos) {
    EXPECT_NE(pos, std::string::npos);
    return std::string();
  }

  std::string message_string = log_message.substr(pos + strlen(kFindString));
  if (message_string.size() < 1) {
    EXPECT_GE(message_string.size(), 1u);
    return std::string();
  }

  constexpr char kEndChar = '\n';
  if (message_string[message_string.size() - 1] != kEndChar) {
    EXPECT_NE(message_string[message_string.size() - 1], kEndChar);
    return std::string();
  }

  message_string.resize(message_string.size() - 1);
  return message_string;
}

TEST(ThreadLogMessages, Basic) {
  // Logging must be enabled at least at this level for this test to work.
  ASSERT_TRUE(LOG_IS_ON(INFO));

  {
    static constexpr const char* kMessages[] = {
      "An info message",
      "A warning message",
      "An error message",
    };

    ThreadLogMessages thread_log_messages;

    LOG(INFO) << kMessages[0];
    LOG(WARNING) << kMessages[1];
    LOG(ERROR) << kMessages[2];

    const std::vector<std::string>& log_messages =
        thread_log_messages.log_messages();

    EXPECT_EQ(log_messages.size(), arraysize(kMessages));
    for (size_t index = 0; index < arraysize(kMessages); ++index) {
      EXPECT_EQ(MessageString(log_messages[index]), kMessages[index])
          << "index " << index;
    }
  }

  {
    static constexpr char kMessage[] = "Sample error message";

    ThreadLogMessages thread_log_messages;

    LOG(ERROR) << kMessage;

    const std::vector<std::string>& log_messages =
        thread_log_messages.log_messages();

    EXPECT_EQ(log_messages.size(), 1u);
    EXPECT_EQ(MessageString(log_messages[0]), kMessage);
  }

  {
    ThreadLogMessages thread_log_messages;

    LOG(INFO) << "I can't believe I " << "streamed" << " the whole thing.";

    const std::vector<std::string>& log_messages =
        thread_log_messages.log_messages();

    EXPECT_EQ(log_messages.size(), 1u);
    EXPECT_EQ(MessageString(log_messages[0]),
              "I can't believe I streamed the whole thing.");
  }
}

class LoggingTestThread : public Thread {
 public:
  LoggingTestThread() : thread_number_(0), start_(0), count_(0) {}
  ~LoggingTestThread() override {}

  void Initialize(size_t thread_number, int start, int count) {
    thread_number_ = thread_number;
    start_ = start;
    count_ = count;
  }

 private:
  void ThreadMain() override {
    ThreadLogMessages thread_log_messages;

    std::vector<std::string> expected_messages;
    for (int index = start_; index < start_ + count_; ++index) {
      std::string message = base::StringPrintf("message %d", index);
      expected_messages.push_back(message);
      LOG(WARNING) << message;
    }

    const std::vector<std::string>& log_messages =
        thread_log_messages.log_messages();

    ASSERT_EQ(log_messages.size(), static_cast<size_t>(count_));
    for (size_t index = 0; index < log_messages.size(); ++index) {
      EXPECT_EQ(MessageString(log_messages[index]), expected_messages[index])
          << "thread_number_ " << thread_number_ << ", index " << index;
    }
  }

  size_t thread_number_;
  int start_;
  int count_;

  DISALLOW_COPY_AND_ASSIGN(LoggingTestThread);
};

TEST(ThreadLogMessages, Multithreaded) {
  // Logging must be enabled at least at this level for this test to work.
  ASSERT_TRUE(LOG_IS_ON(WARNING));

  LoggingTestThread threads[20];
  int start = 0;
  for (size_t index = 0; index < arraysize(threads); ++index) {
    threads[index].Initialize(
        index, static_cast<int>(start), static_cast<int>(index));
    start += static_cast<int>(index);

    ASSERT_NO_FATAL_FAILURE(threads[index].Start());
  }

  for (LoggingTestThread& thread : threads) {
    thread.Join();
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
