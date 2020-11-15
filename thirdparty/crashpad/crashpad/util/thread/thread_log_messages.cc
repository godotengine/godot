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

#include <sys/types.h>

#include "base/logging.h"
#include "base/threading/thread_local_storage.h"

namespace crashpad {

namespace {

// While an object of this class exists, it will be set as the log message
// handler. A thread may register its thread-specific log message list to
// receive messages produced just on that thread.
//
// Only one object of this class may exist in the program at a time as created
// by GetInstance(). There must not be any log message handler in effect when it
// is created, and nothing else can be set as a log message handler while an
// object of this class exists.
class ThreadLogMessagesMaster {
 public:
  void SetThreadMessageList(std::vector<std::string>* message_list) {
    DCHECK_EQ(logging::GetLogMessageHandler(), &LogMessageHandler);
    DCHECK_NE(tls_.Get() != nullptr, message_list != nullptr);
    tls_.Set(message_list);
  }

  static ThreadLogMessagesMaster* GetInstance() {
    static auto master = new ThreadLogMessagesMaster();
    return master;
  }

 private:
  ThreadLogMessagesMaster() {
    DCHECK(!tls_.initialized());
    tls_.Initialize(nullptr);
    DCHECK(tls_.initialized());

    DCHECK(!logging::GetLogMessageHandler());
    logging::SetLogMessageHandler(LogMessageHandler);
  }

  ~ThreadLogMessagesMaster() = delete;

  static bool LogMessageHandler(logging::LogSeverity severity,
                                const char* file_path,
                                int line,
                                size_t message_start,
                                const std::string& string) {
    std::vector<std::string>* log_messages =
        reinterpret_cast<std::vector<std::string>*>(tls_.Get());
    if (log_messages) {
      log_messages->push_back(string);
    }

    // Don’t consume the message. Allow it to be logged as if nothing was set as
    // the log message handler.
    return false;
  }

  static base::ThreadLocalStorage::StaticSlot tls_;

  DISALLOW_COPY_AND_ASSIGN(ThreadLogMessagesMaster);
};

// static
base::ThreadLocalStorage::StaticSlot ThreadLogMessagesMaster::tls_
    = TLS_INITIALIZER;

}  // namespace

ThreadLogMessages::ThreadLogMessages() : log_messages_() {
  ThreadLogMessagesMaster::GetInstance()->SetThreadMessageList(&log_messages_);
}

ThreadLogMessages::~ThreadLogMessages() {
  ThreadLogMessagesMaster::GetInstance()->SetThreadMessageList(nullptr);
}

}  // namespace crashpad
