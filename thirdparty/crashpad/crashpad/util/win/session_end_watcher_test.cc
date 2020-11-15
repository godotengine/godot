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

#include "util/win/session_end_watcher.h"

#include "gtest/gtest.h"
#include "test/errors.h"

namespace crashpad {
namespace test {
namespace {

class SessionEndWatcherTest final : public SessionEndWatcher {
 public:
  SessionEndWatcherTest() : SessionEndWatcher(), called_(false) {}

  ~SessionEndWatcherTest() override {}

  void Run() {
    WaitForStart();

    HWND window = GetWindow();
    ASSERT_TRUE(window);
    EXPECT_TRUE(PostMessage(window, WM_ENDSESSION, 1, 0));

    WaitForStop();

    EXPECT_TRUE(called_);
  }

 private:
  // SessionEndWatcher:
  void SessionEnding() override { called_ = true; }

  bool called_;

  DISALLOW_COPY_AND_ASSIGN(SessionEndWatcherTest);
};

TEST(SessionEndWatcher, SessionEndWatcher) {
  SessionEndWatcherTest test;
  test.Run();
}

TEST(SessionEndWatcher, DoNothing) {
  SessionEndWatcherTest test;
}

}  // namespace
}  // namespace test
}  // namespace crashpad
