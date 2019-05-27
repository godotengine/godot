// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "libshaderc_util/mutex.h"

#include <gmock/gmock.h>
#include <thread>

namespace {

TEST(MutexTest, CanCreateMutex) {
  shaderc_util::mutex mutex;
  mutex.lock();
  mutex.unlock();
}

#ifndef SHADERC_DISABLE_THREADED_TESTS

void increment_by_1000(shaderc_util::mutex& mut, int& i) {
  for(size_t j = 0; j < 1000; ++j) {
    mut.lock();
    i = i + 1;
    mut.unlock();
  }
}

TEST(MutexTest, MutexLocks) {
  shaderc_util::mutex mutex;
  int i = 0;
  std::thread t1([&mutex, &i]() { increment_by_1000(mutex, i); });
  std::thread t2([&mutex, &i]() { increment_by_1000(mutex, i); });
  std::thread t3([&mutex, &i]() { increment_by_1000(mutex, i); });
  t1.join();
  t2.join();
  t3.join();
  EXPECT_EQ(3000, i);
}
#endif // SHADERC_DISABLE_THREADED_TESTS

}  // anonymous namespace
