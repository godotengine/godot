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

#include "util/stdlib/thread_safe_vector.h"

#include "gtest/gtest.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

constexpr int kElementsPerThread = 100;

class ThreadSafeVectorTestThread : public Thread {
 public:
  ThreadSafeVectorTestThread() : thread_safe_vector_(nullptr), start_(0) {}
  ~ThreadSafeVectorTestThread() {}

  void SetTestParameters(ThreadSafeVector<int>* thread_safe_vector, int start) {
    thread_safe_vector_ = thread_safe_vector;
    start_ = start;
  }

  // Thread:
  void ThreadMain() override {
    for (int i = start_; i < start_ + kElementsPerThread; ++i) {
      thread_safe_vector_->PushBack(i);
    }
  }

 private:
  ThreadSafeVector<int>* thread_safe_vector_;
  int start_;

  DISALLOW_COPY_AND_ASSIGN(ThreadSafeVectorTestThread);
};

TEST(ThreadSafeVector, ThreadSafeVector) {
  ThreadSafeVector<int> thread_safe_vector;
  std::vector<int> vector = thread_safe_vector.Drain();
  EXPECT_TRUE(vector.empty());

  ThreadSafeVectorTestThread threads[100];
  for (size_t index = 0; index < arraysize(threads); ++index) {
    threads[index].SetTestParameters(
        &thread_safe_vector, static_cast<int>(index * kElementsPerThread));
  }

  for (size_t index = 0; index < arraysize(threads); ++index) {
    threads[index].Start();

    if (index % 10 == 0) {
      // Drain the vector periodically to test that simultaneous Drain() and
      // PushBack() operations work properly.
      std::vector<int> drained = thread_safe_vector.Drain();
      vector.insert(vector.end(), drained.begin(), drained.end());
    }
  }

  for (ThreadSafeVectorTestThread& thread : threads) {
    thread.Join();
  }

  std::vector<int> drained = thread_safe_vector.Drain();
  vector.insert(vector.end(), drained.begin(), drained.end());
  bool found[arraysize(threads) * kElementsPerThread] = {};
  EXPECT_EQ(vector.size(), arraysize(found));
  for (int element : vector) {
    EXPECT_FALSE(found[element]) << element;
    found[element] = true;
  }

  vector = thread_safe_vector.Drain();
  EXPECT_TRUE(vector.empty());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
