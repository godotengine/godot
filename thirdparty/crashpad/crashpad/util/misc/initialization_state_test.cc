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

#include "util/misc/initialization_state.h"

#include <stdlib.h>

#include <memory>

#include "base/memory/free_deleter.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(InitializationState, InitializationState) {
  // Use placement new so that the buffer used to host the object remains live
  // even after the object is destroyed.
  std::unique_ptr<InitializationState, base::FreeDeleter>
  initialization_state_buffer(
      static_cast<InitializationState*>(malloc(sizeof(InitializationState))));

  InitializationState* initialization_state =
      new (initialization_state_buffer.get()) InitializationState();

  EXPECT_TRUE(initialization_state->is_uninitialized());
  EXPECT_FALSE(initialization_state->is_valid());

  initialization_state->set_invalid();

  EXPECT_FALSE(initialization_state->is_uninitialized());
  EXPECT_FALSE(initialization_state->is_valid());

  initialization_state->set_valid();

  EXPECT_FALSE(initialization_state->is_uninitialized());
  EXPECT_TRUE(initialization_state->is_valid());

  initialization_state->~InitializationState();

  // initialization_state points to something that no longer exists. This
  // portion of the test is intended to check that after an InitializationState
  // object is destroyed, it will not be considered valid on a use-after-free,
  // assuming that nothing else was written to its former home in memory.
  //
  // Because initialization_state was constructed via placement new into a
  // buffer that’s still valid and its destructor was called directly, this
  // approximates use-after-free without risking that the memory formerly used
  // for the InitializationState object has been repurposed.
  EXPECT_FALSE(initialization_state->is_uninitialized());
  EXPECT_FALSE(initialization_state->is_valid());
}

}  // namespace
}  // namespace test
}  // namespace crashpad
