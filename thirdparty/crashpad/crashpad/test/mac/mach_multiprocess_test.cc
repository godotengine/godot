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

#include "test/mac/mach_multiprocess.h"

#include <unistd.h>

#include "base/macros.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

class TestMachMultiprocess final : public MachMultiprocess {
 public:
  TestMachMultiprocess() : MachMultiprocess() {}

  ~TestMachMultiprocess() {}

 private:
  // MachMultiprocess will have already exercised the Mach ports for IPC and the
  // child task port.
  void MachMultiprocessParent() override {}

  void MachMultiprocessChild() override {}

  DISALLOW_COPY_AND_ASSIGN(TestMachMultiprocess);
};

TEST(MachMultiprocess, MachMultiprocess) {
  TestMachMultiprocess mach_multiprocess;
  mach_multiprocess.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
