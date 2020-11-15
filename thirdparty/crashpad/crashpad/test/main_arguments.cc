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

#include "test/main_arguments.h"

#include "base/logging.h"

namespace crashpad {
namespace test {

const std::vector<std::string>* g_arguments;

void InitializeMainArguments(int argc, char* argv[]) {
  CHECK(!g_arguments);
  CHECK(argc);
  CHECK(argv);

  g_arguments = new const std::vector<std::string>(argv, argv + argc);
}

const std::vector<std::string>& GetMainArguments() {
  CHECK(g_arguments);
  return *g_arguments;
}

}  // namespace test
}  // namespace crashpad
