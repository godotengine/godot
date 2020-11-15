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

#include "test/gtest_disabled.h"

#include <stdio.h>

#include "base/format_macros.h"
#include "base/logging.h"
#include "base/strings/stringprintf.h"

namespace crashpad {
namespace test {

namespace {

DisabledTestGtestEnvironment* g_instance;

}  // namespace

// static
DisabledTestGtestEnvironment* DisabledTestGtestEnvironment::Get() {
  if (!g_instance) {
    g_instance = new DisabledTestGtestEnvironment();
  }
  return g_instance;
}

void DisabledTestGtestEnvironment::DisabledTest() {
  const testing::TestInfo* test_info =
      testing::UnitTest::GetInstance()->current_test_info();
  std::string disabled_test = base::StringPrintf(
      "%s.%s", test_info->test_case_name(), test_info->name());

  // Show a DISABLED message using a format similar to gtest, along with a hint
  // explaining that OK or FAILED will also appear.
  printf(
      "This test has been disabled dynamically.\n"
      "It will appear as both DISABLED and OK or FAILED.\n"
      "[ DISABLED ] %s\n",
      disabled_test.c_str());

  disabled_tests_.push_back(disabled_test);
}

DisabledTestGtestEnvironment::DisabledTestGtestEnvironment()
    : testing::Environment(),
      disabled_tests_() {
  DCHECK(!g_instance);
}

DisabledTestGtestEnvironment::~DisabledTestGtestEnvironment() {
  DCHECK_EQ(this, g_instance);
  g_instance = nullptr;
}

void DisabledTestGtestEnvironment::TearDown() {
  if (!disabled_tests_.empty()) {
    printf(
        "[ DISABLED ] %" PRIuS " dynamically disabled test%s, listed below:\n"
        "[ DISABLED ] %s also counted in PASSED or FAILED below.\n",
        disabled_tests_.size(),
        disabled_tests_.size() == 1 ? "" : "s",
        disabled_tests_.size() == 1 ? "This test is" : "These tests are");
    for (const std::string& disabled_test : disabled_tests_) {
      printf("[ DISABLED ] %s\n", disabled_test.c_str());
    }
  }
}

}  // namespace test
}  // namespace crashpad
