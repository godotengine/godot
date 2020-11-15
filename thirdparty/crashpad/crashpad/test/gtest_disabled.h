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

#ifndef CRASHPAD_TEST_GTEST_DISABLED_H_
#define CRASHPAD_TEST_GTEST_DISABLED_H_

#include <string>
#include <vector>

#include "base/macros.h"
#include "gtest/gtest.h"

//! \file

namespace crashpad {
namespace test {

//! \brief Provides support for dynamically disabled gtest tests.
//!
//! A test runner must register this with gtest as follows prior to calling
//! `RUN_ALL_TESTS()`:
//! \code
//!   testing::AddGlobalTestEnvironment(
//!       crashpad::test::DisabledTestGtestEnvironment::Get());
//! \endcode
class DisabledTestGtestEnvironment final : public testing::Environment {
 public:
  //! \brief Returns the DisabledTestGtestEnvironment singleton instance,
  //!     creating it if necessary.
  static DisabledTestGtestEnvironment* Get();

  //! \brief Displays a message about a test being disabled, and arranges for
  //!     this information to be duplicated in TearDown().
  //!
  //! This method is for the internal use of the DISABLED_TEST() macro. Do not
  //! call it directly, use the macro instead.
  void DisabledTest();

 private:
  DisabledTestGtestEnvironment();
  ~DisabledTestGtestEnvironment() override;

  // testing::Environment:
  void TearDown() override;

  std::vector<std::string> disabled_tests_;

  DISALLOW_COPY_AND_ASSIGN(DisabledTestGtestEnvironment);
};

}  // namespace test
}  // namespace crashpad

//! \brief Displays a message about a test being disabled, and returns early.
//!
//! gtest only provides a mechanism for tests to be disabled statically, by
//! prefixing test case names or test names with `DISABLED_`. When it is
//! necessary to disable tests dynamically, gtest provides no assistance. This
//! macro displays a message about the disabled test and returns early. The
//! dynamically disabled test will also be displayed during gtest global test
//! environment tear-down before the test executable exits.
//!
//! This macro may only be invoked from the context of a gtest test.
//!
//! There’s a long-standing <a
//! href="https://groups.google.com/d/topic/googletestframework/Nwh3u7YFuN4">gtest
//! feature request</a> to provide this functionality directly in gtest, but
//! since it hasn’t been implemented, this macro provides a local mechanism to
//! achieve it.
#define DISABLED_TEST()                                                    \
  do {                                                                     \
    ::crashpad::test::DisabledTestGtestEnvironment::Get()->DisabledTest(); \
    return;                                                                \
  } while (false)

#endif  // CRASHPAD_TEST_GTEST_DISABLED_H_
