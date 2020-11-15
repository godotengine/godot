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

#ifndef CRASHPAD_TEST_GTEST_DEATH_H_
#define CRASHPAD_TEST_GTEST_DEATH_H_

#include "base/logging.h"
#include "build/build_config.h"
#include "gtest/gtest.h"

#if defined(OS_MACOSX)
#include "test/mac/exception_swallower.h"
#endif

//! \file

#if defined(OS_MACOSX) || DOXYGEN

//! \brief Wraps the gtest `ASSERT_DEATH_IF_SUPPORTED()` macro to make
//!     assertions about death caused by crashes.
//!
//! On macOS, this macro prevents the system’s crash reporter from handling
//! crashes that occur in \a statement. Crashes are normally visible to the
//! system’s crash reporter, but it is undesirable for intentional
//! ASSERT_DEATH_CRASH() crashes to be handled by any crash reporter.
//!
//! `ASSERT_DEATH_IF_SUPPORTED()` is used instead of `ASSERT_DEATH()` to
//! support platforms where death tests are not implemented by gtest (e.g.
//! Fuchsia). On platforms where death tests are not implemented, a warning
//! will be logged and the remainder of the test body skipped.
//!
//! \sa ASSERT_DEATH_CHECK()
//! \sa EXPECT_DEATH_CRASH()
#define ASSERT_DEATH_CRASH(statement, regex)                     \
  do {                                                           \
    crashpad::test::ExceptionSwallower exception_swallower;      \
    ASSERT_DEATH_IF_SUPPORTED(                                   \
        crashpad::test::ExceptionSwallower::SwallowExceptions(); \
        { statement; }, regex);                                  \
  } while (false)

//! \brief Wraps the gtest `EXPECT_DEATH_IF_SUPPORTED()` macro to make
//!     assertions about death caused by crashes.
//!
//! On macOS, this macro prevents the system’s crash reporter from handling
//! crashes that occur in \a statement. Crashes are normally visible to the
//! system’s crash reporter, but it is undesirable for intentional
//! EXPECT_DEATH_CRASH() crashes to be handled by any crash reporter.
//!
//! `EXPECT_DEATH_IF_SUPPORTED()` is used instead of `EXPECT_DEATH()` to
//! support platforms where death tests are not implemented by gtest (e.g.
//! Fuchsia). On platforms where death tests are not implemented, a warning
//! will be logged and the remainder of the test body skipped.
//!
//! \sa EXPECT_DEATH_CHECK()
//! \sa ASSERT_DEATH_CRASH()
#define EXPECT_DEATH_CRASH(statement, regex)                              \
  do {                                                                    \
    crashpad::test::ExceptionSwallower exception_swallower;               \
    EXPECT_DEATH(crashpad::test::ExceptionSwallower::SwallowExceptions(); \
                 { statement; },                                          \
                 regex);                                                  \
  } while (false)

#else  // OS_MACOSX

#define ASSERT_DEATH_CRASH(statement, regex) \
  ASSERT_DEATH_IF_SUPPORTED(statement, regex)
#define EXPECT_DEATH_CRASH(statement, regex) \
  EXPECT_DEATH_IF_SUPPORTED(statement, regex)

#endif  // OS_MACOSX

#if !(!defined(MINI_CHROMIUM_BASE_LOGGING_H_) && \
      defined(OFFICIAL_BUILD) &&                 \
      defined(NDEBUG)) ||                        \
    DOXYGEN

//! \brief Wraps the ASSERT_DEATH_CRASH() macro to make assertions about death
//!     caused by `CHECK()` failures.
//!
//! In an in-Chromium build in the official configuration, `CHECK()` does not
//! print its condition or streamed messages. In that case, this macro uses an
//! empty \a regex pattern when calling ASSERT_DEATH_CRASH() to avoid looking
//! for any particular output on the standard error stream. In other build
//! configurations, the \a regex pattern is left intact.
//!
//! On macOS, `CHECK()` failures normally show up as crashes to the system’s
//! crash reporter, but it is undesirable for intentional ASSERT_DEATH_CHECK()
//! crashes to be handled by any crash reporter, so this is implemented in
//! terms of ASSERT_DEATH_CRASH() instead of `ASSERT_DEATH()`.
//!
//! \sa EXPECT_DEATH_CHECK()
#define ASSERT_DEATH_CHECK(statement, regex) \
  ASSERT_DEATH_CRASH(statement, regex)

//! \brief Wraps the EXPECT_DEATH_CRASH() macro to make assertions about death
//!     caused by `CHECK()` failures.
//!
//! In an in-Chromium build in the official configuration, `CHECK()` does not
//! print its condition or streamed messages. In that case, this macro uses an
//! empty \a regex pattern when calling EXPECT_DEATH_CRASH() to avoid looking
//! for any particular output on the standard error stream. In other build
//! configurations, the \a regex pattern is left intact.
//!
//! On macOS, `CHECK()` failures normally show up as crashes to the system’s
//! crash reporter, but it is undesirable for intentional EXPECT_DEATH_CHECK()
//! crashes to be handled by any crash reporter, so this is implemented in
//! terms of EXPECT_DEATH_CRASH() instead of `EXPECT_DEATH()`.
//!
//! \sa ASSERT_DEATH_CHECK()
#define EXPECT_DEATH_CHECK(statement, regex) \
  EXPECT_DEATH_CRASH(statement, regex)

#else  // !(!MINI_CHROMIUM_BASE_LOGGING_H_ && OFFICIAL_BUILD && NDEBUG)

#define ASSERT_DEATH_CHECK(statement, regex) ASSERT_DEATH_CRASH(statement, "")
#define EXPECT_DEATH_CHECK(statement, regex) EXPECT_DEATH_CRASH(statement, "")

#endif  // !(!MINI_CHROMIUM_BASE_LOGGING_H_ && OFFICIAL_BUILD && NDEBUG)

#endif  // CRASHPAD_TEST_GTEST_DEATH_H_
