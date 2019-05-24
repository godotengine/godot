// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef UTIL_TEST_H_
#define UTIL_TEST_H_

#include "util/util.h"
#include "util/flags.h"
#include "util/logging.h"

#define TEST(x, y) \
	void x##y(void); \
	TestRegisterer r##x##y(x##y, # x "." # y); \
	void x##y(void)

void RegisterTest(void (*)(void), const char*);

class TestRegisterer {
 public:
  TestRegisterer(void (*fn)(void), const char *s) {
    RegisterTest(fn, s);
  }
};

// fatal assertions
#define ASSERT_TRUE CHECK
#define ASSERT_FALSE(x) CHECK(!(x))
#define ASSERT_EQ CHECK_EQ
#define ASSERT_NE CHECK_NE
#define ASSERT_LT CHECK_LT
#define ASSERT_LE CHECK_LE
#define ASSERT_GT CHECK_GT
#define ASSERT_GE CHECK_GE

// nonfatal assertions
// TODO(rsc): Do a better job?
#define EXPECT_TRUE CHECK
#define EXPECT_FALSE(x) CHECK(!(x))
#define EXPECT_EQ CHECK_EQ
#define EXPECT_NE CHECK_NE
#define EXPECT_LT CHECK_LT
#define EXPECT_LE CHECK_LE
#define EXPECT_GT CHECK_GT
#define EXPECT_GE CHECK_GE

namespace testing {
class MallocCounter {
 public:
  MallocCounter(int x) {}
  static const int THIS_THREAD_ONLY = 0;
  long long HeapGrowth() { return 0; }
  long long PeakHeapGrowth() { return 0; }
  void Reset() {}
};
}  // namespace testing

#endif  // UTIL_TEST_H_
