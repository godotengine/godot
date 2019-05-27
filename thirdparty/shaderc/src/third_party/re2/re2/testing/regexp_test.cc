// Copyright 2006 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test parse.cc, dump.cc, and tostring.cc.

#include <stddef.h>
#include <map>
#include <string>
#include <vector>

#include "util/test.h"
#include "util/logging.h"
#include "re2/regexp.h"

namespace re2 {

// Test that overflowed ref counts work.
TEST(Regexp, BigRef) {
  Regexp* re;
  re = Regexp::Parse("x", Regexp::NoParseFlags, NULL);
  for (int i = 0; i < 100000; i++)
    re->Incref();
  for (int i = 0; i < 100000; i++)
    re->Decref();
  ASSERT_EQ(re->Ref(), 1);
  re->Decref();
}

// Test that very large Concats work.
// Depends on overflowed ref counts working.
TEST(Regexp, BigConcat) {
  Regexp* x;
  x = Regexp::Parse("x", Regexp::NoParseFlags, NULL);
  std::vector<Regexp*> v(90000, x);  // ToString bails out at 100000
  for (size_t i = 0; i < v.size(); i++)
    x->Incref();
  ASSERT_EQ(x->Ref(), 1 + static_cast<int>(v.size())) << x->Ref();
  Regexp* re = Regexp::Concat(v.data(), static_cast<int>(v.size()),
                              Regexp::NoParseFlags);
  ASSERT_EQ(re->ToString(), std::string(v.size(), 'x'));
  re->Decref();
  ASSERT_EQ(x->Ref(), 1) << x->Ref();
  x->Decref();
}

TEST(Regexp, NamedCaptures) {
  Regexp* x;
  RegexpStatus status;
  x = Regexp::Parse(
      "(?P<g1>a+)|(e)(?P<g2>w*)+(?P<g1>b+)", Regexp::PerlX, &status);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(4, x->NumCaptures());
  const std::map<std::string, int>* have = x->NamedCaptures();
  EXPECT_TRUE(have != NULL);
  EXPECT_EQ(2, have->size());  // there are only two named groups in
                               // the regexp: 'g1' and 'g2'.
  std::map<std::string, int> want;
  want["g1"] = 1;
  want["g2"] = 3;
  EXPECT_EQ(want, *have);
  x->Decref();
  delete have;
}

TEST(Regexp, CaptureNames) {
  Regexp* x;
  RegexpStatus status;
  x = Regexp::Parse(
      "(?P<g1>a+)|(e)(?P<g2>w*)+(?P<g1>b+)", Regexp::PerlX, &status);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(4, x->NumCaptures());
  const std::map<int, std::string>* have = x->CaptureNames();
  EXPECT_TRUE(have != NULL);
  EXPECT_EQ(3, have->size());
  std::map<int, std::string> want;
  want[1] = "g1";
  want[3] = "g2";
  want[4] = "g1";

  EXPECT_EQ(want, *have);
  x->Decref();
  delete have;
}

}  // namespace re2
