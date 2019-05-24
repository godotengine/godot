// Copyright 2010 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stddef.h>
#include <string>
#include <vector>

#include "util/test.h"
#include "util/logging.h"
#include "re2/re2.h"
#include "re2/set.h"

namespace re2 {

TEST(Set, Unanchored) {
  RE2::Set s(RE2::DefaultOptions, RE2::UNANCHORED);

  ASSERT_EQ(s.Add("foo", NULL), 0);
  ASSERT_EQ(s.Add("(", NULL), -1);
  ASSERT_EQ(s.Add("bar", NULL), 1);
  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("foobar", NULL), true);
  ASSERT_EQ(s.Match("fooba", NULL), true);
  ASSERT_EQ(s.Match("oobar", NULL), true);

  std::vector<int> v;
  ASSERT_EQ(s.Match("foobar", &v), true);
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);

  ASSERT_EQ(s.Match("fooba", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  ASSERT_EQ(s.Match("oobar", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 1);
}

TEST(Set, UnanchoredFactored) {
  RE2::Set s(RE2::DefaultOptions, RE2::UNANCHORED);

  ASSERT_EQ(s.Add("foo", NULL), 0);
  ASSERT_EQ(s.Add("(", NULL), -1);
  ASSERT_EQ(s.Add("foobar", NULL), 1);
  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("foobar", NULL), true);
  ASSERT_EQ(s.Match("obarfoobaroo", NULL), true);
  ASSERT_EQ(s.Match("fooba", NULL), true);
  ASSERT_EQ(s.Match("oobar", NULL), false);

  std::vector<int> v;
  ASSERT_EQ(s.Match("foobar", &v), true);
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);

  ASSERT_EQ(s.Match("obarfoobaroo", &v), true);
  ASSERT_EQ(v.size(), 2);
  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);

  ASSERT_EQ(s.Match("fooba", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  ASSERT_EQ(s.Match("oobar", &v), false);
  ASSERT_EQ(v.size(), 0);
}

TEST(Set, UnanchoredDollar) {
  RE2::Set s(RE2::DefaultOptions, RE2::UNANCHORED);

  ASSERT_EQ(s.Add("foo$", NULL), 0);
  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("foo", NULL), true);
  ASSERT_EQ(s.Match("foobar", NULL), false);

  std::vector<int> v;
  ASSERT_EQ(s.Match("foo", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  ASSERT_EQ(s.Match("foobar", &v), false);
  ASSERT_EQ(v.size(), 0);
}

TEST(Set, UnanchoredWordBoundary) {
  RE2::Set s(RE2::DefaultOptions, RE2::UNANCHORED);

  ASSERT_EQ(s.Add("foo\\b", NULL), 0);
  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("foo", NULL), true);
  ASSERT_EQ(s.Match("foobar", NULL), false);
  ASSERT_EQ(s.Match("foo bar", NULL), true);

  std::vector<int> v;
  ASSERT_EQ(s.Match("foo", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  ASSERT_EQ(s.Match("foobar", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("foo bar", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);
}

TEST(Set, Anchored) {
  RE2::Set s(RE2::DefaultOptions, RE2::ANCHOR_BOTH);

  ASSERT_EQ(s.Add("foo", NULL), 0);
  ASSERT_EQ(s.Add("(", NULL), -1);
  ASSERT_EQ(s.Add("bar", NULL), 1);
  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("foobar", NULL), false);
  ASSERT_EQ(s.Match("fooba", NULL), false);
  ASSERT_EQ(s.Match("oobar", NULL), false);
  ASSERT_EQ(s.Match("foo", NULL), true);
  ASSERT_EQ(s.Match("bar", NULL), true);

  std::vector<int> v;
  ASSERT_EQ(s.Match("foobar", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("fooba", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("oobar", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("foo", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  ASSERT_EQ(s.Match("bar", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 1);
}

TEST(Set, EmptyUnanchored) {
  RE2::Set s(RE2::DefaultOptions, RE2::UNANCHORED);

  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("", NULL), false);
  ASSERT_EQ(s.Match("foobar", NULL), false);

  std::vector<int> v;
  ASSERT_EQ(s.Match("", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("foobar", &v), false);
  ASSERT_EQ(v.size(), 0);
}

TEST(Set, EmptyAnchored) {
  RE2::Set s(RE2::DefaultOptions, RE2::ANCHOR_BOTH);

  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("", NULL), false);
  ASSERT_EQ(s.Match("foobar", NULL), false);

  std::vector<int> v;
  ASSERT_EQ(s.Match("", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("foobar", &v), false);
  ASSERT_EQ(v.size(), 0);
}

TEST(Set, Prefix) {
  RE2::Set s(RE2::DefaultOptions, RE2::ANCHOR_BOTH);

  ASSERT_EQ(s.Add("/prefix/\\d*", NULL), 0);
  ASSERT_EQ(s.Compile(), true);

  ASSERT_EQ(s.Match("/prefix", NULL), false);
  ASSERT_EQ(s.Match("/prefix/", NULL), true);
  ASSERT_EQ(s.Match("/prefix/42", NULL), true);

  std::vector<int> v;
  ASSERT_EQ(s.Match("/prefix", &v), false);
  ASSERT_EQ(v.size(), 0);

  ASSERT_EQ(s.Match("/prefix/", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);

  ASSERT_EQ(s.Match("/prefix/42", &v), true);
  ASSERT_EQ(v.size(), 1);
  ASSERT_EQ(v[0], 0);
}

TEST(Set, OutOfMemory) {
  RE2::Set s(RE2::DefaultOptions, RE2::UNANCHORED);

  std::string a(10000, 'a');
  ASSERT_EQ(s.Add(a, NULL), 0);
  ASSERT_EQ(s.Compile(), true);

  std::vector<int> v;
  RE2::Set::ErrorInfo ei;
  ASSERT_EQ(s.Match(a, &v, &ei), false);
  ASSERT_EQ(v.size(), 0);
  ASSERT_EQ(ei.kind, RE2::Set::kOutOfMemory);
}

}  // namespace re2
