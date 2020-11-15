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

#include "util/net/url.h"

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(URLEncode, Empty) {
  EXPECT_EQ(URLEncode(""), "");
}

TEST(URLEncode, ReservedCharacters) {
  EXPECT_EQ(URLEncode(" !#$&'()*+,/:;=?@[]"),
            "%20%21%23%24%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D");
}

TEST(URLEncode, UnreservedCharacters) {
  EXPECT_EQ(URLEncode("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
  EXPECT_EQ(URLEncode("0123456789-_.~"), "0123456789-_.~");
}

TEST(URLEncode, SimpleAddress) {
  EXPECT_EQ(
      URLEncode("http://some.address.com/page.html?arg1=value&arg2=value"),
      "http%3A%2F%2Fsome.address.com%2Fpage.html%3Farg1%3Dvalue%26arg2%"
      "3Dvalue");
}

TEST(CrackURL, Unsupported) {
  std::string scheme, host, port, rest;

  // Not HTTP.
  EXPECT_FALSE(CrackURL("file://stuff/things", &scheme, &host, &port, &rest));

  // No resource.
  EXPECT_FALSE(CrackURL("file://stuff", &scheme, &host, &port, &rest));
  EXPECT_FALSE(CrackURL("http://stuff", &scheme, &host, &port, &rest));
  EXPECT_FALSE(CrackURL("https://stuff", &scheme, &host, &port, &rest));
}

TEST(CrackURL, UnsupportedDoesNotModifiedOutArgs) {
  std::string scheme, host, port, rest;

  scheme = "scheme";
  host = "host";
  port = "port";
  rest = "rest";

  // Bad scheme.
  EXPECT_FALSE(CrackURL("file://stuff/things", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "scheme");
  EXPECT_EQ(host, "host");
  EXPECT_EQ(port, "port");
  EXPECT_EQ(rest, "rest");

  scheme = "scheme";
  host = "host";
  port = "port";
  rest = "rest";

  // No resource.
  EXPECT_FALSE(CrackURL("http://stuff", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "scheme");
  EXPECT_EQ(host, "host");
  EXPECT_EQ(port, "port");
  EXPECT_EQ(rest, "rest");
}

TEST(CrackURL, BasicWithDefaultPort) {
  std::string scheme, host, port, rest;

  ASSERT_TRUE(CrackURL("http://stuff/things", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "http");
  EXPECT_EQ(host, "stuff");
  EXPECT_EQ(port, "80");
  EXPECT_EQ(rest, "/things");

  ASSERT_TRUE(CrackURL("https://stuff/things", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "https");
  EXPECT_EQ(host, "stuff");
  EXPECT_EQ(port, "443");
  EXPECT_EQ(rest, "/things");
}

TEST(CrackURL, BasicWithExplicitPort) {
  std::string scheme, host, port, rest;

  ASSERT_TRUE(
      CrackURL("http://stuff:999/things", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "http");
  EXPECT_EQ(host, "stuff");
  EXPECT_EQ(port, "999");
  EXPECT_EQ(rest, "/things");

  ASSERT_TRUE(
      CrackURL("https://stuff:1010/things", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "https");
  EXPECT_EQ(host, "stuff");
  EXPECT_EQ(port, "1010");
  EXPECT_EQ(rest, "/things");
}

TEST(CrackURL, WithURLParams) {
  std::string scheme, host, port, rest;

  ASSERT_TRUE(CrackURL(
      "http://stuff:999/things?blah=stuff:3", &scheme, &host, &port, &rest));
  EXPECT_EQ(scheme, "http");
  EXPECT_EQ(host, "stuff");
  EXPECT_EQ(port, "999");
  EXPECT_EQ(rest, "/things?blah=stuff:3");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
