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

#include "util/mac/xattr.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <limits>

#include "base/files/scoped_file.h"
#include "base/posix/eintr_wrapper.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/scoped_temp_dir.h"

namespace crashpad {
namespace test {
namespace {

class Xattr : public testing::Test {
 protected:
  // testing::Test:

  void SetUp() override {
    path_ = temp_dir_.path().Append("xattr_file");
    base::ScopedFD tmp(
        HANDLE_EINTR(open(path_.value().c_str(),
                          O_WRONLY | O_CREAT | O_TRUNC | O_NOCTTY | O_CLOEXEC,
                          0644)));
    EXPECT_GE(tmp.get(), 0) << ErrnoMessage("open");
  }

  void TearDown() override {
    EXPECT_EQ(unlink(path_.value().c_str()), 0) << ErrnoMessage("unlink");
  }

  const base::FilePath& path() const { return path_; }

 private:
  ScopedTempDir temp_dir_;
  base::FilePath path_;
};

constexpr char kKey[] = "org.chromium.crashpad.test";

TEST_F(Xattr, ReadNonExistentXattr) {
  std::string value;
  EXPECT_EQ(ReadXattr(path(), kKey, &value), XattrStatus::kNoAttribute);
}

TEST_F(Xattr, WriteAndReadString) {
  std::string value = "hello world";
  EXPECT_TRUE(WriteXattr(path(), kKey, value));

  std::string actual;
  EXPECT_EQ(ReadXattr(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, value);
}

TEST_F(Xattr, WriteAndReadVeryLongString) {
  std::string value(533, 'A');
  EXPECT_TRUE(WriteXattr(path(), kKey, value));

  std::string actual;
  EXPECT_EQ(ReadXattr(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, value);
}

TEST_F(Xattr, WriteAndReadBool) {
  EXPECT_TRUE(WriteXattrBool(path(), kKey, true));
  bool actual = false;
  EXPECT_EQ(ReadXattrBool(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_TRUE(actual);

  EXPECT_TRUE(WriteXattrBool(path(), kKey, false));
  EXPECT_EQ(ReadXattrBool(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_FALSE(actual);
}

TEST_F(Xattr, WriteAndReadInt) {
  int expected = 42;
  int actual;

  EXPECT_TRUE(WriteXattrInt(path(), kKey, expected));
  EXPECT_EQ(ReadXattrInt(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, expected);

  expected = std::numeric_limits<int>::max();
  EXPECT_TRUE(WriteXattrInt(path(), kKey, expected));
  EXPECT_EQ(ReadXattrInt(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, expected);
}

TEST_F(Xattr, WriteAndReadTimeT) {
  time_t expected = time(nullptr);
  time_t actual;

  EXPECT_TRUE(WriteXattrTimeT(path(), kKey, expected));
  EXPECT_EQ(ReadXattrTimeT(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, expected);

  expected = std::numeric_limits<time_t>::max();
  EXPECT_TRUE(WriteXattrTimeT(path(), kKey, expected));
  EXPECT_EQ(ReadXattrTimeT(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, expected);
}

TEST_F(Xattr, RemoveAndRead) {
  const std::string value = "hello world";
  EXPECT_TRUE(WriteXattr(path(), kKey, value));

  std::string actual;
  EXPECT_EQ(ReadXattr(path(), kKey, &actual), XattrStatus::kOK);
  EXPECT_EQ(actual, value);

  EXPECT_EQ(RemoveXattr(path(), kKey), XattrStatus::kOK);
  EXPECT_EQ(ReadXattr(path(), kKey, &actual), XattrStatus::kNoAttribute);

  EXPECT_EQ(RemoveXattr(path(), kKey), XattrStatus::kNoAttribute);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
