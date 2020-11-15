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

#include "client/settings.h"

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/scoped_temp_dir.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {
namespace {

class SettingsTest : public testing::Test {
 public:
  SettingsTest() = default;

  base::FilePath settings_path() {
    return temp_dir_.path().Append(FILE_PATH_LITERAL("settings"));
  }

  Settings* settings() { return &settings_; }

  void InitializeBadFile() {
    ScopedFileHandle handle(
        LoggingOpenFileForWrite(settings_path(),
                                FileWriteMode::kTruncateOrCreate,
                                FilePermissions::kWorldReadable));
    ASSERT_TRUE(handle.is_valid());

    static constexpr char kBuf[] = "test bad file";
    ASSERT_TRUE(LoggingWriteFile(handle.get(), kBuf, sizeof(kBuf)));
    handle.reset();
  }

 protected:
  // testing::Test:
  void SetUp() override {
    ASSERT_TRUE(settings()->Initialize(settings_path()));
  }

 private:
  ScopedTempDir temp_dir_;
  Settings settings_;

  DISALLOW_COPY_AND_ASSIGN(SettingsTest);
};

TEST_F(SettingsTest, ClientID) {
  UUID client_id;
  EXPECT_TRUE(settings()->GetClientID(&client_id));
  EXPECT_NE(client_id, UUID());

  Settings local_settings;
  EXPECT_TRUE(local_settings.Initialize(settings_path()));
  UUID actual;
  EXPECT_TRUE(local_settings.GetClientID(&actual));
  EXPECT_EQ(actual, client_id);
}

TEST_F(SettingsTest, UploadsEnabled) {
  bool enabled = true;
  // Default value is false.
  EXPECT_TRUE(settings()->GetUploadsEnabled(&enabled));
  EXPECT_FALSE(enabled);

  EXPECT_TRUE(settings()->SetUploadsEnabled(true));
  EXPECT_TRUE(settings()->GetUploadsEnabled(&enabled));
  EXPECT_TRUE(enabled);

  Settings local_settings;
  EXPECT_TRUE(local_settings.Initialize(settings_path()));
  enabled = false;
  EXPECT_TRUE(local_settings.GetUploadsEnabled(&enabled));
  EXPECT_TRUE(enabled);

  EXPECT_TRUE(settings()->SetUploadsEnabled(false));
  EXPECT_TRUE(settings()->GetUploadsEnabled(&enabled));
  EXPECT_FALSE(enabled);

  enabled = true;
  EXPECT_TRUE(local_settings.GetUploadsEnabled(&enabled));
  EXPECT_FALSE(enabled);
}

TEST_F(SettingsTest, LastUploadAttemptTime) {
  time_t actual = -1;
  EXPECT_TRUE(settings()->GetLastUploadAttemptTime(&actual));
  // Default value is 0.
  EXPECT_EQ(actual, 0);

  const time_t expected = time(nullptr);
  EXPECT_TRUE(settings()->SetLastUploadAttemptTime(expected));
  EXPECT_TRUE(settings()->GetLastUploadAttemptTime(&actual));
  EXPECT_EQ(actual, expected);

  Settings local_settings;
  EXPECT_TRUE(local_settings.Initialize(settings_path()));
  actual = -1;
  EXPECT_TRUE(local_settings.GetLastUploadAttemptTime(&actual));
  EXPECT_EQ(actual, expected);
}

// The following tests write a corrupt settings file and test the recovery
// operation.

TEST_F(SettingsTest, BadFileOnInitialize) {
  InitializeBadFile();

  Settings settings;
  EXPECT_TRUE(settings.Initialize(settings_path()));
}

TEST_F(SettingsTest, BadFileOnGet) {
  InitializeBadFile();

  UUID client_id;
  EXPECT_TRUE(settings()->GetClientID(&client_id));
  EXPECT_NE(client_id, UUID());

  Settings local_settings;
  EXPECT_TRUE(local_settings.Initialize(settings_path()));
  UUID actual;
  EXPECT_TRUE(local_settings.GetClientID(&actual));
  EXPECT_EQ(actual, client_id);
}

TEST_F(SettingsTest, BadFileOnSet) {
  InitializeBadFile();

  EXPECT_TRUE(settings()->SetUploadsEnabled(true));
  bool enabled = false;
  EXPECT_TRUE(settings()->GetUploadsEnabled(&enabled));
  EXPECT_TRUE(enabled);
}

TEST_F(SettingsTest, UnlinkFile) {
  UUID client_id;
  EXPECT_TRUE(settings()->GetClientID(&client_id));
  EXPECT_TRUE(settings()->SetUploadsEnabled(true));
  EXPECT_TRUE(settings()->SetLastUploadAttemptTime(time(nullptr)));

#if defined(OS_WIN)
  EXPECT_EQ(_wunlink(settings_path().value().c_str()), 0)
      << ErrnoMessage("_wunlink");
#else
  EXPECT_EQ(unlink(settings_path().value().c_str()), 0)
      << ErrnoMessage("unlink");
#endif

  Settings local_settings;
  EXPECT_TRUE(local_settings.Initialize(settings_path()));
  UUID new_client_id;
  EXPECT_TRUE(local_settings.GetClientID(&new_client_id));
  EXPECT_NE(new_client_id, client_id);

  // Check that all values are reset.
  bool enabled = true;
  EXPECT_TRUE(local_settings.GetUploadsEnabled(&enabled));
  EXPECT_FALSE(enabled);

  time_t time = -1;
  EXPECT_TRUE(local_settings.GetLastUploadAttemptTime(&time));
  EXPECT_EQ(time, 0);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
