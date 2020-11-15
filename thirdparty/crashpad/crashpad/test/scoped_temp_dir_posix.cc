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

#include "test/scoped_temp_dir.h"

#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "base/logging.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"

namespace crashpad {
namespace test {

void ScopedTempDir::Rename() {
  base::FilePath move_to = CreateTemporaryDirectory();
  PCHECK(rename(path_.value().c_str(), move_to.value().c_str()) == 0);
  path_ = move_to;
}

// static
base::FilePath ScopedTempDir::CreateTemporaryDirectory() {
  char* tmpdir = getenv("TMPDIR");
  std::string dir;
  if (tmpdir && tmpdir[0] != '\0') {
    dir.assign(tmpdir);
  } else {
#if defined(OS_ANDROID)
    dir.assign("/data/local/tmp");
#else
    dir.assign("/tmp");
#endif
  }

  if (dir[dir.size() - 1] != '/') {
    dir.append(1, '/');
  }
  dir.append("org.chromium.crashpad.test.XXXXXX");

  PCHECK(mkdtemp(&dir[0])) << "mkdtemp " << dir;
  return base::FilePath(dir);
}

}  // namespace test
}  // namespace crashpad
