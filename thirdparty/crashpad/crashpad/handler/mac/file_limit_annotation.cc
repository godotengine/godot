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

#include "handler/mac/file_limit_annotation.h"

#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#include <string>

#include "base/format_macros.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "client/crashpad_info.h"
#include "client/simple_string_dictionary.h"
#include "util/posix/scoped_dir.h"

namespace crashpad {

namespace {

// rv is the return value from sysctl() or sysctlbyname(), and value and size
// are the pointers passed as oldp and oldlenp. If sysctl() failed, the returned
// string will be "E" followed by the error number. If there was a size
// mismatch, the returned string will be "Z" followed by the size indicated by
// sysctl(). Normally, a string representation of *value will be returned.
std::string FormatFromSysctl(int rv, const int* value, const size_t* size) {
  if (rv != 0) {
    return base::StringPrintf("E%d", errno);
  }
  if (*size != sizeof(*value)) {
    return base::StringPrintf("Z%zu", *size);
  }
  return base::StringPrintf("%d", *value);
}

// Counts the number of open file descriptors in the process and returns it as a
// string. This /dev/fd and the value returned will include the open file
// descriptor for that directory. If opendir() fails, the returned string will
// be "E" followed by the error number. If readdir() fails, it will be "R"
// followed by the error number.
std::string CountOpenFileDescriptors() {
  DIR* dir = opendir("/dev/fd");
  if (!dir) {
    return base::StringPrintf("E%d", errno);
  }

  ScopedDIR dir_owner(dir);

  dirent* entry;
  int count = 0;
  while ((errno = 0, entry = readdir(dir)) != nullptr) {
    const char* entry_name = entry->d_name;
    if (strcmp(entry_name, ".") == 0 || strcmp(entry_name, "..") == 0) {
      continue;
    }

    ++count;
  }

  if (errno != 0) {
    return base::StringPrintf("R%d", errno);
  }

  return base::StringPrintf("%d", count);
}

// Returns a string for |limit|, or "inf" if |limit| is RLIM_INFINITY.
std::string StringForRLim(rlim_t limit) {
  if (limit == RLIM_INFINITY) {
    return std::string("inf");
  }

  return base::StringPrintf("%" PRIu64, limit);
}

}  // namespace

void RecordFileLimitAnnotation() {
  CrashpadInfo* crashpad_info = CrashpadInfo::GetCrashpadInfo();
  SimpleStringDictionary* simple_annotations =
      crashpad_info->simple_annotations();
  if (!simple_annotations) {
    simple_annotations = new SimpleStringDictionary();
    crashpad_info->set_simple_annotations(simple_annotations);
  }

  int value;
  size_t size = sizeof(value);
  std::string num_files = FormatFromSysctl(
      sysctlbyname("kern.num_files", &value, &size, nullptr, 0), &value, &size);

  int mib[] = {CTL_KERN, KERN_MAXFILES};
  size = sizeof(value);
  std::string max_files = FormatFromSysctl(
      sysctl(mib, arraysize(mib), &value, &size, nullptr, 0), &value, &size);

  std::string open_files = CountOpenFileDescriptors();

  rlimit limit;
  std::string nofile;
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    nofile = base::StringPrintf("E%d,E%d", errno, errno);
  } else {
    nofile =
        StringForRLim(limit.rlim_cur) + "," + StringForRLim(limit.rlim_max);
  }

  std::string annotation = base::StringPrintf("%s,%s,%s,%s",
                                              num_files.c_str(),
                                              max_files.c_str(),
                                              open_files.c_str(),
                                              nofile.c_str());
  simple_annotations->SetKeyValue("file-limits", annotation.c_str());
}

}  // namespace crashpad
