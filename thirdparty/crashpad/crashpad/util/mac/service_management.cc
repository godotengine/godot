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

#include "util/mac/service_management.h"

#include <errno.h>
#include <launch.h>

#include "base/mac/scoped_launch_data.h"
#include "util/mac/launchd.h"
#include "util/misc/clock.h"

namespace crashpad {

namespace {

launch_data_t LaunchDataDictionaryForJob(const std::string& label) {
  base::mac::ScopedLaunchData request(LaunchDataAlloc(LAUNCH_DATA_DICTIONARY));
  LaunchDataDictInsert(
      request.get(), LaunchDataNewString(label.c_str()), LAUNCH_KEY_GETJOB);

  base::mac::ScopedLaunchData response(LaunchMsg(request.get()));
  if (LaunchDataGetType(response.get()) != LAUNCH_DATA_DICTIONARY) {
    return nullptr;
  }

  return response.release();
}

}  // namespace

bool ServiceManagementSubmitJob(CFDictionaryRef job_cf) {
  base::mac::ScopedLaunchData job_launch(CFPropertyToLaunchData(job_cf));
  if (!job_launch.get()) {
    return false;
  }

  base::mac::ScopedLaunchData jobs(LaunchDataAlloc(LAUNCH_DATA_ARRAY));
  LaunchDataArraySetIndex(jobs.get(), job_launch.release(), 0);

  base::mac::ScopedLaunchData request(LaunchDataAlloc(LAUNCH_DATA_DICTIONARY));
  LaunchDataDictInsert(request.get(), jobs.release(), LAUNCH_KEY_SUBMITJOB);

  base::mac::ScopedLaunchData response(LaunchMsg(request.get()));
  if (LaunchDataGetType(response.get()) != LAUNCH_DATA_ARRAY) {
    return false;
  }

  if (LaunchDataArrayGetCount(response.get()) != 1) {
    return false;
  }

  launch_data_t response_element = LaunchDataArrayGetIndex(response.get(), 0);
  if (LaunchDataGetType(response_element) != LAUNCH_DATA_ERRNO) {
    return false;
  }

  int err = LaunchDataGetErrno(response_element);
  if (err != 0) {
    return false;
  }

  return true;
}

bool ServiceManagementRemoveJob(const std::string& label, bool wait) {
  base::mac::ScopedLaunchData request(LaunchDataAlloc(LAUNCH_DATA_DICTIONARY));
  LaunchDataDictInsert(
      request.get(), LaunchDataNewString(label.c_str()), LAUNCH_KEY_REMOVEJOB);

  base::mac::ScopedLaunchData response(LaunchMsg(request.get()));
  if (LaunchDataGetType(response.get()) != LAUNCH_DATA_ERRNO) {
    return false;
  }

  int err = LaunchDataGetErrno(response.get());
  if (err == EINPROGRESS) {
    if (wait) {
      // TODO(mark): Use a kqueue to wait for the process to exit. To avoid a
      // race, the kqueue would need to be set up prior to asking launchd to
      // remove the job. Even so, the job’s PID may change between the time it’s
      // obtained and the time the kqueue is set up, so this is nontrivial.
      do {
        SleepNanoseconds(1E5);  // 100 microseconds
      } while (ServiceManagementIsJobLoaded(label));
    }

    return true;
  }

  if (err != 0) {
    return false;
  }

  return true;
}

bool ServiceManagementIsJobLoaded(const std::string& label) {
  base::mac::ScopedLaunchData dictionary(LaunchDataDictionaryForJob(label));
  if (!dictionary.is_valid()) {
    return false;
  }

  return true;
}

pid_t ServiceManagementIsJobRunning(const std::string& label) {
  base::mac::ScopedLaunchData dictionary(LaunchDataDictionaryForJob(label));
  if (!dictionary.is_valid()) {
    return 0;
  }

  launch_data_t pid = LaunchDataDictLookup(dictionary.get(), LAUNCH_JOBKEY_PID);
  if (!pid) {
    return 0;
  }

  if (LaunchDataGetType(pid) != LAUNCH_DATA_INTEGER) {
    return 0;
  }

  return LaunchDataGetInteger(pid);
}

}  // namespace crashpad
