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

#include "client/crashpad_client.h"

#include <lib/fdio/spawn.h>
#include <lib/zx/job.h>
#include <lib/zx/port.h>
#include <lib/zx/process.h>
#include <zircon/processargs.h>

#include "base/fuchsia/fuchsia_logging.h"
#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "client/client_argv_handling.h"
#include "util/fuchsia/system_exception_port_key.h"

namespace crashpad {

CrashpadClient::CrashpadClient() {}

CrashpadClient::~CrashpadClient() {}

bool CrashpadClient::StartHandler(
    const base::FilePath& handler,
    const base::FilePath& database,
    const base::FilePath& metrics_dir,
    const std::string& url,
    const std::map<std::string, std::string>& annotations,
    const std::vector<std::string>& arguments,
    bool restartable,
    bool asynchronous_start) {
  DCHECK_EQ(restartable, false);  // Not used on Fuchsia.
  DCHECK_EQ(asynchronous_start, false);  // Not used on Fuchsia.

  zx::port exception_port;
  zx_status_t status = zx::port::create(0, &exception_port);
  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "zx_port_create";
    return false;
  }

  status = zx::job::default_job()->bind_exception_port(
      exception_port, kSystemExceptionPortKey, 0);
  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "zx_task_bind_exception_port";
    return false;
  }

  std::vector<std::string> argv_strings = BuildHandlerArgvStrings(
      handler, database, metrics_dir, url, annotations, arguments);

  std::vector<const char*> argv;
  StringVectorToCStringVector(argv_strings, &argv);

  // Follow the same protocol as devmgr and crashlogger in Zircon (that is,
  // process handle as handle 0, with type USER0, exception port handle as
  // handle 1, also with type PA_USER0) so that it's trivial to replace
  // crashlogger with crashpad_handler. The exception port is passed on, so
  // released here. Currently it is assumed that this process's default job
  // handle is the exception port that should be monitored. In the future, it
  // might be useful for this to be configurable by the client.
  constexpr size_t kActionCount = 2;
  fdio_spawn_action_t actions[] = {
      {.action = FDIO_SPAWN_ACTION_ADD_HANDLE,
       .h = {.id = PA_HND(PA_USER0, 0), .handle = ZX_HANDLE_INVALID}},
      {.action = FDIO_SPAWN_ACTION_ADD_HANDLE,
       .h = {.id = PA_HND(PA_USER0, 1), .handle = ZX_HANDLE_INVALID}},
  };

  status = zx_handle_duplicate(
      zx_job_default(), ZX_RIGHT_SAME_RIGHTS, &actions[0].h.handle);
  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "zx_handle_duplicate";
    return false;
  }
  actions[1].h.handle = exception_port.release();

  char error_message[FDIO_SPAWN_ERR_MSG_MAX_LENGTH];
  zx::process child;
  // TODO(scottmg): https://crashpad.chromium.org/bug/196, FDIO_SPAWN_CLONE_ALL
  // is useful during bringup, but should probably be made minimal for real
  // usage.
  status = fdio_spawn_etc(ZX_HANDLE_INVALID,
                          FDIO_SPAWN_CLONE_ALL,
                          argv[0],
                          argv.data(),
                          nullptr,
                          kActionCount,
                          actions,
                          child.reset_and_get_address(),
                          error_message);
  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "fdio_spawn_etc: " << error_message;
    return false;
  }

  return true;
}

}  // namespace crashpad
