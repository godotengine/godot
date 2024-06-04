// Copyright (c) 2014, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

namespace google_breakpad {

void LaunchReporter(const char *reporter_executable_path,
                    const char *config_file_path) {
  const char* argv[] = { reporter_executable_path, config_file_path, NULL };

  // Launch the reporter
  pid_t pid = fork();

  if (pid == -1) {
    perror("fork");
    fprintf(stderr, "Failed to fork reporter process\n");
    return;
  }

  // If we're in the child, load in our new executable and run.
  // The parent will not wait for the child to complete.
  if (pid == 0) {
    execv(argv[0], (char* const*)argv);
    perror("exec");
    fprintf(stderr,
            "Failed to launch reporter process from path %s\n",
            reporter_executable_path);
    unlink(config_file_path);  // launch failed - get rid of config file
    _exit(1);
  }

  // Wait until the Reporter child process exits.
  //

  // We'll use a timeout of one minute.
  int timeout_count = 60;   // 60 seconds

  while (timeout_count-- > 0) {
    int status;
    pid_t result = waitpid(pid, &status, WNOHANG);

    if (result == 0) {
      // The child has not yet finished.
      sleep(1);
    } else if (result == -1) {
      // error occurred.
      break;
    } else {
      // child has finished
      break;
    }
  }
}

}  // namespace google_breakpad
