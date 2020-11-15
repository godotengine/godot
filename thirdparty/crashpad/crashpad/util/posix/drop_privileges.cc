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

#include <unistd.h>

#include "base/logging.h"
#include "build/build_config.h"

namespace crashpad {

void DropPrivileges() {
  gid_t gid = getgid();
  uid_t uid = getuid();

#if defined(OS_MACOSX)
  // Based on the POSIX.1-2008 2013 edition documentation for setreuid() and
  // setregid(), setreuid() and setregid() alone should be sufficient to drop
  // privileges. The standard specifies that the saved ID should be set to the
  // effective ID whenever the real ID is not -1, or whenever the effective ID
  // is set not equal to the real ID. This code never specifies -1, so the
  // setreuid() and setregid() alone should work according to the standard.
  //
  // In practice, on older versions of macOS, setuid() and setgid() (or
  // seteuid() and setegid()) must be called first. Otherwise, setreuid() and
  // setregid() do not alter the saved IDs, leaving open the possibility for
  // future privilege escalation.
  //
  // The problem exists in 10.9.5 xnu-2422.115.4/bsd/kern/kern_prot.c
  // setreuid(). Based on its comments, it purports to set the svuid to the new
  // euid when the old svuid doesn’t match one of the new ruid and euid. This
  // isn’t how POSIX.1-2008 says it should behave, but it should work for this
  // function’s purposes. In reality, setreuid() doesn’t even do this: it sets
  // the svuid to the old euid, which does not drop privileges when the old euid
  // is different from the desired euid. The workaround of calling setuid() or
  // seteuid() before setreuid() works because it sets the euid so that by the
  // time setreuid() runs, the old euid is actually the value that ought to be
  // set as the svuid. setregid() is similar. This bug was reported as radar
  // 18987552, fixed in 10.10.3 and security updates to 10.9.5 and 10.8.5.
  //
  // setuid() and setgid() alone will only set the saved IDs when running as
  // root. When running a setuid non-root or setgid program, they do not alter
  // the saved ID, and do not effect a permanent privilege drop.
  gid_t egid = getegid();
  PCHECK(setgid(gid) == 0) << "setgid";
  PCHECK(setregid(gid, gid) == 0) << "setregid";

  uid_t euid = geteuid();
  PCHECK(setuid(uid) == 0) << "setuid";
  PCHECK(setreuid(uid, uid) == 0) << "setreuid";

  if (uid != 0) {
    // Because the setXid()+setreXid() interface to change IDs is fragile,
    // ensure that privileges cannot be regained. This can only be done if the
    // real user ID (and now the effective user ID as well) is not root, because
    // root always has permission to change identity.
    if (euid != uid) {
      CHECK_EQ(seteuid(euid), -1);
    }
    if (egid != gid) {
      CHECK_EQ(setegid(egid), -1);
    }
  }
#elif defined(OS_LINUX) || defined(OS_ANDROID)
  PCHECK(setresgid(gid, gid, gid) == 0) << "setresgid";
  PCHECK(setresuid(uid, uid, uid) == 0) << "setresuid";

  // Don’t check to see if privileges can be regained on Linux, because on
  // Linux, it’s not as simple as ensuring that this can’t be done if non-root.
  // Instead, the ability to change user and group IDs are controlled by the
  // CAP_SETUID and CAP_SETGID capabilities, which may be granted to non-root
  // processes. Since the setresXid() interface is well-defined, it shouldn’t be
  // necessary to perform any additional checking anyway.
  //
  // TODO(mark): Drop CAP_SETUID and CAP_SETGID if present and non-root?
#else
#error Port this function to your system.
#endif
}

}  // namespace crashpad
