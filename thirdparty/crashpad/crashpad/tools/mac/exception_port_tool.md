<!--
Copyright 2014 The Crashpad Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# exception_port_tool(1)

## Name

exception_port_tool—Show and change Mach exception ports

## Synopsis

**exception_port_tool** [_OPTION…_] [_COMMAND_ [_ARG…_]]

## Description

Shows Mach exception ports registered for a thread, task, or host target with a
__--show-*__ option, changes Mach exception ports with **--set-handler**, shows
changes with a __--show-new-*__ option, and executes _COMMAND_ along with any
arguments specified (_ARG…_) with the changed exception ports in effect.

## Options

 * **-s**, **--set-handler**=_DESCRIPTION_

   Set an exception port to _DESCRIPTION_. This option may appear zero, one, or
   more times.

   _DESCRIPTION_ is formatted as a comma-separated sequence of tokens, where
   each token consists of a key and value separated by an equals sign. These
   keys are recognized:

    * **target**=_TARGET_

      _TARGET_ defines which target’s exception ports to set: **host**,
      **task**, or **thread**. The default value of _TARGET_ is **task**.
      Operations on **host** are restricted to the superuser.

    * **mask**=_MASK_

      _MASK_ defines the mask of exception types to handle, from
      `<mach/exception_types.h>`. This can be **BAD_ACCESS**,
      **BAD_INSTRUCTION**, **ARITHMETIC**, **EMULATION**, **SOFTWARE**,
      **BREAKPOINT**, **SYSCALL**, **MACH_SYSCALL**, **RPC_ALERT**, **CRASH**,
      **RESOURCE**, **GUARD**, or **CORPSE_NOTIFY**. Different exception types
      may be combined by combining them with pipe characters (**|**). The
      special value **ALL** includes each exception type except for **CRASH**
      and **CORPSE_NOTIFY**. To truly specify all exception types including
      these, use **ALL|CRASH|CORPSE_NOTIFY**. The default value of _MASK_ is
      **CRASH**.

    * **behavior**=_BEHAVIOR_

      _BEHAVIOR_ defines the specific exception handler routine to be called
      when an exception occurs. This can be **DEFAULT**, **STATE**, or
      **STATE_IDENTITY**. **MACH** may also be specified by combining them with
      pipe characters (**|**). The most complete set of exception information is
      provided with **STATE_IDENTITY|MACH**. Not all exception servers implement
      all possible behaviors. The default value of _BEHAVIOR_ is
      **DEFAULT|MACH**.

    * **flavor**=_FLAVOR_

      For state-carrying values of _BEHAVIOR_ (those including **STATE** or
      **STATE_IDENTITY**), _FLAVOR_ specifies the architecture-specific thread
      state flavor to be provided to the exception handler. For the x86 family,
      this can be **THREAD**, **THREAD32**, **THREAD64**, **FLOAT**,
      **FLOAT32**, **FLOAT64**, **DEBUG**, **DEBUG32**, or **DEBUG64**. The
      default value of _FLAVOR_ is **NONE**, which is not valid for
      state-carrying values of _BEHAVIOR_.

    * **handler**=_HANDLER_

      _HANDLER_ defines the exception handler. **NULL** indicates that any
      existing exception port should be cleared. _HANDLER_ may also take the
      form **bootstrap**:_SERVICE_, which will look _SERVICE_ up with the
      bootstrap server and set that service as the exception handler. The
      default value of _HANDLER_ is **NULL**.

 * **--show-bootstrap**=_SERVICE_

   Looks up _SERVICE_ with the bootstrap server and shows it. Normally, the
   handler port values displayed by the other __--show-*__ options are
   meaningless handles, but by comparing them to the port values for known
   bootstrap services, it is possible to verify that they are set as intended.

 * **-p**, **--pid**=_PID_

   For operations on the task target, including **--set-handler** with _TARGET_
   set to **task**, **--show-task**, and **--show-new-task**, operates on the
   task associated with process id _PID_ instead of the current task associated
   with the tool. When this option is supplied, _COMMAND_ must not be specified.

   This option uses `task_for_pid()` to access the process’ task port. This
   operation may be restricted to use by the superuser, executables signed by an
   authority trusted by the system, and processes otherwise permitted by
   taskgated(8). Consequently, this program must normally either be signed or be
   invoked by root to use this option. It is possible to install this program as
   a setuid root executable to overcome this limitation. However, it is not
   possible to use this option to operate on processes protected by [System
   Integrity Protection (SIP)](https://support.apple.com/HT204899), including
   those whose “restrict” codesign(1) option is respected.

 * **-h**, **--show-host**

   Shows the original host exception ports before making any changes requested
   by **--set-handler**. This option is restricted to the superuser.

 * **-t**, **--show-task**

   Shows the original task exception ports before making any changes requested
   by **--set-handler**.

 * **--show-thread**

   Shows the original thread exception ports before making any changes requested
   by **--set-handler**.

 * **-H**, **--show-new-host**

   Shows the modified host exception ports after making any changes requested by
   **--set-handler**. This option is restricted to the superuser.

 * **-T**, **--show-new-task**

   Shows the modified task exception ports after making any changes requested by
   **--set-handler**.

 * **--show-new-thread**

   Shows the modified thread exception ports after making any changes requested
   by **--set-handler**

 * **-n**, **--numeric**

   For __--show-*__ options, all values will be displayed numerically only. The
   default is to decode numeric values and display them symbolically as well.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Sets a new task-level exception handler for `EXC_CRASH`-type exceptions to the
handler registered with the bootstrap server as `svc`, showing the task-level
exception ports before and after the change. The old and new exception handlers
are verified by their service names as registered with the bootstrap server.
With the new task-level exception ports in effect, a program is run.

```
$ exception_port_tool --show-task --show-new-task \
      --show-bootstrap=com.apple.ReportCrash --show-bootstrap=svc \
      --set-handler=behavior=DEFAULT,handler=bootstrap:svc crash
service com.apple.ReportCrash 0xe03
service svc 0x1003
task exception port 0, mask 0x400 (CRASH), port 0xe03, behavior 0x80000003 (STATE_IDENTITY|MACH), flavor 7 (THREAD)
new task exception port 0, mask 0x400 (CRASH), port 0x1003, behavior 0x1 (DEFAULT), flavor 13 (NONE)
Illegal instruction: 4
```

Shows the task-level exception ports for the process with PID 1234. This
requires superuser permissions or the approval of taskgated(8), and the process
must not be SIP-protected.

```
# exception_port_tool --pid=1234 --show-task
task exception port 0, mask 0x4e (BAD_ACCESS|BAD_INSTRUCTION|ARITHMETIC|BREAKPOINT), port 0x1503, behavior 0x1 (DEFAULT), flavor 13 (NONE)
task exception port 1, mask 0x1c00 (CRASH|RESOURCE|GUARD), port 0x1403, behavior 0x80000003 (STATE_IDENTITY|MACH), flavor 7 (THREAD)
```

## Exit Status

 * **0**

   Success.

 * **125**

   Failure, with a message printed to the standard error stream.

 * **126**

   The program specified by _COMMAND_ was found, but could not be invoked.

 * **127**

   The program specified by _COMMAND_ could not be found.

## See Also

[catch_exception_tool(1)](catch_exception_tool.md),
[on_demand_service_tool(1)](on_demand_service_tool.md)

## Resources

Crashpad home page: https://crashpad.chromium.org/.

Report bugs at https://crashpad.chromium.org/bug/new.

## Copyright

Copyright 2014 [The Crashpad
Authors](https://chromium.googlesource.com/crashpad/crashpad/+/master/AUTHORS).

## License

Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
