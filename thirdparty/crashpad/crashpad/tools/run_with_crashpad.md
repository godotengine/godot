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

# run_with_crashpad(1)

## Name

run_with_crashpad—Run a program with a Crashpad exception handler

## Synopsis

**run_with_crashpad** [_OPTION…_] _COMMAND_ [_ARG…_]

## Description

Starts a Crashpad exception handler server such as
[crashpad_handler(8)](../../handler/crashpad_handler.md) and becomes its client,
setting an exception port referencing the handler. Then, executes _COMMAND_
along with any arguments specified (_ARG…_) with the new exception port in
effect.

On macOS, the exception port is configured to receive exceptions of type
`EXC_CRASH`, `EXC_RESOURCE`, and `EXC_GUARD`. The exception behavior is
configured as `EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES`. The thread
state flavor is set to `MACHINE_THREAD_STATE`.

Programs that use the Crashpad client library directly will not normally use
this tool. This tool exists to allow programs that are unaware of Crashpad to be
run with a Crashpad exception handler.

## Options

 * **-h**, **--handler**=_HANDLER_

   Invoke _HANDLER_ as the Crashpad handler program instead of the default,
   **crashpad_handler**.

 * **--annotation**=_KEY=VALUE_

   Passed to the Crashpad handler program as an **--annotation** argument.

 * **--database**=_PATH_

   Passed to the Crashpad handler program as its **--database** argument.

 * **--url**=_URL_

   Passed to the Crashpad handler program as its **--url** argument.

 * **-a**, **--argument**=_ARGUMENT_

   Invokes the Crashpad handler program with _ARGUMENT_ as one of its arguments.
   This option may appear zero, one, or more times. If this program has a
   specific option such as **--database** matching the desired Crashpad handler
   program option, the specific option should be used in preference to
   **--argument**. Regardless of this option’s presence, the handler will always
   be invoked with the necessary arguments to perform a handshake.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Starts a Crashpad exception handler server by its default name,
**crashpad_handler**, and runs a program with this handler in effect.

```
$ run_with_crashpad --database=/tmp/crashpad_database crasher
Illegal instruction: 4
```

Starts a Crashpad exception handler server at a nonstandard path, and runs
[exception_port_tool(1)](exception_port_tool.md) to show the task-level
exception ports.

```
$ run_with_crashpad --handler=/tmp/crashpad_handler \
      --database=/tmp/crashpad_database exception_port_tool \
      --show-task
task exception port 0, mask 0x1c00 (CRASH|RESOURCE|GUARD), port 0x30b, behavior 0x80000003 (STATE_IDENTITY|MACH), flavor 7 (THREAD)
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

[crashpad_handler(8)](../../handler/crashpad_handler.md),
[exception_port_tool(1)](exception_port_tool.md)

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
