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

# catch_exception_tool(1)

## Name

catch_exception_tool—Catch Mach exceptions and display information about them

## Synopsis

**catch_exception_tool** **-m** _SERVICE_ [_OPTION…_]

## Description

Runs a Mach exception server registered with the bootstrap server under the name
_SERVICE_. The exception server is capable of receiving exceptions for
“behavior” values of `EXCEPTION_DEFAULT`, `EXCEPTION_STATE`, and
`EXCEPTION_STATE_IDENTITY`, with or without `MACH_EXCEPTION_CODES` set.

## Options

 * **-f**, **--file**=_FILE_

   Information about the exception will be appended to _FILE_ instead of the
   standard output stream.

 * **-m**, **--mach-service**=_SERVICE_

   Check in with the bootstrap server under the name _SERVICE_. This service
   name may already be reserved with the bootstrap server in cases where this
   tool is started by launchd(8) as a result of a message being sent to a
   service declared in a job’s `MachServices` dictionary (see launchd.plist(5)).
   The service name may also be completely unknown to the system.

 * **-p**, **--persistent**

   Continue processing exceptions after the first one. The default mode is
   one-shot, where this tool exits after processing the first exception.

 * **-t**, **--timeout**=_TIMEOUT_

   Run for a maximum of _TIMEOUT_ seconds. Specify `0` to request non-blocking
   operation, in which the tool exits immediately if no exception is received.
   In **--persistent** mode, _TIMEOUT_ applies to the overall duration that this
   tool will run, not to the processing of individual exceptions. When
   **--timeout** is not specified, this tool will block indefinitely while
   waiting for an exception.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Run a one-shot blocking exception server registered with the bootstrap server
under the name `svc`:

```
$ catch_exception_tool --mach-service=svc --file=out &
[1] 1233
$ exception_port_tool --set-handler=handler=bootstrap:svc crasher
Illegal instruction: 4
[1]+  Done    catch_exception_tool --mach-service=svc --file=out
$ cat out
catch_exception_tool: behavior EXCEPTION_DEFAULT|MACH_EXCEPTION_CODES, pid 1234, thread 56789, exception EXC_CRASH, codes[2] 0x4200001, 0, original exception EXC_BAD_INSTRUCTION, original code[0] 1, signal SIGILL
```

Run an on-demand exception server started by launchd(5) available via the
bootstrap server under the name `svc`:

```
$ `on_demand_service_tool --load --label=catch_exception \
      --mach-service=svc \
      $(which catch_exception_tool) --mach-service=svc \
      --file=/tmp/out --persistent --timeout=0
$ exception_port_tool --set-handler=handler=bootstrap:svc crasher
Illegal instruction: 4
$ on_demand_service_tool --unload --label=catch_exception
$ cat /tmp/out
catch_exception_tool: behavior EXCEPTION_DEFAULT|MACH_EXCEPTION_CODES, pid 2468, thread 13579, exception EXC_CRASH, codes[2] 0x4200001, 0, original exception EXC_BAD_INSTRUCTION, original code[0] 1, signal SIGILL
```

## Exit Status

 * **0**

   Success. In **--persistent** mode with a **--timeout** set, it is considered
   successful if at least one exception was caught when the timer expires.

 * **1**

   Failure, with a message printed to the standard error stream.

## See Also

[exception_port_tool(1)](exception_port_tool.md),
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
