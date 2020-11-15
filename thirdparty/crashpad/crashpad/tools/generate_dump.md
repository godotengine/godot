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

# generate_dump(1)

## Name

generate_dump—Generate a minidump file containing a snapshot of a running
process

## Synopsis

**generate_dump** [_OPTION…_] _PID_

## Description

Generates a minidump file containing a snapshot of a running process whose
process identifier is _PID_. By default, the target process will be suspended
while the minidump is generated, and the minidump file will be written to
`minidump.PID`. After the minidump file is generated, the target process resumes
running.

The minidump file will contain information about the process, its threads, its
modules, and the system. It will not contain any exception information because
it will be generated from a live running process, not as a result of an
exception occurring.

On macOS, this program uses `task_for_pid()` to access the process’ task port.
This operation may be restricted to use by the superuser, executables signed by
an authority trusted by the system, and processes otherwise permitted by
taskgated(8). Consequently, this program must normally either be signed or be
invoked by root. It is possible to install this program as a setuid root
executable to overcome this limitation, although it will remain impossible to
generate dumps for processes protected by [System Integrity Protection
(SIP)](https://support.apple.com/HT204899), including those whose “restrict”
codesign(1) option is respected.

This program is similar to the gcore(1) program available on some operating
systems.

## Options

 * **-r**, **--no-suspend**

   The target process will continue running while the minidump file is
   generated. Normally, the target process is suspended during this operation,
   which guarantees that the minidump file will contain an atomic snapshot of
   the process.

   This option may be useful when attempting to generate a minidump from a
   process that dump generation has an interprocess dependency on, such as a
   system server like launchd(8) or opendirectoryd(8) on macOS. Deadlock could
   occur if any portion of the dump generation operation blocks while waiting
   for a response from one of these servers while they are suspended.

 * **-o**, **--output**=_FILE_

   The minidump will be written to _FILE_ instead of `minidump.PID`.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Generate a minidump file in `/tmp/minidump` containing a snapshot of the process
with PID 1234.

```
$ generate_dump --output=/tmp/minidump 1234
```

## Exit Status

 * **0**

   Success.

 * **1**

   Failure, with a message printed to the standard error stream.

## See Also

[catch_exception_tool(1)](mac/catch_exception_tool.md)

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
