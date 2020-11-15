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

# crashpad_handler(8)

## Name

crashpad_handler—Crashpad’s exception handler server

## Synopsis

**crashpad_handler** [_OPTION…_]

## Description

This program is Crashpad’s main exception-handling server. It is responsible for
catching exceptions, writing crash reports, and uploading them to a crash report
collection server. Uploads are disabled by default, and can only be enabled by a
Crashpad client using the Crashpad client library, typically in response to a
user requesting this behavior.

On macOS, this server may be started by its initial client, in which case it
performs a handshake with this client via a pipe established by the client that
is inherited by the server, referenced by the **--handshake-fd** argument.
During the handshake, the server furnishes the client with a send right that the
client may use as an exception port. The server retains the corresponding
receive right, which it monitors for exception messages. When the receive right
loses all senders, the server exits after allowing any upload in progress to
complete.

Alternatively, on macOS, this server may be started from launchd(8), where it
receives the Mach service name in a **--mach-service** argument. It checks in
with the bootstrap server under this service name, and clients may look it up
with the bootstrap server under this service name. It monitors this service for
exception messages. Upon receipt of `SIGTERM`, the server exits after allowing
any upload in progress to complete. `SIGTERM` is normally sent by launchd(8)
when it determines that the server should exit.

On Windows, clients register with this server by communicating with it via the
named pipe identified by the **--pipe-name** argument. Alternatively, the server
can inherit an already-created pipe from a parent process by using the
**--initial-client-data** mechanism. That argument also takes all of the
arguments that would normally be passed in a registration message, and so
constitutes registration of the first client. Subsequent clients may then
register by communicating with the server via the pipe. During registration, a
client provides the server with an OS event object that it will signal should it
crash. The server obtains the client’s process handle and waits on the crash
event object for a crash, as well as the client’s process handle for the client
to exit cleanly without crashing. When a server started via the
**--initial-client-data** mechanism loses all of its clients, it exits after
allowing any upload in progress to complete.

On Windows, this executable is built by default as a Windows GUI app, so no
console will appear in normal usage. This is the version that will typically be
used. A second copy is also made with a `.com` extension, rather than `.exe`. In
this second copy, the PE header is modified to indicate that it’s a console app.
This is useful because the `.com` is found in the path before the `.exe`, so
when run normally from a shell using only the basename (without an explicit
`.com` or `.exe` extension), the `.com` console version will be chosen, and so
stdio will be hooked up as expected to the parent console so that logging output
will be visible.

On Linux/Android, the handler may create a crash dump for its parent process
using **--trace-parent-with-exception**. In this mode, the handler process
creates a crash dump for its parent and exits. Alternatively, the handler may
be launched with **--initial-client-fd** which will start the server connected
to an initial client. The server will exit when all connected client sockets are
closed.

It is not normally appropriate to invoke this program directly. Usually, it will
be invoked by a Crashpad client using the Crashpad client library, or started by
another system service. On macOS, arbitrary programs may be run with a Crashpad
handler by using [run_with_crashpad(1)](../tools/mac/run_with_crashpad.md) to
establish the Crashpad client environment before running a program.

## Options

 * **--annotation**=_KEY_=_VALUE_

   Sets a process-level annotation mapping _KEY_ to _VALUE_ in each crash report
   that is written. This option may appear zero, one, or multiple times.

   Most annotations should be provided by the Crashpad client as module-level
   annotations instead of process-level annotations. Module-level annotations
   are more flexible in that they can be modified and cleared during the client
   program’s lifetime. Module-level annotations can be set via the Crashpad
   client library. Process-level annotations are useful for annotations that the
   collection server requires be present, that have fixed values, and for cases
   where a program that does not use the Crashpad client library is being
   monitored.

   Breakpad-type collection servers only require the `"prod"` and `"ver"`
   annotations, which should be set to the product name or identifier and
   product version, respectively. It is unusual to specify other annotations as
   process-level annotations via this argument.

 * **--database**=_PATH_

   Use _PATH_ as the path to the Crashpad crash report database. This option is
   required. Crash reports are written to this database, and if uploads are
   enabled, uploaded from this database to a crash report collection server. If
   the database does not exist, it will be created, provided that the parent
   directory of _PATH_ exists.

 * **--handshake-fd**=_FD_

   Perform the handshake with the initial client on the file descriptor at _FD_.
   Either this option or **--mach-service**, but not both, is required. This
   option is only valid on macOS.

 * **--no-identify-client-via-url**

   Do not add client-identifying fields to the URL. By default, `"prod"`,
   `"ver"`, and `"guid"` annotations are added to the upload URL as name-value
   pairs `"product"`, `"version"`, and `"guid"`, respectively. Using this
   option disables that behavior.

 * **--initial-client-data**=*HANDLE_request_crash_dump*,*HANDLE_request_non_crash_dump*,*HANDLE_non_crash_dump_completed*,*HANDLE_first_pipe_instance*,*HANDLE_client_process*,*Address_crash_exception_information*,*Address_non_crash_exception_information*,*Address_debug_critical_section*

   Register the initial client using the inherited handles and data provided.
   For more information on the argument’s format, see the implementations of
   `CrashpadClient` and `ExceptionHandlerServer`. Either this option or
   **--pipe-name**, but not both, is required. This option is only valid on
   Windows.

   When this option is present, the server creates a new named pipe at a random
   name and informs its client of the name. The server waits for at least one
   client to register, and exits when all clients have exited, after waiting for
   any uploads in progress to complete.

 * **--mach-service**=_SERVICE_

   Check in with the bootstrap server under the name _SERVICE_. Either this
   option or **--handshake-fd**, but not both, is required. This option is only
   valid on macOS.

   _SERVICE_ may already be reserved with the bootstrap server in cases where
   this tool is started by launchd(8) as a result of a message being sent to a
   service declared in a job’s `MachServices` dictionary (see launchd.plist(5)).
   The service name may also be completely unknown to the system.

 * **--metrics-dir**=_DIR_

   Metrics information will be written to _DIR_. This option only has an effect
   when built as part of Chromium. In non-Chromium builds, and in the absence of
   this option, metrics information will not be written.

 * **--monitor-self**

   Causes a second instance of the Crashpad handler program to be started,
   monitoring the original instance for exceptions. The original instance will
   become a client of the second one. The second instance will be started with
   the same **--annotation**, **--database**, **--monitor-self-annotation**,
   **--no-rate-limit**, **--no-upload-gzip**, and **--url** arguments as the
   original one. The second instance will always be started with a
   **--no-periodic-tasks** argument, and will not be started with a
   **--metrics-dir** argument even if the original instance was.

   Where supported by the underlying operating system, the second instance will
   be restarted should it exit before the first instance. The second instance
   will not be eligible to be started asynchronously.

 * **--monitor-self-annotation**=_KEY_=_VALUE_

   Sets a module-level annotation mapping _KEY_ to _VALUE_ in the Crashpad
   handler. This option may appear zero, one, or more times.

   If **--monitor-self** is in use, the second instance of the Crashpad handler
   program will find these annotations stored in the original instance and will
   include them in any crash reports written for the original instance.

   These annotations will only appear in crash reports written for the Crashpad
   handler itself. To apply a process-level annotation to all crash reports
   written by an instance of the Crashpad handler, use **--annotation** instead.

 * **--monitor-self-argument**=_ARGUMENT_

   When directed by **--monitor-self** to start a second instance of the
   Crashpad handler program, the second instance will be started with _ARGUMENT_
   as one of its arguments. This option may appear zero, one, or more times.
   This option has no effect in the absence of **--monitor-self**.

   This supports embedding the Crashpad handler into a multi-purpose executable
   that dispatches to the desired entry point based on a command-line argument.
   To prevent excessive accumulation of handler processes, _ARGUMENT_ must not
   be `--monitor-self`.

 * **--no-periodic-tasks**

   Do not scan for new pending crash reports or prune the crash report database.
   Only crash reports recorded by this instance of the Crashpad handler will
   become eligible for upload in this instance, and only a single initial upload
   attempt will be made.

   This option is not intended for general use. It is provided to prevent
   multiple instances of the Crashpad handler from duplicating the effort of
   performing the same periodic tasks. In normal use, the first instance of the
   Crashpad handler will assume the responsibility for performing these tasks,
   and will provide this argument to any second instance. See
   **--monitor-self**.

 * **--no-rate-limit**

   Do not rate limit the upload of crash reports. By default uploads are
   throttled to one per hour. Using this option disables that behavior, and
   Crashpad will attempt to upload all captured reports.

 * **--no-upload-gzip**

   Do not use `gzip` compression for uploaded crash reports. Normally, the
   entire request body is compressed into a `gzip` stream and transmitted with
   `Content-Encoding: gzip`. This option disables compression, and is intended
   for use with collection servers that don’t accept uploads compressed in this
   way.

 * **--pipe-name**=_PIPE_

   Listen on the given pipe name for connections from clients. _PIPE_ must be of
   the form `\\.\pipe\<somename>`. Either this option or
   **--initial-client-data**, but not both, is required. This option is only
   valid on Windows.

   When this option is present, the server creates a named pipe at _PIPE_, a
   name known to both the server and its clients. The server continues running
   even after all clients have exited.

 * **--reset-own-crash-exception-port-to-system-default**

   Causes the exception handler server to set its own crash handler to the
   system default before beginning operation. This is only expected to be useful
   in cases where the server inherits an inappropriate crash handler from its
   parent process. This option is only valid on macOS. Use of this option is
   discouraged. It should not be used absent extraordinary circumstances.

 * **--trace-parent-with-exception**=_EXCEPTION-INFORMATION-ADDRESS_

   Causes the handler process to trace its parent process and exit. The parent
   process should have an ExceptionInformation struct at
   _EXCEPTION-INFORMATION-ADDRESS_.

 * **--initial-client-fd**=_FD_

   Starts the excetion handler server with an initial ExceptionHandlerClient
   connected on the socket _FD_. The server will exit when all connected client
   sockets have been closed.

 * **--url**=_URL_

   If uploads are enabled, sends crash reports to the Breakpad-type crash report
   collection server at _URL_. Uploads are disabled by default, and can only be
   enabled for a database by a Crashpad client using the Crashpad client
   library, typically in response to a user requesting this behavior. If this
   option is not specified, this program will behave as if uploads are disabled.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Exit Status

 * **0**

   Success.

 * **1**

   Failure, with a message printed to the standard error stream.

## See Also

[catch_exception_tool(1)](../tools/mac/catch_exception_tool.md),
[crashpad_database_util(1)](../tools/crashpad_database_util.md),
[generate_dump(1)](../tools/generate_dump.md),
[run_with_crashpad(1)](../tools/mac/run_with_crashpad.md)

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
