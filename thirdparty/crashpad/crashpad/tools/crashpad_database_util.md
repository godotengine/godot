<!--
Copyright 2015 The Crashpad Authors. All rights reserved.

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

# crashpad_database_util(1)

## Name

crashpad_database_util—Operate on Crashpad crash report databases

## Synopsis

**crashpad_database_util** [_OPTION…_]

## Description

Operates on Crashpad crash report databases. The database’s settings can be
queried and modified, and information about crash reports stored in the database
can be displayed.

When this program is requested to both show and set information in a single
invocation, all “show” operations will be completed prior to beginning any “set”
operation.

Programs that use the Crashpad client library directly will not normally use
this tool, but may use the database through the programmatic interfaces in the
client library. This tool exists to allow developers to manipulate a Crashpad
database.

## Options

 * **--create**

   Creates the database identified by **--database** if it does not exist,
   provided that the parent directory of _PATH_ exists.

 * **-d**, **--database**=_PATH_

   Use _PATH_ as the path to the Crashpad crash report database. This option is
   required. The database must already exist unless **--create** is also
   specified.

 * **--show-client-id**

   Show the client ID stored in the database’s settings. The client ID is
   formatted as a UUID. The client ID is set when the database is created.
   [crashpad_handler(8)](../handler/crashpad_handler.md) retrieves the client ID
   and stores it in crash reports as they are written.

 * **--show-uploads-enabled**

   Show the status of the uploads-enabled bit stored in the database’s settings.
   [crashpad_handler(8)](../handler/crashpad_handler.md) does not upload reports
   when this bit is false. This bit is false when a database is created, and is
   under an application’s control via the Crashpad client library interface.

   See also **--set-uploads-enabled**.

 * **--show-last-upload-attempt-time**

   Show the last-upload-attempt time stored in the database’s settings. This
   value is `0`, meaning “never,” when the database is created.
   [crashpad_handler(8)](../handler/crashpad_handler.md) consults this value
   before attempting an upload to implement its rate-limiting behavior. The
   database updates this value whenever an upload is attempted.

   See also **--set-last-upload-attempt-time**.

 * **--show-pending-reports**

   Show reports eligible for upload.

 * **--show-completed-reports**

   Show reports not eligible for upload. A report is moved from the “pending”
   state to the “completed” state by
   [crashpad_handler(8)](../handler/crashpad_handler.md). This may happen when a
   report is successfully uploaded, when a report is not uploaded because
   uploads are disabled, or when a report upload attempt fails and will not be
   retried.

 * **--show-all-report-info**

   With **--show-pending-reports** or **--show-completed-reports**, show all
   metadata for each report displayed. Without this option, only report IDs will
   be shown.

 * **--show-report**=_UUID_

   Show a report from the database looked up by its identifier, _UUID_, which
   must be formatted in string representation per RFC 4122 §3. All metadata for
   each report found via a **--show-report** option will be shown. If _UUID_ is
   not found, the string `"not found"` will be printed. If this program is only
   requested to show a single report and it is not found, it will treat this as
   a failure for the purposes of determining its exit status. This option may
   appear multiple times.

 * **--set-report-uploads-enabled**=_BOOL_

   Enable or disable report upload in the database’s settings. _BOOL_ is a
   string representation of a boolean value, such as `"0"` or `"true"`.

   See also **--show-uploads-enabled**.

 * **--set-last-upload-attempt-time**=_TIME_

   Set the last-upload-attempt time in the database’s settings. _TIME_ is a
   string representation of a time, which may be in _yyyy-mm-dd hh:mm:ss_
   format, a numeric `time_t` value, or the special strings `"never"` or
   `"now"`.

   See also **--show-last-upload-attempt-time**.

 * **--new-report**=_PATH_

   Submit a new report located at _PATH_ to the database. If _PATH_ is `"-"`,
   the new report will be read from standard input. The new report will be in
   the “pending” state. The UUID assigned to the new report will be printed.
   This option may appear multiple times.

 * **--utc**

   When showing times, do so in UTC as opposed to the local time zone. When
   setting times, interpret ambiguous time strings in UTC as opposed to the
   local time zone.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Shows all crash reports in a crash report database that are in the “completed”
state.

```
$ crashpad_database_util --database /tmp/crashpad_database \
      --show-completed-reports
23f9512b-63e1-4ead-9dcd-e2e21fbccc68
4bfca440-039f-4bc6-bbd4-6933cef5efd4
56caeff8-b61a-43b2-832d-9e796e6e4a50
```

Disables report upload in a crash report database’s settings, and then verifies
that the change was made.

```
$ crashpad_database_util --database /tmp/crashpad_database \
      --set-uploads-enabled false
$ crashpad_database_util --database /tmp/crashpad_database \
      --show-uploads-enabled
false
```

## Exit Status

 * **0**

   Success.

 * **1**

   Failure, with a message printed to the standard error stream.

## See Also

[crashpad_handler(8)](../handler/crashpad_handler.md)

## Resources

Crashpad home page: https://crashpad.chromium.org/.

Report bugs at https://crashpad.chromium.org/bug/new.

## Copyright

Copyright 2015 [The Crashpad
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
