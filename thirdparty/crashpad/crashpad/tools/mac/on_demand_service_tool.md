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

# on_demand_service_tool(1)

## Name

on_demand_service_tool—Load and unload on-demand Mach services registered with
launchd(8)

## Synopsis

**on_demand_service_tool** **-L** **-l** _LABEL_ [_OPTION…_] _COMMAND_
[_ARG…_]<br/>
**on_demand_service_tool** **-U** **-l** _LABEL_

## Description

On-demand services may be registered with launchd(8) by using the **--load**
form. One or more service names may be registered with the bootstrap server by
specifying **--mach-service**. When a Mach message is sent to any of these
services, launchd(8) will invoke _COMMAND_ along with any arguments specified
(_ARG…_). _COMMAND_ must be an absolute pathname.

The **--unload** form unregisters jobs registered with launchd(8).

## Options

 * **-L**, **--load**

   Registers a job with launchd(8). **--label**=_LABEL_ and _COMMAND_ are
   required. This operation may also be referred to as “load” or “submit”.

 * **-U**, **--unload**

   Unregisters a job with launchd(8). **--label**=_LABEL_ is required. This
   operation may also be referred to as “unload” or “remove”.

 * **-l**, **--label**=_LABEL_

   _LABEL_ is used as the job label to identify the job to launchd(8). _LABEL_
   must be unique within a launchd(8) context.

 * **-m**, **--mach-service**=_SERVICE_

   In conjunction with **--load**, registers _SERVICE_ with the bootstrap
   server. Clients will be able to obtain a send right by looking up the
   _SERVICE_ name with the bootstrap server. When a message is sent to such a
   Mach port, launchd(8) will invoke _COMMAND_ along with any arguments
   specified (_ARG…_) if it is not running. This forms the “on-demand” nature
   referenced by this tool’s name. This option may appear zero, one, or more
   times. _SERVICE_ must be unique within a bootstrap context.

 * **--help**

   Display help and exit.

 * **--version**

   Output version information and exit.

## Examples

Registers an on-demand server that will execute
[catch_exception_tool(1)](catch_exception_tool.md) when a Mach message is sent
to a Mach port obtained from the bootstrap server by looking up the name `svc`:

```
$ on_demand_service_tool --load --label=catch_exception \
      --mach-service=svc \
      $(which catch_exception_tool) --mach-service=svc \
      --file=/tmp/out --persistent --timeout=0
```

Unregisters the on-demand server installed above:

```
$ on_demand_service_tool --unload --label=catch_exception
```

## Exit Status

 * **0**

   Success.

 * **1**

   Failure, with a message printed to the standard error stream.

## See Also

[catch_exception_tool(1)](catch_exception_tool.md),
[exception_port_tool(1)](exception_port_tool.md),
launchctl(1)

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
