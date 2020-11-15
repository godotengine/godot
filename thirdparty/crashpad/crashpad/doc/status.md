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

# Project Status

## Completed

Crashpad currently consists of a crash-reporting client and some related tools
for macOS and Windows. The core client work for both platforms is substantially
complete. Crashpad became the crash reporter client for
[Chromium](https://www.chromium.org/Home) on macOS as of [March
2015](https://chromium.googlesource.com/chromium/src/\+/d413b2dcb54d523811d386f1ff4084f677a6d089),
and on Windows as of [November
2015](https://chromium.googlesource.com/chromium/src/\+/cfa5b01bb1d06bf96967bd37e21a44752801948c).

## In Progress

Initial work on a Crashpad client for
[Android](https://crashpad.chromium.org/bug/30) has begun. This is currently in
the early implementation phase.

## Future

There are plans to bring Crashpad clients to other operating systems in the
future, including a more generic non-Android Linux implementation. There are
also plans to implement a [crash report
processor](https://crashpad.chromium.org/bug/29) as part of Crashpad. No
timeline for completing this work has been set yet.
