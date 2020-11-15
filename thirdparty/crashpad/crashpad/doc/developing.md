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

# Developing Crashpad

## Status

[Project status](status.md) information has moved to its own page.

## Introduction

Crashpad is a [Chromium project](https://www.chromium.org/Home). Most of its
development practices follow Chromium’s. In order to function on its own in
other projects, Crashpad uses
[mini_chromium](https://chromium.googlesource.com/chromium/mini_chromium/), a
small, self-contained library that provides many of Chromium’s useful low-level
base routines. [mini_chromium’s
README](https://chromium.googlesource.com/chromium/mini_chromium/+/master/README.md)
provides more detail.

## Prerequisites

To develop Crashpad, the following tools are necessary, and must be present in
the `$PATH` environment variable:

 * Appropriate development tools.
    * On macOS, install [Xcode](https://developer.apple.com/xcode/). The latest
      version is generally recommended.
    * On Windows, install [Visual Studio](https://www.visualstudio.com/) with
      C++ support and the Windows SDK. MSVS 2015 and MSVS 2017 are both
      supported. Some tests also require the CDB debugger, installed with
      [Debugging Tools for
      Windows](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/).
 * Chromium’s
   [depot_tools](https://www.chromium.org/developers/how-tos/depottools).
 * [Git](https://git-scm.com/). This is provided by Xcode on macOS and by
   depot_tools on Windows.
 * [Python](https://www.python.org/). This is provided by the operating system
   on macOS, and by depot_tools on Windows.

## Getting the Source Code

The main source code repository is a Git repository hosted at
https://chromium.googlesource.com/crashpad/crashpad. Although it is possible to
check out this repository directly with `git clone`, Crashpad’s dependencies are
managed by
[`gclient`](https://www.chromium.org/developers/how-tos/depottools#TOC-gclient)
instead of Git submodules, so to work on Crashpad, it is best to use `fetch` to
get the source code.

`fetch` and `gclient` are part of the
[depot_tools](https://www.chromium.org/developers/how-tos/depottools). There’s
no need to install them separately.

### Initial Checkout

```
$ mkdir ~/crashpad
$ cd ~/crashpad
$ fetch crashpad
```

`fetch crashpad` performs the initial `git clone` and `gclient sync`,
establishing a fully-functional local checkout.

### Subsequent Checkouts

```
$ cd ~/crashpad/crashpad
$ git pull -r
$ gclient sync
```

## Building

### Windows, Mac, Linux, Fuchsia

On Windows, Mac, Linux, and Fuchsia Crashpad uses
[GN](https://gn.googlesource.com/gn) to generate
[Ninja](https://ninja-build.org/) build files. For example,

```
$ cd ~/crashpad/crashpad
$ gn gen out/Default
$ ninja -C out/Default
```

You can then use `gn args out/Default` or edit `out/Default/args.gn` to
configure the build, for example things like `is_debug=true` or
`target_cpu="x86"`.

GN and Ninja are part of the
[depot_tools](https://www.chromium.org/developers/how-tos/depottools). There’s
no need to install them separately.

### Android

Crashpad’s Android port is in its early stages. This build relies on
cross-compilation. It’s possible to develop Crashpad for Android on any platform
that the [Android NDK (Native Development
Kit)](https://developer.android.com/ndk/) runs on.

If it’s not already present on your system, [download the NDK package for your
system](https://developer.android.com/ndk/downloads/) and expand it to a
suitable location. These instructions assume that it’s been expanded to
`~/android-ndk-r16`.

To build Crashpad, portions of the NDK must be reassembled into a [standalone
toolchain](https://developer.android.com/ndk/guides/standalone_toolchain.html).
This is a repackaged subset of the NDK suitable for cross-compiling for a single
Android architecture (such as `arm`, `arm64`, `x86`, and `x86_64`) targeting a
specific [Android API
level](https://source.android.com/source/build-numbers.html). The standalone
toolchain only needs to be built from the NDK one time for each set of options
desired. To build a standalone toolchain targeting 64-bit ARM and API level 21
(Android 5.0 “Lollipop”), run:

```
$ cd ~
$ python android-ndk-r16/build/tools/make_standalone_toolchain.py \
      --arch=arm64 --api=21 --install-dir=android-ndk-r16_arm64_api21
```

Note that Chrome uses Android API level 21 for 64-bit platforms and 16 for
32-bit platforms. See Chrome’s
[`build/config/android/config.gni`](https://chromium.googlesource.com/chromium/src/+/master/build/config/android/config.gni)
which sets `_android_api_level` and `_android64_api_level`.

To configure a Crashpad build for Android using the standalone toolchain
assembled above, use `gyp_crashpad_android.py`. This script is a wrapper for
`gyp_crashpad.py` that sets several environment variables directing the build to
the standalone toolchain, and several GYP options to identify an Android build.
This must be done after any `gclient sync`, or instead of any `gclient runhooks`
operation.

```
$ cd ~/crashpad/crashpad
$ python build/gyp_crashpad_android.py \
      --ndk ~/android-ndk-r16_arm64_api21 \
      --generator-output out/android_arm64_api21
```

`gyp_crashpad_android.py` detects the build type based on the characteristics of
the standalone toolchain given in its `--ndk` argument.

`gyp_crashpad_android.py` sets the build up to use Clang by default. It’s also
possible to use GCC by providing the `--compiler=gcc` argument to
`gyp_crashpad_android.py`.

The Android port is incomplete, but targets known to be working include
`crashpad_test`, `crashpad_util`, and their tests. This list will grow over
time. To build, direct `ninja` to the specific `out` directory chosen by the
`--generator-output` argument to `gyp_crashpad_android.py`.

```
$ ninja -C out/android_arm64_api21/out/Debug \
      crashpad_test_test crashpad_util_test
```

## Testing

Crashpad uses [Google Test](https://github.com/google/googletest/) as its
unit-testing framework, and some tests use [Google
Mock](https://github.com/google/googletest/tree/master/googlemock/) as well. Its
tests are currently split up into several test executables, each dedicated to
testing a different component. This may change in the future. After a successful
build, the test executables will be found at `out/Debug/crashpad_*_test`.

```
$ cd ~/crashpad/crashpad
$ out/Debug/crashpad_minidump_test
$ out/Debug/crashpad_util_test
```

A script is provided to run all of Crashpad’s tests. It accepts a single
argument, a path to the directory containing the test executables.

```
$ cd ~/crashpad/crashpad
$ python build/run_tests.py out/Debug
```

To run a subset of the tests, use the --gtest\_filter flag, e.g., to run all the
tests for MinidumpStringWriter:

```sh
$ python build/run_tests.py out/Debug --gtest_filter MinidumpStringWriter*
```

### Windows

On Windows, `end_to_end_test.py` requires the CDB debugger, installed with
[Debugging Tools for
Windows](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/).
This can be installed either as part of the [Windows Driver
Kit](https://go.microsoft.com/fwlink/p?LinkID=239721) or the [Windows
SDK](https://go.microsoft.com/fwlink/p?LinkID=271979). If the Windows SDK has
already been installed (possibly with Visual Studio) but Debugging Tools for
Windows is not present, it can be installed from Add or remove programs→Windows
Software Development Kit.

### Android

To test on Android, [ADB (Android Debug
Bridge)](https://developer.android.com/studio/command-line/adb.html) from the
[Android SDK](https://developer.android.com/sdk/) must be in the `PATH`. Note
that it is sufficient to install just the command-line tools from the Android
SDK. The entire Android Studio IDE is not necessary to obtain ADB.

When asked to test an Android build directory, `run_tests.py` will detect a
single connected Android device (including an emulator). If multiple devices are
connected, one may be chosen explicitly with the `ANDROID_DEVICE` environment
variable. `run_tests.py` will upload test executables and data to a temporary
location on the detected or selected device, run them, and clean up after itself
when done.

### Fuchsia

To test on Fuchsia, you need a connected device running Fuchsia and then run:

```sh
$ gn gen out/fuchsia --args='target_os="fuchsia" target_cpu="x64" is_debug=true'
$ ninja -C out/fuchsia
$ python build/run_tests.py out/fuchsia
```

If you have multiple devices running, you will need to specify which device you
want using their hostname, for instance:

```sh
$ export ZIRCON_NODENAME=scare-brook-skip-dried; \
  python build/run_tests.py out/fuchsia; \
  unset ZIRCON_NODENAME
```

## Contributing

Crashpad’s contribution process is very similar to [Chromium’s contribution
process](https://www.chromium.org/developers/contributing-code).

### Code Review

A code review must be conducted for every change to Crashpad’s source code. Code
review is conducted on [Chromium’s
Gerrit](https://chromium-review.googlesource.com/) system, and all code reviews
must be sent to an appropriate reviewer, with a Cc sent to
[crashpad-dev](https://groups.google.com/a/chromium.org/group/crashpad-dev). The
[`codereview.settings`](https://chromium.googlesource.com/crashpad/crashpad/+/master/codereview.settings)
file specifies this environment to `git-cl`.

`git-cl` is part of the
[depot_tools](https://www.chromium.org/developers/how-tos/depottools). There’s
no need to install it separately.

```
$ cd ~/crashpad/crashpad
$ git checkout -b work_branch origin/master
…do some work…
$ git add …
$ git commit
$ git cl upload
```

Uploading a patch to Gerrit does not automatically request a review. You must
select a reviewer on the Gerrit review page after running `git cl upload`. This
action notifies your reviewer of the code review request. If you have lost track
of the review page, `git cl issue` will remind you of its URL. Alternatively,
you can request review when uploading to Gerrit by using `git cl upload
--send-mail`.

Git branches maintain their association with Gerrit reviews, so if you need to
make changes based on review feedback, you can do so on the correct Git branch,
committing your changes locally with `git commit`. You can then upload a new
patch set with `git cl upload` and let your reviewer know you’ve addressed the
feedback.

The most recently uploaded patch set on a review may be tested on a [try
server](https://www.chromium.org/developers/testing/try-server-usage) by running
`git cl try` or by clicking the “CQ Dry Run” button in Gerrit. These set the
“Commit-Queue: +1” label. This does not mean that the patch will be committed,
but the try server and commit queue share infrastructure and a Gerrit label. The
patch will be tested on try bots in a variety of configurations. Status
information will be available on Gerrit. Try server access is available to
Crashpad and Chromium committers.

### Landing Changes

After code review is complete and “Code-Review: +1” has been received from all
reviewers, the patch can be submitted to Crashpad’s [commit
queue](https://www.chromium.org/developers/testing/commit-queue) by clicking the
“Submit to CQ” button in Gerrit. This sets the “Commit-Queue: +2” label, which
tests the patch on the try server before landing it. Commit queue access is
available to Crashpad and Chromium committers.

Although the commit queue is recommended, if needed, project members can bypass
the commit queue and land patches without testing by using the “Submit” button
in Gerrit or by committing via `git cl land`:

```
$ cd ~/crashpad/crashpad
$ git checkout work_branch
$ git cl land
```

### External Contributions

Copyright holders must complete the [Individual Contributor License
Agreement](https://developers.google.com/open-source/cla/individual) or
[Corporate Contributor License
Agreement](https://developers.google.com/open-source/cla/corporate) as
appropriate before any submission can be accepted, and must be listed in the
[`AUTHORS`](https://chromium.googlesource.com/crashpad/crashpad/+/master/AUTHORS)
file. Contributors may be listed in the
[`CONTRIBUTORS`](https://chromium.googlesource.com/crashpad/crashpad/+/master/CONTRIBUTORS)
file.

## Buildbot

The [Crashpad Buildbot](https://build.chromium.org/p/client.crashpad/) performs
automated builds and tests of Crashpad. Before checking out or updating the
Crashpad source code, and after checking in a new change, it is prudent to check
the Buildbot to ensure that “the tree is green.”
