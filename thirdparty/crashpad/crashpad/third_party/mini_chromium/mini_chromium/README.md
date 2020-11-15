<!--
// Copyright 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
-->

# mini_chromium

This is mini_chromium, a small collection of useful low-level (“base”) routines
from the [Chromium open-source project](https://dev.chromium.org/Home). Chromium
is large, sprawling, full of dependencies, and a web browser. mini_chromium is
small, self-contained, and a library. mini_chromium is especially useful as a
dependency of other code that wishes to use Chromium’s base routines. By using
mini_chromium, other projects’ code can function in a standalone environment
outside of Chromium without having to treat all of Chromium as a dependency.
When building as part of Chromium, those projects’ code can use Chromium’s own
(non-mini_chromium) base implementation.

Code provided in mini_chromium provides the same interface as the equivalent
code in Chromium.

While it’s a goal of mini_chromium to maintain interface compatibility with
Chromium’s base library for the interfaces it does implement, there’s no
requirement that it use the same implementations as Chromium’s base library.
Many of the implementations used in mini_chromium are identical to Chromium’s,
but many others have been modified to eliminate dependencies that are not
desired in mini_chromium, and a few are completely distinct from Chromium’s
altogether. Additionally, when mini_chromium provides an interface in the form
of a file or class present in Chromium, it’s not bound to provide all functions,
methods, or types that the Chromium equivalent does. The differences noted above
notwithstanding, the interfaces exposed by mini_chromium’s base are and must
remain a strict subset of Chromium’s.

[Crashpad](https://crashpad.chromium.org/) is the chief consumer of
mini_chromium.

Mark Mentovai<br/>
mark@chromium.org
