![Supported Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows%20%7C%20FreeBSD-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
[![CMake Build Matrix](https://github.com/gulrak/filesystem/actions/workflows/build_cmake.yml/badge.svg?branch=master)](https://github.com/gulrak/filesystem/actions/workflows/build_cmake.yml)
[![Build Status](https://ci.appveyor.com/api/projects/status/t07wp3k2cddo0hpo/branch/master?svg=true)](https://ci.appveyor.com/project/gulrak/filesystem)
[![Build Status](https://api.cirrus-ci.com/github/gulrak/filesystem.svg?branch=master)](https://cirrus-ci.com/github/gulrak/filesystem)
[![Build Status](https://cloud.drone.io/api/badges/gulrak/filesystem/status.svg?ref=refs/heads/master)](https://cloud.drone.io/gulrak/filesystem)
[![Coverage Status](https://coveralls.io/repos/github/gulrak/filesystem/badge.svg?branch=master)](https://coveralls.io/github/gulrak/filesystem?branch=master)
[![Latest Release Tag](https://img.shields.io/github/tag/gulrak/filesystem.svg)](https://github.com/gulrak/filesystem/tree/v1.5.12)

- [Filesystem](#filesystem)
  - [Motivation](#motivation)
  - [Why the namespace GHC?](#why-the-namespace-ghc)
  - [Platforms](#platforms)
  - [Tests](#tests)
  - [Usage](#usage)
    - [Downloads](#downloads)
    - [Using it as Single-File-Header](#using-it-as-single-file-header)
    - [Using it as Forwarding-/Implementation-Header](#using-it-as-forwarding-implementation-header)
    - [Git Submodule and CMake](#git-submodule-and-cmake)
    - [Versioning](#versioning)
  - [Documentation](#documentation)
    - [`ghc::filesystem::ifstream`, `ghc::filesystem::ofstream`, `ghc::filesystem::fstream`](#ghcfilesystemifstream-ghcfilesystemofstream-ghcfilesystemfstream)
    - [`ghc::filesystem::u8arguments`](#ghcfilesystemu8arguments)
  - [Differences](#differences)
    - [LWG Defects](#lwg-defects)
    - [Not Implemented on C++ before C++17](#not-implemented-on-c-before-c17)
    - [Differences in API](#differences-in-api)
      - [Differences of Specific Interfaces](#differences-of-specific-interfaces)
    - [Differences in Behavior](#differences-in-behavior)
      - [fs.path](#fspath-refhttpsencppreferencecomwcppfilesystempath)
  - [Open Issues](#open-issues)
    - [Windows](#windows)
      - [Symbolic Links on Windows](#symbolic-links-on-windows)
      - [Permissions](#permissions)
  - [Release Notes](#release-notes)
  
# Filesystem

This is a header-only single-file `std::filesystem` compatible helper library,
based on the C++17 and C++20 specs, but implemented for C++11, C++14, C++17 or C++20
(tightly following  the C++17 standard with very few documented exceptions). It is currently tested on
macOS 10.12/10.14/10.15/11.6, Windows 10, Ubuntu 18.04, Ubuntu 20.04, CentOS 7, CentOS 8, FreeBSD 12,
Alpine ARM/ARM64 Linux and Solaris 10 but should work on other systems too, as long as you have
at least a C++11 compatible compiler. It should work with Android NDK, Emscripten and I even
had reports of it being used on iOS (within sandboxing constraints) and with v1.5.6 there
is experimental support for QNX. The support of Android NDK, Emscripten and QNX is not
backed up by automated testing but PRs and bug reports are welcome for those too.
It is of course in its own namespace `ghc::filesystem` to not interfere with a regular `std::filesystem`
should you use it in a mixed C++17 environment (which is possible).

*Test coverage is well above 90%, and starting with v1.3.6 and in v1.5.0
more time was invested in benchmarking and optimizing parts of the library. I'll try
to continue to optimize some parts and refactor others, striving
to improve it as long as it doesn't introduce additional C++17/C++20 compatibility
issues. Feedback is always welcome. Simply open an issue if you see something missing
or wrong or not behaving as expected and I'll comment.*


## Motivation

I'm often in need of filesystem functionality, mostly `fs::path`, but directory
access too, and when beginning to use C++11, I used that language update
to try to reduce my third-party dependencies. I could drop most of what
I used, but still missed some stuff that I started implementing for the
fun of it. Originally I based these helpers on my own coding- and naming
conventions. When C++17 was finalized, I wanted to use that interface,
but it took a while, to push myself to convert my classes.

The implementation is closely based on chapter 30.10 from the C++17 standard
and a draft close to that version is
[Working Draft N4687](https://github.com/cplusplus/draft/raw/master/papers/n4687.pdf).
It is from after the standardization of C++17 but it contains the latest filesystem
interface changes compared to the
[Working Draft N4659](https://github.com/cplusplus/draft/raw/master/papers/n4659.pdf).
Staring with v1.4.0, when compiled using C++20, it adapts to the changes according to path sorting order
and `std::u8string` handling from [Working Draft N4860](https://isocpp.org/files/papers/N4860.pdf).

I want to thank the people working on improving C++, I really liked how the language
evolved with C++11 and the following standards. Keep on the good work!

## Why the namespace GHC?
If you ask yourself, what `ghc` is standing for, it is simply
`gulraks helper classes`, yeah, I know, not very imaginative, but I wanted a
short namespace and I use it in some of my private classes (so **it has nothing
to do with Haskell**, sorry for the name clash).

## Platforms

`ghc::filesystem` is developed on macOS but CI tested on macOS, Windows,
various Linux Distributions, FreeBSD and starting with v1.5.12 on Solaris.
It should work on any of these with a C++11-capable  compiler. Also there are some
checks to hopefully better work on Android, but  as I currently don't test with the
Android NDK, I wouldn't call it a  supported platform yet, same is valid for using
it with Emscripten. It is now  part of the detected platforms, I fixed the obvious
issues and ran some tests with  it, so it should be fine. All in all, I don't see it
replacing `std::filesystem` where full C++17 or C++20 is available, it doesn't try
to be a "better" `std::filesystem`, just an almost drop-in if you can't use it
(with the exception  of the UTF-8 preference).

:information_source: **Important:** _This implementation is following the ["UTF-8 Everywhere" philosophy](https://utf8everywhere.org/) in that all
`std::string` instances will be interpreted the same as `std::u8string` encoding
wise and as being in UTF-8. The `std::u16string` will be seen as UTF-16. See *Differences in API*
for more information._

Unit tests are currently run with:

* macOS 10.12: Xcode 9.2 (clang-900.0.39.2), GCC 9.2, Clang 9.0, macOS 10.13: Xcode 10.1, macOS 10.14: Xcode 11.2, macOS 10.15: Xcode 11.6, Xcode 12.4
* Windows: Visual Studio 2017, Visual Studio 2015, Visual Studio 2019, MinGW GCC 6.3 (Win32), GCC 7.2 (Win64), Cygwin GCC 10.2 (no CI yet)
* Linux (Ubuntu): GCC (5.5, 6.5, 7.4, 8.3, 9.2), Clang (5.0, 6.0, 7.1, 8.0, 9.0)
* Linux (Alpine ARM/ARM64): GCC 9.2.0
* FreeBSD: Clang 8.0
* Solaris: GCC 5.5 


## Tests

The header comes with a set of unit-tests and uses [CMake](https://cmake.org/)
as a build tool and [Catch2](https://github.com/catchorg/Catch2) as test framework.
All tests are registered with in CMake, so the [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
commando can be used to run the tests. 

All tests against this implementation should succeed, depending on your environment
it might be that there are some warnings, e.g. if you have no rights to create
Symlinks on Windows or at least the test thinks so, but these are just informative.

To build the tests from inside the project directory under macOS or Linux just:

```cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
ctest
```

This generates the test binaries that run the tests and the last command executes
them.

If the default compiler is a GCC 8 or newer, or Clang 7 or newer, it
additionally tries to build a version of the test binary compiled against GCCs/Clangs
`std::filesystem` implementation, named `std_filesystem_test`
as an additional test of conformance. Ideally all tests should compile and
succeed with all filesystem implementations, but in reality, there are
some differences in behavior, sometimes due to room for interpretation in
in the standard, and there might be issues in these implementations too.


## Usage

### Downloads

The latest release version is [v1.5.12](https://github.com/gulrak/filesystem/tree/v1.5.12) and
source archives can be found [here](https://github.com/gulrak/filesystem/releases/tag/v1.5.12).

The latest pre-native-backend version is [v1.4.0](https://github.com/gulrak/filesystem/tree/v1.4.0) and
source archives can be found [here](https://github.com/gulrak/filesystem/releases/tag/v1.4.0).

The latest pre-C++20-support release version is [v1.3.10](https://github.com/gulrak/filesystem/tree/v1.3.10) and
source archives can be found [here](https://github.com/gulrak/filesystem/releases/tag/v1.3.10).

Currently only the latest minor release version receives bugfixes, so if possible,
you should use the latest release.

### Using it as Single-File-Header

As `ghc::filesystem` is at first a header-only library, it should be enough to copy the header
or the `include/ghc` directory into your project folder or point your include path to this place and
simply include the `filesystem.hpp` header (or `ghc/filesystem.hpp` if you use the subdirectory).

Everything is in the namespace `ghc::filesystem`, so one way to use it only as
a fallback could be:

```cpp
#ifdef __APPLE__
#include <Availability.h> // for deployment target to support pre-catalina targets without std::fs 
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
#if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#endif
```

**Note that this code uses a two-stage preprocessor condition because Visual Studio 2015
doesn't like the `(<...>)` syntax, even if it could cut evaluation early before. This code also
used the minimum deployment target to detect if `std::filesystem` really is available on macOS
compilation.**

**Note also, this detection now works on MSVC versions prior to 15.7 on, or without setting
the `/Zc:__cplusplus` compile switch that would fix `__cplusplus` on MSVC. (Without the switch
the compiler always reports `199711L`
([see](https://blogs.msdn.microsoft.com/vcblog/2018/04/09/msvc-now-correctly-reports-__cplusplus/)),
but `_MSVC_LANG` works without it.**

If you want to also use the `fstream` wrapper with `path` support as fallback,
you might use:

```cpp
#ifdef __APPLE__
#include <Availability.h> // for deployment target to support pre-catalina targets without std::fs 
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
#if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs {
using namespace std::filesystem;
using ifstream = std::ifstream;
using ofstream = std::ofstream;
using fstream = std::fstream;
}
#endif
#endif
#ifndef GHC_USE_STD_FS
#include <ghc/filesystem.hpp>
namespace fs {
using namespace ghc::filesystem;
using ifstream = ghc::filesystem::ifstream;
using ofstream = ghc::filesystem::ofstream;
using fstream = ghc::filesystem::fstream;
}
#endif
```

Now you have e.g. `fs::ofstream out(somePath);` and it is either the wrapper or
the C++17 `std::ofstream`.

:information_source: **Be aware, as a header-only library, it is not hiding the fact, that it
uses system includes, so they "pollute" your global namespace. Use the
forwarding-/implementation-header based approach (see below) to avoid this.
For Windows it needs `Windows.h` and it might be a good idea to define
`WIN32_LEAN_AND_MEAN` or `NOMINMAX` prior to including `filesystem.hpp` or
`fs_std.hpp` headers  to reduce pollution of your global namespace and compile
time. They are not defined by `ghc::filesystem` to allow combination with contexts
where the full `Windows.h`is needed, e.g. for UI elements.**

:information_source: **Hint:** There is an additional header named `ghc/fs_std.hpp` that implements this
dynamic selection of a filesystem implementation, that you can include
instead of `ghc/filesystem.hpp` when you want `std::filesystem` where
available and `ghc::filesystem` where not.


### Using it as Forwarding-/Implementation-Header

Alternatively, starting from v1.1.0 `ghc::filesystem` can also be used by
including one of two additional wrapper headers. These allow to include
a forwarded version in most places (`ghc/fs_fwd.hpp`) while hiding the
implementation details in a single cpp file that includes `ghc/fs_impl.hpp` to
implement the needed code. Using `ghc::filesystem` this way makes sure
system includes are only visible from inside the cpp file, all other places are clean.

Be aware, that it is currently not supported to hide the implementation
into a Windows-DLL, as a DLL interface with C++ standard templates in interfaces
is a different beast. If someone is willing to give it a try, I might integrate
a PR but currently working on that myself is not a priority.

If you use the forwarding/implementation approach, you can still use the dynamic
switching like this:

```cpp
#ifdef __APPLE__
#include <Availability.h> // for deployment target to support pre-catalina targets without std::fs
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
#if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs {
using namespace std::filesystem;
using ifstream = std::ifstream;
using ofstream = std::ofstream;
using fstream = std::fstream;
}
#endif
#endif
#ifndef GHC_USE_STD_FS
#include <ghc/fs-fwd.hpp>
namespace fs {
using namespace ghc::filesystem;
using ifstream = ghc::filesystem::ifstream;
using ofstream = ghc::filesystem::ofstream;
using fstream = ghc::filesystem::fstream;
}
#endif
```

and in the implementation hiding cpp, you might use (before any include that includes `ghc/fs_fwd.hpp`
to take precedence:

```cpp
#ifdef __APPLE__ // for deployment target to support pre-catalina targets without std::fs 
#include <Availability.h>
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
#if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
#define GHC_USE_STD_FS
#endif
#endif
#ifndef GHC_USE_STD_FS
#define GHC_FILESYSTEM_IMPLEMENTATION
#include <ghc/filesystem.hpp>
#endif
```

:information_source: **Hint:** There are additional helper headers, named `ghc/fs_std_fwd.hpp` and
`ghc/fs_std_impl.hpp` that use this technique, so you can simply include them
if you want to dynamically select the filesystem implementation. they also
enable the `wchar_t` support on `ghc::filesystem` on Windows, so the resulting
implementation in the `fs` namespace will be compatible.



### Git Submodule and CMake

Starting from v1.1.0, it is possible to add `ghc::filesystem`
as a git submodule, add the directory to your `CMakeLists.txt` with
`add_subdirectory()` and then simply use `target_link_libraries(your-target ghc_filesystem)`
to ensure correct include path that allow `#include <ghc/filesystem.hpp>`
to work.

The `CMakeLists.txt` offers a few options to customize its behavior:

* `GHC_FILESYSTEM_BUILD_TESTING` - Compile tests, default is `OFF` when used as
  a submodule, else `ON`.
* `GHC_FILESYSTEM_BUILD_EXAMPLES` - Compile the examples, default is `OFF` when used as
  a submodule, else `ON`.
* `GHC_FILESYSTEM_WITH_INSTALL` - Add install target to build, default is `OFF` when used as
  a submodule, else `ON`.
* `GHC_FILESYSTEM_BUILD_STD_TESTING` -  Compile `std_filesystem_test`, the variant of
  the test suite running against `std::filesystem`, defaulting to `GHC_FILESYSTEM_BUILD_TESTING`.
  This is only done if the compiler is detected as being able to do it.
* `GHC_FILESYSTEM_TEST_COMPILE_FEATURES` can be set to a list of features to override
  `CMAKE_CXX_COMPILE_FEATURES` when the detection of C++17 or C++20 for additional tests
  is not working (e.g. `cxx_std_20` to enforce building a `filesystem_test_cpp20` with C++20).

### Versioning

There is a version macro `GHC_FILESYSTEM_VERSION` defined in case future changes
might make it needed to react on the version, but I don't plan to break anything.
It's the version as decimal number `(major * 10000 + minor * 100 + patch)`.

:information_source: **Note:** Only even patch versions will be used for releases
and odd patch version will only be used for in between commits while working on
the next version.


## Documentation

There is almost no documentation in this release, as any `std::filesystem`
documentation would work, besides the few differences explained in the next
section. So you might head over to https://en.cppreference.com/w/cpp/filesystem
for a description of the components of this library.

When compiling with C++11, C++14 or C++17, the API is following the C++17
standard, where possible, with the exception that `std::string_view` parameters
are only supported on C++17. When Compiling with C++20, `ghc::filesysytem`
defaults to the C++20 API, with the `char8_t` and `std::u8string` interfaces
and the deprecated `fs::u8path` factory method.

:information_source: **Note:** If the C++17 API should be enforced even in C++20 mode,
use the define `GHC_FILESYSTEM_ENFORCE_CPP17_API`.
Even then it is possible to create `fws::path` from `std::u8string` but
`fs::path::u8string()` and `fs::path::generic_u8string()` return normal
UTF-8 encoded `std::string` instances, so code written for C++17 could
still work with `ghc::filesystem` when compiled with C++20.

The only additions to the standard are documented here:


### `ghc::filesystem::ifstream`, `ghc::filesystem::ofstream`, `ghc::filesystem::fstream`

These are simple wrappers around `std::ifstream`, `std::ofstream` and `std::fstream`.
They simply add an `open()` method and a constructor with an `ghc::filesystem::path`
argument as the `fstream` variants in C++17 have them.

### `ghc::filesystem::u8arguments`

This is a helper class that currently checks for UTF-8 encoding on non-Windows platforms but on Windows it
fetches the command line arguments as Unicode strings from the OS with

```cpp
::CommandLineToArgvW(::GetCommandLineW(), &argc)
```

and then converts them to UTF-8, and replaces `argc` and `argv`. It is a guard-like
class that reverts its changes when going out of scope.

So basic usage is:

```cpp
namespace fs = ghc::filesystem;

int main(int argc, char* argv[])
{
    fs::u8arguments u8guard(argc, argv);
    if(!u8guard.valid()) {
        std::cerr << "Bad encoding, needs UTF-8." << std::endl;
        exit(EXIT_FAILURE);
    }

    // now use argc/argv as usual, they have utf-8 enconding on windows
    // ...

    return 0;
}
```

That way `argv` is UTF-8 encoded as long as the scope from `main` is valid.

**Note:** On macOS, while debugging under Xcode the code currently will return
`false` as Xcode starts the application with `US-ASCII` as encoding, no matter what
encoding is actually used and even setting `LC_ALL` in the product scheme doesn't
change anything. I still need to investigate this.


## Differences

As this implementation is based on existing code from my private helper
classes, it derived some constraints of it. Starting from v1.5.0 most of the
differences between this and the standard C++17/C++20 API where removed.


### LWG Defects

This implementation has switchable behavior for the LWG defects
[#2682](https://wg21.cmeerw.net/lwg/issue2682),
[#2935](http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2935),
[#2936](http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2936) and
[#2937](http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2937).
The currently selected behavior (starting from v1.4.0) is following
[#2682](https://wg21.cmeerw.net/lwg/issue2682), [#2936](http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2936),
[#2937](http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2937) but
not following [#2935](http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2935),
as I feel it is a bug to report no error on a `create_directory()` or `create_directories()`
where a regular file of the same name prohibits the creation of a directory and forces
the user of those functions to double-check via `fs::is_directory` if it really worked.
The more intuitive approach to directory creation of treating a file with that name as an
error is also advocated by the newer paper
[WG21 P1164R0](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1164r0.pdf), the revision
P1161R1 was agreed upon on Kona 2019 meeting [see merge](https://github.com/cplusplus/draft/issues/2703)
and GCC by now switched to following its proposal
([GCC #86910](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86910)).

### Not Implemented on C++ before C++17

```cpp
// methods in ghc::filesystem::path:
path& operator+=(basic_string_view<value_type> x);
int compare(basic_string_view<value_type> s) const;
```

These are not implemented under C++11 and C++14, as there is no
`std::basic_string_view` available and I did want to keep this
implementation self-contained and not write a full C++17-upgrade for
C++11/14. Starting with v1.1.0 these are supported when compiling
`ghc::filesystem` under C++17 of C++20.

Starting with v1.5.2 `ghc::filesystem` will try to allow the use of
`std::experimental::basic_string_view` where it detects is availability.
Additionally if you have a `basic_string_view` compatible c++11
implementation it can be used instead of `std::basic_string_view`
by defining `GHC_HAS_CUSTOM_STRING_VIEW` and importing the
implementation into the `ghc::filesystem` namespace with:

```cpp
namespace ghc {
    namespace filesystem {
        using my::basic_string_view;
    }
}
```

before including the filesystem header.

### Differences in API

To not depend on any external third party libraries and still stay portable and
compact, this implementation is following the ["UTF-8 Everywhere" philosophy](https://utf8everywhere.org/) in that all
`std::string` instances will be interpreted the same as `std::u8string` encoding
wise and as being in UTF-8. The `std::u16string` will be seen as UTF-16 and `std::u32string` will be
seen as Unicode codepoints. Depending on the size of `std::wstring` characters, it will handle
`std::wstring` as being UTF-16 (e.g. Windows) or `char32_t` Unicode codepoints
(currently all other platforms).

#### Differences of Specific Interfaces

Starting with v1.5.0 `ghc::filesystem` is following the C++17 standard in
using `wchar_t` and `std::wstring` on Windows as the types internally used
for path representation. It is still possible to get the old behavior by defining
`GHC_WIN_DISABLE_WSTRING_STORAGE_TYPE` and get `filesystem::path::string_type` as
`std::string` and `filesystem::path::value_type` as `wchar_t`.

If you need to call some Windows API, with v1.5.0 and above, simply
use the W-variant of the Windows-API call (e.g. `GetFileAttributesW(p.c_str())`).

:information_source: **Note:** _When using the old behavior by defining
`GHC_WIN_DISABLE_WSTRING_STORAGE_TYPE`, use the `path::wstring()` member
(e.g. `GetFileAttributesW(p.wstring().c_str())`). This gives you the
Unicode variant independent of the `UNICODE` macro and makes sharing code
between Windows, Linux and macOS easier and works with `std::filesystem` and
`ghc::filesystem`._

```cpp
std::string path::u8string() const;
std::string path::generic_u8string() const;
vs.
std::u8string path::u8string() const;
std::u8string path::generic_u8string() const;
```

The return type of these two methods is depending on the used C++ standard
and if `GHC_FILESYSTEM_ENFORCE_CPP17_API` is defined. On C++11, C++14 and
C++17 or when `GHC_FILESYSTEM_ENFORCE_CPP17_API` is defined, the return
type is `std::string`, and on C++20 without the define it is `std::u8string`.

### Differences in Behavior

I created a wiki entry about quite a lot of [behavioral differences](https://github.com/gulrak/filesystem/wiki/Differences-to-Standard-Filesystem-Implementations)
between different `std::filesystem` implementations that could result in a
mention here, but this readme only tries to address the design choice
differences between `ghc::filesystem` and those. I try to update the wiki page
from time to time.

Any additional observations are welcome!

#### fs.path ([ref](https://en.cppreference.com/w/cpp/filesystem/path))

Since v1.5.0 the complete inner mechanics of this implementations `fs::path`
where changed to the _native_ format as the internal representation.
Creating any mixed slash `fs::path` object under Windows (e.g. with `"C:\foo/bar"`)
will lead clean path with `"C:\foo\bar"` via `native()` and `"C:/foo/bar"` via
`generic_string()` API. On all platforms redundant additional separators are
removed, even if this is not enforced by the standard and other implementations
mostly not do this.

Additionally this implementation follows the standards suggestion to handle
posix paths of the form `"//host/path"` and USC path on windows also as having
a root-name (e.g. `"//host"`). The GCC implementation didn't choose to do that
while testing on Ubuntu 18.04 and macOS with GCC 8.1.0 or Clang 7.0.0. This difference
will show as warnings under `std::filesystem`. This leads to a change in the
algorithm described in the standard for `operator/=(path& p)` where any path
`p` with `p.is_absolute()` will degrade to an assignment, while this implementation
has the exception where `*this == *this.root_name()` and `p == preferred_separator`
a normal append will be done, to allow:

```cpp
fs::path p1 = "//host/foo/bar/file.txt";
fs::path p2;
for (auto p : p1) p2 /= p;
ASSERT(p1 == p2);
```

For all non-host-leading paths the behavior will match the one described by
the standard.


## Open Issues

### Windows

#### Symbolic Links on Windows

As symbolic links on Windows, while being supported more or less since
Windows Vista (with some strict security constraints) and fully since some earlier
build of Windows 10, when "Developer Mode" is activated, are at time of writing
(2018) rarely used, still they are supported wiit th this implementation.

#### Permissions

The Windows ACL permission feature translates badly to the POSIX permission
bit mask used in the interface of C++17 filesystem. The permissions returned
in the `file_status` are therefore currently synthesized for the `user`-level
and copied to the `group`- and `other`-level. There is still some potential
for more interaction with the Windows permission system, but currently setting
or reading permissions with this implementation will most certainly not lead
to the expected behavior.


## Release Notes

### [v1.5.12](https://github.com/gulrak/filesystem/releases/tag/v1.5.12)

* Fix for [#142](https://github.com/gulrak/filesystem/issues/142), removed need
  for `GHC_NO_DIRENT_D_TYPE` on systems that don't support `dirent::d_type` and
  fixed build configuration and tests to support Solaris as new platform.
* Pull request [#138](https://github.com/gulrak/filesystem/pull/138), if the
  platform uses the POSIX backend and has no `PATH_MAX`, one is defined.
* Pull request [#137](https://github.com/gulrak/filesystem/pull/137), update
  of Catch2 to version v2.13.7
* Added macOS 11 to the automatically tested platforms.

### [v1.5.10](https://github.com/gulrak/filesystem/releases/tag/v1.5.10)

* Pull request [#136](https://github.com/gulrak/filesystem/pull/136), the Windows
  implementation used some unnecessary expensive shared pointer for resource
  management and these where replaced by a dedicated code.
* Fix for [#132](https://github.com/gulrak/filesystem/issues/132), pull request
  [#135](https://github.com/gulrak/filesystem/pull/135), `fs::remove_all` now
  just deletes symbolic links instead of following them.
* Pull request [#133](https://github.com/gulrak/filesystem/pull/133), fix for
  `fs::space` where a numerical overflow could happen in a multiplication.
* Replaced _travis-ci.org_  with GitHub Workflow for the configurations:
  Ubuntu 20.04: GCC 9.3, Ubuntu 18.04: GCC 7.5, GCC 8.4, macOS 10.15: Xcode 12.4,
  Windows 10: Visual Studio 2019

### [v1.5.8](https://github.com/gulrak/filesystem/releases/tag/v1.5.8)

* Fix for [#125](https://github.com/gulrak/filesystem/issues/124), where
  `fs::create_directories` on Windows no longer breaks on long filenames.

### [v1.5.6](https://github.com/gulrak/filesystem/releases/tag/v1.5.6)

* Fix for [#124](https://github.com/gulrak/filesystem/issues/124),
  `ghc::filesystem` treated mounted folder/volumes erroneously as symlinks,
  leading `fs::canonical` to fail on paths containing those.
* Fix for [#122](https://github.com/gulrak/filesystem/issues/122), incrementing
  the `recursive_directory_iterator` will not try to enter dead symlinks.
* Fix for [#121](https://github.com/gulrak/filesystem/issues/121), on Windows
  backend the `fs::remove` failed when the path pointed to a read-only entry,
  see also ([microsoft/STL#1511](https://github.com/microsoft/STL/issues/1511))
  for the corresponding issue in `std::fs` on windows.
* Fix for [#119](https://github.com/gulrak/filesystem/issues/119), added missing
  support for char16_t and char32_t and on C++20 char8_t literals.
* Pull request [#118](https://github.com/gulrak/filesystem/pull/118), when
  running tests as root, disable tests that would not work.
* Pull request [#117](https://github.com/gulrak/filesystem/pull/117), added
  checks to tests to detect the clang/libstdc++ combination.
* Fix for [#116](https://github.com/gulrak/filesystem/issues/116), internal
  macro `GHC_NO_DIRENT_D_TYPE` allows os detection to support systems without
  the `dirent.d_type` member, experimental first QNX compile support as
  initial use case, fixed issue with filesystems returning DT_UNKNOWN
  (e.g. reiserfs).
* Pull request [#115](https://github.com/gulrak/filesystem/pull/115), added
  `string_view` support when clang with libstdc++ is detected.
* Fix for [#114](https://github.com/gulrak/filesystem/issues/114), for macOS
  the pre-Catalina deployment target detection worked only if `<Availability.h>`
  was included before `<ghc/fs_std.hpp>` or `<ghc/fs_std_fwd.hpp>`/`<ghc/fs_std_impl.hpp>`.
* Fix for [#113](https://github.com/gulrak/filesystem/issues/113), the use of
  standard chapter numbers was misleading since C++17 and C++20 `std::filesystem`
  features are supported, and was replaced by the tag-like chapter names that
  stay (mostly) consistent over the versions.

### [v1.5.4](https://github.com/gulrak/filesystem/releases/tag/v1.5.4)

* Pull request [#112](https://github.com/gulrak/filesystem/pull/112), lots
  of cleanup work on the readme, thanks!
* Enhancement for [#111](https://github.com/gulrak/filesystem/issues/111),
  further optimization of directory iteration, performance for
  `recursive_directory_iterator` over large trees now somewhere  between
  libc++ and libstdc++.
* Enhancement for [#110](https://github.com/gulrak/filesystem/issues/110),
  `ghc::filesystem` now has preliminary support for Cygwin. Changes where
  made to allow the tests to compile and run successfully (tested with GCC
  10.2.0), feedback and additional PRs welcome as it is currently not
  part of the CI configuration.
* Pull request [#109](https://github.com/gulrak/filesystem/pull/109), various
  spelling errors in error messages and comments fixed.
* Pull request [#108](https://github.com/gulrak/filesystem/pull/108), old
  style casts removed.
* Fix for [#107](https://github.com/gulrak/filesystem/issues/107), the error
  handling for status calls was suppressing errors on symlink targets.
* Pull request [#106](https://github.com/gulrak/filesystem/pull/106), fixed
  detection of AppleClang for compile options.
* Pull request [#105](https://github.com/gulrak/filesystem/pull/105), added
  option `GHC_FILESYSTEM_BUILD_STD_TESTING` to override additional build of
  `std::filesystem` versions of the tests for comparison and the possibility
  to use `GHC_FILESYSTEM_TEST_COMPILE_FEATURES` to prefill the used compile
  features defaulting to `CMAKE_CXX_COMPILE_FEATURES` when not given.

### [v1.5.2](https://github.com/gulrak/filesystem/releases/tag/v1.5.2)

* Enhancement [#104](https://github.com/gulrak/filesystem/issues/104),
  on POSIX backend: optimized reuse of status information and reduced
  `directory_entry` creation leads to about 20%-25% in tests with
  `recursive_directory_iterator` over a larger directory tree.
* Pull request [#103](https://github.com/gulrak/filesystem/pull/103), `wchar_t`
  was not in the list of supported char types on non-Windows backends.
* Pull request [#102](https://github.com/gulrak/filesystem/pull/102), improved
  `string_view` support makes use of `<string_view>` or `<experimental/string_view>`
  when available, and allows use of custom `basic_string_view` implementation
  when defining `GHC_HAS_CUSTOM_STRING_VIEW` and importing the string view
  into the `ghc::filesystem` namespace before including filesystem header.
* Pull request [#101](https://github.com/gulrak/filesystem/pull/101), fix for
  [#100](https://github.com/gulrak/filesystem/issues/100), append and concat
  type of operations on path called redundant conversions.
* Pull request [#98](https://github.com/gulrak/filesystem/pull/98), on older
  linux variants (GCC 7/8), the comparison `std::filesystem` tests now link
  with `-lrt` to avoid issues.
* Fix for [#97](https://github.com/gulrak/filesystem/issues/97), on BTRFS the
  test case for `fs::hard_link_count` failed due to the filesystems behavior,
  the test case was adapted to take that into account.
* Pull request [#96](https://github.com/gulrak/filesystem/pull/96), the export
  attribute defines  `GHC_FS_API` and `GHC_FS_API_CLASS` are now honored when when
  set from outside to allow override of behavior.
* Fix for [#95](https://github.com/gulrak/filesystem/issues/95), the syntax for
  disabling the deprecated warning in tests in MSVC was wrong.
* Pull request [#93](https://github.com/gulrak/filesystem/pull/93), now the
  CMake configuration file is configured and part of the `make install` files.

### [v1.5.0](https://github.com/gulrak/filesystem/releases/tag/v1.5.0)

* Fix for [#91](https://github.com/gulrak/filesystem/issues/91), the way
  the CMake build options `GHC_FILESYSTEM_BUILD_TESTING`, `GHC_FILESYSTEM_BUILD_EXAMPLES`
  and `GHC_FILESYSTEM_WITH_INSTALL` where implemented, prohibited setting them
  from a parent project when using this via `add_subdirectory`, this fix
  allows to set them again.
* Major refactoring for [#90](https://github.com/gulrak/filesystem/issues/90),
  the way, the Windows version of `fs::path` was originally created from the
  POSIX based implementation was, by adaption of the incoming and outgoing
  strings. This resulted in a mutable cache inside `fs::path`on Windows, that
  was inherently not thread-safe, even for `const` methods.
  To not add additional patches to a suboptimal solution, this time I reworked
  the `path` code to now store _native_ path-representation. This changed a
  lot of code, but when combined with `wchar_t` as `value_type` helped to avoid
  lots of conversion for calls to Win-API.<br>
  As interfaces where changed, it had to be released in a new minor version.
  The set of refactorings resulted in the following changes:
  * `fs::path::native()` and `fs::path::c_str()` can now be `noexcept` as the
    standard mandates
  * On Windows `wchar_t` is now the default for `fs::path::value_type` and
    `std::wstring` is the default for `fs::path::string_type`.
  * This allows the implementation to call Win-API without allocating
    conversions
  * Thread-safety on `const` methods of `fs::path` is no longer an issue
  * Some code could be simplified during this refactoring
  * Automatic prefixing of long path on Windows can now be disabled with
    defining `GHC_WIN_DISABLE_AUTO_PREFIXES`, for all other types of prefixes
    or namespaces the behavior follows that of MSVC `std::filesystem::path`
  * In case the old `char`/`std::string` based approach for Windows is still
    needed, it can be activated with `GHC_WIN_DISABLE_WSTRING_STORAGE_TYPE`
* Enhancement for [#89](https://github.com/gulrak/filesystem/issues/89), `fs::file_status`
  now supports `operator==` introduced in `std::filesystem` with C++20.
* Refactoring for [#88](https://github.com/gulrak/filesystem/issues/88), `fs::path::parent_path()`
  had a performance issue, as it was still using a loop based approach to recreate
  the parent from elements. This created lots of temporaries and was too slow
  especially on long paths.

### [v1.4.0](https://github.com/gulrak/filesystem/releases/tag/v1.4.0)

* Enhancements for [#71](https://github.com/gulrak/filesystem/issues/71), when compiled with C++20:
  * `char8_t` and `std::u8string` are supported where `Source` is the parameter type
  * `fs::path::u8string()` and `fs::path::generic_u8string()` now return a `std::u8string`
  * The _spaceship operator_ `<=>` is now supported for `fs::path`
  * With the define `GHC_FILESYSTEM_ENFORCE_CPP17_API` `ghc::filesystem` will fall back
    to the old `fs::path::u8string()` and `fs::path::generic_u8string()` API if preferred
* Bugfix for `fs::proximate(p, ec)` where the internal call to `fs::current_path()` was not
  using the `error_code` variant, throwing possible exceptions instead of setting `ec`.
* Enhancement `LWG_2936_BEHAVIOUR` is now on by default.
* Some cleanup work to reduce preprocessor directives for better readability and remove unneeded
  template specializations.

### [v1.3.10](https://github.com/gulrak/filesystem/releases/tag/v1.3.10)

* Fix for [#81](https://github.com/gulrak/filesystem/issues/81), fixed issues with
  handling `Source` parameters that are string views.
* Fix for [#79](https://github.com/gulrak/filesystem/issues/79), the bit operations
  for filesystem bitmasks that should be are now `constexpr`.

### [v1.3.8](https://github.com/gulrak/filesystem/releases/tag/v1.3.8)

* Refactoring for [#78](https://github.com/gulrak/filesystem/issues/78), the dynamic
  switching helper includes are now using `__MAC_OS_X_VERSION_MIN_REQUIRED` to
  ensure that `std::filesystem` is only selected on macOS if the deployment target is
  at least Catalina.
* Bugfix for [#77](https://github.com/gulrak/filesystem/issues/77), the `directory_iterator`
  and the `recursive_directory_iterator` had an issue with the `skip_permission_denied`
  option, that leads to the inability to skip SIP protected folders on macOS.
* Enhancement for [#76](https://github.com/gulrak/filesystem/issues/76), `_MSVC_LANG` is
  now used when available, additionally to `__cplusplus`, in the helping headers to
  allow them to work even when `/Zc:__cplusplus` is not used.
* Bugfix for [#75](https://github.com/gulrak/filesystem/issues/75), NTFS reparse points
  to mapped volumes where handled incorrect, leading to `false` on `fs::exists` or
  not-found-errors on `fs::status`. Namespaced paths are not filtered anymore.

### [v1.3.6](https://github.com/gulrak/filesystem/releases/tag/v1.3.6)

* Pull request [#74](https://github.com/gulrak/filesystem/pull/74), on Windows symlink
  evaluation used the wrong reparse struct information and was not handling the case
  of relative paths well, thanks for the contribution.
* Refactoring for [#73](https://github.com/gulrak/filesystem/issues/73), enhanced performance
  in path handling. the changes lead to much fewer path/string creations or copies, speeding
  up large directory iteration or operations on many path instances.
* Bugfix for [#72](https://github.com/gulrak/filesystem/issues/72), the `TestAllocator` in
  `filesystem_test.cpp` was completed to fulfill the requirements to build on CentOS 7 with
  `devtoolset-9`. CentOS 7 and CentOS 8 are now part of the CI builds.
* Bugfix for [#70](https://github.com/gulrak/filesystem/issues/70), root names are now case
  insensitive on Windows. This fix also adds the new behavior switch `LWG_2936_BEHAVIOUR`
  that allows to enable post C++17 `fs::path::compare` behavior, where the comparison is as
  if it was an element wise path comparison as described in
  [LWG 2936](https://cplusplus.github.io/LWG/issue2936) and C++20 `[fs.path.compare]`.
  It is default off in v1.3.6 and will be default starting from v1.4.0 as it changes ordering.

### [v1.3.4](https://github.com/gulrak/filesystem/releases/tag/v1.3.4)

* Pull request [#69](https://github.com/gulrak/filesystem/pull/69), use `wchar_t` versions of
  `std::fstream` from `ghc::filesystem::fstream` wrappers on Windows if using GCC with libc++.
* Bugfix for [#68](https://github.com/gulrak/filesystem/issues/68), better handling of
  permission issues for directory iterators when using `fs::directory_options::skip_permission_denied`
  and initial support for compilation with emscripten.
* Refactoring for [#66](https://github.com/gulrak/filesystem/issues/63), unneeded shared_ptr guards
  where removed and the file handles closed where needed to avoid unnecessary allocations.
* Bugfix for [#63](https://github.com/gulrak/filesystem/issues/63), fixed issues on Windows
  with clang++ and C++17.
* Pull request [#62](https://github.com/gulrak/filesystem/pull/62), various fixes for
  better Android support, thanks for the PR
* Pull request [#61](https://github.com/gulrak/filesystem/pull/61), `ghc::filesystem` now
  supports use in projects with disabled exceptions. API signatures using exceptions for
  error handling are not available in this mode, thanks for the PR (this resolves
  [#60](https://github.com/gulrak/filesystem/issues/60) and
  [#43](https://github.com/gulrak/filesystem/issues/43))

### [v1.3.2](https://github.com/gulrak/filesystem/releases/tag/v1.3.2)

* Bugfix for [#58](https://github.com/gulrak/filesystem/issues/58), on MinGW the
  compilation could fail with an error about an undefined `ERROR_FILE_TOO_LARGE`
  constant.
* Bugfix for [#56](https://github.com/gulrak/filesystem/issues/58), `fs::lexically_relative`
  didn't ignore trailing slash on the base parameter, thanks for PR
  [#57](https://github.com/gulrak/filesystem/pull/57).
* Bugfix for [#55](https://github.com/gulrak/filesystem/issues/55), `fs::create_directories`
  returned `true` when nothing needed to be created, because the directory already existed.
* Bugfix for [#54](https://github.com/gulrak/filesystem/issues/54), `error_code`
  was not reset, if cached result was returned.
* Pull request [#53](https://github.com/gulrak/filesystem/pull/53), fix for wrong
  handling of leading whitespace when reading `fs::path` from a stream.
* Pull request [#52](https://github.com/gulrak/filesystem/pull/52), an ARM Linux
  target is now part of the CI infrastructure with the service of Drone CI.
* Pull request [#51](https://github.com/gulrak/filesystem/pull/51), FreeBSD is now
  part of the CI infrastructure with the service of Cirrus CI.
* Pull request [#50](https://github.com/gulrak/filesystem/pull/50), adaptive cast to
  `timespec` fields to avoid warnings.

### [v1.3.0](https://github.com/gulrak/filesystem/releases/tag/v1.3.0)

* **Important: `ghc::filesystem` is re-licensed from BSD-3-Clause to MIT license.** (see
  [#47](https://github.com/gulrak/filesystem/issues/47))
* Pull request [#46](https://github.com/gulrak/filesystem/pull/46), suppresses
  unused parameter warning on Android.
* Bugfix for [#44](https://github.com/gulrak/filesystem/issues/44), fixes
  for warnings from newer Xcode versions.

### [v1.2.10](https://github.com/gulrak/filesystem/releases/tag/v1.2.10)

* The Visual Studio 2019 compiler, GCC 9.2 and Clang 9.0 where added to the
  CI configuration.
* Bugfix for [#41](https://github.com/gulrak/filesystem/issues/41), `fs::rename`
  on Windows didn't replace an existing regular file as required by the standard,
  but gave an error. New tests and a fix as provided in the issue was implemented.
* Bugfix for [#39](https://github.com/gulrak/filesystem/issues/39), for the
  forwarding use via `fs_fwd.hpp` or `fs_std_fwd.hpp` there was a use of
  `DWORD` in the forwarding part leading to an error if `Windows.h` was not
  included before the header. The tests were changed to give an error in that
  case too and the useage of `DWORD` was removed.
* Bugfix for [#38](https://github.com/gulrak/filesystem/issues/38), casting the
  return value of `GetProcAddress` gave a warning with `-Wcast-function-type`
  on MSYS2 and MinGW GCC 9 builds.

### [v1.2.8](https://github.com/gulrak/filesystem/releases/tag/v1.2.8)

* Pull request [#30](https://github.com/gulrak/filesystem/pull/30), the
  `CMakeLists.txt` will automatically exclude building examples and tests when
  used as submodule, the configuration options now use a prefixed name to
  reduce risk of conflicts.
* Pull request [#24](https://github.com/gulrak/filesystem/pull/24), install
  target now creates a `ghcFilesystemConfig.cmake` in
  `${CMAKE_INSTALL_LIBDIR}/cmake/ghcFilesystem` for `find_package` that
  exports a target as `ghcFilesystem::ghc_filesystem`.
* Pull request [#31](https://github.com/gulrak/filesystem/pull/31), fixes
  `error: redundant redeclaration of 'constexpr' static data member` deprecation
  warning in C++17 mode.
* Pull request [#32](https://github.com/gulrak/filesystem/pull/32), fixes
  old-style-cast warnings.
* Pull request [#34](https://github.com/gulrak/filesystem/pull/34), fixes
  [TOCTOU](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use) situation
  on `fs::create_directories`, thanks for the PR!
* Feature [#35](https://github.com/gulrak/filesystem/issues/35), new CMake
  option to add an install target `GHC_FILESYSTEM_WITH_INSTALL` that is
  defaulted to OFF if `ghc::filesystem` is used via `add_subdirectory`.
* Bugfix for [#33](https://github.com/gulrak/filesystem/issues/33), fixes
  an issue with `fs::path::lexically_normal()` that leaves a trailing separator
  in case of a resulting path ending with `..` as last element.
* Bugfix for [#36](https://github.com/gulrak/filesystem/issues/36), warnings
  on Xcode 11.2 due to unhelpful references in path element iteration.

### [v1.2.6](https://github.com/gulrak/filesystem/releases/tag/v1.2.6)

* Pull request [#23](https://github.com/gulrak/filesystem/pull/23), tests and
  examples can now be disabled in CMake via setting `BUILD_TESTING` and
  `BUILD_EXAMPLES` to `NO`, `OFF` or `FALSE`.
* Pull request [#25](https://github.com/gulrak/filesystem/pull/25),
  missing specialization for construction from `std::string_view` when
  available was added.
* Additional test case when `std::string_view` is available.
* Bugfix for [#27](https://github.com/gulrak/filesystem/issues/27), the
  `fs::path::preferred_separator` declaration was not compiling on pre
  C++17 compilers and no test accessed it, to show the problem. Fixed
  it to an construction C++11 compiler should accept and added a test that
  is successful on all combinations tested.
* Bugfix for [#29](https://github.com/gulrak/filesystem/issues/29), stricter
  warning settings where chosen and resulting warnings where fixed.

### [v1.2.4](https://github.com/gulrak/filesystem/releases/tag/v1.2.4)

* Enabled stronger warning switches and resulting fixed issues on GCC and MinGW
* Bugfix for #22, the `fs::copy_options` where not forwarded from `fs::copy` to
  `fs::copy_file` in one of the cases.

### [v1.2.2](https://github.com/gulrak/filesystem/releases/tag/v1.2.2)

* Fix for ([#21](https://github.com/gulrak/filesystem/pull/21)), when compiling
  on Alpine Linux with musl instead of glibc, the wrong `strerror_r` signature
  was expected. The complex preprocessor define mix was dropped in favor of
  the usual dispatch by overloading a unifying wrapper.

### [v1.2.0](https://github.com/gulrak/filesystem/releases/tag/v1.2.0)

* Added MinGW 32/64 and Visual Studio 2015 builds to the CI configuration.
* Fixed additional compilation issues on MinGW.
* Pull request ([#13](https://github.com/gulrak/filesystem/pull/13)), set
  minimum required CMake version to 3.7.2 (as in Debian 8).
* Pull request ([#14](https://github.com/gulrak/filesystem/pull/14)), added
  support for a make install target.
* Bugfix for ([#15](https://github.com/gulrak/filesystem/issues/15)), the
  forward/impl way of using `ghc::filesystem` missed a `<vector>` include
  in the windows case.
* Bugfix for ([#16](https://github.com/gulrak/filesystem/issues/16)),
  VS2019 didn't like the old size dispatching in the utf8 decoder, so it
  was changed to a sfinae based approach.
* New feature ([#17](https://github.com/gulrak/filesystem/issues/17)), optional
  support for standard conforming `wchar_t/std::wstring` interface when
  compiling on Windows with defined `GHC_WIN_WSTRING_STRING_TYPE`, this is
  default when using the `ghc/fs_std*.hpp` header, to enhance compatibility.
* New feature ([#18](https://github.com/gulrak/filesystem/issues/18)), optional
  filesystem exceptions/errors on Unicode errors with defined
  `GHC_RAISE_UNICODE_ERRORS` (instead of replacing invalid code points or
  UTF-8 encoding errors with the replacement character `U+FFFD`).
* Pull request ([#20](https://github.com/gulrak/filesystem/pull/20)), fix for
  file handle leak in `fs::copy_file`.
* Coverage now checked in CI (~95% line coverage).

### [v1.1.4](https://github.com/gulrak/filesystem/releases/tag/v1.1.4)

* Additional Bugfix for ([#12](https://github.com/gulrak/filesystem/issues/12)),
  error in old unified `readdir/readdir_r` code of `fs::directory_iterator`;
  as `readdir_r` is now deprecated, I decided to drop it and the resulting
  code is much easier, shorter and due to more refactoring faster
* Fix for crashing unit tests against MSVC C++17 `std::filesystem`
* Travis-CI now additionally test with Xcode 10.2 on macOS
* Some minor refactorings

### [v1.1.2](https://github.com/gulrak/filesystem/releases/tag/v1.1.2)

* Bugfix for ([#11](https://github.com/gulrak/filesystem/issues/11)),
  `fs::path::lexically_normal()` had some issues with `".."`-sequences.
* Bugfix for ([#12](https://github.com/gulrak/filesystem/issues/12)),
  `fs::recursive_directory_iterator` could run into endless loops,
  the methods depth() and pop() had issues and the copy behavior and
  `input_iterator_tag` conformance was broken, added tests
* Restructured some CMake code into a macro to ease the support for
  C++17 `std::filesystem` builds of tests and examples for interoperability
  checks.
* Some fixes on Windows tests to ease interoperability test runs.
* Reduced noise on `fs::weakly_canonical()` tests against `std::fs`
* Added simple `du` example showing the `recursive_directory_iterator`
  used to add the sizes of files in a directory tree.
* Added error checking in `fs::file_time_type` test helpers
* `fs::copy()` now conforms LWG #2682, disallowing the use of
  `copy_option::create_symlinks' to be used on directories

### [v1.1.0](https://github.com/gulrak/filesystem/releases/tag/v1.1.0)

* Restructuring of the project directory. The header files are now using
  `hpp` as extension to be marked as c++ and they where moved to
  `include/ghc/` to be able to include by `<ghc/filesystem.hpp>` as the
  former include name might have been to generic and conflict with other
  files.
* Better CMake support: `ghc::filesystem` now can be used as a submodul
  and added with `add_subdirectory` and will export itself as `ghc_filesystem`
  target. To use it, only `target_link_libraries(your-target ghc_filesystem)`
  is needed and the include directories will be set so `#include <ghc/filesystem.hpp>`
  will be a valid directive.
  Still you can simply only add the header file to you project and include it
  from there.
* Enhancement ([#10](https://github.com/gulrak/filesystem/issues/10)),
  support for separation of implementation and forwarded api: Two
  additional simple includes are added, that can be used to forward
  `ghc::filesystem` declarations (`fs_fwd.hpp`) and to wrap the
  implementation into a single cpp (`fs_impl.hpp`)
* The `std::basic_string_view` variants of the `fs::path` api are
  now supported when compiling with C++17.
* Added CI integration for Travis-CI and Appveyor.
* Fixed MinGW compilation issues.
* Added long filename support for Windows.

### [v1.0.10](https://github.com/gulrak/filesystem/releases/tag/v1.0.10)

* Bugfix for ([#9](https://github.com/gulrak/filesystem/issues/9)), added
  missing return statement to `ghc::filesystem::path::generic_string()`
* Added checks to hopefully better compile against Android NDK. There where
  no tests run yet, so feedback is needed to actually call this supported.
* `filesystem.h` was renamed `filesystem.hpp` to better reflect that it is
  a c++ language header.

### [v1.0.8](https://github.com/gulrak/filesystem/releases/tag/v1.0.8)

* Bugfix for ([#6](https://github.com/gulrak/filesystem/issues/6)), where
  `ghc::filesystem::remove()` and `ghc::filesystem::remove_all()` both are
  now able to remove a single file and both will not raise an error if the
  path doesn't exist.
* Merged pull request ([#7](https://github.com/gulrak/filesystem/pull/7)),
  a typo leading to setting error code instead of comparing it in
  `ghc::filesystem::remove()` under Windows.
* Bugfix for (([#8](https://github.com/gulrak/filesystem/issues/8)), the
  Windows version of `ghc::filesystem::directory_iterator` now releases
  resources when reaching `end()` like the POSIX one does.


### [v1.0.6](https://github.com/gulrak/filesystem/releases/tag/v1.0.6)

* Bugfix for ([#4](https://github.com/gulrak/filesystem/issues/4)), missing error_code
  propagation in `ghc::filesystem::copy()` and `ghc::filesystem::remove_all` fixed.
* Bugfix for ([#5](https://github.com/gulrak/filesystem/issues/5)), added missing std
  namespace in `ghc::filesystem::recursive_directory_iterator::difference_type`.

### [v1.0.4](https://github.com/gulrak/filesystem/releases/tag/v1.0.4)

* Bugfix for ([#3](https://github.com/gulrak/filesystem/issues/3)), fixed missing inlines
  and added test to ensure including into multiple implementation files works as expected.
* Building tests with `-Wall -Wextra -Werror` and fixed resulting issues.

### [v1.0.2](https://github.com/gulrak/filesystem/releases/tag/v1.0.2)

* Updated catch2 to v2.4.0.
* Refactored `fs.op.permissions` test to work with all tested `std::filesystem`
  implementations (gcc, clang, msvc++).
* Added helper class `ghc::filesystem::u8arguments` as `argv` converter, to
  help follow the UTF-8 path on windows. Simply instantiate it with `argc` and
  `argv` and it will fetch the Unicode version of the command line and convert
  it to UTF-8. The destructor reverts the change.
* Added `examples` folder with hopefully some usefull example usage. Examples are
  tested (and build) with `ghc::filesystem` and C++17 `std::filesystem` when
  available.
* Starting with this version, only even patch level versions will be tagged and
  odd patch levels mark in-between non-stable wip states.
* Tests can now also be run against MS version of `std::filesystem` for comparison.
* Added missing `fstream` include.
* Removed non-conforming C99 `timespec`/`timeval` usage.
* Fixed some integer type mismatches that could lead to warnings.
* Fixed `chrono` conversion issues in test and example on clang 7.0.0.

### [v1.0.1](https://github.com/gulrak/filesystem/releases/tag/v1.0.1)

* Bugfix: `ghc::filesystem::canonical` now sees empty path as non-existant and reports
  an error. Due to this `ghc::filesystem::weakly_canonical` now returns relative
  paths for non-existant argument paths. ([#1](https://github.com/gulrak/filesystem/issues/1))
* Bugfix: `ghc::filesystem::remove_all` now also counts directories removed ([#2](https://github.com/gulrak/filesystem/issues/2))
* Bugfix: `recursive_directory_iterator` tests didn't respect equality domain issues
  and dereferencapable constraints, leading to fails on `std::filesystem` tests.
* Bugfix: Some `noexcept` tagged methods and functions could indirectly throw exceptions
  due to UFT-8 decoding issues.
* `std_filesystem_test` is now also generated if LLVM/clang 7.0.0 is found.


### [v1.0.0](https://github.com/gulrak/filesystem/releases/tag/v1.0.0)

This was the first public release version. It implements the full range of
C++17 `std::filesystem`, as far as possible without other C++17 dependencies.

