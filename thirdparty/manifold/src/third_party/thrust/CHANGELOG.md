# Changelog

## Thrust 1.17.0

### Summary

Thrust 1.17.0 is the final minor release of the 1.X series. This release
provides GDB pretty-printers for device vectors/references, a new `unique_count`
algorithm, and an easier way to create tagged Thrust iterators. Several
documentation fixes are included, which can be found on the new Thrust
documentation site at https://nvidia.github.io/thrust. We'll be migrating
existing documentation sources to this new location over the next few months.

### New Features

- NVIDIA/thrust#1586: Add new `thrust::make_tagged_iterator` convenience
  function. Thanks to @karthikeyann for this contribution.
- NVIDIA/thrust#1619: Add `unique_count` algorithm. Thanks to @upsj for this
  contribution.
- NVIDIA/thrust#1631: Add GDB pretty-printers for device vectors/references
  to `scripts/gdb-pretty-printers.py`. Thanks to @upsj for this contribution.

### Bug Fixes

- NVIDIA/thrust#1671: Fixed `reduce_by_key` when called with 2^31 elements.

### Other Enhancements

- NVIDIA/thrust#1512: Use CUB to implement `adjacent_difference`.
- NVIDIA/thrust#1555: Use CUB to implement `scan_by_key`.
- NVIDIA/thrust#1611: Add new doxybook-based Thrust documentation
  at https://nvidia.github.io/thrust.
- NVIDIA/thrust#1639: Fixed broken link in documentation. Thanks to @jrhemstad
  for this contribution.
- NVIDIA/thrust#1644: Increase contrast of search input text in new doc site.
  Thanks to @bdice for this contribution.
- NVIDIA/thrust#1647: Add `__forceinline__` annotations to a functor wrapper.
  Thanks to @mkuron for this contribution.
- NVIDIA/thrust#1660: Fixed typo in documentation example for
  `permutation_iterator`.
- NVIDIA/thrust#1669: Add a new `explicit_cuda_stream.cu` example that shows how
  to use explicit CUDA streams and `par`/`par_nosync` execution policies.

## Thrust 1.16.0

### Summary

Thrust 1.16.0 provides a new “nosync” hint for the CUDA backend, as well as
numerous bugfixes and stability improvements.

#### New `thrust::cuda::par_nosync` Execution Policy

Most of Thrust’s parallel algorithms are fully synchronous and will block the
calling CPU thread until all work is completed. This design avoids many pitfalls
associated with asynchronous GPU programming, resulting in simpler and
less-error prone usage for new CUDA developers. Unfortunately, this improvement
in user experience comes at a performance cost that often frustrates more
experienced CUDA programmers.

Prior to this release, the only synchronous-to-asynchronous migration path for
existing Thrust codebases involved significant refactoring, replacing calls
to `thrust` algorithms with a limited set of `future`-based `thrust::async`
algorithms or lower-level CUB kernels. The new `thrust::cuda::par_nosync`
execution policy provides a new, less-invasive entry point for asynchronous
computation.

`par_nosync` is a hint to the Thrust execution engine that any non-essential
internal synchronizations should be skipped and that an explicit synchronization
will be performed by the caller before accessing results.

While some Thrust algorithms require internal synchronization to safely compute
their results, many do not. For example, multiple `thrust::for_each` invocations
can be launched without waiting for earlier calls to complete:

```cpp
// Queue three `for_each` kernels:
thrust::for_each(thrust::cuda::par_nosync, vec1.begin(), vec1.end(), Op{});
thrust::for_each(thrust::cuda::par_nosync, vec2.begin(), vec2.end(), Op{});
thrust::for_each(thrust::cuda::par_nosync, vec3.begin(), vec3.end(), Op{});

// Do other work while kernels execute:
do_something();

// Must explictly synchronize before accessing `for_each` results:
cudaDeviceSynchronize();
```

Thanks to @fkallen for this contribution.

### Deprecation Notices

#### CUDA Dynamic Parallelism Support

**A future version of Thrust will remove support for CUDA Dynamic Parallelism
(CDP).**

This will only affect calls to Thrust algorithms made from CUDA device-side code
that currently launches a kernel; such calls will instead execute sequentially
on the calling GPU thread instead of launching a device-wide kernel.

### Breaking Changes

- Thrust 1.14.0 included a change that aliased the `cub` namespace
  to `thrust::cub`. This has caused issues with ambiguous namespaces for
  projects that declare `using namespace thrust;` from the global namespace. We
  recommend against this practice.
- NVIDIA/thrust#1572: Removed several unnecessary header includes. Downstream
  projects may need to update their includes if they were relying on this
  behavior.

### New Features

- NVIDIA/thrust#1568: Add `thrust::cuda::par_nosync` policy. Thanks to @fkallen
  for this contribution.

### Enhancements

- NVIDIA/thrust#1511: Use CUB’s new `DeviceMergeSort` API and remove Thrust’s
  internal implementation.
- NVIDIA/thrust#1566: Improved performance of `thrust::shuffle`. Thanks to
  @djns99 for this contribution.
- NVIDIA/thrust#1584: Support user-defined `CMAKE_INSTALL_INCLUDEDIR` values in
  Thrust’s CMake install rules. Thanks to @robertmaynard for this contribution.

### Bug Fixes

- NVIDIA/thrust#1496: Fix some issues affecting `icc` builds.
- NVIDIA/thrust#1552: Fix some collisions with the `min`/`max`  macros defined
  in `windows.h`.
- NVIDIA/thrust#1582: Fix issue with function type alias on 32-bit MSVC builds.
- NVIDIA/thrust#1591: Workaround issue affecting compilation with `nvc++`.
- NVIDIA/thrust#1597: Fix some collisions with the `small` macro defined
  in `windows.h`.
- NVIDIA/thrust#1599, NVIDIA/thrust#1603: Fix some issues with version handling
  in Thrust’s CMake packages.
- NVIDIA/thrust#1614: Clarify that scan algorithm results are non-deterministic
  for pseudo-associative operators (e.g. floating-point addition).

## Thrust 1.15.0

### Summary

Thrust 1.15.0 provides numerous bugfixes, including non-numeric
`thrust::sequence` support, several MSVC-related compilation fixes, fewer
conversion warnings, `counting_iterator` initialization, and documentation
updates.

### Deprecation Notices

**A future version of Thrust will remove support for CUDA Dynamic Parallelism
(CDP).**

This will only affect calls to Thrust algorithms made from CUDA device-side code
that currently launches a kernel; such calls will instead execute sequentially
on the calling GPU thread instead of launching a device-wide kernel.

### Bug Fixes

- NVIDIA/thrust#1507: Allow `thrust::sequence` to work with non-numeric types.
  Thanks to Ben Jude (@bjude) for this contribution.
- NVIDIA/thrust#1509: Avoid macro collision when calling `max()` on MSVC. Thanks
  to Thomas (@tomintheshell) for this contribution.
- NVIDIA/thrust#1514: Initialize all members in `counting_iterator`'s default
  constructor.
- NVIDIA/thrust#1518: Fix `std::allocator_traits` on MSVC + C++17.
- NVIDIA/thrust#1530: Fix several `-Wconversion` warnings. Thanks to Matt
  Stack (@matt-stack) for this contribution.
- NVIDIA/thrust#1539: Fixed typo in `thrust::for_each` documentation. Thanks to
  Salman (@untamedImpala) for this contribution.
- NVIDIA/thrust#1548: Avoid name collision with `B0` macro in termios.h system
  header. Thanks to Philip Deegan (@PhilipDeegan) for this contribution.

## Thrust 1.14.0 (NVIDIA HPC SDK 21.9)

Thrust 1.14.0 is a major release accompanying the NVIDIA HPC SDK 21.9.

This release adds the ability to wrap the `thrust::` namespace in an external
namespace, providing a workaround for a variety of shared library linking
issues. Thrust also learned to detect when CUB's symbols are in a wrapped
namespace and properly import them. To enable this feature, use
`#define THRUST_CUB_WRAPPED_NAMESPACE foo` to wrap both Thrust and CUB in the
`foo::` namespace. See `thrust/detail/config/namespace.h` for details and more
namespace options.

Several bugfixes are also included: The `tuple_size` and `tuple_element` helpers
now support cv-qualified types. `scan_by_key` uses less memory.
`thrust::iterator_traits` is better integrated with `std::iterator_traits`.
See below for more details and references.

### Breaking Changes

- Thrust 1.14.0 included a change that aliased the `cub` namespace
  to `thrust::cub`. This has caused issues with ambiguous namespaces for
  projects that declare `using namespace thrust;` from the global namespace. We
  recommend against this practice.

### New Features

- NVIDIA/thrust#1464: Add preprocessor hooks that allow `thrust::` to be wrapped
  in an external namespace, and support cases when CUB is wrapped in an external
  namespace.

### Bug Fixes

- NVIDIA/thrust#1457: Support cv-qualified types in `thrust::tuple_size` and
  `thrust::tuple_element`. Thanks to Jake Hemstad for this contribution.
- NVIDIA/thrust#1471: Fixed excessive memory allocation in `scan_by_key`. Thanks
  to Lilo Huang for this contribution.
- NVIDIA/thrust#1476: Removed dead code from the `expand` example. Thanks to
  Lilo Huang for this contribution.
- NVIDIA/thrust#1488: Fixed the path to the installed CUB headers in the CMake
  `find_package` configuration files.
- NVIDIA/thrust#1491: Fallback to `std::iterator_traits` when no
  `thrust::iterator_traits` specialization exists for an iterator type. Thanks
  to Divye Gala for this contribution.

## Thrust 1.13.1 (CUDA Toolkit 11.5)

Thrust 1.13.1 is a minor release accompanying the CUDA Toolkit 11.5.

This release provides a new hook for embedding the `thrust::` namespace inside a
custom namespace. This is intended to work around various issues related to
linking multiple shared libraries that use Thrust. The existing `CUB_NS_PREFIX`
and `CUB_NS_POSTFIX` macros already provided this capability for CUB; this
update provides a simpler mechanism that is extended to and integrated with
Thrust. Simply define `THRUST_CUB_WRAPPED_NAMESPACE` to a namespace name, and
both `thrust::` and `cub::` will be placed inside the new namespace. Using
different wrapped namespaces for each shared library will prevent issues like
those reported in NVIDIA/thrust#1401.

### New Features

- NVIDIA/thrust#1464: Add `THRUST_CUB_WRAPPED_NAMESPACE` hooks.

### Bug Fixes

- NVIDIA/thrust#1488: Fix path to installed CUB in Thrust's CMake config files.

## Thrust 1.13.0 (NVIDIA HPC SDK 21.7)

Thrust 1.13.0 is the major release accompanying the NVIDIA HPC SDK 21.7 release.
Notable changes include `bfloat16` radix sort support (via `thrust::sort`) and
  memory handling fixes in the `reserve` method of Thrust's vectors.
The `CONTRIBUTING.md` file has been expanded to include instructions for
  building CUB as a component of Thrust, and API documentation now refers to
  [cppreference](https://cppreference.com) instead of SGI's old STL reference.

### Breaking Changes

- NVIDIA/thrust#1459: Remove deprecated aliases `thrust::host_space_tag` and
  `thrust::device_space_tag`. Use the equivalent `thrust::host_system_tag` and
  `thrust::device_system_tag` instead.

### New Features

- NVIDIA/cub#306: Add radix-sort support for `bfloat16` in `thrust::sort`.
  Thanks to Xiang Gao (@zasdfgbnm) for this contribution.
- NVIDIA/thrust#1423: `thrust::transform_iterator` now supports non-copyable
  types. Thanks to Jake Hemstad (@jrhemstad) for this contribution.
- NVIDIA/thrust#1459: Introduce a new `THRUST_IGNORE_DEPRECATED_API` macro that
  disables deprecation warnings on Thrust and CUB APIs.

### Bug Fixes

- NVIDIA/cub#277: Fixed sanitizer warnings when `thrust::sort` calls
  into `cub::DeviceRadixSort`. Thanks to Andy Adinets (@canonizer) for this
  contribution.
- NVIDIA/thrust#1442: Reduce extraneous comparisons in `thrust::sort`'s merge
  sort implementation.
- NVIDIA/thrust#1447: Fix memory leak and avoid overallocation when
  calling `reserve` on Thrust's vector containers. Thanks to Kai Germaschewski
  (@germasch) for this contribution.

### Other Enhancements

- NVIDIA/thrust#1405: Update links to standard C++ documentations from sgi to
  cppreference. Thanks to Muhammad Adeel Hussain (@AdeilH) for this
  contribution.
- NVIDIA/thrust#1432: Updated build instructions in `CONTRIBUTING.md` to include
  details on building CUB's test suite as part of Thrust.

## Thrust 1.12.1 (CUDA Toolkit 11.4)

Thrust 1.12.1 is a trivial patch release that slightly changes the phrasing of
a deprecation message.

## Thrust 1.12.0 (NVIDIA HPC SDK 21.3)

Thrust 1.12.0 is the major release accompanying the NVIDIA HPC SDK 21.3
  and the CUDA Toolkit 11.4.
It includes a new `thrust::universal_vector`, which holds data that is
  accessible from both host and device. This allows users to easily leverage
  CUDA's unified memory with Thrust.
New asynchronous `thrust::async:exclusive_scan` and `inclusive_scan` algorithms
  have been added, and the synchronous versions of these have been updated to
  use `cub::DeviceScan` directly.
CUB radix sort for floating point types is now stable when both +0.0 and -0.0
  are present in the input. This affects some usages of `thrust::sort` and
  `thrust::stable_sort`.
Many compilation warnings and subtle overflow bugs were fixed in the device
  algorithms, including a long-standing bug that returned invalid temporary
  storage requirements when `num_items` was close to (but not
  exceeding) `INT32_MAX`.
This release deprecates support for Clang < 7.0 and MSVC < 2019 (aka
  19.20/16.0/14.20).

### Breaking Changes

- NVIDIA/thrust#1372: Deprecate Clang < 7 and MSVC < 2019.
- NVIDIA/thrust#1376: Standardize `thrust::scan_by_key` functors / accumulator
    types.
  This may change the results from `scan_by_key` when input, output, and
    initial value types are not the same type.

### New Features

- NVIDIA/thrust#1251: Add two new `thrust::async::` algorithms: `inclusive_scan`
    and `exclusive_scan`.
- NVIDIA/thrust#1334: Add `thrust::universal_vector`, `universal_ptr`,
    and `universal_allocator`.

### Bug Fixes

- NVIDIA/thrust#1347: Qualify calls to `make_reverse_iterator`.
- NVIDIA/thrust#1359: Enable stricter warning flags. This fixes several
  outstanding issues:
  - NVIDIA/cub#221: Overflow in `temp_storage_bytes` when `num_items` close to
      (but not over) `INT32_MAX`.
  - NVIDIA/cub#228: CUB uses non-standard C++ extensions that break strict
      compilers.
  - NVIDIA/cub#257: Warning when compiling `GridEvenShare` with unsigned
      offsets.
  - NVIDIA/thrust#974: Conversion warnings in `thrust::transform_reduce`.
  - NVIDIA/thrust#1091: Conversion warnings in `thrust::counting_iterator`.
- NVIDIA/thrust#1373: Fix compilation error when a standard library type is
    wrapped in `thrust::optional`.
  Thanks to Vukasin Milovanovic for this contribution.
- NVIDIA/thrust#1388: Fix `signbit(double)` implementation on MSVC.
- NVIDIA/thrust#1389: Support building Thrust tests without CUDA enabled.

### Other Enhancements

- NVIDIA/thrust#1304: Use `cub::DeviceScan` to implement
    `thrust::exclusive_scan` and `thrust::inclusive_scan`.
- NVIDIA/thrust#1362, NVIDIA/thrust#1370: Update smoke test naming.
- NVIDIA/thrust#1380: Fix typos in `set_operation` documentation.
    Thanks to Hongyu Cai for this contribution.
- NVIDIA/thrust#1383: Include FreeBSD license in LICENSE.md for
  `thrust::complex` implementation.
- NVIDIA/thrust#1384: Add missing precondition to `thrust::gather`
    documentation.

## Thrust 1.11.0 (CUDA Toolkit 11.3)

Thrust 1.11.0 is a major release providing bugfixes and performance
  enhancements.
It includes a new sort algorithm that provides up to 2x more performance
  from `thrust::sort` when used with certain key types and hardware.
The new `thrust::shuffle` algorithm has been tweaked to improve the randomness
  of the output.
Our CMake package and build system continue to see improvements with
  better `add_subdirectory` support, installation rules, status messages, and
  other features that make Thrust easier to use from CMake projects.
The release includes several other bugfixes and modernizations, and received
  updates from 12 contributors.

### New Features

- NVIDIA/cub#204: New implementation for `thrust::sort` on CUDA when using
    32/64-bit numeric keys on Pascal and up (SM60+).
  This improved radix sort algorithm provides up to 2x more performance.
  Thanks for Andy Adinets for this contribution.
- NVIDIA/thrust#1310, NVIDIA/thrust#1312: Various tuple-related APIs have been
    updated to use variadic templates.
  Thanks for Andrew Corrigan for these contributions.
- NVIDIA/thrust#1297: Optionally add install rules when included with
    CMake's `add_subdirectory`.
  Thanks to Kai Germaschewski for this contribution.

### Bug Fixes

- NVIDIA/thrust#1309: Fix `thrust::shuffle` to produce better quality random
    distributions.
  Thanks to Rory Mitchell and Daniel Stokes for this contribution.
- NVIDIA/thrust#1337: Fix compile-time regression in `transform_inclusive_scan`
    and `transform_exclusive_scan`.
- NVIDIA/thrust#1306: Fix binary search `middle` calculation to avoid overflows.
    Thanks to Richard Barnes for this contribution.
- NVIDIA/thrust#1314: Use `size_t` for the index type parameter
    in `thrust::tuple_element`.
  Thanks to Andrew Corrigan for this contribution.
- NVIDIA/thrust#1329: Fix runtime error when copying an empty
    `thrust::device_vector` in MSVC Debug builds.
  Thanks to Ben Jude for this contribution.
- NVIDIA/thrust#1323: Fix and add test for cmake package install rules.
  Thanks for Keith Kraus and Kai Germaschewski for testing and discussion.
- NVIDIA/thrust#1338: Fix GCC version checks in `thrust::detail::is_pod`
    implementation.
  Thanks to Anatoliy Tomilov for this contribution.
- NVIDIA/thrust#1289: Partial fixes for Clang 10 as host compiler.
  Filed an NVCC bug that will be fixed in a future version of the CUDA Toolkit
    (NVBug 3136307).
- NVIDIA/thrust#1272: Fix ambiguous `iter_swap` call when
    using `thrust::partition` with STL containers.
  Thanks to Isaac Deutsch for this contribution.
- NVIDIA/thrust#1281: Update our bundled `FindTBB.cmake` module to support
    latest MSVC.
- NVIDIA/thrust#1298: Use semantic versioning rules for our CMake package's
    compatibility checks.
  Thanks to Kai Germaschewski for this contribution.
- NVIDIA/thrust#1300: Use `FindPackageHandleStandardArgs` to print standard
    status messages when our CMake package is found.
  Thanks to Kai Germaschewski for this contribution.
- NVIDIA/thrust#1320: Use feature-testing instead of a language dialect check
    for `thrust::remove_cvref`.
  Thanks to Andrew Corrigan for this contribution.
- NVIDIA/thrust#1319: Suppress GPU deprecation warnings.

### Other Enhancements

- NVIDIA/cub#213: Removed some tuning policies for unsupported hardware (<SM35).
- References to the old Github repository and branch names were updated.
  - Github's `thrust/cub` repository is now `NVIDIA/cub`.
  - Development has moved from the `master` branch to the `main` branch.

## Thrust 1.10.0 (NVIDIA HPC SDK 20.9, CUDA Toolkit 11.2)

Thrust 1.10.0 is the major release accompanying the NVIDIA HPC SDK 20.9 release
  and the CUDA Toolkit 11.2 release.
It drops support for C++03, GCC < 5, Clang < 6, and MSVC < 2017.
It also overhauls CMake support.
Finally, we now have a Code of Conduct for contributors:
https://github.com/NVIDIA/thrust/blob/main/CODE_OF_CONDUCT.md

### Breaking Changes

- C++03 is no longer supported.
- GCC < 5, Clang < 6, and MSVC < 2017 are no longer supported.
- C++11 is deprecated.
  Using this dialect will generate a compile-time warning.
  These warnings can be suppressed by defining
    `THRUST_IGNORE_DEPRECATED_CPP_DIALECT` or `THRUST_IGNORE_DEPRECATED_CPP_11`.
  Suppression is only a short term solution.
  We will be dropping support for C++11 in the near future.
- Asynchronous algorithms now require C++14.
- CMake < 3.15 is no longer supported.
- The default branch on GitHub is now called `main`.
- Allocator and vector classes have been replaced with alias templates.

### New Features

- NVIDIA/thrust#1159: CMake multi-config support, which allows multiple
    combinations of host and device systems to be built and tested at once.
  More details can be found here: https://github.com/NVIDIA/thrust/blob/main/CONTRIBUTING.md#multi-config-cmake-options
- CMake refactoring:
  - Added install targets to CMake builds.
  - Added support for CUB tests and examples.
  - Thrust can be added to another CMake project by calling `add_subdirectory`
      with the Thrust source root (see NVIDIA/thrust#976).
    An example can be found here:
      https://github.com/NVIDIA/thrust/blob/main/examples/cmake/add_subdir/CMakeLists.txt
  - CMake < 3.15 is no longer supported.
  - Dialects are now configured through target properties.
    A new `THRUST_CPP_DIALECT` option has been added for single config mode.
    Logic that modified `CMAKE_CXX_STANDARD` and `CMAKE_CUDA_STANDARD` has been
      eliminated.
  - Testing related CMake code has been moved to `testing/CMakeLists.txt`
  - Example related CMake code has been moved to `examples/CMakeLists.txt`
  - Header testing related CMake code has been moved to `cmake/ThrustHeaderTesting.cmake`
  - CUDA configuration CMake code has been moved to to `cmake/ThrustCUDAConfig.cmake`.
  - Now we explicitly `include(cmake/*.cmake)` files rather than searching
      `CMAKE_MODULE_PATH` - we only want to use the ones in the repo.
- `thrust::transform_input_output_iterator`, a variant of transform iterator
    adapter that works as both an input iterator and an output iterator.
  The given input function is applied after reading from the wrapped iterator
    while the output function is applied before writing to the wrapped iterator.
  Thanks to Trevor Smith for this contribution.

### Other Enhancements

- Contributor documentation: https://github.com/NVIDIA/thrust/blob/main/CONTRIBUTING.md
- Code of Conduct: https://github.com/NVIDIA/thrust/blob/main/CODE_OF_CONDUCT.md.
  Thanks to Conor Hoekstra for this contribution.
- Support for all combinations of host and device systems.
- C++17 support.
- NVIDIA/thrust#1221: Allocator and vector classes have been replaced with
    alias templates.
  Thanks to Michael Francis for this contribution.
- NVIDIA/thrust#1186: Use placeholder expressions to simplify the definitions
    of a number of algorithms.
  Thanks to Michael Francis for this contribution.
- NVIDIA/thrust#1170: More conforming semantics for scan algorithms:
  - Follow P0571's guidance regarding intermediate types.
    - https://wg21.link/P0571
    - The accumulator's type is now:
      - The type of the user-supplied initial value (if provided), or
      - The input iterator's value type if no initial value.
  - Follow C++ standard guidance for default binary operator type.
    - https://eel.is/c++draft/exclusive.scan#1
    - Thrust binary/unary functors now specialize a default void template
        parameter.
      Types are deduced and forwarded transparently.
    - Updated the scan's default binary operator to the new `thrust::plus<>`
        specialization.
  - The `thrust::intermediate_type_from_function_and_iterators` helper is no
      longer needed and has been removed.
- NVIDIA/thrust#1255: Always use `cudaStreamSynchronize` instead of
    `cudaDeviceSynchronize` if the execution policy has a stream attached to it.
  Thanks to Rong Ou for this contribution.
- NVIDIA/thrust#1201: Tests for correct handling of legacy and per-thread
    default streams.
  Thanks to Rong Ou for this contribution.

### Bug Fixes

- NVIDIA/thrust#1260: Fix `thrust::transform_inclusive_scan` with heterogeneous
    types.
  Thanks to Rong Ou for this contribution.
- NVIDIA/thrust#1258, NVC++ FS #28463: Ensure the CUDA radix sort backend
    synchronizes before returning; otherwise, copies from temporary storage will
    race with destruction of said temporary storage.
- NVIDIA/thrust#1264: Evaluate `CUDA_CUB_RET_IF_FAIL` macro argument only once.
  Thanks to Jason Lowe for this contribution.
- NVIDIA/thrust#1262: Add missing `<stdexcept>` header.
- NVIDIA/thrust#1250: Restore some `THRUST_DECLTYPE_RETURNS` macros in async
    test implementations.
- NVIDIA/thrust#1249: Use `std::iota` in `CUDATestDriver::target_devices`.
  Thanks to Michael Francis for this contribution.
- NVIDIA/thrust#1244: Check for macro collisions with system headers during
    header testing.
- NVIDIA/thrust#1224: Remove unnecessary SFINAE contexts from asynchronous
    algorithms.
- NVIDIA/thrust#1190: Make `out_of_memory_recovery` test trigger faster.
- NVIDIA/thrust#1187: Elminate superfluous iterators specific to the CUDA
    backend.
- NVIDIA/thrust#1181: Various fixes for GoUDA.
  Thanks to Andrei Tchouprakov for this contribution.
- NVIDIA/thrust#1178, NVIDIA/thrust#1229: Use transparent functionals in
    placeholder expressions, fixing issues with `thrust::device_reference` and
    placeholder expressions and `thrust::find` with asymmetric equality
    operators.
- NVIDIA/thrust#1153: Switch to placement new instead of assignment to
    construct items in uninitialized memory.
  Thanks to Hugh Winkler for this contribution.
- NVIDIA/thrust#1050: Fix compilation of asynchronous algorithms when RDC is
    enabled.
- NVIDIA/thrust#1042: Correct return type of
    `thrust::detail::predicate_to_integral` from `bool` to `IntegralType`.
  Thanks to Andreas Hehn for this contribution.
- NVIDIA/thrust#1009: Avoid returning uninitialized allocators.
  Thanks to Zhihao Yuan for this contribution.
- NVIDIA/thrust#990: Add missing `<thrust/system/cuda/memory.h>` include to
    `<thrust/system/cuda/detail/malloc_and_free.h>`.
  Thanks to Robert Maynard for this contribution.
- NVIDIA/thrust#966: Fix spurious MSVC conversion with loss of data warning in
    sort algorithms.
  Thanks to Zhihao Yuan for this contribution.
- Add more metadata to mock specializations for testing iterator in
   `testing/copy.cu`.
- Add missing include to shuffle unit test.
- Specialize `thrust::wrapped_function` for `void` return types because MSVC is
    not a fan of the pattern `return static_cast<void>(expr);`.
- Replace deprecated `tbb/tbb_thread.h` with `<thread>`.
- Fix overcounting of initial value in TBB scans.
- Use `thrust::advance` instead of `+=` for generic iterators.
- Wrap the OMP flags in `-Xcompiler` for NVCC
- Extend `ASSERT_STATIC_ASSERT` skip for the OMP backend.
- Add missing header caught by `tbb.cuda` configs.
- Fix "unsafe API" warnings in examples on MSVC: `s/fopen/fstream/`
- Various C++17 fixes.

## Thrust 1.9.10-1 (NVIDIA HPC SDK 20.7, CUDA Toolkit 11.1)

Thrust 1.9.10-1 is the minor release accompanying the NVIDIA HPC SDK 20.7 release
  and the CUDA Toolkit 11.1 release.

### Bug Fixes

- #1214, NVBug 200619442: Stop using `std::allocator` APIs deprecated in C++17.
- #1216, NVBug 200540293: Make `thrust::optional` work with Clang when used
    with older libstdc++.
- #1207, NVBug 200618218: Don't force C++14 with older compilers that don't
    support it.
- #1218: Wrap includes of `<memory>` and `<algorithm>` to avoid circular
    inclusion with NVC++.

## Thrust 1.9.10 (NVIDIA HPC SDK 20.5)

Thrust 1.9.10 is the release accompanying the NVIDIA HPC SDK 20.5 release.
It adds CMake support for compilation with NVC++ and a number of minor bug fixes
  for NVC++.
It also adds CMake `find_package` support, which replaces the broken 3rd-party
  legacy `FindThrust.cmake` script.
C++03, C++11, GCC < 5, Clang < 6, and MSVC < 2017 are now deprecated.
Starting with the upcoming 1.10.0 release, C++03 support will be dropped
  entirely.

### Breaking Changes

- #1082: Thrust now checks that it is compatible with the version of CUB found
    in your include path, generating an error if it is not.
  If you are using your own version of CUB, it may be too old.
  It is recommended to simply delete your own version of CUB and use the
    version of CUB that comes with Thrust.
- #1089: C++03 and C++11 are deprecated.
  Using these dialects will generate a compile-time warning.
  These warnings can be suppressed by defining
    `THRUST_IGNORE_DEPRECATED_CPP_DIALECT` (to suppress C++03 and C++11
    deprecation warnings) or `THRUST_IGNORE_DEPRECATED_CPP11` (to suppress C++11
    deprecation warnings).
  Suppression is only a short term solution.
  We will be dropping support for C++03 in the 1.10.0 release and C++11 in the
    near future.
- #1089: GCC < 5, Clang < 6, and MSVC < 2017 are deprecated.
  Using these compilers will generate a compile-time warning.
  These warnings can be suppressed by defining
    `THRUST_IGNORE_DEPRECATED_COMPILER`.
  Suppression is only a short term solution.
  We will be dropping support for these compilers in the near future.

### New Features

- #1130: CMake `find_package` support.
  This is significant because there is a legacy `FindThrust.cmake` script
    authored by a third party in widespread use in the community which has a
    bug in how it parses Thrust version numbers which will cause it to
    incorrectly parse 1.9.10.
  This script only handles the first digit of each part of the Thrust version
    number correctly: for example, Thrust 17.17.17 would be interpreted as
    Thrust 1.1.1701717.
  You can find directions for using the new CMake `find_package` support and
    migrating away from the legacy `FindThrust.cmake` [here](https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md)
- #1129: Added `thrust::detail::single_device_tls_caching_allocator`, a
    convenient way to get an MR caching allocator for device memory, which is
    used by NVC++.

### Other Enhancements

- #1129: Refactored RDC handling in CMake to be a global option and not create
    two targets for each example and test.

### Bug Fixes

- #1129: Fix the legacy `thrust::return_temporary_buffer` API to support
    passing a size.
  This was necessary to enable usage of Thrust caching MR allocators with
    synchronous Thrust algorithms.
  This change has allowed NVC++’s C++17 Parallel Algorithms implementation to
    switch to use Thrust caching MR allocators for device temporary storage,
    which gives a 2x speedup on large multi-GPU systems such as V100 and A100
    DGX where `cudaMalloc` is very slow.
- #1128: Respect `CUDA_API_PER_THREAD_DEFAULT_STREAM`.
  Thanks to Rong Ou for this contribution.
- #1131: Fix the one-policy overload of `thrust::async::copy` to not copy the
    policy, resolving use-afer-move issues.
- #1145: When cleaning up type names in `unittest::base_class_name`, only call
    `std::string::replace` if we found the substring we are looking to replace.
- #1139: Don't use `cxx::__demangle` in NVC++.
- #1102: Don't use `thrust::detail::normal_distribution_nvcc` for Feta because
    it uses `erfcinv`, a non-standard function that Feta doesn't have.

## Thrust 1.9.9 (CUDA Toolkit 11.0)

Thrust 1.9.9 adds support for NVC++, which uses Thrust to implement
  GPU-accelerated C++17 Parallel Algorithms.
`thrust::zip_function` and `thrust::shuffle` were also added.
C++03, C++11, GCC < 5, Clang < 6, and MSVC < 2017 are now deprecated.
Starting with the upcoming 1.10.0 release, C++03 support will be dropped
  entirely.
All other deprecated platforms will be dropped in the near future.

### Breaking Changes

- #1082: Thrust now checks that it is compatible with the version of CUB found
    in your include path, generating an error if it is not.
  If you are using your own version of CUB, it may be too old.
  It is recommended to simply delete your own version of CUB and use the
    version of CUB that comes with Thrust.
- #1089: C++03 and C++11 are deprecated.
  Using these dialects will generate a compile-time warning.
  These warnings can be suppressed by defining
    `THRUST_IGNORE_DEPRECATED_CPP_DIALECT` (to suppress C++03 and C++11
    deprecation warnings) or `THRUST_IGNORE_DEPRECATED_CPP_11` (to suppress C++11
    deprecation warnings).
  Suppression is only a short term solution.
  We will be dropping support for C++03 in the 1.10.0 release and C++11 in the
    near future.
- #1089: GCC < 5, Clang < 6, and MSVC < 2017 are deprecated.
  Using these compilers will generate a compile-time warning.
  These warnings can be suppressed by defining
  `THRUST_IGNORE_DEPRECATED_COMPILER`.
  Suppression is only a short term solution.
  We will be dropping support for these compilers in the near future.

### New Features

- #1086: Support for NVC++ aka "Feta".
  The most significant change is in how we use `__CUDA_ARCH__`.
  Now, there are four macros that must be used:
  - `THRUST_IS_DEVICE_CODE`, which should be used in an `if` statement around
      device-only code.
  - `THRUST_INCLUDE_DEVICE_CODE`, which should be used in an `#if` preprocessor
      directive inside of the `if` statement mentioned in the prior bullet.
  - `THRUST_IS_HOST_CODE`, which should be used in an `if` statement around
      host-only code.
  - `THRUST_INCLUDE_HOST_CODE`, which should be used in an `#if` preprocessor
      directive inside of the `if` statement mentioned in the prior bullet.
- #1085: `thrust::shuffle`.
  Thanks to Rory Mitchell for this contribution.
- #1029: `thrust::zip_function`, a facility for zipping functions that take N
    parameters instead of a tuple of N parameters as `thrust::zip_iterator`
    does.
  Thanks to Ben Jude for this contribution.
- #1068: `thrust::system::cuda::managed_memory_pointer`, a universal memory
    strongly typed pointer compatible with the ISO C++ Standard Library.

### Other Enhancements

- #1029: Thrust is now built and tested with NVCC warnings treated as errors.
- #1029: MSVC C++11 support.
- #1029: `THRUST_DEPRECATED` abstraction for generating compile-time
    deprecation warning messages.
- #1029: `thrust::pointer<T>::pointer_to(reference)`.
- #1070: Unit test for `thrust::inclusive_scan` with a user defined types.
  Thanks to Conor Hoekstra for this contribution.

### Bug Fixes

- #1088: Allow `thrust::replace` to take functions that have non-`const`
    `operator()`.
- #1094: Add missing `constexpr` to `par_t` constructors.
  Thanks to Patrick Stotko for this contribution.
- #1077: Remove `__device__` from CUDA MR-based device allocators to fix
    obscure "host function called from host device function" warning that occurs
    when you use the new Thrust MR-based allocators.
- #1029: Remove inconsistently-used `THRUST_BEGIN`/`END_NS` macros.
- #1029: Fix C++ dialect detection on newer MSVC.
- #1029 Use `_Pragma`/`__pragma` instead of `#pragma` in macros.
- #1029: Replace raw `__cplusplus` checks with the appropriate Thrust macros.
- #1105: Add a missing `<math.h>` include.
- #1103: Fix regression of `thrust::detail::temporary_allocator` with non-CUDA
    back ends.
- #1111: Use Thrust's random number engine instead of `std::`s in device code.
- #1108: Get rid of a GCC 9 warning about deprecated generation of copy ctors.

## Thrust 1.9.8-1 (NVIDIA HPC SDK 20.3)

Thrust 1.9.8-1 is a variant of 1.9.8 accompanying the NVIDIA HPC SDK 20.3
  release.
It contains modifications necessary to serve as the implementation of NVC++'s
  GPU-accelerated C++17 Parallel Algorithms when using the CUDA Toolkit 11.0
  release.

## Thrust 1.9.8 (CUDA Toolkit 11.0 Early Access)

Thrust 1.9.8, which is included in the CUDA Toolkit 11.0 release, removes
  Thrust's internal derivative of CUB, upstreams all relevant changes too CUB,
  and adds CUB as a Git submodule.
It will now be necessary to do `git clone --recursive` when checking out
  Thrust, and to update the CUB submodule when pulling in new Thrust changes.
Additionally, CUB is now included as a first class citizen in the CUDA toolkit.
Thrust 1.9.8 also fixes bugs preventing most Thrust algorithms from working
  with more than `2^31-1` elements.
Now, `thrust::reduce`, `thrust::*_scan`, and related algorithms (aka most of
  Thrust) work with large element counts.

### Breaking Changes

- Thrust will now use the version of CUB in your include path instead of its own
    internal copy.
  If you are using your own version of CUB, it may be older and incompatible
    with Thrust.
  It is recommended to simply delete your own version of CUB and use the
    version of CUB that comes with Thrust.

### Other Enhancements

- Refactor Thrust and CUB to support 64-bit indices in most algorithms.
  In most cases, Thrust now selects between kernels that use 32-bit indices and
    64-bit indices at runtime depending on the size of the input.
  This means large element counts work, but small element counts do not have to
    pay for the register usage of 64-bit indices if they are not needed.
  Now, `thrust::reduce`, `thrust::*_scan`, and related algorithms (aka most of
    Thrust) work with more than `2^31-1` elements.
  Notably, `thrust::sort` is still limited to less than `2^31-1` elements.
- CUB is now a submodule and the internal copy of CUB has been removed.
- #1051: Stop specifying the `__launch_bounds__` minimum blocks parameter
    because it messes up register allocation and increases register pressure,
    and we don't actually know at compile time how many blocks we will use
    (aside from single tile kernels).

### Bug Fixes

- #1020: After making a CUDA API call, always clear the global CUDA error state
    by calling `cudaGetLastError`.
- #1021: Avoid calling destroy in the destructor of a Thrust vector if the
    vector is empty.
- #1046: Actually throw `thrust::bad_alloc` when `thrust::system::cuda::malloc`
    fails instead of just constructing a temporary and doing nothing with it.
- Add missing copy constructor or copy assignment operator to all classes that
    GCC 9's `-Wdeprecated-copy` complains about
- Add missing move operations to `thrust::system::cuda::vector`.
- #1015: Check that the backend is CUDA before using CUDA-specifics in
    `thrust::detail::temporary_allocator`.
  Thanks to Hugh Winkler for this contribution.
- #1055: More correctly detect the presence of aligned/sized `new`/`delete`.
- #1043: Fix ill-formed specialization of `thrust::system::is_error_code_enum`
    for `thrust::event_errc`.
  Thanks to Toru Niina for this contribution.
- #1027: Add tests for `thrust::tuple_for_each` and `thrust::tuple_subset`.
  Thanks to Ben Jude for this contribution.
- #1027: Use correct macro in `thrust::tuple_for_each`.
  Thanks to Ben Jude for this contribution.
- #1026: Use correct MSVC version formatting in CMake.
  Thanks to Ben Jude for this contribution.
- Workaround an NVCC issue with type aliases with template template arguments
    containing a parameter pack.
- Remove unused functions from the CUDA backend which call slow CUDA attribute
    query APIs.
- Replace `CUB_RUNTIME_FUNCTION` with `THRUST_RUNTIME_FUNCTION`.
- Correct typo in `thrust::transform` documentation.
  Thanks to Eden Yefet for this contribution.

### Known Issues

- `thrust::sort` remains limited to `2^31-1` elements for now.

## Thrust 1.9.7-1 (CUDA Toolkit 10.2 for Tegra)

Thrust 1.9.7-1 is a minor release accompanying the CUDA Toolkit 10.2 release
  for Tegra.
It is nearly identical to 1.9.7.

### Bug Fixes

- Remove support for GCC's broken nodiscard-like attribute.

## Thrust 1.9.7 (CUDA Toolkit 10.2)

Thrust 1.9.7 is a minor release accompanying the CUDA Toolkit 10.2 release.
Unfortunately, although the version and patch numbers are identical, one bug
  fix present in Thrust 1.9.7 (NVBug 2646034: Fix incorrect dependency handling
  for stream acquisition in `thrust::future`) was not included in the CUDA
  Toolkit 10.2 preview release for AArch64 SBSA.
The tag `cuda-10.2aarch64sbsa` contains the exact version of Thrust present
  in the CUDA Toolkit 10.2 preview release for AArch64 SBSA.

### Bug Fixes

- #967, NVBug 2448170: Fix the CUDA backend `thrust::for_each` so that it
    supports large input sizes with 64-bit indices.
- NVBug 2646034: Fix incorrect dependency handling for stream acquisition in
    `thrust::future`.
  - Not present in the CUDA Toolkit 10.2 preview release for AArch64 SBSA.
- #968, NVBug 2612102: Fix the `thrust::mr::polymorphic_adaptor` to actually
    use its template parameter.

## Thrust 1.9.6-1 (NVIDIA HPC SDK 20.3)

Thrust 1.9.6-1 is a variant of 1.9.6 accompanying the NVIDIA HPC SDK 20.3
  release.
It contains modifications necessary to serve as the implementation of NVC++'s
  GPU-accelerated C++17 Parallel Algorithms when using the CUDA Toolkit 10.1
  Update 2 release.

## Thrust 1.9.6 (CUDA Toolkit 10.1 Update 2)

Thrust 1.9.6 is a minor release accompanying the CUDA Toolkit 10.1 Update 2
  release.

### Bug Fixes

- NVBug 2509847: Inconsistent alignment of `thrust::complex`
- NVBug 2586774: Compilation failure with Clang + older libstdc++ that doesn't
    have `std::is_trivially_copyable`
- NVBug 200488234: CUDA header files contain Unicode characters which leads
    compiling errors on Windows
- #949, #973, NVBug 2422333, NVBug 2522259, NVBug 2528822:
    `thrust::detail::aligned_reinterpret_cast` must be annotated with
    `__host__ __device__`.
- NVBug 2599629: Missing include in the OpenMP sort implementation
- NVBug 200513211: Truncation warning in test code under VC142

## Thrust 1.9.5 (CUDA Toolkit 10.1 Update 1)

Thrust 1.9.5 is a minor release accompanying the CUDA Toolkit 10.1 Update 1
  release.

### Bug Fixes

- NVBug 2502854: Fixed assignment of
    `thrust::device_vector<thrust::complex<T>>` between host and device.

## Thrust 1.9.4 (CUDA Toolkit 10.1)

Thrust 1.9.4 adds asynchronous interfaces for parallel algorithms, a new
  allocator system including caching allocators and unified memory support, as
  well as a variety of other enhancements, mostly related to
  C++11/C++14/C++17/C++20 support.
The new asynchronous algorithms in the `thrust::async` namespace return
  `thrust::event` or `thrust::future` objects, which can be waited upon to
  synchronize with the completion of the parallel operation.

### Breaking Changes

Synchronous Thrust algorithms now block until all of their operations have
  completed.
Use the new asynchronous Thrust algorithms for non-blocking behavior.

### New Features

- `thrust::event` and `thrust::future<T>`, uniquely-owned asynchronous handles
    consisting of a state (ready or not ready), content (some value; for
    `thrust::future` only), and an optional set of objects that should be
    destroyed only when the future's value is ready and has been consumed.
  - The design is loosely based on C++11's `std::future`.
  - They can be `.wait`'d on, and the value of a future can be waited on and
      retrieved with `.get` or `.extract`.
  - Multiple `thrust::event`s and `thrust::future`s can be combined with
      `thrust::when_all`.
  - `thrust::future`s can be converted to `thrust::event`s.
  - Currently, these primitives are only implemented for the CUDA backend and
      are C++11 only.
- New asynchronous algorithms that return `thrust::event`/`thrust::future`s,
    implemented as C++20 range style customization points:
    - `thrust::async::reduce`.
    - `thrust::async::reduce_into`, which takes a target location to store the
        reduction result into.
    - `thrust::async::copy`, including a two-policy overload that allows
        explicit cross system copies which execution policy properties can be
        attached to.
    - `thrust::async::transform`.
    - `thrust::async::for_each`.
    - `thrust::async::stable_sort`.
    - `thrust::async::sort`.
    - By default the asynchronous algorithms use the new caching allocators.
        Deallocation of temporary storage is deferred until the destruction of
        the returned `thrust::future`. The content of `thrust::future`s is
        stored in either device or universal memory and transferred to the host
        only upon request to prevent unnecessary data migration.
    - Asynchronous algorithms are currently only implemented for the CUDA
        system and are C++11 only.
- `exec.after(f, g, ...)`, a new execution policy method that takes a set of
    `thrust::event`/`thrust::future`s and returns an execution policy that
    operations on that execution policy should depend upon.
- New logic and mindset for the type requirements for cross-system sequence
    copies (currently only used by `thrust::async::copy`), based on:
  - `thrust::is_contiguous_iterator` and `THRUST_PROCLAIM_CONTIGUOUS_ITERATOR`
      for detecting/indicating that an iterator points to contiguous storage.
  - `thrust::is_trivially_relocatable` and
      `THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE` for detecting/indicating that a
      type is `memcpy`able (based on principles from
      [P1144](https://wg21.link/P1144)).
  - The new approach reduces buffering, increases performance, and increases
      correctness.
  - The fast path is now enabled when copying CUDA `__half` and vector types with
      `thrust::async::copy`.
- All Thrust synchronous algorithms for the CUDA backend now actually
    synchronize. Previously, any algorithm that did not allocate temporary
    storage (counterexample: `thrust::sort`) and did not have a
    computation-dependent result (counterexample: `thrust::reduce`) would
    actually be launched asynchronously. Additionally, synchronous algorithms
    that allocated temporary storage would become asynchronous if a custom
    allocator was supplied that did not synchronize on allocation/deallocation,
    unlike `cudaMalloc`/`cudaFree`. So, now `thrust::for_each`,
    `thrust::transform`, `thrust::sort`, etc are truly synchronous. In some
    cases this may be a performance regression; if you need asynchrony, use the
    new asynchronous algorithms.
- Thrust's allocator framework has been rewritten. It now uses a memory
    resource system, similar to C++17's `std::pmr` but supporting static
    polymorphism. Memory resources are objects that allocate untyped storage and
    allocators are cheap handles to memory resources in this new model. The new
    facilities live in `<thrust/mr/*>`.
  - `thrust::mr::memory_resource<Pointer>`, the memory resource base class,
      which takes a (possibly tagged) pointer to `void` type as a parameter.
  - `thrust::mr::allocator<T, MemoryResource>`, an allocator backed by a memory
      resource object.
  - `thrust::mr::polymorphic_adaptor_resource<Pointer>`, a type-erased memory
      resource adaptor.
  - `thrust::mr::polymorphic_allocator<T>`, a C++17-style polymorphic allocator
      backed by a type-erased memory resource object.
  - New tunable C++17-style caching memory resources,
      `thrust::mr::(disjoint_)?(un)?synchronized_pool_resource`, designed to
      cache both small object allocations and large repetitive temporary
      allocations. The disjoint variants use separate storage for management of
      the pool, which is necessary if the memory being allocated cannot be
      accessed on the host (e.g.  device memory).
  - System-specific allocators were rewritten to use the new memory resource
      framework.
  - New `thrust::device_memory_resource` for allocating device memory.
  - New `thrust::universal_memory_resource` for allocating memory that can be
      accessed from both the host and device (e.g. `cudaMallocManaged`).
  - New `thrust::universal_host_pinned_memory_resource` for allocating memory
      that can be accessed from the host and the device but always resides in
      host memory (e.g. `cudaMallocHost`).
  - `thrust::get_per_device_resource` and `thrust::per_device_allocator`, which
      lazily create and retrieve a per-device singleton memory resource.
  - Rebinding mechanisms (`rebind_traits` and `rebind_alloc`) for
      `thrust::allocator_traits`.
  - `thrust::device_make_unique`, a factory function for creating a
      `std::unique_ptr` to a newly allocated object in device memory.
  - `<thrust/detail/memory_algorithms>`, a C++11 implementation of the C++17
      uninitialized memory algorithms.
  - `thrust::allocate_unique` and friends, based on the proposed C++23
      [`std::allocate_unique`](https://wg21.link/P0211).
- New type traits and metaprogramming facilities. Type traits are slowly being
    migrated out of `thrust::detail::` and `<thrust/detail/*>`; their new home
    will be `thrust::` and `<thrust/type_traits/*>`.
  - `thrust::is_execution_policy`.
  - `thrust::is_operator_less_or_greater_function_object`, which detects
      `thrust::less`, `thrust::greater`, `std::less`, and `std::greater`.
  - `thrust::is_operator_plus_function_object``, which detects `thrust::plus`
      and `std::plus`.
  - `thrust::remove_cvref(_t)?`, a C++11 implementation of C++20's
      `thrust::remove_cvref(_t)?`.
  - `thrust::void_t`, and various other new type traits.
  - `thrust::integer_sequence` and friends, a C++11 implementation of C++20's
      `std::integer_sequence`
  - `thrust::conjunction`, `thrust::disjunction`, and `thrust::disjunction`, a
      C++11 implementation of C++17's logical metafunctions.
  - Some Thrust type traits (such as `thrust::is_constructible`) have been
      redefined in terms of C++11's type traits when they are available.
- `<thrust/detail/tuple_algorithms.h>`, new `std::tuple` algorithms:
  - `thrust::tuple_transform`.
  - `thrust::tuple_for_each`.
  - `thrust::tuple_subset`.
- Miscellaneous new `std::`-like facilities:
  - `thrust::optional`, a C++11 implementation of C++17's `std::optional`.
  - `thrust::addressof`, an implementation of C++11's `std::addressof`.
  - `thrust::next` and `thrust::prev`, an implementation of C++11's `std::next`
      and `std::prev`.
  - `thrust::square`, a `<functional>` style unary function object that
      multiplies its argument by itself.
  - `<thrust/limits.h>` and `thrust::numeric_limits`, a customized version of
      `<limits>` and `std::numeric_limits`.
- `<thrust/detail/preprocessor.h>`, new general purpose preprocessor facilities:
  - `THRUST_PP_CAT[2-5]`, concatenates two to five tokens.
  - `THRUST_PP_EXPAND(_ARGS)?`, performs double expansion.
  - `THRUST_PP_ARITY` and `THRUST_PP_DISPATCH`, tools for macro overloading.
  - `THRUST_PP_BOOL`, boolean conversion.
  - `THRUST_PP_INC` and `THRUST_PP_DEC`, increment/decrement.
  - `THRUST_PP_HEAD`, a variadic macro that expands to the first argument.
  - `THRUST_PP_TAIL`, a variadic macro that expands to all its arguments after
      the first.
  - `THRUST_PP_IIF`, bitwise conditional.
  - `THRUST_PP_COMMA_IF`, and `THRUST_PP_HAS_COMMA`, facilities for adding and
      detecting comma tokens.
  - `THRUST_PP_IS_VARIADIC_NULLARY`, returns true if called with a nullary
      `__VA_ARGS__`.
  - `THRUST_CURRENT_FUNCTION`, expands to the name of the current function.
- New C++11 compatibility macros:
  - `THRUST_NODISCARD`, expands to `[[nodiscard]]` when available and the best
      equivalent otherwise.
  - `THRUST_CONSTEXPR`, expands to `constexpr` when available and the best
      equivalent otherwise.
  - `THRUST_OVERRIDE`, expands to `override` when available and the best
      equivalent otherwise.
  - `THRUST_DEFAULT`, expands to `= default;` when available and the best
      equivalent otherwise.
  - `THRUST_NOEXCEPT`, expands to `noexcept` when available and the best
      equivalent otherwise.
  - `THRUST_FINAL`, expands to `final` when available and the best equivalent
      otherwise.
  - `THRUST_INLINE_CONSTANT`, expands to `inline constexpr` when available and
      the best equivalent otherwise.
- `<thrust/detail/type_deduction.h>`, new C++11-only type deduction helpers:
  - `THRUST_DECLTYPE_RETURNS*`, expand to function definitions with suitable
      conditional `noexcept` qualifiers and trailing return types.
  - `THRUST_FWD(x)`, expands to `::std::forward<decltype(x)>(x)`.
  - `THRUST_MVCAP`, expands to a lambda move capture.
  - `THRUST_RETOF`, expands to a decltype computing the return type of an
      invocable.
- New CMake build system.

### New Examples

- `mr_basic` demonstrates how to use the new memory resource allocator system.

### Other Enhancements

- Tagged pointer enhancements:
  - New `thrust::pointer_traits` specialization for `void const*`.
  - `nullptr` support to Thrust tagged pointers.
  - New `explicit operator bool` for Thrust tagged pointers when using C++11
      for `std::unique_ptr` interoperability.
  - Added `thrust::reinterpret_pointer_cast` and `thrust::static_pointer_cast`
      for casting Thrust tagged pointers.
- Iterator enhancements:
  - `thrust::iterator_system` is now SFINAE friendly.
  - Removed cv qualifiers from iterator types when using
      `thrust::iterator_system`.
- Static assert enhancements:
  - New `THRUST_STATIC_ASSERT_MSG`, takes an optional string constant to be
      used as the error message when possible.
  - Update `THRUST_STATIC_ASSERT(_MSG)` to use C++11's `static_assert` when
      it's available.
  - Introduce a way to test for static assertions.
- Testing enhancements:
  - Additional scalar and sequence types, including non-builtin types and
      vectors with unified memory allocators, have been added to the list of
      types used by generic unit tests.
  - The generation of random input data has been improved to increase the range
      of values used and catch more corner cases.
  - New `unittest::truncate_to_max_representable` utility for avoiding the
      generation of ranges that cannot be represented by the underlying element
      type in generic unit test code.
  - The test driver now synchronizes with CUDA devices and check for errors
      after each test, when switching devices, and after each raw kernel launch.
  - The `warningtester` uber header is now compiled with NVCC to avoid needing
      to disable CUDA-specific code with the preprocessor.
  - Fixed the unit test framework's `ASSERT_*` to print `char`s as `int`s.
  - New `DECLARE_INTEGRAL_VARIABLE_UNITTEST` test declaration macro.
  - New `DECLARE_VARIABLE_UNITTEST_WITH_TYPES_AND_NAME` test declaration macro.
  - `thrust::system_error` in the CUDA backend now print out its `cudaError_t`
      enumerator in addition to the diagnostic message.
  - Stopped using conditionally signed types like `char`.

### Bug Fixes

- #897, NVBug 2062242: Fix compilation error when using `__device__` lambdas
    with `thrust::reduce` on MSVC.
- #908, NVBug 2089386: Static assert that `thrust::generate`/`thrust::fill`
    isn't operating on const iterators.
- #919 Fix compilation failure with `thrust::zip_iterator` and
    `thrust::complex`.
- #924, NVBug 2096679, NVBug 2315990: Fix dispatch for the CUDA backend's
    `thrust::reduce` to use two functions (one with the pragma for disabling
    exec checks, one with `THRUST_RUNTIME_FUNCTION`) instead of one. This fixes
    a regression with device compilation that started in CUDA Toolkit 9.2.
- #928, NVBug 2341455: Add missing `__host__ __device__` annotations to a
    `thrust::complex::operator=` to satisfy GoUDA.
- NVBug 2094642: Make `thrust::vector_base::clear` not depend on the element
    type being default constructible.
- NVBug 2289115: Remove flaky `simple_cuda_streams` example.
- NVBug 2328572: Add missing `thrust::device_vector` constructor that takes an
    allocator parameter.
- NVBug 2455740: Update the `range_view` example to not use device-side launch.
- NVBug 2455943: Ensure that sized unit tests that use
    `thrust::counting_iterator` perform proper truncation.
- NVBug 2455952: Refactor questionable `thrust::copy_if` unit tests.

## Thrust 1.9.3 (CUDA Toolkit 10.0)

Thrust 1.9.3 unifies and integrates CUDA Thrust and GitHub Thrust.

### Bug Fixes

- #725, #850, #855, #859, #860: Unify the `thrust::iter_swap` interface and fix
    `thrust::device_reference` swapping.
- NVBug 2004663: Add a `data` method to `thrust::detail::temporary_array` and
    refactor temporary memory allocation in the CUDA backend to be exception
    and leak safe.
- #886, #894, #914: Various documentation typo fixes.
- #724: Provide `NVVMIR_LIBRARY_DIR` environment variable to NVCC.
- #878: Optimize `thrust::min/max_element` to only use
    `thrust::detail::get_iterator_value` for non-numeric types.
- #899: Make `thrust::cuda::experimental::pinned_allocator`'s comparison
    operators `const`.
- NVBug 2092152: Remove all includes of `<cuda.h>`.
- #911: Fix default comparator element type for `thrust::merge_by_key`.

### Acknowledgments

- Thanks to Andrew Corrigan for contributing fixes for swapping interfaces.
- Thanks to Francisco Facioni for contributing optimizations for
    `thrust::min/max_element`.

## Thrust 1.9.2 (CUDA Toolkit 9.2)

Thrust 1.9.2 brings a variety of performance enhancements, bug fixes and test
  improvements.
CUB 1.7.5 was integrated, enhancing the performance of `thrust::sort` on
  small data types and `thrust::reduce`.
Changes were applied to `complex` to optimize memory access.
Thrust now compiles with compiler warnings enabled and treated as errors.
Additionally, the unit test suite and framework was enhanced to increase
  coverage.

### Breaking Changes

- The `fallback_allocator` example was removed, as it was buggy and difficult
    to support.

### New Features

- `<thrust/detail/alignment.h>`, utilities for memory alignment:
  - `thrust::aligned_reinterpret_cast`.
  - `thrust::aligned_storage_size`, which computes the amount of storage needed
      for an object of a particular size and alignment.
  - `thrust::alignment_of`, a C++03 implementation of C++11's
      `std::alignment_of`.
  - `thrust::aligned_storage`, a C++03 implementation of C++11's
      `std::aligned_storage`.
  - `thrust::max_align_t`, a C++03 implementation of C++11's
      `std::max_align_t`.

### Bug Fixes

- NVBug 200385527, NVBug 200385119, NVBug 200385113, NVBug 200349350, NVBug
    2058778: Various compiler warning issues.
- NVBug 200355591: `thrust::reduce` performance issues.
- NVBug 2053727: Fixed an ADL bug that caused user-supplied `allocate` to be
    overlooked but `deallocate` to be called with GCC <= 4.3.
- NVBug 1777043: Fixed `thrust::complex` to work with `thrust::sequence`.

## Thrust 1.9.1-2 (CUDA Toolkit 9.1)

Thrust 1.9.1-2 integrates version 1.7.4 of CUB and introduces a new CUDA backend
  for `thrust::reduce` based on CUB.

### Bug Fixes

- NVBug 1965743: Remove unnecessary static qualifiers.
- NVBug 1940974: Fix regression causing a compilation error when using
    `thrust::merge_by_key` with `thrust::constant_iterator`s.
- NVBug 1904217: Allow callables that take non-const refs to be used with
    `thrust::reduce` and `thrust::*_scan`.

## Thrust 1.9.0-5 (CUDA Toolkit 9.0)

Thrust 1.9.0-5 replaces the original CUDA backend (bulk) with a new one
  written using CUB, a high performance CUDA collectives library.
This brings a substantial performance improvement to the CUDA backend across
  the board.

### Breaking Changes

- Any code depending on CUDA backend implementation details will likely be
    broken.

### New Features

- New CUDA backend based on CUB which delivers substantially higher performance.
- `thrust::transform_output_iterator`, a fancy iterator that applies a function
    to the output before storing the result.

### New Examples

- `transform_output_iterator` demonstrates use of the new fancy iterator
    `thrust::transform_output_iterator`.

### Other Enhancements

- When C++11 is enabled, functors do not have to inherit from
    `thrust::(unary|binary)_function` anymore to be used with
    `thrust::transform_iterator`.
- Added C++11 only move constructors and move assignment operators for
    `thrust::detail::vector_base`-based classes, e.g. `thrust::host_vector`,
    `thrust::device_vector`, and friends.

### Bug Fixes

- `sin(thrust::complex<double>)` no longer has precision loss to float.

### Acknowledgments

- Thanks to Manuel Schiller for contributing a C++11 based enhancement
    regarding the deduction of functor return types, improving the performance
    of `thrust::unique` and implementing `thrust::transform_output_iterator`.
- Thanks to Thibault Notargiacomo for the implementation of move semantics for
    the `thrust::vector_base`-based classes.
- Thanks to Duane Merrill for developing CUB and helping to integrate it into
    Thrust's backend.

## Thrust 1.8.3 (CUDA Toolkit 8.0)

Thrust 1.8.3 is a small bug fix release.

### New Examples

- `range_view` demonstrates the use of a view (a non-owning wrapper for an
    iterator range with a container-like interface).

### Bug Fixes

- `thrust::(min|max|minmax)_element` can now accept raw device pointers when
    an explicit device execution policy is used.
- `thrust::clear` operations on vector types no longer requires the element
    type to have a default constructor.

## Thrust 1.8.2 (CUDA Toolkit 7.5)

Thrust 1.8.2 is a small bug fix release.

### Bug Fixes

- Avoid warnings and errors concerning user functions called from
    `__host__ __device__` functions.
- #632: Fix an error in `thrust::set_intersection_by_key` with the CUDA backend.
- #651: `thrust::copy` between host and device now accepts execution policies
    with streams attached, i.e. `thrust::::cuda::par.on(stream)`.
- #664: `thrust::for_each` and algorithms based on it no longer ignore streams
    attached to execution policys.

### Known Issues

- #628: `thrust::reduce_by_key` for the CUDA backend fails for Compute
    Capability 5.0 devices.

## Thrust 1.8.1 (CUDA Toolkit 7.0)

Thrust 1.8.1 is a small bug fix release.

### Bug Fixes

- #615, #620: Fixed `thrust::for_each` and `thrust::reduce` to no longer fail on
    large inputs.

### Known Issues

- #628: `thrust::reduce_by_key` for the CUDA backend fails for Compute
    Capability 5.0 devices.

## Thrust 1.8.0

Thrust 1.8.0 introduces support for algorithm invocation from CUDA device
  code, support for CUDA streams, and algorithm performance improvements.
Users may now invoke Thrust algorithms from CUDA device code, providing a
  parallel algorithms library to CUDA programmers authoring custom kernels, as
  well as allowing Thrust programmers to nest their algorithm calls within
  functors.
The `thrust::seq` execution policy allows users to require sequential algorithm
  execution in the calling thread and makes a sequential algorithms library
  available to individual CUDA threads.
The `.on(stream)` syntax allows users to request a CUDA stream for kernels
  launched during algorithm execution.
Finally, new CUDA algorithm implementations provide substantial performance
  improvements.

### New Features

- Algorithms in CUDA Device Code:
    - Thrust algorithms may now be invoked from CUDA `__device__` and
        `__host__` __device__ functions.
      Algorithms invoked in this manner must be invoked with an execution
        policy as the first parameter.
      The following execution policies are supported in CUDA __device__ code:
      - `thrust::seq`
      - `thrust::cuda::par`
      - `thrust::device`, when THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA.
  - Device-side algorithm execution may not be parallelized unless CUDA Dynamic
      Parallelism is available.
- Execution Policies:
  - CUDA Streams
    - The `thrust::cuda::par.on(stream)` syntax allows users to request that
        CUDA kernels launched during algorithm execution should occur on a given
        stream.
    - Algorithms executed with a CUDA stream in this manner may still
        synchronize with other streams when allocating temporary storage or
        returning results to the CPU.
  - `thrust::seq`, which allows users to require that an algorithm execute
      sequentially in the calling thread.
- `thrust::complex`, a complex number data type.

### New Examples

- simple_cuda_streams demonstrates how to request a CUDA stream during
    algorithm execution.
- async_reduce demonstrates ways to achieve algorithm invocations which are
    asynchronous with the calling thread.

### Other Enhancements

- CUDA sort performance for user-defined types is 300% faster on Tesla K20c for
    large problem sizes.
- CUDA merge performance is 200% faster on Tesla K20c for large problem sizes.
- CUDA sort performance for primitive types is 50% faster on Tesla K20c for
    large problem sizes.
- CUDA reduce_by_key performance is 25% faster on Tesla K20c for large problem
    sizes.
- CUDA scan performance is 15% faster on Tesla K20c for large problem sizes.
- fallback_allocator example is simpler.

### Bug Fixes

- #364: Iterators with unrelated system tags may be used with algorithms invoked
    with an execution policy
- #371: Do not redefine `__CUDA_ARCH__`.
- #379: Fix crash when dereferencing transform_iterator on the host.
- #391: Avoid use of uppercase variable names.
- #392: Fix `thrust::copy` between `cusp::complex` and `std::complex`.
- #396: Program compiled with gcc < 4.3 hangs during comparison sort.
- #406: `fallback_allocator.cu` example checks device for unified addressing support.
- #417: Avoid using `std::less<T>` in binary search algorithms.
- #418: Avoid various warnings.
- #443: Including version.h no longer configures default systems.
- #578: NVCC produces warnings when sequential algorithms are used with CPU systems.

### Known Issues

- When invoked with primitive data types, thrust::sort, thrust::sort_by_key,
    thrust::stable_sort, & thrust::stable_sort_by_key may
- Sometimes linking fails when compiling with `-rdc=true` with NVCC.
- The CUDA implementation of thrust::reduce_by_key incorrectly outputs the last
    element in a segment of equivalent keys instead of the first.

### Acknowledgments

- Thanks to Sean Baxter for contributing faster CUDA reduce, merge, and scan
    implementations.
- Thanks to Duane Merrill for contributing a faster CUDA radix sort implementation.
- Thanks to Filipe Maia for contributing the implementation of thrust::complex.

## Thrust 1.7.2 (CUDA Toolkit 6.5)

Thrust 1.7.2 is a minor bug fix release.

### Bug Fixes

- Avoid use of `std::min` in generic find implementation.

## Thrust 1.7.1 (CUDA Toolkit 6.0)

Thrust 1.7.1 is a minor bug fix release.

### Bug Fixes

- Eliminate identifiers in `set_operations.cu` example with leading underscore.
- Eliminate unused variable warning in CUDA `reduce_by_key` implementation.
- Avoid deriving function objects from `std::unary_function` and
    `std::binary_function`.

## Thrust 1.7.0 (CUDA Toolkit 5.5)

Thrust 1.7.0 introduces a new interface for controlling algorithm execution as
  well as several new algorithms and performance improvements.
With this new interface, users may directly control how algorithms execute as
  well as details such as the allocation of temporary storage.
Key/value versions of thrust::merge and the set operation algorithms have been
  added, as well stencil versions of partitioning algorithms.
thrust::tabulate has been introduced to tabulate the values of functions taking
  integers.
For 32b types, new CUDA merge and set operations provide 2-15x faster
  performance while a new CUDA comparison sort provides 1.3-4x faster
  performance.
Finally, a new TBB reduce_by_key implementation provides 80% faster
  performance.

### Breaking Changes

- Dispatch:
  - Custom user backend systems' tag types must now inherit from the
      corresponding system's execution_policy template (e.g.
      thrust::cuda::execution_policy) instead of the tag struct (e.g.
      thrust::cuda::tag). Otherwise, algorithm specializations will silently go
      unfound during dispatch. See examples/minimal_custom_backend.cu and
      examples/cuda/fallback_allocator.cu for usage examples.
  - thrust::advance and thrust::distance are no longer dispatched based on
      iterator system type and thus may no longer be customized.
- Iterators:
  - iterator_facade and iterator_adaptor's Pointer template parameters have
      been eliminated.
  - iterator_adaptor has been moved into the thrust namespace (previously
      thrust::experimental::iterator_adaptor).
  - iterator_facade has been moved into the thrust namespace (previously
      thrust::experimental::iterator_facade).
  - iterator_core_access has been moved into the thrust namespace (previously
      thrust::experimental::iterator_core_access).
  - All iterators' nested pointer typedef (the type of the result of
      operator->) is now void instead of a pointer type to indicate that such
      expressions are currently impossible.
  - Floating point counting_iterators' nested difference_type typedef is now a
      signed integral type instead of a floating point type.
- Other:
  - normal_distribution has been moved into the thrust::random namespace
      (previously thrust::random::experimental::normal_distribution).
  - Placeholder expressions may no longer include the comma operator.

### New Features
- Execution Policies:
  - Users may directly control the dispatch of algorithm invocations with
      optional execution policy arguments.
    For example, instead of wrapping raw pointers allocated by cudaMalloc with
      thrust::device_ptr, the thrust::device execution_policy may be passed as
      an argument to an algorithm invocation to enable CUDA execution.
  - The following execution policies are supported in this version:
    - `thrust::host`
    - `thrust::device`
    - `thrust::cpp::par`
    - `thrust::cuda::par`
    - `thrust::omp::par`
    - `thrust::tbb::par`
- Algorithms:
  - `thrust::merge_by_key`
  - `thrust::partition` with stencil
  - `thrust::partition_copy` with stencil
  - `thrust::set_difference_by_key`
  - `thrust::set_intersection_by_key`
  - `thrust::set_symmetric_difference_by_key`
  - `thrust::set_union_by_key`
  - `thrust::stable_partition with stencil`
  - `thrust::stable_partition_copy with stencil`
  - `thrust::tabulate`
- Memory Allocation:
	- `thrust::malloc`
	- `thrust::free`
  - `thrust::get_temporary_buffer`
  - `thrust::return_temporary_buffer`

### New Examples

- uninitialized_vector demonstrates how to use a custom allocator to avoid the
    automatic initialization of elements in thrust::device_vector.

### Other Enhancements

- Authors of custom backend systems may manipulate arbitrary state during
    algorithm dispatch by incorporating it into their execution_policy parameter.
- Users may control the allocation of temporary storage during algorithm
    execution by passing standard allocators as parameters via execution policies
    such as thrust::device.
- THRUST_DEVICE_SYSTEM_CPP has been added as a compile-time target for the
    device backend.
- CUDA merge performance is 2-15x faster.
- CUDA comparison sort performance is 1.3-4x faster.
- CUDA set operation performance is 1.5-15x faster.
- TBB reduce_by_key performance is 80% faster.
- Several algorithms have been parallelized with TBB.
- Support for user allocators in vectors has been improved.
- The sparse_vector example is now implemented with merge_by_key instead of
    sort_by_key.
- Warnings have been eliminated in various contexts.
- Warnings about __host__ or __device__-only functions called from __host__
    __device__ functions have been eliminated in various contexts.
- Documentation about algorithm requirements have been improved.
- Simplified the minimal_custom_backend example.
- Simplified the cuda/custom_temporary_allocation example.
- Simplified the cuda/fallback_allocator example.

### Bug Fixes

- #248: Fix broken `thrust::counting_iterator<float>` behavior with OpenMP.
- #231, #209: Fix set operation failures with CUDA.
- #187: Fix incorrect occupancy calculation with CUDA.
- #153: Fix broken multi GPU behavior with CUDA.
- #142: Eliminate warning produced by `thrust::random::taus88` and MSVC 2010.
- #208: Correctly initialize elements in temporary storage when necessary.
- #16: Fix compilation error when sorting bool with CUDA.
- #10: Fix ambiguous overloads of `thrust::reinterpret_tag`.

### Known Issues

- GCC 4.3 and lower may fail to dispatch thrust::get_temporary_buffer correctly
    causing infinite recursion in examples such as
    cuda/custom_temporary_allocation.

### Acknowledgments

- Thanks to Sean Baxter, Bryan Catanzaro, and Manjunath Kudlur for contributing
    a faster merge implementation for CUDA.
- Thanks to Sean Baxter for contributing a faster set operation implementation
    for CUDA.
- Thanks to Cliff Woolley for contributing a correct occupancy calculation
    algorithm.

## Thrust 1.6.0

Thrust 1.6.0 provides an interface for customization and extension and a new
  backend system based on the Threading Building Blocks library.
With this new interface, programmers may customize the behavior of specific
  algorithms as well as control the allocation of temporary storage or invent
  entirely new backends.
These enhancements also allow multiple different backend systems
  such as CUDA and OpenMP to coexist within a single program.
Support for TBB allows Thrust programs to integrate more naturally into
  applications which may already employ the TBB task scheduler.

### Breaking Changes

- The header <thrust/experimental/cuda/pinned_allocator.h> has been moved to
    <thrust/system/cuda/experimental/pinned_allocator.h>
- thrust::experimental::cuda::pinned_allocator has been moved to
    thrust::cuda::experimental::pinned_allocator
- The macro THRUST_DEVICE_BACKEND has been renamed THRUST_DEVICE_SYSTEM
- The macro THRUST_DEVICE_BACKEND_CUDA has been renamed THRUST_DEVICE_SYSTEM_CUDA
- The macro THRUST_DEVICE_BACKEND_OMP has been renamed THRUST_DEVICE_SYSTEM_OMP
- thrust::host_space_tag has been renamed thrust::host_system_tag
- thrust::device_space_tag has been renamed thrust::device_system_tag
- thrust::any_space_tag has been renamed thrust::any_system_tag
- thrust::iterator_space has been renamed thrust::iterator_system

### New Features

- Backend Systems
  - Threading Building Blocks (TBB) is now supported
- Algorithms
  - `thrust::for_each_n`
  - `thrust::raw_reference_cast`
- Types
  - `thrust::pointer`
  - `thrust::reference`

### New Examples

- `cuda/custom_temporary_allocation`
- `cuda/fallback_allocator`
- `device_ptr`
- `expand`
- `minimal_custom_backend`
- `raw_reference_cast`
- `set_operations`

### Other Enhancements

- `thrust::for_each` now returns the end of the input range similar to most
    other algorithms.
- `thrust::pair` and `thrust::tuple` have swap functionality.
- All CUDA algorithms now support large data types.
- Iterators may be dereferenced in user `__device__` or `__global__` functions.
- The safe use of different backend systems is now possible within a single
  binary

### Bug Fixes

- #469 `min_element` and `max_element` algorithms no longer require a const comparison operator

### Known Issues

- NVCC may crash when parsing TBB headers on Windows.

## Thrust 1.5.3 (CUDA Toolkit 5.0)

Thrust 1.5.3 is a minor bug fix release.

### Bug Fixes

- Avoid warnings about potential race due to `__shared__` non-POD variable

## Thrust 1.5.2 (CUDA Toolkit 4.2)

Thrust 1.5.2 is a minor bug fix release.

### Bug Fixes

- Fixed warning about C-style initialization of structures

## Thrust 1.5.1 (CUDA Toolkit 4.1)

Thrust 1.5.1 is a minor bug fix release.

### Bug Fixes

- Sorting data referenced by permutation_iterators on CUDA produces invalid results

## Thrust 1.5.0

Thrust 1.5.0 provides introduces new programmer productivity and performance
  enhancements.
New functionality for creating anonymous "lambda" functions has been added.
A faster host sort provides 2-10x faster performance for sorting arithmetic
  types on (single-threaded) CPUs.
A new OpenMP sort provides 2.5x-3.0x speedup over the host sort using a
  quad-core CPU.
When sorting arithmetic types with the OpenMP backend the combined performance
  improvement is 5.9x for 32-bit integers and ranges from 3.0x (64-bit types) to
  14.2x (8-bit types).
A new CUDA `reduce_by_key` implementation provides 2-3x faster
  performance.

### Breaking Changes
- device_ptr<void> no longer unsafely converts to device_ptr<T> without an
    explicit cast.
  Use the expression device_pointer_cast(static_cast<int*>(void_ptr.get())) to
    convert, for example, device_ptr<void> to device_ptr<int>.

### New Features

- Algorithms:
  - Stencil-less `thrust::transform_if`.
- Lambda placeholders

### New Examples
- lambda

### Other Enhancements

- Host sort is 2-10x faster for arithmetic types
- OMP sort provides speedup over host sort
- `reduce_by_key` is 2-3x faster
- `reduce_by_key` no longer requires O(N) temporary storage
- CUDA scan algorithms are 10-40% faster
- `host_vector` and `device_vector` are now documented
- out-of-memory exceptions now provide detailed information from CUDART
- improved histogram example
- `device_reference` now has a specialized swap
- `reduce_by_key` and scan algorithms are compatible with `discard_iterator`

### Bug Fixes

- #44: Allow `thrust::host_vector` to compile when `value_type` uses
    `__align__`.
- #198: Allow `thrust::adjacent_difference` to permit safe in-situ operation.
- #303: Make thrust thread-safe.
- #313: Avoid race conditions in `thrust::device_vector::insert`.
- #314: Avoid unintended ADL invocation when dispatching copy.
- #365: Fix merge and set operation failures.

### Known Issues

- None

### Acknowledgments

- Thanks to Manjunath Kudlur for contributing his Carbon library, from which
    the lambda functionality is derived.
- Thanks to Jean-Francois Bastien for suggesting a fix for #303.

## Thrust 1.4.0 (CUDA Toolkit 4.0)

Thrust 1.4.0 is the first release of Thrust to be included in the CUDA Toolkit.
Additionally, it brings many feature and performance improvements.
New set theoretic algorithms operating on sorted sequences have been added.
Additionally, a new fancy iterator allows discarding redundant or otherwise
  unnecessary output from algorithms, conserving memory storage and bandwidth.

### Breaking Changes

- Eliminations
  - `thrust/is_sorted.h`
  - `thrust/utility.h`
  - `thrust/set_intersection.h`
  - `thrust/experimental/cuda/ogl_interop_allocator.h` and the functionality
      therein
  - `thrust::deprecated::copy_when`
  - `thrust::deprecated::absolute_value`
  - `thrust::deprecated::copy_when`
  - `thrust::deprecated::absolute_value`
  - `thrust::deprecated::copy_when`
  - `thrust::deprecated::absolute_value`
  - `thrust::gather` and `thrust::scatter` from host to device and vice versa
      are no longer supported.
  - Operations which modify the elements of a thrust::device_vector are no longer
      available from source code compiled without nvcc when the device backend
      is CUDA.
    Instead, use the idiom from the cpp_interop example.

### New Features

- Algorithms:
  - `thrust::copy_n`
  - `thrust::merge`
  - `thrust::set_difference`
  - `thrust::set_symmetric_difference`
  - `thrust::set_union`

- Types
  - `thrust::discard_iterator`

- Device Support:
  - Compute Capability 2.1 GPUs.

### New Examples

- run_length_decoding

### Other Enhancements

- Compilation warnings are substantially reduced in various contexts.
- The compilation time of thrust::sort, thrust::stable_sort,
    thrust::sort_by_key, and thrust::stable_sort_by_key are substantially
    reduced.
- A fast sort implementation is used when sorting primitive types with
    thrust::greater.
- The performance of thrust::set_intersection is improved.
- The performance of thrust::fill is improved on SM 1.x devices.
- A code example is now provided in each algorithm's documentation.
- thrust::reverse now operates in-place

### Bug Fixes

- #212: `thrust::set_intersection` works correctly for large input sizes.
- #275: `thrust::counting_iterator` and `thrust::constant_iterator` work
    correctly with OpenMP as the backend when compiling with optimization.
- #256: `min` and `max` correctly return their first argument as a tie-breaker
- #248: `NDEBUG` is interpreted incorrectly

### Known Issues

- NVCC may generate code containing warnings when compiling some Thrust
    algorithms.
- When compiling with `-arch=sm_1x`, some Thrust algorithms may cause NVCC to
    issue benign pointer advisories.
- When compiling with `-arch=sm_1x` and -G, some Thrust algorithms may fail to
    execute correctly.
- `thrust::inclusive_scan`, `thrust::exclusive_scan`,
    `thrust::inclusive_scan_by_key`, and `thrust::exclusive_scan_by_key` are
    currently incompatible with `thrust::discard_iterator`.

### Acknowledgments

- Thanks to David Tarjan for improving the performance of set_intersection.
- Thanks to Duane Merrill for continued help with sort.
- Thanks to Nathan Whitehead for help with CUDA Toolkit integration.

## Thrust 1.3.0

Thrust 1.3.0 provides support for CUDA Toolkit 3.2 in addition to many feature
  and performance enhancements.
Performance of the sort and sort_by_key algorithms is improved by as much as 3x
  in certain situations.
The performance of stream compaction algorithms, such as copy_if, is improved
  by as much as 2x.
CUDA errors are now converted to runtime exceptions using the system_error
  interface.
Combined with a debug mode, also new in 1.3, runtime errors can be located with
  greater precision.
Lastly, a few header files have been consolidated or renamed for clarity.
See the deprecations section below for additional details.

### Breaking Changes

- Promotions
  - thrust::experimental::inclusive_segmented_scan has been renamed
      thrust::inclusive_scan_by_key and exposes a different interface
  - thrust::experimental::exclusive_segmented_scan has been renamed
      thrust::exclusive_scan_by_key and exposes a different interface
  - thrust::experimental::partition_copy has been renamed
      thrust::partition_copy and exposes a different interface
  - thrust::next::gather has been renamed thrust::gather
  - thrust::next::gather_if has been renamed thrust::gather_if
  - thrust::unique_copy_by_key has been renamed thrust::unique_by_key_copy
- Deprecations
  - thrust::copy_when has been renamed thrust::deprecated::copy_when
  - thrust::absolute_value has been renamed thrust::deprecated::absolute_value
  - The header thrust/set_intersection.h is now deprecated; use
      thrust/set_operations.h instead
  - The header thrust/utility.h is now deprecated; use thrust/swap.h instead
  - The header thrust/swap_ranges.h is now deprecated; use thrust/swap.h instead
- Eliminations
  - thrust::deprecated::gather
  - thrust::deprecated::gather_if
  - thrust/experimental/arch.h and the functions therein
  - thrust/sorting/merge_sort.h
  - thrust/sorting/radix_sort.h
- NVCC 2.3 is no longer supported

### New Features

- Algorithms:
  - `thrust::exclusive_scan_by_key`
  - `thrust::find`
  - `thrust::find_if`
  - `thrust::find_if_not`
  - `thrust::inclusive_scan_by_key`
  - `thrust::is_partitioned`
  - `thrust::is_sorted_until`
  - `thrust::mismatch`
  - `thrust::partition_point`
  - `thrust::reverse`
  - `thrust::reverse_copy`
  - `thrust::stable_partition_copy`

- Types:
  - `thrust::system_error` and related types.
  - `thrust::experimental::cuda::ogl_interop_allocator`.
  - `thrust::bit_and`, `thrust::bit_or`, and `thrust::bit_xor`.

- Device Support:
  - GF104-based GPUs.

### New Examples

- opengl_interop.cu
- repeated_range.cu
- simple_moving_average.cu
- sparse_vector.cu
- strided_range.cu

### Other Enhancements

- Performance of thrust::sort and thrust::sort_by_key is substantially improved
    for primitive key types
- Performance of thrust::copy_if is substantially improved
- Performance of thrust::reduce and related reductions is improved
- THRUST_DEBUG mode added
- Callers of Thrust functions may detect error conditions by catching
    thrust::system_error, which derives from std::runtime_error
- The number of compiler warnings generated by Thrust has been substantially
    reduced
- Comparison sort now works correctly for input sizes > 32M
- min & max usage no longer collides with <windows.h> definitions
- Compiling against the OpenMP backend no longer requires nvcc
- Performance of device_vector initialized in .cpp files is substantially
    improved in common cases
- Performance of thrust::sort_by_key on the host is substantially improved

### Bug Fixes

- Debug device code now compiles correctly
- thrust::uninitialized_copy and thrust::uninitialized_fill now dispatch
    constructors on the device rather than the host

### Known Issues

- #212 set_intersection is known to fail for large input sizes
- partition_point is known to fail for 64b types with nvcc 3.2

Acknowledgments
- Thanks to Duane Merrill for contributing a fast CUDA radix sort implementation
- Thanks to Erich Elsen for contributing an implementation of find_if
- Thanks to Andrew Corrigan for contributing changes which allow the OpenMP
    backend to compile in the absence of nvcc
- Thanks to Andrew Corrigan, Cliff Wooley, David Coeurjolly, Janick Martinez
    Esturo, John Bowers, Maxim Naumov, Michael Garland, and Ryuta Suzuki for
    bug reports
- Thanks to Cliff Woolley for help with testing

## Thrust 1.2.1

Thrust 1.2.1 is a small bug fix release that is compatible with the CUDA
  Toolkit 3.1 release.

### Known Issues

- `thrust::inclusive_scan` and `thrust::exclusive_scan` may fail with very
    large types.
- MSVC may fail to compile code using both sort and binary search algorithms.
- `thrust::uninitialized_fill` and `thrust::uninitialized_copy` dispatch
    constructors on the host rather than the device.
- #109: Some algorithms may exhibit poor performance with the OpenMP backend
    with large numbers (>= 6) of CPU threads.
- `thrust::default_random_engine::discard` is not accelerated with NVCC 2.3
- NVCC 3.1 may fail to compile code using types derived from
    `thrust::subtract_with_carry_engine`, such as `thrust::ranlux24` and
    `thrust::ranlux48`.

## Thrust 1.2.0

Thrust 1.2.0 introduces support for compilation to multicore CPUs and the Ocelot
  virtual machine, and several new facilities for pseudo-random number
  generation.
New algorithms such as set intersection and segmented reduction have also been
  added.
Lastly, improvements to the robustness of the CUDA backend ensure correctness
  across a broad set of (uncommon) use cases.

### Breaking Changes

- `thrust::gather`'s interface was incorrect and has been removed.
  The old interface is deprecated but will be preserved for Thrust version 1.2
    at `thrust::deprecated::gather` and `thrust::deprecated::gather_if`.
  The new interface is provided at `thrust::next::gather` and
    `thrust::next::gather_if`.
  The new interface will be promoted to `thrust::` in Thrust version 1.3.
  For more details, please refer to [this thread](http://groups.google.com/group/thrust-users/browse_thread/thread/f5f0583cb97b51fd).
- The `thrust::sorting` namespace has been deprecated in favor of the top-level
    sorting functions, such as `thrust::sort` and `thrust::sort_by_key`.
- Removed support for `thrust::equal` between host & device sequences.
- Removed support for `thrust::scatter` between host & device sequences.

### New Features

- Algorithms:
  - `thrust::reduce_by_key`
  - `thrust::set_intersection`
  - `thrust::unique_copy`
  - `thrust::unique_by_key`
  - `thrust::unique_copy_by_key`
- Types
- Random Number Generation:
  - `thrust::discard_block_engine`
  - `thrust::default_random_engine`
  - `thrust::linear_congruential_engine`
  - `thrust::linear_feedback_shift_engine`
  - `thrust::subtract_with_carry_engine`
  - `thrust::xor_combine_engine`
  - `thrust::minstd_rand`
  - `thrust::minstd_rand0`
  - `thrust::ranlux24`
  - `thrust::ranlux48`
  - `thrust::ranlux24_base`
  - `thrust::ranlux48_base`
  - `thrust::taus88`
  - `thrust::uniform_int_distribution`
  - `thrust::uniform_real_distribution`
  - `thrust::normal_distribution` (experimental)
- Function Objects:
  - `thrust::project1st`
  - `thrust::project2nd`
- `thrust::tie`
- Fancy Iterators:
  - `thrust::permutation_iterator`
  - `thrust::reverse_iterator`
- Vector Functions:
  - `operator!=`
  - `rbegin`
  - `crbegin`
  - `rend`
  - `crend`
  - `data`
  - `shrink_to_fit`
- Device Support:
  - Multicore CPUs via OpenMP.
  - Fermi-class GPUs.
  - Ocelot virtual machines.
- Support for NVCC 3.0.

### New Examples

- `cpp_integration`
- `histogram`
- `mode`
- `monte_carlo`
- `monte_carlo_disjoint_sequences`
- `padded_grid_reduction`
- `permutation_iterator`
- `row_sum`
- `run_length_encoding`
- `segmented_scan`
- `stream_compaction`
- `summary_statistics`
- `transform_iterator`
- `word_count`

### Other Enhancements

- Integer sorting performance is improved when max is large but (max - min) is
    small and when min is negative
- Performance of `thrust::inclusive_scan` and `thrust::exclusive_scan` is
    improved by 20-25% for primitive types.

### Bug Fixes

- #8 cause a compiler error if the required compiler is not found rather than a
    mysterious error at link time
- #42 device_ptr & device_reference are classes rather than structs,
    eliminating warnings on certain platforms
- #46 gather & scatter handle any space iterators correctly
- #51 thrust::experimental::arch functions gracefully handle unrecognized GPUs
- #52 avoid collisions with common user macros such as BLOCK_SIZE
- #62 provide better documentation for device_reference
- #68 allow built-in CUDA vector types to work with device_vector in pure C++
    mode
- #102 eliminated a race condition in device_vector::erase
- various compilation warnings eliminated

### Known Issues

- inclusive_scan & exclusive_scan may fail with very large types
- MSVC may fail to compile code using both sort and binary search algorithms
- uninitialized_fill & uninitialized_copy dispatch constructors on the host
    rather than the device
- #109 some algorithms may exhibit poor performance with the OpenMP backend
    with large numbers (>= 6) of CPU threads
- default_random_engine::discard is not accelerated with nvcc 2.3

### Acknowledgments

- Thanks to Gregory Diamos for contributing a CUDA implementation of
    set_intersection
- Thanks to Ryuta Suzuki & Gregory Diamos for rigorously testing Thrust's unit
    tests and examples against Ocelot
- Thanks to Tom Bradley for contributing an implementation of normal_distribution
- Thanks to Joseph Rhoads for contributing the example summary_statistics

## Thrust 1.1.1

Thrust 1.1.1 is a small bug fix release that is compatible with the CUDA
  Toolkit 2.3a release and Mac OSX Snow Leopard.

## Thrust 1.1.0

Thrust 1.1.0 introduces fancy iterators, binary search functions, and several
  specialized reduction functions.
Experimental support for segmented scans has also been added.

### Breaking Changes

- `thrust::counting_iterator` has been moved into the `thrust` namespace
    (previously `thrust::experimental`).

### New Features

- Algorithms:
  - `thrust::copy_if`
  - `thrust::lower_bound`
  - `thrust::upper_bound`
  - `thrust::vectorized lower_bound`
  - `thrust::vectorized upper_bound`
  - `thrust::equal_range`
  - `thrust::binary_search`
  - `thrust::vectorized binary_search`
  - `thrust::all_of`
  - `thrust::any_of`
  - `thrust::none_of`
  - `thrust::minmax_element`
  - `thrust::advance`
  - `thrust::inclusive_segmented_scan` (experimental)
  - `thrust::exclusive_segmented_scan` (experimental)
- Types:
  - `thrust::pair`
  - `thrust::tuple`
  - `thrust::device_malloc_allocator`
- Fancy Iterators:
  - `thrust::constant_iterator`
  - `thrust::counting_iterator`
  - `thrust::transform_iterator`
  - `thrust::zip_iterator`

### New Examples

- Computing the maximum absolute difference between vectors.
- Computing the bounding box of a two-dimensional point set.
- Sorting multiple arrays together (lexicographical sorting).
- Constructing a summed area table.
- Using `thrust::zip_iterator` to mimic an array of structs.
- Using `thrust::constant_iterator` to increment array values.

### Other Enhancements

- Added pinned memory allocator (experimental).
- Added more methods to host_vector & device_vector (issue #4).
- Added variant of remove_if with a stencil argument (issue #29).
- Scan and reduce use cudaFuncGetAttributes to determine grid size.
- Exceptions are reported when temporary device arrays cannot be allocated.

### Bug Fixes

- #5: Make vector work for larger data types
- #9: stable_partition_copy doesn't respect OutputIterator concept semantics
- #10: scans should return OutputIterator
- #16: make algorithms work for larger data types
- #27: Dispatch radix_sort even when comp=less<T> is explicitly provided

### Known Issues

- Using functors with Thrust entry points may not compile on Mac OSX with gcc
    4.0.1.
- `thrust::uninitialized_copy` and `thrust::uninitialized_fill` dispatch
    constructors on the host rather than the device.
- `thrust::inclusive_scan`, `thrust::inclusive_scan_by_key`,
    `thrust::exclusive_scan`, and `thrust::exclusive_scan_by_key` may fail when
    used with large types with the CUDA Toolkit 3.1.

## Thrust 1.0.0

First production release of Thrust.

### Breaking Changes

- Rename top level namespace `komrade` to `thrust`.
- Move `thrust::partition_copy` & `thrust::stable_partition_copy` into
    `thrust::experimental` namespace until we can easily provide the standard
    interface.
- Rename `thrust::range` to `thrust::sequence` to avoid collision with
    Boost.Range.
- Rename `thrust::copy_if` to `thrust::copy_when` due to semantic differences
    with C++0x `std::copy_if`.

### New Features

- Add C++0x style `cbegin` & `cend` methods to `thrust::host_vector` and
    `thrust::device_vector`.
- Add `thrust::transform_if` function.
- Add stencil versions of `thrust::replace_if` & `thrust::replace_copy_if`.
- Allow `counting_iterator` to work with `thrust::for_each`.
- Allow types with constructors in comparison `thrust::sort` and
    `thrust::reduce`.

### Other Enhancements

- `thrust::merge_sort` and `thrust::stable_merge_sort` are now 2x to 5x faster
    when executed on the parallel device.

### Bug Fixes

- Komrade 6: Workaround an issue where an incremented iterator causes NVCC to
    crash.
- Komrade 7: Fix an issue where `const_iterator`s could not be passed to
    `thrust::transform`.

