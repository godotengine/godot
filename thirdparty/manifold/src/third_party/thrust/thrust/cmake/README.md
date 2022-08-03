# Using Thrust with CMake

Thrust provides configuration files that simplify using Thrust
from other CMake projects. Requirements:

- Thrust >= 1.9.10
- CMake >= 3.15

See the [Fixing Legacy FindThrust.cmake](#fixing-legacy-findthrustcmake)
section for solutions that work on older Thrust versions.

## User Guide

#### Default Configuration (CUDA)

Thrust is configured using a `thrust_create_target` CMake function that
assembles a complete interface to the Thrust library:

```cmake
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
target_link_libraries(MyProgram Thrust)
```

The first argument is the name of the interface target to create, and any
additional options will be used to configure the target. By default,
`thrust_create_target` will configure its result to use CUDA acceleration.

If desired, `thrust_create_target` may be called multiple times to build
several unique Thrust interface targets with different configurations, as
detailed below.

**Note:** If CMake is unable to locate Thrust, specify the path to Thrust's CMake
configuration directory (where this README file is located) as `Thrust_DIR`.
If cloning Thrust from github, this would be

```
$ cmake . -DThrust_DIR=<thrust git repo root>/thrust/cmake/
```

#### TBB / OpenMP

To explicitly specify host/device systems, `HOST` and `DEVICE` arguments can be
passed to `thrust_create_target`. If an explicit system is not specified, the
target will default to using CPP for host and/or CUDA for device.

```cmake
thrust_create_target(ThrustTBB DEVICE TBB)
thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)
```

will create targets `ThrustTBB` and `ThrustOMP`. Both will use the serial `CPP`
host system, but will find and use TBB or OpenMP for the device system.

#### Configure Target from Cache Options

To allow a Thrust target to be configurable easily via `cmake-gui` or
`ccmake`, pass the `FROM_OPTIONS` flag to `thrust_create_target`. This will add
`THRUST_HOST_SYSTEM` and `THRUST_DEVICE_SYSTEM` options to the CMake cache that
allow selection from the systems supported by this version of Thrust.

```cmake
thrust_create_target(Thrust FROM_OPTIONS
  [HOST_OPTION <option name>]
  [DEVICE_OPTION <option name>]
  [HOST_OPTION_DOC <doc string>]
  [DEVICE_OPTION_DOC <doc string>]
  [HOST <default host system name>]
  [DEVICE <default device system name>]
  [ADVANCED]
)
```

The optional arguments have sensible defaults, but may be configured per
`thrust_create_target` call:

| Argument            | Default                 | Description                     |
|---------------------|-------------------------|---------------------------------|
| `HOST_OPTION`       | `THRUST_HOST_SYSTEM`    | Name of cache option for host   |
| `DEVICE_OPTION`     | `THRUST_DEVICE_SYSTEM`  | Name of cache option for device |
| `HOST_OPTION_DOC`   | Thrust's host system.   | Docstring for host option       |
| `DEVICE_OPTION_DOC` | Thrust's device system. | Docstring for device option     |
| `HOST`              | `CPP`                   | Default host system             |
| `DEVICE`            | `CUDA`                  | Default device system           |
| `ADVANCED`          | *N/A*                   | Mark cache options advanced     |

### Specifying Thrust Version Requirements

A specific version of Thrust may be required in the `find_package` call:

```cmake
find_package(Thrust 1.9.10)
```

will only consider Thrust installations with version `1.9.10.X`. An exact match
down to the patch version can be forced by using `EXACT` matching:

```cmake
find_package(Thrust 1.9.10.1 EXACT)
```

would only match the 1.9.10.1 release.

#### Using a Specific TBB or OpenMP Environment

When `thrust_create_target` is called, it will lazily load the requested
systems on-demand through internal `find_package` calls. If a project already
uses TBB or OpenMP, it may specify a CMake target for Thrust to share instead:

```cmake
thrust_set_TBB_target(MyTBBTarget)
thrust_set_OMP_target(MyOMPTarget)
```

These functions must be called **before** `thrust_create_target`, and will
have no effect if the dependency is loaded as a
`find_package(Thrust COMPONENT [...])` component.

#### Testing for Systems

The following functions check if a system has been found, either by lazy loading
through `thrust_create_target` or as a `find_package` `COMPONENT` /
`OPTIONAL_COMPONENT`:

```cmake
# Set var_name to TRUE or FALSE if an individual system has been found:
thrust_is_cuda_system_found(<var_name>)
thrust_is_cpp_system_found(<var_name>)
thrust_is_tbb_system_found(<var_name>)
thrust_is_omp_system_found(<var_name>)

# Generic version that takes a component name from CUDA, CPP, TBB, OMP:
thrust_is_system_found(<component_name> <var_name>)

# Defines `THRUST_*_FOUND` variables in the current scope that reflect the
# state of all known systems. Can be used to refresh these flags after
# lazy system loading.
thrust_update_system_found_flags()
```

#### Debugging

Thrust will produce a detailed log describing its targets, cache options, and
interfaces when `--log-level=VERBOSE` is passed to CMake 3.15.7 or newer:

```
$ cmake . --log-level=VERBOSE
```

This can be handy for inspecting interface and dependency information.

## Fixing Legacy FindThrust.cmake

A community-created `FindThrust.cmake` module exists and is necessary to find
Thrust installations prior to Thrust 1.9.10. Its usage is discouraged whenever
possible and the config files in this directory should be strongly preferred.
However, projects that need to support old versions of Thrust may still need to
use the legacy `FindThrust.cmake` with pre-1.9.10 installations.

One popular flavor of this find module has a version parsing bug. Projects that
rely on `FindThrust.cmake` should check for this and patch their copies as
follows.

Replace:

```cmake
string( REGEX MATCH "^[0-9]" major ${version} )
string( REGEX REPLACE "^${major}00" "" version "${version}" )
string( REGEX MATCH "^[0-9]" minor ${version} )
string( REGEX REPLACE "^${minor}0" "" version "${version}" )
```

with:

```cmake
math(EXPR major "${version} / 100000")
math(EXPR minor "(${version} / 100) % 1000")
math(EXPR version "${version} % 100")
```

# Thrust Developer Documentation

This portion of the file contains descriptions of Thrust's internal CMake target
structure for Thrust developers. It should not be necessary for users
who just want to use Thrust from their projects.

## Internal Targets

By default, `find_package(Thrust)` will only create a single `Thrust::Thrust`
target that describes where the actual Thrust headers are located. It does not
locate or create configurations for any dependencies; these are lazily loaded
on-demand by calls to `create_thrust_target`, or when explicitly requested via
`find_package`'s component mechanism.

As mentioned, the basic Thrust interface is described by the `Thrust::Thrust`
target.

Each backend system (`CPP`, `CUDA`, `TBB`, `OMP`) is described by multiple
targets:

- `Thrust::${system}`
  - Specifies an interface configured to build against all
    dependencies for this backend (including `Thrust::Thrust`).
  - For example, the `Thrust::CUDA` target is an interface
    target that combines the interfaces of both Thrust and CUB.
- `Thrust::${system}::Host`
  - Configures an interface for using a specific host system.
  - Multiple `::Host` targets cannot be combined in the same library/executable.
    Attempting to do so will produce a CMake configuration error.
  - Only defined for systems that support being used as the host.
- `Thrust::${system}::Device`
  - Configures an interface for using a specific device system.
  - Multiple `::Device` targets cannot be combined in the same library/executable.
    Attempting to do so will produce a CMake configuration error.
  - Only defined for systems that support being used as the device.
