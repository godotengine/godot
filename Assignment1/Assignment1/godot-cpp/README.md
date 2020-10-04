# godot-cpp

**C++ bindings for the Godot script API.**

The instructions below feature the new NativeScript 1.1 class structure and will only work for modules created for Godot 3.1 and later. Use the following branches for older implementations:

Version | Branch
--- | ---
**Godot 3.0 Nativescript 1.0** | [3.0](https://github.com/GodotNativeTools/godot-cpp/tree/3.0)
**Godot 3.1 Nativescript 1.0** | [nativescript-1.0](https://github.com/GodotNativeTools/godot-cpp/tree/nativescript-1.0)

## Table of contents

- [**Contributing**](#contributing)
- [**Getting Started**](#getting-started)
- [**Creating a simple class**](#creating-a-simple-class)

## Contributing

We greatly appreciate help in maintaining and extending this project. If you
wish to help out, ensure you have an account on GitHub and create a "fork" of
this repository. Rémi "Akien" Verschelde wrote an excellent bit of documentation
for the main Godot project on this:
[Pull request workflow](https://docs.godotengine.org/en/3.0/community/contributing/pr_workflow.html)

Please install clang-format and copy the files in `misc/hooks` into `.git/hooks`
so formatting is done before your changes are submitted.

## Getting Started

| **Build latest version of Godot** | [**GitHub**](https://github.com/godotengine/godot) | [**Docs**](https://godot.readthedocs.io/en/latest/development/compiling/index.html) |
| --- | --- | --- |

### Setting up a new project

We recommend using Git for managing your project. The instructions below assume
you're using Git. Alternatively, you can download the source code directly from
GitHub. In this case, you need to download both
[godot-cpp](https://github.com/GodotNativeTools/godot-cpp) and
[godot_headers](https://github.com/GodotNativeTools/godot_headers).

```bash
mkdir SimpleLibrary
cd SimpleLibrary
mkdir bin
mkdir src
git clone --recursive https://github.com/GodotNativeTools/godot-cpp
```

If you wish to use a specific branch, add the -b option to the clone command:

```bash
git clone --recursive https://github.com/GodotNativeTools/godot-cpp -b 3.0
```

If your project is an existing repository, use a Git submodule instead:

```bash
git submodule add https://github.com/GodotNativeTools/godot-cpp
git submodule update --init --recursive
```

Right now, our directory structure should look like this:

```text
SimpleLibrary/
├─godot-cpp/
| └─godot_headers/
├─bin/
└─src/
```

### Updating the `api.json` file

Our `api.json` file contains metadata for all the classes that are part of the
Godot core. This metadata is required to generate the C++ binding classes for
use in GDNative modules.

This file is supplied with our
[godot_headers](https://github.com/GodotNativeTools/godot_headers) repository
for your convenience. However, if you're running a custom build of Godot and
need access to classes that have recent changes, you must generate a new
`api.json` file. You do this by starting your Godot executable with the
following parameters:

```bash
godot --gdnative-generate-json-api api.json
```

Now copy the `api.json` file into your folder structure to make it easier to
access.

See the remark below for the extra ```custom_api_file``` SCons argument, which
is required to tell SCons where to find your file.

### Compiling the C++ bindings library

The final step is to compile our C++ bindings library:

```bash
cd godot-cpp
scons platform=<your platform> generate_bindings=yes
cd ..
```

Replace `<your platform>` with either `windows`, `linux`, `osx` or `android`. If
you leave out `platform`, the target platform will automatically be detected
from the host platform.

The resulting library will be created in `godot-cpp/bin/`, take note of its name
as it'll differ depending on the target platform.

#### Compiling for Android

Download the latest [Android NDK](https://developer.android.com/ndk/downloads)
and set the NDK path.

```bash
scons platform=android generate_bindings=yes ANDROID_NDK_ROOT="/PATH-TO-ANDROID-NDK/" android_arch=<arch>
```

The value of `android_arch` can be `armv7, arm64v8, x86, x86_64`. Most Android
devices in use nowadays use an ARM architecture, so compiling for `armv7` and
`arm64v8` is often enough when distributing an application.

`ANDROID_NDK_ROOT` can also be set in the environment variables of your PC if
you don't want to include it in your SCons call.

#### Compilation options

You can optionally add the following options to the SCons command line:

- When targeting Linux, add `use_llvm=yes` to use Clang instead of GCC.
- When targeting Windows, add `use_mingw=yes` to use MinGW instead of MSVC.
- When targeting Windows, include `target=runtime` to build a runtime build.
- To use an alternative `api.json` file, add `use_custom_api_file=yes
  custom_api_file=../api.json`. Be sure to specify the correct location where
  you placed your file (it can be a relative or absolute path).

## Creating a simple class

Create `init.cpp` under `SimpleLibrary/src/` and add the following code:

```cpp
#include <Godot.hpp>
#include <Reference.hpp>

using namespace godot;

class SimpleClass : public Reference {
    GODOT_CLASS(SimpleClass, Reference);
public:
    SimpleClass() { }

    /** `_init` must exist as it is called by Godot. */
    void _init() { }

    void test_void_method() {
        Godot::print("This is test");
    }

    Variant method(Variant arg) {
        Variant ret;
        ret = arg;

        return ret;
    }

    static void _register_methods() {
        register_method("method", &SimpleClass::method);

        /**
         * The line below is equivalent to the following GDScript export:
         *     export var _name = "SimpleClass"
         **/
        register_property<SimpleClass, String>("base/name", &SimpleClass::_name, String("SimpleClass"));

        /** Alternatively, with getter and setter methods: */
        register_property<SimpleClass, int>("base/value", &SimpleClass::set_value, &SimpleClass::get_value, 0);

        /** Registering a signal: **/
        // register_signal<SimpleClass>("signal_name");
        // register_signal<SimpleClass>("signal_name", "string_argument", GODOT_VARIANT_TYPE_STRING)
    }

    String _name;
    int _value;

    void set_value(int p_value) {
        _value = p_value;
    }

    int get_value() const {
        return _value;
    }
};

/** GDNative Initialize **/
extern "C" void GDN_EXPORT godot_gdnative_init(godot_gdnative_init_options *o) {
    godot::Godot::gdnative_init(o);
}

/** GDNative Terminate **/
extern "C" void GDN_EXPORT godot_gdnative_terminate(godot_gdnative_terminate_options *o) {
    godot::Godot::gdnative_terminate(o);
}

/** NativeScript Initialize **/
extern "C" void GDN_EXPORT godot_nativescript_init(void *handle) {
    godot::Godot::nativescript_init(handle);

    godot::register_class<SimpleClass>();
}
```

### Compiling the GDNative library

Once you've compiled the GDNative C++ bindings (see above), you can compile the GDNative library we've just created.

#### Linux

```bash
cd SimpleLibrary
clang++ -fPIC -o src/init.o -c src/init.cpp -g -O3 -std=c++14 -Igodot-cpp/include -Igodot-cpp/include/core -Igodot-cpp/include/gen -Igodot-cpp/godot_headers
clang++ -o bin/libtest.so -shared src/init.o -Lgodot-cpp/bin -l<name of the godot-cpp>
```

You'll need to replace `<name of the godot-cpp>` with the file that was created in [**Compiling the cpp bindings library**](#compiling-the-cpp-bindings-library).

This creates the file `libtest.so` in your `SimpleLibrary/bin` directory.

#### Windows

```bash
cd SimpleLibrary
cl /Fosrc/init.obj /c src/init.cpp /nologo -EHsc -DNDEBUG /MDd /Igodot-cpp\include /Igodot-cpp\include\core /Igodot-cpp\include\gen /Igodot-cpp\godot_headers
link /nologo /dll /out:bin\libtest.dll /implib:bin\libsimple.lib src\init.obj godot-cpp\bin\<name of the godot-cpp>
```

You'll need to replace `<name of the godot-cpp>` with the file that was created
in [**Compiling the cpp bindingslibrary**](#compiling-the-cpp-bindings-library).
Replace `/MDd` with `/MD` to create a release build, which will run faster and
be smaller.

This creates the file `libtest.dll` in your `SimpleLibrary/bin` directory.

#### macOS

For macOS, you'll need to find out which compiler flags need to be used. These
are likely similar to Linux when using Clang, but may not be identical.

If you find suitable compiler flags for this example library, feel free to
submit a pull request :slightly_smiling_face:

#### Android

```bash
cd SimpleLibrary
aarch64-linux-android29-clang++ -fPIC -o src/init.o -c src/init.cpp -g -O3 -std=c++14 -Igodot-cpp/include -Igodot-cpp/include/core -Igodot-cpp/include/gen -Igodot-cpp/godot_headers
aarch64-linux-android29-clang++ -o bin/libtest.so -shared src/init.o -Lgodot-cpp/bin -l<name of the godot-cpp>
```

You'll need to replace `<name of the godot-cpp>` with the file that was created in [**Compiling the cpp bindings library**](#compiling-the-cpp-bindings-library). The command above targets `arm64v8`. To target `armv7`, use `armv7a-linux-androideabi29-clang++` instead of `aarch64-linux-android29-clang++`.

This creates the file `libtest.so` in your `SimpleLibrary/bin` directory.

#### iOS

GDNative isn't supported on iOS yet. This is because iOS only allows linking
static libraries, not dynamic libraries. In theory, it would be possible to link
a GDNative library statically, but some of GDNative's convenience would be lost
in the process as one would have to recompile the engine on every change. See
[issue #30](https://github.com/GodotNativeTools/godot_headers/issues/30) in the
Godot headers repository for more information.

#### HTML5

GDNative isn't supported on the HTML5 platform yet. Support is being tracked on
[issue #12243](https://github.com/godotengine/godot/issues/12243) in the main
Godot repository.

### Creating `.gdnlib` and `.gdns` files

Follow the instructions in
[godot_header/README.md](https://github.com/GodotNativeTools/godot_headers/blob/master/README.md#how-do-i-use-native-scripts-from-the-editor)
to create the `.gdns` file. This file contains paths to GDNative libraries for
various platforms. This makes the library usable from Godot in a
platform-independent manner.

### Implementing with GDScript

Once your GDNative library is compiled and referenced in a `.gdns` file, you can use it in GDScript or C#. Here's an example with GDScript:

```gdscript
var simpleclass = load("res://simpleclass.gdns").new();
simpleclass.method("Test argument");
```

### Using Godot classes in C++

Godot expects you to manage its classes the same way the engine does. These rules apply to all Godot classes, including your NativeScripts, but not to any normal C++ classes used in your library.

- Instantiate Objects using `_new()`, not C++'s `new` operator.

```cpp
Sprite *sprite = Sprite::_new();
```

- Destroy Nodes using `queue_free()`, not C++'s `delete` operator.

```cpp
some_old_node->queue_free();
```

- Wrap References in `Ref` instead of passing around raw pointers. They are reference-counted and don't need to be freed manually.

```cpp
Ref<Texture> texture = resource_loader->load("res://icon.png");
```

- Pass core types that do *not* inherit Object by value. The containers (Array, Dictionary, PoolArray, String) manage their own memory and do not need to be explicitly initialized or freed.

```cpp
Array ints;
ints.append(123);
return ints;
```

- Initialize your NativeScript classes in their `_init()` method, not their constructor. The constructor can't access the base class's methods.

- Cast objects using `Object::cast_to`, not unsafe C-style casts or `static_cast`.

```cpp
MeshInstance *m = Object::cast_to<MeshInstance>(get_node("ChildNode"));
// `m` will be null if it's not a MeshInstance
if (m) { ... }
```

- **Never** use Godot types in static or global variables. The Godot API isn't loaded until after their constructors are called.

```cpp
String s; // crashes
class SomeClass {
    static Dictionary d; // crashes

    static Node *node_a = NULL; // fine, it's just a pointer
    static Node *node_b = Node::_new(); // crashes
};
```
