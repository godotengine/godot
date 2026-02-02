<p align="center">
  <img height="90" src="img/logo.png"/>  
</p>

<p align="center">
  Unix-style pathname pattern expansion
</p>

## Table of Contents

- [Quick Start](#quick-start)
  * [Build Library and Standalone Sample](#build-library-and-standalone-sample)
- [Usage](#usage)
- [API](#api)
- [Wildcards](#wildcards)
- [Examples](#examples)
  * [Match file extensions](#match-file-extensions)
  * [Match files in absolute pathnames](#match-files-in-absolute-pathnames)
  * [Wildcards: Match a range of characters listed in brackets ('[]')](#wildcards-match-a-range-of-characters-listed-in-brackets-)
  * [Exclude files from the matching](#exclude-files-from-the-matching)
  * [Wildcards: Match any one character with question mark ('?')](#wildcards-match-any-one-character-with-question-mark-)
  * [Case sensitivity](#case-sensitivity)
  * [Tilde expansion](#tilde-expansion)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

* This library is available in two flavors:
  1. Two file version: `glob.h` and `glob.cpp`
  2. Single header file version in `single_include/`
* No external dependencies - just the standard library
* Requires C++17 `std::filesystem`
  - If you can't use `C++17`, you can integrate [gulrak/filesystem](https://github.com/gulrak/filesystem) with minimal effort.
* MIT License

### Build Library and Standalone Sample

```bash
cmake -Hall -Bbuild
cmake --build build

# run standalone `glob` sample
./build/standalone/glob --help
```

### Usage

```cpp
// Match on a single pattern
for (auto& p : glob::glob("~/.b*")) {                // e.g., .bash_history, .bashrc
  // do something with `p`
}

// Match on multiple patterns
for (auto& p : glob::glob({"*.png", "*.jpg"})) {     // e.g., foo.png, bar.jpg
  // do something with `p`
}

// Match recursively with `rglob`
for (auto& p : glob::rglob("**/*.hpp")) {            // e.g., include/foo.hpp, include/foo/bar.hpp
  // do something with `p`
}
```

## API

```cpp
/// e.g., glob("*.hpp")
/// e.g., glob("**/*.cpp")
/// e.g., glob("test_files_02/[0-9].txt")
/// e.g., glob("/usr/local/include/nc*.h")
/// e.g., glob("test_files_02/?.txt")
vector<filesystem::path> glob(string pathname);

/// Globs recursively
/// e.g., rglob("Documents/Projects/Foo/**/*.hpp")
/// e.g., rglob("test_files_02/*[0-9].txt")
vector<filesystem::path> rglob(string pathname);
```

There are also two convenience functions to `glob` on a list of patterns:

```cpp
/// e.g., glob({"*.png", "*.jpg"})
vector<filesystem::path> glob(vector<string> pathnames);

/// Globs recursively
/// e.g., rglob({"**/*.h", "**/*.hpp", "**/*.cpp"})
vector<filesystem::path> rglob(vector<string> pathnames);
```

## Wildcards

| Wildcard | Matches | Example
|--- |--- |--- |
| `*` | any characters | `*.txt` matches all files with the txt extension |
| `?` | any one character | `???` matches files with 3 characters long |
| `[]` | any character listed in the brackets | `[ABC]*` matches files starting with A,B or C | 
| `[-]` | any character in the range listed in brackets | `[A-Z]*` matches files starting with capital letters |
| `[!]` | any character not listed in the brackets | `[!ABC]*` matches files that do not start with A,B or C |

## Examples

The following examples use the [standalone](standalone/source/main.cpp) sample that is part of this repository to illustrate the library functionality.

```console
foo@bar:~$ ./build/standalone/glob -h
Run glob to find all the pathnames matching a specified pattern
Usage:
  ./build/standalone/glob [OPTION...]

  -h, --help       Show help
  -v, --version    Print the current version number
  -r, --recursive  Run glob recursively
  -i, --input arg  Patterns to match
```

### Match file extensions

```console
foo@bar:~$ tree
.
├── include
│   └── foo
│       ├── bar.hpp
│       ├── baz.hpp
│       └── foo.hpp
└── test
    ├── bar.cpp
    ├── doctest.hpp
    ├── foo.cpp
    └── main.cpp

3 directories, 7 files

foo@bar:~$ ./glob -i "**/*.hpp"
"test/doctest.hpp"

foo@bar:~$ ./glob -i "**/**/*.hpp"
"include/foo/baz.hpp"
"include/foo/foo.hpp"
"include/foo/bar.hpp"
```

***NOTE*** If you run glob recursively, i.e., using `rglob`:

```console
foo@bar:~$ ./glob -r -i "**/*.hpp"
"test/doctest.hpp"
"include/foo/baz.hpp"
"include/foo/foo.hpp"
"include/foo/bar.hpp"
```

### Match files in absolute pathnames

```console
foo@bar:~$ ./glob -i '/usr/local/include/nc*.h'
"/usr/local/include/ncCheck.h"
"/usr/local/include/ncGroupAtt.h"
"/usr/local/include/ncUshort.h"
"/usr/local/include/ncByte.h"
"/usr/local/include/ncString.h"
"/usr/local/include/ncUint64.h"
"/usr/local/include/ncGroup.h"
"/usr/local/include/ncUbyte.h"
"/usr/local/include/ncvalues.h"
"/usr/local/include/ncInt.h"
"/usr/local/include/ncAtt.h"
"/usr/local/include/ncVar.h"
"/usr/local/include/ncUint.h"
```

### Wildcards: Match a range of characters listed in brackets ('[]')

```console
foo@bar:~$ ls test_files_02
1.txt 2.txt 3.txt 4.txt

foo@bar:~$ ./glob -i 'test_files_02/[0-9].txt'
"test_files_02/4.txt"
"test_files_02/3.txt"
"test_files_02/2.txt"
"test_files_02/1.txt"

foo@bar:~$ ./glob -i 'test_files_02/[1-2]*'
"test_files_02/2.txt"
"test_files_02/1.txt"
```

```console
foo@bar:~$ ls test_files_03
file1.txt file2.txt file3.txt file4.txt

foo@bar:~$ ./glob -i 'test_files_03/file[0-9].*'
"test_files_03/file2.txt"
"test_files_03/file3.txt"
"test_files_03/file1.txt"
"test_files_03/file4.txt"
```

### Exclude files from the matching

```console
foo@bar:~$ ls test_files_01
__init__.py     bar.py      foo.py

foo@bar:~$ ./glob -i 'test_files_01/*[!__init__].py'
"test_files_01/bar.py"
"test_files_01/foo.py"

foo@bar:~$ ./glob -i 'test_files_01/*[!__init__][!bar].py'
"test_files_01/foo.py"

foo@bar:~$ ./glob -i 'test_files_01/[!_]*.py'
"test_files_01/bar.py"
"test_files_01/foo.py"
```

### Wildcards: Match any one character with question mark ('?')

```console
foo@bar:~$ ls test_files_02
1.txt 2.txt 3.txt 4.txt

foo@bar:~$ ./glob -i 'test_files_02/?.txt'
"test_files_02/4.txt"
"test_files_02/3.txt"
"test_files_02/2.txt"
"test_files_02/1.txt"
```

```console
foo@bar:~$ ls test_files_03
file1.txt file2.txt file3.txt file4.txt

foo@bar:~$ ./glob -i 'test_files_03/????[3-4].txt'
"test_files_03/file3.txt"
"test_files_03/file4.txt"
```

### Case sensitivity

`glob` matching is case-sensitive:

```console
foo@bar:~$ ls test_files_05
file1.png file2.png file3.PNG file4.PNG

foo@bar:~$ ./glob -i 'test_files_05/*.png'
"test_files_05/file2.png"
"test_files_05/file1.png"

foo@bar:~$ ./glob -i 'test_files_05/*.PNG'
"test_files_05/file3.PNG"
"test_files_05/file4.PNG"

foo@bar:~$ ./glob -i "test_files_05/*.png","test_files_05/*.PNG"
"test_files_05/file2.png"
"test_files_05/file1.png"
"test_files_05/file3.PNG"
"test_files_05/file4.PNG"
```

### Tilde expansion

```console
foo@bar:~$ ./glob -i "~/.b*"
"/Users/pranav/.bashrc"
"/Users/pranav/.bash_sessions"
"/Users/pranav/.bash_profile"
"/Users/pranav/.bash_history"

foo@bar:~$ ./glob -i "~/Documents/Projects/glob/**/glob/*.h"
"/Users/pranav/Documents/Projects/glob/include/glob/glob.h"
```

## Contributing
Contributions are welcome, have a look at the [CONTRIBUTING.md](CONTRIBUTING.md) document for more information.

## License
The project is available under the [MIT](https://opensource.org/licenses/MIT) license.
