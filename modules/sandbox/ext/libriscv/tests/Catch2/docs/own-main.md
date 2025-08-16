<a id="top"></a>
# Supplying main() yourself

**Contents**<br>
[Let Catch2 take full control of args and config](#let-catch2-take-full-control-of-args-and-config)<br>
[Amending the Catch2 config](#amending-the-catch2-config)<br>
[Adding your own command line options](#adding-your-own-command-line-options)<br>
[Version detection](#version-detection)<br>

The easiest way to use Catch2 is to use its own `main` function, and let
it handle the command line arguments. This is done by linking against
Catch2Main library, e.g. through the [CMake target](cmake-integration.md#cmake-targets),
or pkg-config files.

If you want to provide your own `main`, then you should link against
the static library (target) only, without the main part. You will then
have to write your own `main` and call into Catch2 test runner manually.

Below are some basic recipes on what you can do supplying your own main.


## Let Catch2 take full control of args and config

This is useful if you just need to have code that executes before/after
Catch2 runs tests.

```cpp
#include <catch2/catch_session.hpp>

int main( int argc, char* argv[] ) {
  // your setup ...

  int result = Catch::Session().run( argc, argv );

  // your clean-up...

  return result;
}
```

_Note that if you only want to run some set up before tests are run, it
might be simpler to use [event listeners](event-listeners.md#top) instead._


## Amending the Catch2 config

If you want Catch2 to process command line arguments, but also want to
programmatically change the resulting configuration of Catch2 run,
you can do it in two ways:

```c++
int main( int argc, char* argv[] ) {
  Catch::Session session; // There must be exactly one instance

  // writing to session.configData() here sets defaults
  // this is the preferred way to set them

  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
    return returnCode;

  // writing to session.configData() or session.Config() here
  // overrides command line args
  // only do this if you know you need to

  returnCode = session.run();

  // returnCode encodes the type of error that occured. See the
  // integer constants in catch_session.hpp for more information
  // on what each return code means.

  return numFailed;
}
```

If you want full control of the configuration, don't call `applyCommandLine`.


## Adding your own command line options

You can add new command line options to Catch2, by composing the premade
CLI parser (called Clara), and add your own options.

```cpp
int main( int argc, char* argv[] ) {
  Catch::Session session; // There must be exactly one instance

  int height = 0; // Some user variable you want to be able to set

  // Build a new parser on top of Catch2's
  using namespace Catch::Clara;
  auto cli
    = session.cli()           // Get Catch2's command line parser
    | Opt( height, "height" ) // bind variable to a new option, with a hint string
        ["-g"]["--height"]    // the option names it will respond to
        ("how high?");        // description string for the help output

  // Now pass the new composite back to Catch2 so it uses that
  session.cli( cli );

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
      return returnCode;

  // if set on the command line then 'height' is now set at this point
  if( height > 0 )
      std::cout << "height: " << height << std::endl;

  return session.run();
}
```

See the [Clara documentation](https://github.com/catchorg/Clara/blob/master/README.md)
for more details on how to use the Clara parser.


## Version detection

Catch2 provides a triplet of macros providing the header's version,

* `CATCH_VERSION_MAJOR`
* `CATCH_VERSION_MINOR`
* `CATCH_VERSION_PATCH`

these macros expand into a single number, that corresponds to the appropriate
part of the version. As an example, given single header version v2.3.4,
the macros would expand into `2`, `3`, and `4` respectively.


---

[Home](Readme.md#top)
