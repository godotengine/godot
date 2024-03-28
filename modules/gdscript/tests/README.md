# GDScript integration tests

The `scripts/` folder contains integration tests in the form of GDScript files
and output files.

See the
[Integration tests for GDScript documentation](https://docs.godotengine.org/en/latest/contributing/development/core_and_modules/unit_testing.html#integration-tests-for-gdscript)
for information about creating and running GDScript integration tests.

# GDScript Autocompletion tests

The `script/completion` folder contains test for the GDScript autocompletion.

Each test case consists of at least one `.gd` file, which contains the code, and one `.cfg` file, which contains expected results and configuration. Inside of the GDScript file the character `➡` represents the cursor position, at which autocompletion is invoked.

The script files won't be parsable GDScript since it contains an invalid char and and often the code is not complete during autocompletion. To allow for a valid base when used with a scene, the
runner will remove the line which contains `➡`. Therefore the scripts need to be valid if this line is removed, otherwise the test might behave in unexpected ways. This may for example require
adding an additional `pass` statement.

This also means, that the runner will add the script to its owner node, so the script should not be loaded through the scene file.

The config file contains two section:

`[input]` contains keys that configure the test environment. The following keys are possible:

- `cs: boolean = false`: If `true`, the test will be skipped when running a non C# build.
- `use_single_quotes: boolean = false`: Configures the corresponding editor setting for the test.
- `scene: String`: Allows to specify a scene which is opened while autocompletion is performed. If this is not set the test runner will search for a `.tscn` file with the same basename as the GDScript file. If that isn't found either, autocompletion will behave as if no scene was opened.
- `node_path: String`: The node path of the node which holds the current script inside of the scene. Defaults to the scene root node.

`[output]` specifies the expected results for the test. The following key are supported:

- `include: Array`: An unordered list of suggestions that should be in the result. Each entry is one dictionary with the following keys: `display`, `insert_text`, `kind`, `location`, which correspond to the suggestion struct which is used in the code. The runner only tests against specified keys, so in most cases `display` will suffice.
- `exclude: Array`: An array of suggestions which should not be in the result. The entries take the same form as for `include`.
- `call_hint: String`: The expected call hint returned by autocompletion.
- `forced: boolean`: Whether autocompletion is expected to force opening a completion window.

Tests will only test against entries in `[output]` that were specified.

## Writing autocompletion tests

To avoid failing edge cases a certain behavior needs to be tested multiple times. Some things that tests should account for:

- All possible types: Test with all possible types that apply to the tested behavior. (For the last points testing against `SCRIPT` and `CLASS` should suffice. `CLASS` can be obtained through C#, `SCRIPT` through GDScript. Relying on autoloads to be of type `SCRIPT` is not good, since this might change in the future.)

  - `BUILTIN`
  - `NATIVE`
  - GDScripts (with `class_name` as well as `preload`ed)
  - C# (as standin for all other language bindings) (with `class_name` as well as `preload`ed)
  - Autoloads

- Possible contexts: the completion might be placed in different places of the program. e.g:
  - initializers of class members
  - directly inside a suite
  - assignments inside a suite
  - as parameter to a call
