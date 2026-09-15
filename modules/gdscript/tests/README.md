# GDScript integration tests

The `scripts/` folder contains integration tests in the form of GDScript files
and output files.

See the
[Integration tests for GDScript documentation](https://docs.godotengine.org/en/latest/engine_details/architecture/unit_testing.html#integration-tests-for-gdscript)
for information about creating and running GDScript integration tests.

# GDScript Autocompletion tests

The `scripts/completion` folder contains tests for the GDScript autocompletion.

Each test case consists of at least one `.gd` file, which contains the code, and one `.cfg` file, which contains expected results and configuration. Inside of the GDScript file the character `➡` represents the cursor position, at which autocompletion is invoked.

The script files won't be parsable GDScript since it contains an invalid char and often the code is not complete during autocompletion. To allow for a valid base when used with a scene, the
runner will remove the line which contains `➡`. Therefore the scripts need to be valid if this line is removed, otherwise the test might behave in unexpected ways. This may for example require
adding an additional `pass` statement.

This also means, that the runner will add the script to its owner node, so the script should not be loaded through the scene file.

The config file contains two section:

`[input]` contains keys that configure the test environment. The following keys are possible:

- `cs: boolean = false`: If `true`, the test will be skipped when running a non C# build.
- `use_single_quotes: boolean = false`: Configures the corresponding editor setting for the test.
- `add_node_path_literals: boolean = false`: Configures the corresponding editor setting for the test.
- `add_string_name_literals: boolean = false`: Configures the corresponding editor setting for the test.
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

# GDScript Code action tests

The `scripts/code_actions` folder contains tests for GDScript code actions.

Within an individual subfolder, a test case should have a base name shared between three files:

- A `.cfg` file that configures the GDScript compiler and analyzer, and instructs the test runner on what code action should be performed.
- A `.gd` file with GDScript code, before a particular code action is performed.
  For quick fixes to warnings (in the `warning_fixes` folder), the code should be parsable and valid.
- An `.out.gd` file with GDScript code _after_ the code action is performed.

## Configuring a test with a `.cfg` file

Each test's `.cfg` file should have a `[warnings]` section with the key `warnings_only` that corresponds to a list of warning string names. The GDScript analyzer will only report these warnings for the given test. For example, if testing quick fixes for the `UNUSED_VARIABLE` warning, the `[warnings]` section should look like:

```toml
[warnings]
include_only=["UNUSED_VARIABLE"]
```

Additionally, there should be an `[apply]` section with two keys, `group_idx` and `action_idx`, corresponding to integer values. The code actions system reports groups of code actions, and these two values index into the groups to identify a single code action to perform. For example, when getting code actions for a script with an `UNUSED_VARIABLE` warning, the code actions returned may look like so:

```text
- Group: UNUSED_VARIABLE
  - Action: Add underscore to variable name
  - Action: Remove variable declaration
  - Action: Ignore "UNUSED_VARIABLE"
```

To test the action `Remove variable declaration`, we would go to the first group (`group_idx=0`) and the second action within that group (`action_idx=1`). Thus, the `[apply]` section of the `.cfg` file would look like:

```toml
[apply]
group_idx=0
action_idx=1
```
