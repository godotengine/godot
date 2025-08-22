# Hooks

GUT has a pre-run and post-run hook that allows you to take any initialization steps or verify the results of the run.

Hook scripts can be set through the editor, through a command line option or in the `.gutconfig.json` file.

All Hook scripts must inherit from [GutHookScript](class_GutHookScript).  If the pre-run or post-run scripts specified do not exist or do not extend [GutHookScript](class_GutHookScript) then the run will be aborted before any tests are run.

All Hook scripts have access to the GUT instance that is running the tests via the `gut` property defined in `hook_script.gd`.  This is set after initializing the script.

GUT executes the virtual method `run()` when the hook should be executed.  Place your custom code in there.

**Note:** All Hook scripts are instantiated at the start of the run and later `run()` is called on each instance at the appropriate time.  The `_init()` method of your Hook Script will not have access to the GUT instance since it is set  later.




## Setup
Create scripts that inherit from [GutHookScript](class_GutHookScript), implement the `run()` method.  Set the path to your scripts through the panel or  `.gutconfig.json`, depending on how you are running your tests.


You can specify `pre_run_script` and `post_run_script` in the `.gutconfig.json` file.  You can also specify these options directly at the command line using the `-gpre_run_script` and `-gpost_run_script` options.



## Features
The following features are available to scripts that inherit from [GutHookScript](class_GutHookScript).  Not all features are usable by all hooks.  Details below.
* `gut` - the GUT instance running tests.
* `abort()` - abort the test run.
* `set_exit_code(code)` - Set the code to be returned when the command line finishes.
* The `JunitXmlExport` class can be used to create an export object to export results.  See [Export-Test-Results](Export-Test-Results)

### Access GUT instance
Each Hook script can access the GUT instance via the `gut` variable.  Useful for getting to summary info or manipulating the GUT instance for reasons I can't think of (which is probably a bad idea but who am I to judge).

### Abort (pre-run only)
The built in `abort()` method will cause the run to end immediately after the `run()` method of the pre-run script finishes.  The post-run script will NOT be executed.  Calling this in the post-run script will have no effect.

### Exit Code (post-run only)
The `set_exit_code(code)` method will set an exit code that will be used when running from the command line.  The default behavior is to return `0` when all tests pass and `1` if any  tests fail (pending tests do not affect the exit code).  If you call `set_exit_code` then the value passed will be used.

**Note** Calling `set_exit_code` in the pre-run script will not affect the actual exit code.  You could use `gut.get_pre_run_script_instance().get_exit_code()` in your post-run script to get you any value you've set via `set_exit_code` in your pre-run script.




## Pre-Run Hook
The pre-run hook is run just before any tests are executed.  This can be useful in setting global variables or performing any setup required for all your tests.

The post-run hook can access the pre-run hook instance via `gut.get_pre_run_script_instance()`.

**Things to do in your pre-run script**:
* mute all sounds `AudioServer.set_bus_volume_db(0, -INF)`
* set flags you've implemented to prevent actions from occurring during tests
  * flags to prevent files from being saved like user stats (my personal catalyst for all these features)
  * logging levels for your application
* other things I haven't thought of.




## Post-Run Hook
The post-run hook is run after all tests are run and all output has been generated.  The post-run hook can access the pre-run script instance (if one was specified) via `gut.get_pre_run_script_instance()`.

The post-run hook could be useful in writing files used by CICD pipelines to verify the status of the run.




## Summary Info
GUT tracks the results of all the scripts and tests that are run.  There is a Summary object that you can access via the `gut` variable.  Using this information you can take actions in the post-run hook.

Reading the documentation/code in [summary.gd](https://github.com/bitwes/Gut/blob/master/addons/gut/summary.gd) will get you the full details, but here are a few examples of how to get the summary data.

### Full Summary
```
# Returns GUT's summary.gd instance holding all the data about the run.
gut.get_summary()
```

### Counts
```
# This will return a dictionary containing the following counts:
#   passing = 0,
#   pending = 0,
#   failing = 0,
#   tests = 0,
#   scripts = 0
gut.get_summary().get_totals()
```

### All Scripts
```
# Returns and array of Summary.gd.TestScript objects that have detailed information
# about each script/inner class that was ran.  See summary.gd for more details.
gut.get_summary().get_scripts()
```
