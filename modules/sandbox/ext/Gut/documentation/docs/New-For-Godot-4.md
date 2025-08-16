# Godot 4 Changes
These are changes to Godot that affect how GUT is used/implemented.  There is more information about these changes and how GUT has been altered below.

* `setget` has been replaced with a completely new syntax.  More info at [#380](https://github.com/bitwes/Gut/issues/380).  Examples of the new way and the new `assert_property` method below.
* `connect` has been significantly altered.  The signal related asserts will likely change to use `Callable` parameters instead of strings.  It is possible to use strings, so this may remain in some form.  More info in [#383](https://github.com/bitwes/Gut/issues/383).
* `yield` has been replaced with `await`.  `yield_to`, `yield_for`, and `yield_frames` have been deprecated, the new methods are below.
* Arrays are pass by reference now.
* Dictionaries are compared by value now.
* `File` and `Directory` have been replaced with `FileAccess` and `DirAccess`.
* `is` no longer accepts a variable for a class.  You must use `is_instance_of` instead.


## What's new/changed in GUT 9.0.0 for Godot 4.0
* Any methods that were deprecated in GUT 7.x have been removed.
* `assert_setget` no longer works (it now just fails with a message).  `assert_property` has been altered to work with the new setter/getter syntax.  `assert_set_property`, `assert_readonly_property`, and `assert_property_with_backing_variable` have been added.
* To aid refactoring, `assert_property` and `assert_property_with_backing_variable` will warn if any "public accessors" are found for the property ('get_' and 'set_' methods).
* `assert_property` now requires an instance instead of also working with a loaded objects.
* Doubling strategy flags have been renamed to `INCLUDE_NATIVE` (was `FULL`) and `SCRIPT_ONLY` (was `PARTIAL`).  The default is `SCRIPT_ONLY`.  I wanted something more descriptive and less likely to be confused with partial doubles.
* The various `yield_` methods have been deprecated but are still supported to make conversions easier.  The new syntax for `yield_to`, `yield_for`, or `yield_frames` is:
```gdscript
await yield_to(signaler, 'the_signal_name', 5, 'optional message')
await yield_for(1.5, 'optional message')
await yield_frames(30, 'optional message')
```
* The replacement methods for the various `yield_` methods are `wait_seconds`, `wait_idle_frames`, `wait_physics_frames`, and `wait_for_signal`.
```gdscript
await wait_for_signal(signaler.the_signal, 5, 'optional message') # wait for signal or 5 seconds
await wait_seconds(1.5, 'optional message')
await wait_physics_frames(30, 'optional message')
```
* Doubling no longer supports paths to a script or scene.  Load the script or scene first and pass that to `double`.  See the "Doubling Changes" section for more details.
* Doubling Inner Classes now requires you to call `register_inner_classes` first.  See the "Doubling Changes" section for more details.
* Comparing Dictionary/Arrays with `assert_eq`, `assert_eq_deep`, and the new `assert_same` and `assert_not_same` for comparing references.  See Godot's new `is_same` function for details on how `assert_same` works (it's just an assertion wrapper around `is_same`).  See the section "Comparing Dictionaries and Arrays" below for more details.


## Comparing Dictionaries and Arrays
In Godot 3.x dictionaries were compared by reference and arrays were compared by value.  In 4.0 both are compared by value.  Godot 4.0 introduces the `is_same` method which (amongst other things) will compare dictionaries and arrays by reference.

GUT's `assert_eq` and `assert_ne` changed to match Godot's behavior.  To compare by reference you can use the new `assert_same` or `assert_not_same`.  This works with arrays and dictionaries.  Review Godot's documentation for `is_same`.  When comparing dictionaries and arrays it is recommended that you use `assert_eq_deep` since it provides more detailed output than `assert_eq`.

The shallow compare functionality has been removed since it no longer applies.  Shallow compares would compare the elements of an array or dictionary by value.  In Godot 3.x this meant that dictionaries inside of arrays or dictionaries would be compared by reference and everything else would be compared by value.  Since arrays and dictionaries are both compared by value now, shallow compares are no different (functionally) than deep compares.  The following methods have been removed.  Calling these methods will generate a failure and an error.
* `compare_shallow` (causes failure and returns `null` which will likely result in a runtime error)
* `assert_eq_shallow`
* `assert_ne_shallow`

Final note: `assert_eq` does not use `assert_eq_deep` since `assert_eq_deep` compares each element of both arrays/dictionaries and provides detailed info about what is different.  This can be slow for large arrays/dictionaries.  Godot's `==` operator uses a hashing function which is much faster but does not provide information about what is different in each array/dictionary.  With `assert_eq`, `assert_eq_deep`, and `assert_same` (and their inverses) you have fine grained control over the type of comparison that is performed.


## Doubling Changes
### Doubling scripts and scenes
The `double` method no longer supports paths to scripts or scenes.  You should `load` the script or scene first, and then pass that to `double` instead.
```
var MyScript = load('res://my_script.gd')
var dbl = double(MyScript).new()
```
If you pass a string then an error message will be printed and `double` will return `null`.  This will most likely result in a runtime error when you attempt to instantiate your double.
```
'Invalid call. Nonexistent function 'new' in base 'Nil'.'
```

### Doubling Inner Classes
The `double` method no longer supports strings for the path of the base script or a string of the name of the Inner Class.  You must call `register_inner_classes` then pass the Inner Class to `double`.  You only have to do this once, so it is best to call it in `before_all` or a pre-hook script.  Registering multiple times does nothing.  Failing to call `register_inner_classes` will result in a GUT error and a runtime error.
```gdscript
# Given that SomeScript contains the class InnerClass that you wish to double:
var SomeScript = load('res://some_script.gd')

func before_all():
    register_inner_classes(SomeScript)

func test_foo():
    var dbl = double(SomeScript.InnerClass).new()
```
This approach was used to make tests cleaner and less susceptible to typos.  If Godot adds meta data to inner classes that point back to the source script, then `register_inner_classes` can be removed later and no other changes will need to be made.


## setget vs set: and get:
In godot 4.0 `setget` has been replaced with `set(val):` and `get():` pseudo methods which make properties more concrete.  This is a welcome change, but comes with a few caveats.

Here's an example of usage:
```
var foo = 10:
    get():
        return foo
    set(val):
        foo = val
```
This means you no longer need to define methods for your accessors.  Though you may still want to for organizational purposes.

One downside to this approach is that there is no way to set the `foo` without going through the accessor.  Many times, internally, you will want to set a value for a property without going through the setter.  This is still possible, but you have to make a backing variable.
```
var _foo = 10
var foo = 10:
    get():
        return _foo
    set(val):
        _foo = val
        foo_changed.emit()
```
With this approach you can set `_foo` internally in your class without triggering the `foo_changed` signal.  When you see `foo =` anywhere in your code, it will be going through the accessor.  When you see `_foo =` you are only setting the backing variable.

To test this new paradigm `assert_setget` has been removed.  `assert_property` has changed to work with the new syntax.  `assert_property_with_backing_variable` was added to validate the backing variable wiring.  If you use `assert_property_with_backing_variable` it will verify that the property has accessors and will also look for a `_<varname>` variable with the same name and verify it is being set.

`assert_property` will generate a warning when it finds "public" accessors for these properties (`get_foo`, `set_foo`).


## Implementation Changes
* The `Gut` control has been removed.  Adding a `Gut` node to a scene to run tests will no longer work.  This control dates back to Godot 2.x days.  With GUT 7.4.1 I believe the in-editor Gut Panel has enough features to discontinue using a Scene/`Gut` control to run tests.
* The GUI for GUT has been simplified to reflect that it is no longer used to run tests, just display progress and output.  It has also been decoupled from `gut.gd`.  `gut.gd` is now a `Node` instead of a `Control` and all GUI logic has been removed.  New signals have been added so that a GUI can be made without `gut.gd` having to know anything about it.  As a result, GUT can now be run without a GUI if that ever becomes something we want to do.
* Replaced the old `yield_between_tests` flag with `paint_after`.  This property (initially set to .1s) tells GUT how long to wait before it pauses for 1 frame to allow for painting the screen.  This value is checked after each test, so longer tests can still cause a delay in the painting of the screen.  This has made the painting a little choppier but has cut down the time it takes to run tests (200 simple tests in 20 scripts dropped from 2+ seconds to .5 seconds to run).  This feature is settable from the command line, .gutconfig.json, and GutPanel.
* Doubling has changed significantly.
