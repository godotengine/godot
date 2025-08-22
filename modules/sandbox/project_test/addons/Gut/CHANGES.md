# Release notes
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).


# 9.5.0
* All the `.uid` files you could ever want!


# 9.4.0

## Potentially Breaking Changes
* The deprecated `wait_frames` and the new `wait_idle_frames`, `wait_physics_frames` now count frames when the `SceneTree.process_frame` and `SceneTree.physics_frame` signals are emitted.  This may cause some of your awaits to wait a frame too long/short.  This approach is more reliable as it occurs before the `_process` or `_physics_process` is called on anything in the tree.  This means that tree order will not matter and all objects will have finished `_process`/`_physics_process` by the time the await ends.


## Features
* Utilized the adapted Godot tools that generate HTML from code comments, moving some documentation to code comments.  This makes more documentation easily accessible from the editor and cuts down on some duplicate documentation.
* `wait_idle_frames` added.  This counts frames idle/process frames instead of `_physics_process`.  `wait_frames` has been renamed (deprecated) to `wait_physics_frames`.
* `wait_while` added.  This waits while a `Callable` returns `true`.
* New `class_name`s:
  * `GutInputFactory` for `res://addons/gut/gut_input_factory.gd` static class.
* Signal related methods now accept a reference to a signal as well as an object/signal name:  `get_signal_emit_count`, `assert_connected`, `assert_not_connected`, `assert_signal_emitted`, `assert_signal_not_emitted`, `assert_signal_emitted_with_parameters`, `assert_signal_emit_count`, `get_signal_parameters`.

## Bug Fixes
* I192 in just under 5 years I moved the two lines of code up some to finally fix this.
* I666 Using a doubled scene as the value of a property with type `PackedScene` could cause errors.


## Deprecations
* `wait_frames` has been deprecated in lieu of the more specific `wait_physics_frames` and `wait_idle_frames`.





# 9.3.1
A small collection of bug fixes and documentation.  GUT can now generate documentation from code comments.

## Features
* added class_name to InputSender by @bitwes in https://github.com/bitwes/Gut/pull/651
* add await in pre-run script, post-run scrpt, and should_skip_script. by @LowFire in https://github.com/bitwes/Gut/pull/671
* Doctools generation by @bitwes in https://github.com/bitwes/Gut/pull/672

## Bug Fixes
* Parsed Native Refcounted Objects are not freed. by @bitwes in https://github.com/bitwes/Gut/pull/648
* Housekeeping by @bitwes in https://github.com/bitwes/Gut/pull/652
* I650 more warnings by @bitwes in https://github.com/bitwes/Gut/pull/653
* fix: Mocking-Input link by @HotariTobu in https://github.com/bitwes/Gut/pull/659
* Fixed typos in error messages. by @xmoby in https://github.com/bitwes/Gut/pull/663
* Fix docs typo by @MSWS in https://github.com/bitwes/Gut/pull/665
* Font import files: Add disable_embedded_bitmaps by @manuq in https://github.com/bitwes/Gut/pull/674
* Update is_almost_eq() to use built-in vector comparison by @onegm in https://github.com/bitwes/Gut/pull/668

## New Contributors
* @HotariTobu made their first contribution in https://github.com/bitwes/Gut/pull/659
* @xmoby made their first contribution in https://github.com/bitwes/Gut/pull/663
* @MSWS made their first contribution in https://github.com/bitwes/Gut/pull/665
* @LowFire made their first contribution in https://github.com/bitwes/Gut/pull/671
* @manuq made their first contribution in https://github.com/bitwes/Gut/pull/674
* @onegm made their first contribution in https://github.com/bitwes/Gut/pull/668

**Full Changelog**: https://github.com/bitwes/Gut/compare/v9.3.0...9.4.0




# 9.3.0

## Features
* You can Monkey Patch your doubles!  You can make any method in a double call a specified `Callable` using `.to_call(callable)` on `stub`.  Details are on the [Stubbing](https://gut.readthedocs.io/en/latest/Stubbing.html) wiki page.
```gdscript
var dbl = double(MyScript)
stub(dbl.some_method).to_call(func(): print("Monkey Patched!"))
```
* You can now use callables to `stub` insetad of passing the object and method name.  Binding arguments adds an implicit `when_passed` to the stub.  Less strings, less typing!
```gdscript
var dbl = double(MyScript)
# same as stub(dbl, "some_method").to_return(111)
stub(dbl.some_method).to_return(111)

# same as stub(dbl, 'some_method').to_return(999).when_passed("a")
stub(dbl.some_method.bind("a")).to_return(999)
```
* @WebF0x GUT can now wait on a `Callable` to return `true` (aka predicate method)  via the new `wait_until`:
```
# Call the function once per frame until it returns 5 or one second has elapsed.
await wait_until(func(): return randi_range(0, 20)==5, 1)
```
* `wait_for_signal` and the new `wait_until` return `true` if they did not timeout, and `false` otherwise.  This means waiting on, and asserting a signal has been emitted can now be written as
```
assert_true(await wait_for_signal(my_obj.my_singal, 2),
    'signal should emit before 2 seconds')
```
* @mphe GUT now automatically enables the "Exclude Addons" option when running tests.  This means you don't have to keep enabling/disabling this option if GUT does not conform to your warning/error settings.
* GUT disables warnings at key points in execution and then re-enables them.  This makes running GUT possible (or at least easier) with warning levels incompatable with GUT source code.  This also makes the ouput less noisy.
* @plink-plonk-will Elapsed time is now included in the XML export.
* __Issue__ #612 `InputSender` now sets the `button_mask` property for generated mouse motion events when mouse buttons have been pressed but not released prior to a motion event.
* __Issue__ #598 Added the virtual method `should_skip_script` to `GutTest`.  If you impelement this method and return `true` or a `String`, then GUT will skip the script.  Skipped scripts are marked as "risky" in the final counts.  This can be useful when skipping scripts that should not be run under certiain circumstances such as:
    * You are porting tests from 3.x to 4.x and you don't want to comment everything out.
    * Skipping tests that should not be run when in `headless` mode.
    ``` gdscript
    func should_skip_script():
        if DisplayServer.get_name() == "headless":
            return "Skip Input tests when running headless"
    ```
    * If you have tests that would normally cause the debugger to break on an error, you can skip the script if the debugger is enabled so that the run is not interrupted.
    ``` gdscript
    func should_skip_script():
        return EngineDebugger.is_active()
    ```
* The CLI got an update to its Option Parser.  There's more info in #623:
    * options that take a value can now be specified with a space (`option value`) instead of using `option=value`.
    * `-gh` option now has headings for the different options. It looks a lot better.
    * `-gdir` and `-gtest` can be specified multiple times instead of using a comma delimited list.
    * You can use `-gconfig=` to not use a config file.
* Minor niceties such as showing that GUT is exiting in the title bar (takes a bit sometimes) and switching to full display at the end of a run if GUT does not automatically exit.

## Bug Fixes
* __Issue__ #601 doubles now get a resource path that makes Godot ignore them when "Exclude Addons" is enabled (res://adddons/gut/not_a_real_file/...).
* __Issue__ #594 An error is generated if GUT cannot find the double template files.  This can happen if you export your game with tests, but do not include *.txt files.
* __Issue__ #595 When no tests are run GUT no longer displays "All Tests Passed!" and exits (based off of settings).
* __Issue__ #578 Fix `InputSender` so it works with `Input.is_action_just_pressed`, `Input.is_action_just_released` and `Input.is_action_pressed`.  Thanks @lxkarp and @edearth for you work on this.


## Deprecations
* The optional `GutTest` script variable `skip_script` has been deprecated.  Use the new `should_skip_script` method instead.
* GUT now warns if you have overridden `_ready` in your test script without calling `super._ready`.




# 9.2.1
* __Issue__ #570 Doubling scripts that contain a statically typed variable of another class_name script (`var foo := Foo.new()` where foo is a `class_name` in another script) could cause errors.
* Add support for running tests through the debugger via VSCode via the gut-extension.

# 9.2.0
## Configuration Changes
* The GUT Panel config is now auto-saved/loaded to `user://` instead of `res://`.  This file changes a lot and is very annoying with version control and teams that have more than one person (which is all teams since there is no "I" in team).
    * The new location is `user://gut_temp_directory/gut_editor_config.json`
    * When you open your project, GUT will check to see if there is a file in the new location.  If not, it will copy it there.
    * GUT prints a warning to `Output`` telling you that you can delete the old file.
* You can now Save/Load configs to/from anywhere through the Settings Subpanel.
    * Saving/Loading does not change where the GUT panel auto-saves/loads to.
    * This allows you to define standard config files for your project, but not save any changes in a version controlled file (unless you explicitly resave it using the cool new Save As button).
* The GUT Panel Shortcuts config file has also been moved.  GUT also moves this file automatically and prints a warning.
    * The new location is `user://gut_temp_directory/gut_editor_shortcuts.cfg`
* All files that were being saved in `user://` have been moved to `user://gut_temp_directory` for better house keeping.


## Features
* The Settings Subpanel now has on/off switches for directories, so you can turn them off if you want to run a subset of tests.
* Wiki moved to https://gut.readthedocs.io


## Bug Fixes
* __Issue__ #479 source_code_pro.fnt was malformed, is now bienformed.
* __Issue__ #549 @andrejp88 debug/gdscript/warnings/untyped_declaration as error would break GUT due to dynamic code generation.
* __Issue__ #536 Theme refernces font instead of embedding it.
* __Issue__ #523 "got" values are printed with extra precision for float, Vector2, and Vector3 when using `assert_almost_eq`, `assert_almost_ne`, `assert_between` and `assert_not_between`.
* __Issue__ #436 Doubled Scenes now retain export variable values that were set in the editor.
* __Issue__ #547 The output_font_name and output_font_size for the GutPanel are now saved.
* __PR__ #544 (@xorblo-doitus) InputSender will now emit the `gui_input` signal on receivers.
* __Issue__ #473 Moved gut panel settings and gut options out of res:// so that multiple devs won't fight over files that are really user preferences.
    * Created some Editor Preferences for Gut to handle user only settings.
    * When running GUT from the editor, the config used by the runner is saved to `user://` now.
    * You can load and save configs through the editor, so you can have a base set of settings that are not overwritten when running Gut.
    * Moved all files that Gut creates in `user://` to `user://gut_temp_directory`.
    * Output Subanel related settings have moved to the Output Subpanel.  Use the "..." button.
* __Issue__ #557 Tests are now found in exported projects.
* Fixed issue where the panel was not loading the double strategy correctly.
* __Issue__ #542 GUT no longer generates orphans...again.



# 9.1.1
* Fixed numerous issues with doubling that were caused by the port from 3.x.  Most of these involved using the INCLUDE_NATIVE doubling strategy.
* Added errors and better failure messages when trying to stub or spy on an invalid method.  For example, if your script does not implement `_ready` and you try to spy on it, your test will now fail since `_ready` is virtual and you didn't overload it.
* Doubled methods that have a vararg argument are now auto detected and extra parameters (up to 10) are added to the method signature to handle most use cases (i.e. `rpc_id`, `emit_signal`).  If you call a doubled method that has a vararg argument and you have not stubbed `param_count` on the object's script then a warning is generated.
* Fixed an issue where command line would not launch in 4.2rc1.
* __Issue #510__ Added all types to strutils to address #510.
* __Issue #525__ Signals are now disconnected when waiting on signals that do not fire in the expected amount of time.

# 9.1.0 (requires Godot 4.1)
* GUT generated errors now cause tests to fail (not engine errors, just things GUT thinks are bad).  You can disable this through the CLI, .gutconfig, or the panel.
* Changes to Double Strategy and Double/Partial Double creation to fix #482.
    * See [Double-Strategy](https://bitwes.github.io/GutWiki/Godot4/Double-Strategy.html) in the wiki for more information.
    * The default strategy has been changed back to `SCRIPT_ONLY` (a bug caused it to change).  Due to how the Godot Engine calls native methods, the overrides may not be called by the engine so spying and stubbing may not work in some scenarios.
    * Doubling now disables the Native Method Override warning/error when creating Doubles and Partial Doubles.  The warning/error is turned off and then restored to previous value after a Double or Partial Double has been loaded.
    * The doubling strategy `INCLUDE_SUPER` has been renamed to `INCLUDE_NATIVE`.
    * If you have an invalid Double Strategy set via command line or gutconfig, the default will be used.  So if you are explicity setting it to the old `INCLUDE_SUPER`, it will use `SCRIPT_ONLY`.
    * You can now set the default double strategy in the GutPanel in the Editor.
* Added `GutControl` to aid in running tests in a deployed game.  Instructions and sample code can be found [in the wiki](https://bitwes.github.io/GutWiki/Godot4/Running-On-Devices.html).
* __Issue 485__ GUT prints a warning and ignores scripts that do not extend `GutTest`.
* A lot of internal reworkings to simplify logging and info about test statuses.  The summary changed and the final line printed by GUT is now the highest severity status of the run (i.e. failed > pending/risky > passed).
* __Issue 503__ Fixed issue where GUT would not find script object when doubling PackedScenes.
* __Port PR 409__ GUT's simulate function can now check `is_processing` and `is_physics_processing` when running thier respective methods.


# 9.0.1
* Fix #475, you can now double scripts that use the new accessors.


# 9.0.0
9.0.0 is the first version of GUT released for Godot 4.  Any version below 9.0.0 is for 3.x.  See the [GODOT_4_README.md](https://github.com/bitwes/Gut/blob/godot_4/GODOT_4_README.md) in the `godot_4` branch for changes to GUT from 3.x.

The wiki has not been updated yet for GUT 9.0.0, but it has been moved to the `godot_4` branch so it can be edited via this repo.  Changes to the wiki will be pushed to https://bitwes.github.io/GutWiki/Godot4/.
