# Doubling Strategy
By default Godot 4 does not allow you to override native methods defined in Godot Objects (you can change this in the project settings: Debug->GDScript->Native Method Override).  For example, overriding `set_position` on a Node2D will cause an error.  This is because the engine may or may not use overrides in some cases for performance reasons.  It has something to do with pointers and functions and efficiency and stuff.  I don't fully understand it, but it's true ([Here's a Github Issue with some info](https://github.com/godotengine/godot/issues/55024)).  The important thing for us, is that this means you cannot spy-on or stub these functions...or can you?

<hr>

__Warning:__  `INCLUDE_NATIVE` is not compatible with static typing.  When you statically type a variable, Godot will call all of its native methods directly, bypassing the overrides in the double.  See [this issue](https://github.com/bitwes/Gut/issues/633#issuecomment-2198440346) for more information and some examples.

<hr>

In most cases Doubles are not used in a manner that would cause the engine to directly interact with them.  You may want to verify that some object you created calls `set_position` on a double.  This is where changing the Double Strategy can help.  When using `DOUBLE_STRATEGY.INCLUDE_NATIVE` GUT will override all native functions which means you can spy-on and stub them for your tests.  It will also disable the error and then restore the original setting after the Double or Partial Double is created so that GUT does not blow up.  Only direct calls that you make in your objects are guaranteed use the overrides.  A tween probably won't call your `set_position` override.

The default Double Strategy is `SCRIPT_ONLY`, meaning no overrides will be included.  You can change the strategy globally, at the script level, or for an individual Double or Partial Double.


## Set the Default Strategy
You can change the default through the GutPanel in the editor, the `.gutconfig.json` file for the command line, or as a command line option.  Note that the editor and the command line configuration are separate, so you must set it in both places if you are using both.

### .gutconfig
Valid values are `SCRIPT_ONLY`(default) or `INCLUDE_NATIVE`
```json
"double_strategy":"INCLUDE_NATIVE"
```

### Command Line
Use the `-gdouble_strategy` option with the values `INCLUDE_NATIVE` or `SCRIPT_ONLY`
```bash
-gdouble_strategy='SCRIPT_ONLY'
```

## Overriding the Default

### Script Level
From within a `GutTest` script you can call `set_double_strategy` to change the strategy for that script/inner-class ONLY.  This value will be reset to the default after the script has finished.  You should call this in `before_all`.
```gdscript
set_double_strategy(DOUBLE_STRATEGY.INCLUDE_NATIVE)
set_double_strategy(DOUBLE_STRATEGY.SCRIPT_ONLY)
```

### Individual Double/Partial Double
When calling `double` or `partial_double` you can pass an optional parameter to set the double strategy for just that double.
```gdscript
double(Foo, DOUBLE_STRATEGY.SCRIPT_ONLY)
partial_double(Bar, DOUBLE_STRATEGY.INCLUDE_NATIVE)
double(MyScene, DOUBLE_STRATEGY.SCRIPT_ONLY)
```