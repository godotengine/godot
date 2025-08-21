# Memory Management
You may have noticed errors similar to this at the end of your run:
```sh
ERROR: ~List: Condition "_first != __null" is true.
   At: ./core/self_list.h:112.
WARNING: cleanup: ObjectDB Instances still exist!
   At: core/object.cpp:2071.
ERROR: clear: Resources Still in use at Exit!
   At: core/resource.cpp:476.
```
These indicate that when the tests finished running there were existing objects that had not been freed.

GUT will try to detect when an orphan is created and will log how many orphans it finds after each test/script.  To make life a little easier, `GutTest` provides the following methods that makes freeing Nodes.  Each of these methods return what is passed in, so you can save a line or two of code.
  * `autofree` - calls `free` after test finishes
  * `autoqfree` - calls `queue_free` after test finishes
  * `add_child_autofree` - calls `add_child` right away, and `free` after test finishes.
  * `add_child_autoqfree` - calls `add_child` right away, and `queue_free` after teest finishes.

More info can be found in "Freeing Test Objects" below.

Quick Example:
``` gdscript
func test_something():
  # add_child_autofree will add the result of SuperNeatNode.new to the tree,
  # mark it to be freed after the test, and return the instance created by
  # SuperNeatNode.new().
  var my_node = add_child_autofree(SuperNeatNode.new())
  assert_not_null(my_node)
```



At the bottom of the page I attempt to describe how Godot does memory management.  If any of these sections seem confusing, read that part first.  The [Godot docs](https://docs.godotengine.org/en/stable/getting_started/scripting/gdscript/gdscript_basics.html#memory-management) has some information.  Also, here's a [tutorial](https://www.youtube.com/watch?v=cl2PxGkpJdo) on memory management I found.


## GUT Orphan Count
GUT, by default, will print a count of any orphans that are created by a test.  Depending on the log level these counts will appear after each test or may just appear after each script.  GUT counts the orphans before each test and warns when the value changes when a test is done.  These counts are summed up for each script as well and a grand total is printed at the end of the run.  You can disable this feature if you want to.

Godot considers an object to be orphaned if it extends `Object` (but not `Reference`) and has not been added to the tree.  Godot provides the `print_stray_nodes` which will print a list of all nodes (and their children) that are not in the tree.  This information is a bit cryptic and can only be printed to the console and command line.  You can also use the `Performance` object to get a count of orphans at any particular time.
```
var count = Performance.get_monitor(Performance.OBJECT_ORPHAN_NODE_COUNT)
```


## Freeing Test Objects
Freeing up objects in your tests is tedious.  It adds additional lines of code that don't add anything to the test.  To aid in this GUT provides the `autofree` and `autoqfree` functions.  Anything passed to these methods will be freed up after the test runs.  These methods also `return` whatever is passed into them so you can chain them together to cut down on space.
```
var Foo = load('res://foo.gd')
var node = autofree(Node.new())
var bar_scene = autofree(load('res://bar.tscn').instance())
assert_null(autofree(Foo.new()).get_value(), 'default value is null')
```
`autofree` will call `free` on the object (if it is valid to) at the end of the test.  `autoqfree` will call `queue_free` on the object at the end of the test.  These objects are freed before the orphans are counted and will now show up in the count.  If `autoqfree` is called at anytime during a test GUT will pause briefly to give the `queue_free` time to execute, then it will count orphans.

These functions can be used in a test or the `before_each` but should NOT be used in `before_all`.  If you create an object in `before_all` you must free it yourself in `after_all`.  Using either flavor of `autofree` in `before_all` will cause the object to be freed after the first test is run.


## Using `add_child` in Tests
When you call `add_child` from within a test the object is added as a child of the test script.  The test script is a child of the GUT GUI.  Any child you add to a test will not be freed until the end of the test run.  GUT holds onto the test scripts until the end for summary information.  GUT will output a warning if a test script has children when it finishes running.

It is best to free any children you add in a test in that same test.  GUT has two helper functions that will add the child and free the child after the test.  These are `add_child_autofree` and `add_child_autoqfree`.  These work the same way as `autofree` and `autoqfree` but take the additional step of calling `add_child`.  These methods also return whatever is passed to them so you can cut down on lines of code.
```
func test_foo():
  var node = add_child_autofree(Node.new())
  var node2 = add_child_autoqfree(Node.new())
```

These functions can be used in a test or the `before_each` but should NOT be used in `before_all`.  If you have an object you want to add as a child in `before_all` you must free it yourself in `after_all`.  Using either flavor of `add_child_autofree` in `before_all` will cause the object to be freed after the first test is run.

## Freeing Globals
You can use a [post-run hook](Hooks) to clean up any global objects you have created.  If you are running your tests through a scene then you may have to recreate these objects if you want to be able to perform multiple test runs through the GUI.  You could use a [pre-run hook](Hooks) to do this, but it all starts getting messy at that point.

## Automatically Freed Objects
To help with freeing up objects GUT will automatically free the following objects at the end of the test.
* [Doubles](Doubles)
* [Partial Doubles](Partial-Doubles)

Calling `autofree` with one of these objects, or manually freeing them yourself will not have any adverse effects.


## Testing for Leaks
GUT provides the `assert_no_new_orphans` method that will assert that the test has not created any new orphans.  Using this can be a little tricky in complicated test scripts.

`assert_no_new_orphans` strictly validates that at the time it executes the count of of orphans found is the same as before the test was run.  The "before" count is taken prior to executing `before_each`.  It is recommended that these types of tests are done in an [Inner Test Class](Inner-Test-Classes) or standalone script where it is less likely that a `before_each` or `before_all` would introduce orphans causing false positives.

`assert_no_new_orphans` cannot take into account anything you have called `autofree` on.  For one, it's impossible, and it wouldn't tell you much since freeing that object could cause leaks.

A standard memory leak test will create an object, free it, and then verify that you have not created any new orphans.  Based on some bad practices I've done myself I would advise creating tests with and without using `add_child`.

```gdscript
# res://test/unit/test_foo.gd
extends GutTest
...

class TestLeaks:
    extends GutTest
    var Foo = load('res://foo.gd')

    func test_no_leaks():
        var to_free = Foo.new()
        to_free.free()
        assert_not_new_orphans()

    func test_no_leaks_with_add_child():
        var to_free = Foo.new()
        add_child(to_free)
        to_free.free()
        assert_no_new_orphans()
```
If you must use `queue_free` instead of `free` in your test then you will have to pause before asserting that no orphans have been created.  You can do this with `await`
``` gdscript
func test_no_orphans_queue_free();
  var node = Node.new()
  node.queue_free()
  assert_no_new_orphans('this will fail')
  await wait_seconds(.2)
  assert_no_new_orphans('this one passes')
```


## Godot Memory Management
Godot treats `Object` and `Reference` instances differently for memory management.  The confusing part is that `Reference` extends `Object`.  Most of the time you don't have to worry about anything that extends `Reference`.  But if it extends `Object` (and not `Reference`) you must free it yourself or it, and all its parts, will stay around until you exit the game.

### Reference
Any class/script that extends Reference is managed via reference counts.  As you add variables that "point" to an object, the reference count increases.  As these variables go out of scope or are changed the reference count is decreased.  When the count hits 0, the object is freed automatically.

For the most part you don't have to worry about this.  It just works.  These reference counts can become an indirect leak if you do not free an `Object` that has variables pointing to `Reference` instances.  References can become leaks if there are cyclical references of objects that prevent each other from being freed.  `weakref` is useful in these cases.  Currently there is no way to get any additional information about instances of Reference from the engine.

```
# makes a new Reference object and returns a reference to it.  The reference
# count is now 1
var ref1 = Reference.new() # lets call this [Reference:1]
# Make another variable that "points" to the object above.
var ref1_1 = ref1 # [Reference:1] now has a reference count of 2

if(true):
  var ref_in_if = ref1 # now the count is 3

# now we are back to 2 since ref_in_if went out of scope
print('out of if')

ref1_1 = null # the count is now 1
ref1 = null # [Reference:1] will now be freed
```

### Weakref
There can be situations where cyclical references will prevent a `Reference` from being freed.  To help in these situations Godot has the [`weakref`](https://docs.godotengine.org/en/stable/classes/class_weakref.html) function.

The `weakref` function will return you an object that points to a reference but does not increase the reference count.  This allows you to point at an object without preventing it from being freed automatically.  You must take this into account though and verify that the object still exists when you want to get the reference back out of the `weakref`.
```
var foo = Foo.new() # where Foo extends Reference
foo.set_bar('hello world')
var wref = weakref(foo)

...

# this line increases the reference count until ref_of_wref goes out of scope
var ref_of_wref = wref.get_ref()
if(is_instance_valid(ref_of_wref)):
  print(ref_of_wref.get_bar()) # prints 'hello world' if ref hasn't been freed already.
```

### Object (except Reference)
Anything that extends `Object` (directly or indirectly) but DOES NOT extend `Reference` must be freed manually.  Any of these objects that are not in the main tree are considered orphaned.  Any children of these objects are also considered orphans.  You can free these objects by calling `free` or `queue_free`.  You can see a list of orphans by using `print_stray_nodes`.
