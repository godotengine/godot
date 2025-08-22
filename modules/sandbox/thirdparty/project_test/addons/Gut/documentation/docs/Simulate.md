# Simulate
`simulate(obj, times, delta, check_is_processing: bool = false)`

The simulate method will call the `_process` and `_physics_process` on a tree of objects.  It will check each object to see if they have either method and run it if it exists.  In cases where the object has both it will call `_process` and then `_physics_process` and then move on to the next node in the tree.

`simulate` takes in the base object, the number of times to call the methods and the delta value to be passed to `_process` or `_physics_process` (if the object has one).  It starts calling it on the passed in object and then moves through the tree recursively calling `_process` and `_physics_process`.  The order that the children are processed is determined by the order that `get_children` returns

By default `simulate` will ignore if the object is "processing" or not.  When the optional `check_is_processing` is `true`, GUT will check `is_processing` and `is_physics_processing` on each object and will not call their respective methods if the object is not "processing".  Remeber, that nodes not in the tree will not be "processing" by default, so you have to add them (use `add_child_autofree` or `add_child_autoqfree`) or call `set_process` or `set_physics_process` before calling `simulate`.

`simulate` will only cause code directly related to the `_process` and `_physics_process` methods to run.  Signals will be sent, methods will be called but timers, for example, will not fire since the main loop of the game is not actually running.  Creating a test that uses `await` is a better solution for testing such things.

Example
``` gdscript

# --------------------------------
# res://scripts/my_object.gd
# --------------------------------
extends Node2D
  var a_number = 1

  func _process(delta):
    a_number += 1

# --------------------------------
# res://scripts/another_object.gd
# --------------------------------
extends Node2D
  var another_number = 1

  func _physics_process(delta):
    another_number += 1

# --------------------------------
# res://test/unit/test_my_object.gd
# --------------------------------

# ...

var MyObject = load('res://scripts/my_object.gd')
var AnotherObject = load('res://scripts/another_object')

# ...

# Given that SomeCoolObj has a _process method that increments a_number by 1
# each time _process is called, and that the number starts at 0, this test
# should pass
func test_does_something_each_loop():
  var my_obj = MyObject.new()
  add_child_autofree(my_obj)
  gut.simulate(my_obj, 20, .1)
  assert_eq(my_obj.a_number, 20, 'Since a_number is incremented in _process, it should be 20 now')

# Let us also assume that AnotherObj acts exactly the same way as
# but has SomeCoolObj but has a _physics_process method instead of
# _process.  In that case, this test will pass too since all child objects
# have the _process or _physics_process method called.
func test_does_something_each_loop():
  var my_obj = MyObject.new()
  var other_obj = AnotherObj.new()

  add_child_autofree(my_obj)
  my_obj.add_child(other_obj)

  gut.simulate(my_obj, 20, .1)

  assert_eq(my_obj.a_number, 20, 'Since a_number is incremented in _process, \
                                  it should be 20 now')
  assert_eq(other_obj.another_number, 20, 'Since other_obj is a child of my_obj \
                                           and another_number is incremented in \
                                           _physics_process then it should be 20 now')

```

## Where to next?
* [Awaiting](Awaiting)<br/>
