# Partial Doubles

A Partial Double is the same thing as a normal Double except all methods retain their functionality by default.

You can create Partial Doubles for scripts, packed scenes, and inner classes.  It works the same way as `double` works, so read up on that and then apply all that new knowledge to `partial_double`.

Under the covers, whether you call `double` or `partial_double`, GUT makes the same thing.  If you call `partial_double` though, it will setup the object's methods to be stubbed `to_call_super` instead of not being stubbed at all.

After you have your partial double, you can stub methods to return values instead of doing what they normally do.  You can also spy on any of the methods.

__NOTE__ All Doubles and Partial Doubles are freed when a test finishes.  This means you do not have to free them manually and you should not be created in `before_all` or referenced in `after_all`.

## Script Example

Given
``` gdscript
# res://foo.gd
extends Node2D

var _value = 10

func set_value(val):
  _value = val

func get_value():
  return _value
```

Then

```gdscript
var Foo = load('res://script.gd')

func test_things():
  var partial = partial_double(Foo).new()
  stub(partial, 'set_value').to_do_nothing()
  partial.set_value(20) # stubbed so implementation bypassed.

  # since set_value was stubbed, and get_value was not, and since
  # this is a partial stub, then the original functionality of
  # get_value will be executed and _value is returned.
  assert_eq(partial.get_value(), 10)
  # unstubbed partial methods can be spied on.
  assert_called(partial, 'get_value')
  # stubbed methods can be spied on as well
  assert_called(partial, 'set_value', [20])
```
