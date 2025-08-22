# Stubbing

The `stub` function allows you to define behavior for methods of a Doubled instance.  Stubs can be layered to address general and specific circumstances.  You can use `stub` to
* Force a method to do nothing and return a specific value.
* Call `super`'s version of the method, allowing the double to retain original functionality.
* Call a `Callable` of your choosing (aka Monkey Patching).
* Force the method to take no action (useful when using Partial Doubles).


You can also use `stub` to change the signature of a method by
* Set default parameter values.  All doubles default all parameters on all methods to `null` because GUT can't do anything else yet.  If your method has default values, and it has been stubbed to `call_super`, you may need to specify these values.  See `param_defaults` below.
* Changing the parameter count (useful in very specific cases, mostly involving `vararg`).


All Stubs are cleared between tests.  If you want to stub a method for all your tests do this in `before_each` using `stub(MyScript, "method_name")...`.  You should not stub anything in `before_all`, `after_each`, or `after_all`.




## Syntax
`stub` creates a stub for a method on a specific instance of a Double or any instance of a Doubled script.  After calling `stub` you chain calls to the various actioins available to tell GUT what the stub should do.  `stub` can be called numerous ways:

### stub(callable)
This will create a stub for the Double instance and method of the callable.  If the Callable has bound parameters the stub will only be used when the parameters used when calling the method match the bound parameters.  Using bound parameters is the same as using `when_passed`.
``` gdscript
var inst = double(MyScript).new()
stub(inst.some_method).to_return(111)
stub(inst.some_method.bind("a")).to_return(999)

assert_eq(inst.some_method("not a"), 111)
assert_eq(inst.some_method("a"), 999)
```

### stub(double_instance, method_name)
This creates a stub for the object and method.  This is the same as `stub(callable)`.  If you wish to stub for specific parameter values, use `when_passed`.
``` gdscript
var inst = double(MyScript).new()
stub(inst, "some_method").to_return(111)
stub(inst, "some_method").when_passed("a").to_return(999)

assert_eq(inst.some_method("not a"), 111)
assert_eq(inst.some_method("a"), 999)
```

### stub(script, method_name) / stub(path_to_script, method_name):
Creates a stub for `method_name` on all Doubles of a `script`.  This stub will be used if a stub is not added for an instance.  It is best to do this in `before_each`.
``` gdscript
func before_each():
  stub(MyScript, "some_method").to_return(111)

func test_script_level_stubs():
  var uses_script_stub = double(MyScript).new()
  var has_instance_stub = double(MyScript).new()
  stub(has_instance_stub.some_method).to_return(555)
  stub(has_instance_stub.some_method.bind("a")).to_return(999)

  assert_eq(uses_script_stub.some_method("anything"), 111)
  assert_eq(uses_script_stub.some_method("a"), 111)

  assert_eq(has_instance_stub.some_method("anything"), 555)
  assert_eq(has_instance_stub.some_method("a"), 999)
```




## Stub Actions
You can stub a method to take the following actions when called.  These actions will be taken based on the best matched stub found for the method.

__Only one action should be specified as only one will be used.__

Chain an action to the end of a call to `stub`
``` gdscript
stub(MyScript, "some_method").to_return(9)
```

### to_return(value)
This stubs the method to do nothing and return a specific value when called.


### to_do_nothing()
This is the same as `to_return(null)` but has a nice explicit name that is easy to read.  This is mostly used with Partial Doubles to make them not "call super".
```gdscript
var inst = partial_double(MyScript).new()
stub(inst._set).to_do_nothing()

inst.some_property = 9
assert_ne(inst.some_property, 9)
```

### to_call_super()
This will cause a method to punch through to `super`'s implementation of the method retaining the original functionality of the method.
```gdscript
var dbl_inst = double(MyScript).new()
var inst = double(MyScript).new()

stub(dbl_inst.some_method).to_call_super()

assert_eq(dbl_inst.some_method(), inst.some_method())
```

### to_call(callable)
This will cause the double to call the method specified.  Any value returned from the `Callable` will be returned by the original method.

The parameters passed to the original method will be sent to the `Callable.`  Be sure the parameters of the `Callable` are compatable with the parameters of the original method or a runtime error will occur.

Any parameters bound to the `Callable` will be passed __after__ the parameters sent to the original method.  Bound parameters will not be included when spying.

```gdscript
func return_passed(p1=null, p2=null, p3=null, p4=null, p5=null, p6=null):
  return [p1, p2, p3, p4, p5, p6]

func test_illustrate_stub_action_to_call():
  var dbl_inst = double(MyScript).new()

  stub(MyScript, "foo")\
    .to_call(func(value):  print("We did not get a seven."))

  stub(MyScript, "foo")\
    .when_passed(7)\
    .to_call(func(value):  print("We got a 7"))


  # This illustrates the ordering of bound parameters, return values, and
  # argument spying.
  stub(dbl_inst.bar).to_call(return_passed.bind("three", "four"))
  var result = dbl_inst.bar("one", "two")
  assert_eq(result, ["one", "two", "three", "four", null, null])
  assert_called(dbl_inst, "bar", ["one", "two"],
    "This passes because bound arguments are not spied on.")
```

All the greatness and super weirdness of using lambdas applies.
```gdscript
func test_using_local_variable_in_callable():
  var this_var = "some value"
  var d = double(DoubleMe).new()
  stub(d.has_one_param).to_call(
    func(value):
      this_var = "another value"
      return this_var)

  var result = d.has_one_param("asdf")

  # These all pass
  assert_eq(result, "another value", "Seems reasonable")
  assert_ne(result, this_var, "Why would this pass?")
  assert_eq(this_var, "some value", "Ohhh, well ok.")
```


## Stub Qualifiers
There is only one qualifier, `when_passed`.  Only the last `when_passed` qualifier will be used.  If you wish to stub the same action for multiple argument patterns you must make multiple stubs.  See the various examples above.




## Method Signature Modifiers
You alter a method's signature with:
* `param_defaults([default1, default2, ...])`
* `param_count(x)`

There are very few cases where you would want to do this.  They are discussed further near the end of this page.




## Stubbing at the Script level vs Instance level
The `stub` method is pretty smart about what you pass it.  You can pass it a path, an Object or an instance.  If you pass an instance, it __must__ be an instance of a double.

When passed an Object or a path, it will stub the value for __all__ instances that do not have explicitly defined stubs.  When you pass it an instance of a doubled class, then the stubbed return will only be set for that instance.

```gdscript
var DoubleThis = load('res://scripts/double_this.gd')
var Doubled = double(DoubleThis)
var inst = Doubled.new()

# These two are equivalent, and stub returns_seven for any doubles of
# DoubleThis to return 500.
stub('res://scripts/double_this.gd', 'returns_seven').to_return(500)
# or
stub(DoubleThis, 'returns_seven').to_return(500)
assert_eq(inst.returns_seven(), 500)

# This will stub returns_seven on the passed in instance ONLY.
# Any other instances will return 500 from the lines above.
var stub_again = Doubled.new()
stub(stub_again, 'returns_seven').to_return('words')
assert_eq(stub_again.returns_seven(), 'words')
assert_eq(inst.returns_seven, 500)
```




## Stubbing based off of parameter values
You can stub a method to return a specific value based on what was passed to it.
```gdscript
var DoubleThis = load('res://scripts/double_this.gd')
var Doubled = double(DoubleThis)
var inst = Doubled.new()

# Script level using when_passed
stub(DoubleThis, "return_hello").to_return("world").when_passed("hello")
# Instance level using bound callable
stub(inst.return_hello.bind("foo")).to_return("bar")

assert_eq(inst.return_hello(), "hello")
assert_eq(inst.return_hello("hello"), "world")
assert_eq(inst.return_hello("foo"), "bar")
```
The ordering of `when_passed` and `to_return` does not matter.




## Stubbing Packed Scenes
When stubbing doubled scenes, use the path to the scene, __not__ the path to the scene's script.  If you double and stub the script used by the scene, the `instance` you make from `double` will not return values stubbed for the script.  It will only return values stubbed for the scene.

In order for a scene to be doubled, the scene's script must be able to be instantiated with `new` with zero parameters passed.

### Example
Given the script `res://the_script.gd`:
``` gdscript
func return_hello():
  return 'hello'
```
And given a scene with the path `res://double_this_scene.tscn` which has its script set to `res://the_script.gd`.

``` gdscript
var DoubleThisScene = load('res://double_this_scene.tscn')

func test_illustrate_stubbing_scenes():
  var doubled_scene = double(DoubleThisScene).instantiate()
  stub(doubled_scene, 'return_hello').to_return('world')

  assert_eq(doubled_scene.return_hello(), 'world')
```




## Stubbing Method Parameter Defaults
Godot only provides information about default values for built in methods so Gut doesn't know what any default values are for methods you have created.  Since it can't know, Gut defaults all parameters to `null`.  This can cause issues in specific cases (probably all involving calling super).  You can use `.param_defaults` to specify default values to be used.

Here's an example where things go wrong
```
# res://foo.gd
var _sum  = 0
func increment(inc_by=1):
  _sum += inc_by

func go_up_one():
  increment()

func get_sum():
  return _sum
```

The following test will cause a `Invalid operands 'int' and 'Nil'` error.  This is because increment's `inc_by` parameter is defaulted to `null` in the double.
```
var Foo = load('res://foo.gd')
test_go_up_one_increments_sum_by_1():
  var dbl_foo = double(Foo).new()
  stub(dbl_foo, 'go_up_one').to_call_super()

  dbl_foo.go_up_one()
  assert_called(dbl_foo, 'increment', [1])
```

The fix is to add a `param_defaults` stub
```
stub(dbl_foo, 'increment').param_defaults([1])
```




## Stubbing Method Parameter Count
<u>__Changing the number of parameters must be done before `double` is called__</u>

Some built-in methods  have `vararg` parameters.  This makes the parameter list dynamic.  Godot does not provide this information.  This can cause errors due to a signature mismatch.  Your code might be calling a method using 10 parameter values but Gut only sees two.

Let's take `Node.rpc_id` for example.  It has two normal parameters and then a vararg of strings as the last parameter.
```
Variant rpc_id(peer_id: int, method: String, ...) vararg
```
If this method gets called in a partial double with more than 2 parameters Godot will will throw _Invalid call to function 'rpc_id' in base 'Control ()'. Expected 2 arguments._

You can use `.param_count(x)` to tell Gut to give the method any number of extra parameters.  You cannot make the method have less parameters.  You must do this before you call `double`.
``` gdscript
func test_issue_246_rpc_id_varargs():
  # must happen before double is called
  stub(Node, 'rpc_id').to_do_nothing().param_count(5)

  var inst = double(Node).new()
  inst.rpc_id(1, 'foo', '3', '4', '5')
  assert_called(inst, 'rpc_id', [1, 'foo', '3', '4', '5'])
```

You can also use `.param_defaults` to specify extra parameters if you supply more defaults than the method has parameters.

``` gdscript
func test_issue_246_rpc_id_varargs_with_defaults():
  # must happen before double is called
  stub(Node, 'rpc_id').to_do_nothing().param_defaults([null, null, 'a', 'b', 'c'])

  var inst = double(Node).new()
  inst.rpc_id(1, 'foo', 'z')
  assert_called(inst, 'rpc_id', [1, 'foo', 'z', 'b', 'c'])
```
You cannot make a method have less parameters, only more.




## Stubbing Accessors
It is not possible to stub the accessors for properties if you do not use a secondary method for the accessors.  This means that doubles retain the functionality of the accessors, and it cannot be changed.
``` gdscript
# The get and set for my_property cannot be stubbed.  Doubles retain the
# functionality of the get and set methods.
var my_property = 'foo' :
  get: return my_property
  set(val): my_property = val
```

If you use secondary methods, you can stub the behavior, but all doubles will not have any functionality for the accessors by default.
```gdscript
# You can stub _get_my_property and _set_my_property.  Doubles of this do not
# retain the functionality of the accessors.  _get_my_property and
# _set_my_property must be stubbed to_call_super to actually return or set
# the value of my_property.
var my_property = 'foo' :
  get: _get_my_property, set: _set_my_property

func _get_my_property():
  return my_property

func _set_my_property(val):
  my_property = val
```