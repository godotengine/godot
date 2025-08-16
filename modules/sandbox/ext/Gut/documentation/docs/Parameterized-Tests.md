# Parameterized Tests
There are some scenarios where it is desirable to run a test numerous times with different parameters.  You can do this in GUT by creating a test that has a single parameter that is defaulted to the GUT method `use_parameters`.

`use_parameters` expects an array.  The test will be called once for each element in the array, passing the value of each element to the parameter of the function.

## Requirements:
* The test must have one and only one parameter.
* The parameter must be defaulted to call `use_parameters`.
* You must pass an array to `use_parameters`.  The test will be called once for each element in the array.
* If the parameter is not defaulted then it will cause a runtime error.
* If `use_parameters` is not called then GUT will only run the test once and will generate an error.


## Example

``` gdscript
extends GutTest

class Foo:
  func add(p1, p2):
    return p1 + p2

# Define one array, with two arrays as the elements in the array.
# This will cause the test to be called twice.  The first
# call will get [1, 2, 3] as the value of the parameter.
# The second call will get ['a', 'b', 'c']
var foo_params = [[1, 2, 3], ['a', 'b', 'c']]

# This test will add the first two elements in params together,
# using Foo, and assert that they equal the third element in params.
func test_foo(params=use_parameters(foo_params)):
  var foo = Foo.new()
  var result = foo.add(params[0], params[1])
  assert_eq(result, params[2])
```
Running this test will result in:
* One passing test (`1 + 2 = 3`)
* One failing test (`'a' + 'b'  != 'c'`, it actually equals `'ab'`)

## ParameterFactory
`GutTest` scripts have access to the `ParameterFactory` static class which has helper methods for defining parameters for parameterized tests.

### Methods
There's only one right now.  If you have any suggestions, open up an [issue on Github](https://github.com/bitwes/Gut/issues).

#### named_parameters
`named_parameters(names, values)`<br>
Creates an array of dictionaries.  It pairs up the names array with each set of values in values.  If more names than values are specified then the missing values will be filled with nulls.  If more values than names are specified those values will be ignored.

Example:
``` gdscript
# With this setup, you can use `params.p1`, `params.p2`, and
# `params.result` in the test below.
var foo_params = ParameterFactory.named_parameters(
    ['p1', 'p2', 'result'], # names
    [                       # values
        [1, 2, 3],
        ['a', 'b', 'c']
    ])

func test_foo(params = use_parameters(foo_params)):
    var foo = Foo.new()
    var result = foo.add(params.p1, params.p2)
    assert_eq(result, params.result)
```

Here are some invalid setups and what results.
``` gdscript
# Example of extra values.  This returns:
# [{a:1}, {a:3}]
ParameterFactory.named_parameters(
    ['a'],
    [
        [1, 2],
        [3, 4]
    ])

# Example of not enough values.  This returns:
# [{a:1, b:null}, {a:'oops', b:null}, {a:'one', b:'two'}]
ParameterFactory.named_parameters(
    ['a', 'b'],
    [
        [1],
        'oops',
        ['one', 'two']
    ])
```