# Creating Tests

## Quick Sample
Here's a sample test script.  Copy the contents into the file `res://test/unit/test_example.gd` then run the script.  If everything is setup correctly then you'll see some passing and failing tests.

``` gdscript
extends GutTest
func before_each():
	gut.p("ran setup", 2)

func after_each():
	gut.p("ran teardown", 2)

func before_all():
	gut.p("ran run setup", 2)

func after_all():
	gut.p("ran run teardown", 2)

func test_assert_eq_number_not_equal():
	assert_eq(1, 2, "Should fail.  1 != 2")

func test_assert_eq_number_equal():
	assert_eq('asdf', 'asdf', "Should pass")

func test_assert_true_with_true():
	assert_true(true, "Should pass, true is true")

func test_assert_true_with_false():
	assert_true(false, "Should fail")

func test_something_else():
	assert_true(false, "didn't work")
```

## Details
All test scripts must extend the test class.
* `extends GutTest`

Each test script has optional setup and teardown methods that you can provide an implementation for.  These are called by Gut at various stages of execution.  They take no parameters.
 * `before_each()`:  Runs before each test.
 * `after_each()`:  Runs after each test.
 * `before_all()`:  Runs once before any test starts running.
 * `after_all()`:  Runs once after all tests finish running.

All tests in the test script must start with the prefix `test_` in order for them to be run.  The methods must not have any parameters (except [Parameterized-Tests](Parameterized-Tests)).
* `func test_this_is_only_a_test():`

Each test should perform at least one assert or call `pending`, `pass_test` or `fail_test` or it will be marked risky.

A list of all `asserts` and other helper functions available in your test script can be found in [GutTest](class_GutTest).


## Inner Test Classes
You can group tests together using Inner Classes. These classes must start with the prefix `'Test'` (this is configurable) and they must also extend `GutTest`.  You cannot create Inner Test Classes inside Inner Test Classes.  More info can be found at [Inner Test Classes](Inner-Test-Classes).

### Simple Example
``` gdscript
extends GutTest

class TestFeatureA:
	extends GutTest

	var Obj = load('res://scripts/object.gd')
	var _obj = null

	func before_each():
		_obj = Obj.new()

	func test_something():
		assert_true(_obj.is_something_cool(), 'Should be cool.')

class TestFeatureB:
	extends GutTest

	var Obj = load('res://scripts/object.gd')
	var _obj = null

	func before_each():
		_obj = Obj.new()

	func test_foobar():
		assert_eq(_obj.foo(), 'bar', 'Foo should return bar')
```
## Where to next?
* [GutTest](class_GutTest)
* [Inner Test Classes](Inner-Test-Classes)
* [Command Line](Command-Line)
