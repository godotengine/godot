# Inner Test Classes

You can define test classes inside a test script that will be treated as test scripts themselves.  This allows you to create different contexts for your tests in a single script.  These Inner Classes have their own `before_all`, `before_each`, `after_each`, and `after_all` methods that will be called.  Only the methods defined in the class are used, the methods defined in the containing script will not be called.

The Inner Classes must also extend `GutTest` and their constructor cannot take any parameters.  The Classes will be loaded and ran in the order they are defined _after_ all the tests in the containing script are run.  If the script does not contain any tests then only the Inner Classes will be listed in the output.

The order the tests are run are not guaranteed to be in the same order they are defined (I don't know why yet).

Inner Classes are parsed out of the script by looking for a classes that start with `'Test'` and also extend `test.gd`.  You can change the name that Gut looks for using the `inner_class_prefix` property.

## Example
Given the following test script defined at `res://test/unit/some_example.gd`
```
extends GutTest

func before_all():
	gut.p('script:  pre-run')

func before_each():
	gut.p('script:  setup')

func after_each():
	gut.p('script:  teardown')

func after_all():
	gut.p('script:  post-run')

func test_something():
	assert_true(true)

class TestClass1:
	extends GutTest

	func before_all():
		gut.p('TestClass1:  pre-run')

	func before_each():
		gut.p('TestClass1:  setup')

	func after_each():
		gut.p('TestClass1:  teardown')

	func after_all():
		gut.p('TestClass1:  post-run')

	func test_context1_one():
		assert_true(true)

	func test_context1_two():
		pending()
```

Gut will generate this following when running the test script.

```
/-----------------------------------------
Running Script res://test/unit/some_sample.gd
-----------------------------------------/
script:  pre-run
* test_something
    script:  setup
    PASSED:
    script:  teardown

/-----------------------------------------
Running Class [TestClass1] in res://test/unit/some_sample.gd
-----------------------------------------/
TestClass1:  pre-run
* test_context1_two
    TestClass1:  setup
    Pending
    TestClass1:  teardown
* test_context1_one
    TestClass1:  setup
    PASSED:
    TestClass1:  teardown
    TestClass1:  post-run
```
