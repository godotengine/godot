class_name JavaClassWrapperTests
extends BaseTest

func run_tests():
	print("JavaClassWrapper tests starting..")

	__exec_test(test_exceptions)

	__exec_test(test_multiple_signatures)
	__exec_test(test_array_arguments)
	__exec_test(test_array_return)

	__exec_test(test_dictionary)

	__exec_test(test_object_overload)

	__exec_test(test_variant_conversion_safe_from_stack_overflow)

	__exec_test(test_big_integers)

	__exec_test(test_callable)

	print("JavaClassWrapper tests finished.")
	print("Tests started: " + str(_test_started))
	print("Tests completed: " + str(_test_completed))


func test_exceptions() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')
	#print(TestClass.get_java_method_list())

	assert_equal(JavaClassWrapper.get_exception(), null)

	assert_equal(TestClass.testExc(27), 0)
	assert_equal(str(JavaClassWrapper.get_exception()), '<JavaObject:java.lang.NullPointerException "java.lang.NullPointerException">')

	assert_equal(JavaClassWrapper.get_exception(), null)

func test_multiple_signatures() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')

	var ai := [1, 2]
	assert_equal(TestClass.testMethod(1, ai), "IntArray: [1, 2]")

	var astr := ["abc"]
	assert_equal(TestClass.testMethod(2, astr), "IntArray: [0]")

	var atstr: Array[String] = ["abc"]
	assert_equal(TestClass.testMethod(3, atstr), "StringArray: [abc]")

	var TestClass2: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass2')
	var aobjl: Array[Object] = [
		TestClass2.TestClass2(27),
		TestClass2.TestClass2(135),
	]
	assert_equal(TestClass.testMethod(3, aobjl), "testObjects: 27 135")

func test_array_arguments() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')

	assert_equal(TestClass.testArgBoolArray([true, false, true]), "[true, false, true]")
	assert_equal(TestClass.testArgByteArray(PackedByteArray([1, 2, 3])), "[1, 2, 3]")
	assert_equal(TestClass.testArgCharArray("abc".to_utf16_buffer()), "abc");
	assert_equal(TestClass.testArgShortArray(PackedInt32Array([27, 28, 29])), "[27, 28, 29]")
	assert_equal(TestClass.testArgShortArray([27, 28, 29]), "[27, 28, 29]")
	assert_equal(TestClass.testArgIntArray(PackedInt32Array([7, 8, 9])), "[7, 8, 9]")
	assert_equal(TestClass.testArgIntArray([7, 8, 9]), "[7, 8, 9]")
	assert_equal(TestClass.testArgLongArray(PackedInt64Array([17, 18, 19])), "[17, 18, 19]")
	assert_equal(TestClass.testArgLongArray([17, 18, 19]), "[17, 18, 19]")
	assert_equal(TestClass.testArgFloatArray(PackedFloat32Array([17.1, 18.2, 19.3])), "[17.1, 18.2, 19.3]")
	assert_equal(TestClass.testArgFloatArray([17.1, 18.2, 19.3]), "[17.1, 18.2, 19.3]")
	assert_equal(TestClass.testArgDoubleArray(PackedFloat64Array([37.1, 38.2, 39.3])), "[37.1, 38.2, 39.3]")
	assert_equal(TestClass.testArgDoubleArray([37.1, 38.2, 39.3]), "[37.1, 38.2, 39.3]")

func test_array_return() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')
	#print(TestClass.get_java_method_list())

	assert_equal(TestClass.testRetBoolArray(), [true, false, true])
	assert_equal(TestClass.testRetWrappedBoolArray(), [true, false, true])

	assert_equal(TestClass.testRetByteArray(), PackedByteArray([1, 2, 3]))
	assert_equal(TestClass.testRetWrappedByteArray(), PackedByteArray([1, 2, 3]))

	assert_equal(TestClass.testRetCharArray().get_string_from_utf16(), "abc")
	assert_equal(TestClass.testRetWrappedCharArray().get_string_from_utf16(), "abc")

	assert_equal(TestClass.testRetShortArray(), PackedInt32Array([11, 12, 13]))
	assert_equal(TestClass.testRetWrappedShortArray(), PackedInt32Array([11, 12, 13]))

	assert_equal(TestClass.testRetIntArray(), PackedInt32Array([21, 22, 23]))
	assert_equal(TestClass.testRetWrappedIntArray(), PackedInt32Array([21, 22, 23]))

	assert_equal(TestClass.testRetLongArray(), PackedInt64Array([41, 42, 43]))
	assert_equal(TestClass.testRetWrappedLongArray(), PackedInt64Array([41, 42, 43]))

	assert_equal(TestClass.testRetFloatArray(), PackedFloat32Array([31.1, 32.2, 33.3]))
	assert_equal(TestClass.testRetWrappedFloatArray(), PackedFloat32Array([31.1, 32.2, 33.3]))

	assert_equal(TestClass.testRetDoubleArray(), PackedFloat64Array([41.1, 42.2, 43.3]))
	assert_equal(TestClass.testRetWrappedDoubleArray(), PackedFloat64Array([41.1, 42.2, 43.3]))

	var obj_array = TestClass.testRetObjectArray()
	assert_equal(str(obj_array[0]), '<JavaObject:com.godot.game.test.javaclasswrapper.TestClass2 "51">')
	assert_equal(str(obj_array[1]), '<JavaObject:com.godot.game.test.javaclasswrapper.TestClass2 "52">')

	assert_equal(TestClass.testRetStringArray(), PackedStringArray(["I", "am", "String"]))
	assert_equal(TestClass.testRetCharSequenceArray(), PackedStringArray(["I", "am", "CharSequence"]))

func test_dictionary():
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')
	assert_equal(TestClass.testDictionary({a = 1, b = 2}), "{a=1, b=2}")
	assert_equal(TestClass.testRetDictionary(), {a = 1, b = 2})
	assert_equal(TestClass.testRetDictionaryArray(), [{a = 1, b = 2}])
	assert_equal(TestClass.testDictionaryNested({a = 1, b = [2, 3], c = 4}), "{a: 1, b: [2, 3], c: 4}")

func test_object_overload():
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')
	var TestClass2: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass2')
	var TestClass3: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass3')

	var t2 = TestClass2.TestClass2(33)
	var t3 = TestClass3.TestClass3("thirty three")

	assert_equal(TestClass.testObjectOverload(t2), "TestClass2: 33")
	assert_equal(TestClass.testObjectOverload(t3), "TestClass3: thirty three")

	var arr_of_t2 = [t2, TestClass2.TestClass2(34)]
	var arr_of_t3 = [t3, TestClass3.TestClass3("thirty four")]

	assert_equal(TestClass.testObjectOverloadArray(arr_of_t2), "TestClass2: [33, 34]")
	assert_equal(TestClass.testObjectOverloadArray(arr_of_t3), "TestClass3: [thirty three, thirty four]")

func test_variant_conversion_safe_from_stack_overflow():
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')
	var arr: Array = [42]
	var dict: Dictionary = {"arr": arr}
	arr.append(dict)
	# The following line will crash with stack overflow if not handled property:
	TestClass.testDictionary(dict)

func test_big_integers():
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.javaclasswrapper.TestClass')
	assert_equal(TestClass.testArgLong(4242424242), "4242424242")
	assert_equal(TestClass.testArgLong(-4242424242), "-4242424242")
	assert_equal(TestClass.testDictionary({a = 4242424242, b = -4242424242}), "{a=4242424242, b=-4242424242}")

func test_callable():
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	assert_true(android_runtime != null)

	var cb1_data := {called = false}
	var cb1 = func():
		cb1_data['called'] = true
		return null
	android_runtime.createRunnableFromGodotCallable(cb1).run()
	assert_equal(cb1_data['called'], true)
