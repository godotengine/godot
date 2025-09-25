extends Node2D

var _plugin_name = "GodotAppInstrumentedTestPlugin"
var _android_plugin

var _test_passes := 0
var _test_failures := 0

func _ready():
	if Engine.has_singleton(_plugin_name):
		_android_plugin = Engine.get_singleton(_plugin_name)
		_android_plugin.connect("launch_javaclasswrapper_tests", launch_javaclasswrapper_tests)
	else:
		printerr("Couldn't find plugin " + _plugin_name)
		get_tree().quit()

func _reset_tests():
	_test_passes = 0
	_test_failures = 0

func launch_javaclasswrapper_tests():
	_reset_tests()

	test_exceptions()

	test_multiple_signatures()
	test_array_arguments()
	test_array_return()

	test_dictionary()

	test_object_overload()

	test_variant_conversion_safe_from_stack_overflow()

	print("All tests finished.")

	if _android_plugin:
		_android_plugin.onJavaClassWrapperTestsCompleted(_test_passes, _test_failures)

func __get_stack_frame():
	var me = get_script()
	for s in get_stack():
		if not s.function.begins_with('__') and s.function != "assert_equal":
			return s
	return null

func __assert_pass():
	_test_passes += 1
	pass

func __assert_fail():
	_test_failures += 1
	var s = __get_stack_frame()
	if s != null:
		print_rich ("[color=red] == FAILURE: In function %s() from '%s' on line %s[/color]" % [s.function, s.source, s.line])
	else:
		print_rich ("[color=red] == FAILURE (run with --debug to get more information!) ==[/color]")

func assert_equal(actual, expected):
	if actual == expected:
		__assert_pass()
	else:
		__assert_fail()
		print ("    |-> Expected '%s' but got '%s'" % [expected, actual])

func test_exceptions() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass')
	#print(TestClass.get_java_method_list())

	assert_equal(JavaClassWrapper.get_exception(), null)

	assert_equal(TestClass.testExc(27), 0)
	assert_equal(str(JavaClassWrapper.get_exception()), '<JavaObject:java.lang.NullPointerException "java.lang.NullPointerException">')

	assert_equal(JavaClassWrapper.get_exception(), null)

func test_multiple_signatures() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass')

	var ai := [1, 2]
	assert_equal(TestClass.testMethod(1, ai), "IntArray: [1, 2]")

	var astr := ["abc"]
	assert_equal(TestClass.testMethod(2, astr), "IntArray: [0]")

	var atstr: Array[String] = ["abc"]
	assert_equal(TestClass.testMethod(3, atstr), "StringArray: [abc]")

	var TestClass2: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass2')
	var aobjl: Array[Object] = [
		TestClass2.TestClass2(27),
		TestClass2.TestClass2(135),
	]
	assert_equal(TestClass.testMethod(3, aobjl), "testObjects: 27 135")

func test_array_arguments() -> void:
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass')

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
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass')
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
	assert_equal(str(obj_array[0]), '<JavaObject:com.godot.game.test.TestClass2 "51">')
	assert_equal(str(obj_array[1]), '<JavaObject:com.godot.game.test.TestClass2 "52">')

	assert_equal(TestClass.testRetStringArray(), PackedStringArray(["I", "am", "String"]))
	assert_equal(TestClass.testRetCharSequenceArray(), PackedStringArray(["I", "am", "CharSequence"]))

func test_dictionary():
	assert_equal(_android_plugin.testDictionary({a = 1, b = 2}), "{a=1, b=2}")
	assert_equal(_android_plugin.testRetDictionary(), {a = 1, b = 2})
	assert_equal(_android_plugin.testRetDictionaryArray(), [{a = 1, b = 2}])
	assert_equal(_android_plugin.testDictionaryNested({a = 1, b = [2, 3], c = 4}), "{a: 1, b: [2, 3], c: 4}")

func test_object_overload():
	var TestClass: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass')
	var TestClass2: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass2')
	var TestClass3: JavaClass = JavaClassWrapper.wrap('com.godot.game.test.TestClass3')

	var t2 = TestClass2.TestClass2(33)
	var t3 = TestClass3.TestClass3("thirty three")

	assert_equal(TestClass.testObjectOverload(t2), "TestClass2: 33")
	assert_equal(TestClass.testObjectOverload(t3), "TestClass3: thirty three")

	var arr_of_t2 = [t2, TestClass2.TestClass2(34)]
	var arr_of_t3 = [t3, TestClass3.TestClass3("thirty four")]

	assert_equal(TestClass.testObjectOverloadArray(arr_of_t2), "TestClass2: [33, 34]")
	assert_equal(TestClass.testObjectOverloadArray(arr_of_t3), "TestClass3: [thirty three, thirty four]")

func test_variant_conversion_safe_from_stack_overflow():
	var arr: Array = [42]
	var dict: Dictionary = {"arr": arr}
	arr.append(dict)
	# The following line will crash with stack overflow if not handled property:
	_android_plugin.testDictionary(dict)

func _on_plugin_toast_button_pressed() -> void:
	if _android_plugin:
		_android_plugin.helloWorld()

func _on_vibration_button_pressed() -> void:
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	if android_runtime:
		print("Checking if the device supports vibration")
		var vibrator_service = android_runtime.getApplicationContext().getSystemService("vibrator")
		if vibrator_service:
			if vibrator_service.hasVibrator():
				print("Vibration is supported on device! Vibrating now...")
				var VibrationEffect = JavaClassWrapper.wrap("android.os.VibrationEffect")
				var effect = VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE)
				vibrator_service.vibrate(effect)
			else:
				printerr("Vibration is not supported on device")
		else:
			printerr("Unable to retrieve the vibrator service")
	else:
		printerr("Couldn't find AndroidRuntime singleton")

func _on_gd_script_toast_button_pressed() -> void:
	var android_runtime = Engine.get_singleton("AndroidRuntime")
	if android_runtime:
		var activity = android_runtime.getActivity()

		var toastCallable = func ():
			var ToastClass = JavaClassWrapper.wrap("android.widget.Toast")
			ToastClass.makeText(activity, "Toast from GDScript", ToastClass.LENGTH_LONG).show()

		activity.runOnUiThread(android_runtime.createRunnableFromGodotCallable(toastCallable))
