extends Node2D

var _plugin_name = "GodotAppInstrumentedTestPlugin"
var _android_plugin

func _ready():
	if Engine.has_singleton(_plugin_name):
		_android_plugin = Engine.get_singleton(_plugin_name)
		_android_plugin.connect("launch_tests", _launch_tests)
	else:
		printerr("Couldn't find plugin " + _plugin_name)
		get_tree().quit()

func _launch_tests(test_label: String) -> void:
	var test_instance: BaseTest = null
	match test_label:
		"javaclasswrapper_tests":
			test_instance = JavaClassWrapperTests.new()
		"file_access_tests":
			test_instance = FileAccessTests.new()

	if test_instance:
		test_instance.__reset_tests()
		test_instance.run_tests()
		var incomplete_tests = test_instance._test_started - test_instance._test_completed
		_android_plugin.onTestsCompleted(test_label, test_instance._test_completed, test_instance._test_assert_failures + incomplete_tests)
	else:
		_android_plugin.onTestsFailed(test_label, "Unable to launch tests")


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
