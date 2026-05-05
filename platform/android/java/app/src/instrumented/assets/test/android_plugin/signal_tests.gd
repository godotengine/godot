class_name AndroidPluginSignalTests
extends BaseTest

var _plugin: JNISingleton
const emission_test_signal = "emission_test_signal"
const launch_test_signal = "launch_tests"

signal emission_test_signal_emitted

func _init(plugin: JNISingleton) -> void:
	_plugin = plugin

func run_tests():
	print("Android plugin signal tests starting...")

	__exec_test(test_plugin_exists)
	__exec_test(test_signal_registration)
	await __exec_test(test_signal_emission)

	print("Android plugin signal tests completed.")

func test_plugin_exists() -> bool:
	if _plugin == null:
		printerr("ERROR: Couldn't find SignalTestPlugin plugin; _plugin is null")
		return false

	return true

func test_signal_registration() -> bool:
	var signal_registered = _plugin.has_signal(emission_test_signal)
	assert_true(signal_registered)

	var launch_signal_registered = _plugin.has_signal(launch_test_signal)
	assert_true(launch_signal_registered)

	return true

func test_signal_emission() -> bool:
	var err1 = _plugin.connect(emission_test_signal, _on_emission_test_signal_emitted)
	assert_equal(err1, OK)
	_plugin.triggerTestSignal1()
	await emission_test_signal_emitted

	# Test case: Same signal name, but different type and number of parameters
	# The "launch_tests" signal is registered by both GodotAppInstrumentedTestPlugin and SignalTestPlugin.
	# SignalTestPlugin emits it with a boolean and a string arguments, while GodotAppInstrumentedTestPlugin emits it with one string.
	var err2 = _plugin.connect(launch_test_signal, _on_launch_tests_emitted)
	assert_equal(err2, OK)
	_plugin.triggerLaunchTestSignal()
	await emission_test_signal_emitted

	return true

func _on_emission_test_signal_emitted() -> void:
	emission_test_signal_emitted.emit()

func _on_launch_tests_emitted(param1: bool, param2: String) -> void:
	assert_true(param1)
	assert_equal(param2, "second message")
	emission_test_signal_emitted.emit()
