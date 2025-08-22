extends Node2D
@onready var _gut_control = $GutControl

var _current_script_object = null
var _current_test_name = null


func _ready():
	_gut_control.load_config_file('res://test/resources/GutControlDirs/gut_control_config.json')

	# Returns a gut_config.gd instance.
	var config = _gut_control.get_config()
	# Override soecific values for the purposes of this
	# scene.  You can see all the options available in
	# the default_options dictionary in gut_config.gd
	config.options.should_exit = false
	config.options.compact_mode = false

	call_deferred('_post_ready_setup')


func _post_ready_setup():
	var gut = _gut_control.get_gut()
	gut.start_run.connect(_on_gut_run_start)
	
	gut.start_script.connect(_on_gut_start_script)
	gut.end_script.connect(_on_gut_end_script)
	
	gut.start_test.connect(_on_gut_start_test)
	gut.end_test.connect(_on_gut_end_test)
	
	gut.end_run.connect(_on_gut_run_end)


func _on_gut_run_start():
	print('Starting tests')


# This signal passes a TestCollector.gd/TestScript instance
func _on_gut_start_script(script_obj):
	print(script_obj.get_full_name(), ' has ', script_obj.tests.size(), ' tests')
	_current_script_object = script_obj


func _on_gut_end_script():
	var pass_count = 0
	for test in _current_script_object.tests:
		if(test.did_pass()):
			pass_count += 1
	print(pass_count, '/', _current_script_object.tests.size(), " passed\n")
	_current_script_object = null


func _on_gut_start_test(test_name):
	_current_test_name = test_name
	print('  ', test_name)


func _on_gut_end_test():
	# get_test_named returns a TestCollector.gd/Test instance for the name 
	# passed in.
	var test_object = _current_script_object.get_test_named(_current_test_name)
	var status = "failed"
	if(test_object.did_pass()):
		status = "passed"
	elif(test_object.pending):
		status = "pending"
		
	print('    ', status)
	_current_test_name = null
	

func _on_gut_run_end():
	print('Tests Done')
#
