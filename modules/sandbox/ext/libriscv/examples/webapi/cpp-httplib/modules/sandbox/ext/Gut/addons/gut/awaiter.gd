extends Node

signal timeout
signal wait_started

var _wait_time := 0.0
var _wait_process_frames := 0
var _wait_physics_frames := 0
var _signal_to_wait_on = null

var _predicate_method = null
var _waiting_for_predicate_to_be = null

var _predicate_time_between := 0.0
var _predicate_time_between_elpased := 0.0

var _did_last_wait_timeout = false
var did_last_wait_timeout = false :
	get: return _did_last_wait_timeout
	set(val): push_error("Cannot set did_last_wait_timeout")

var _elapsed_time := 0.0
var _elapsed_frames := 0

func _ready() -> void:
	get_tree().process_frame.connect(_on_tree_process_frame)
	get_tree().physics_frame.connect(_on_tree_physics_frame)


func _on_tree_process_frame():
	# Count frames here instead of in _process so that tree order never
	# makes a difference and the count/signaling happens outside of
	# _process being called.
	if(_wait_process_frames > 0):
		_elapsed_frames += 1
		if(_elapsed_frames > _wait_process_frames):
			_end_wait()


func _on_tree_physics_frame():
	# Count frames here instead of in _physics_process so that tree order never
	# makes a difference and the count/signaling happens outside of
	# _physics_process being called.
	if(_wait_physics_frames != 0):
		_elapsed_frames += 1
		if(_elapsed_frames > _wait_physics_frames):
			_end_wait()


func _physics_process(delta):
	if(_wait_time != 0.0):
		_elapsed_time += delta
		if(_elapsed_time >= _wait_time):
			_end_wait()

	if(_predicate_method != null):
		_predicate_time_between_elpased += delta
		if(_predicate_time_between_elpased >= _predicate_time_between):
			_predicate_time_between_elpased = 0.0
			var result = _predicate_method.call()
			if(_waiting_for_predicate_to_be == false):
				if(typeof(result) != TYPE_BOOL or result != true):
					_end_wait()
			else:
				if(typeof(result) == TYPE_BOOL and result == _waiting_for_predicate_to_be):
					_end_wait()


func _end_wait():
	# Check for time before checking for frames so that the extra frames added
	# when waiting on a signal do not cause a false negative for timing out.
	if(_wait_time > 0):
		_did_last_wait_timeout = _elapsed_time >= _wait_time
	elif(_wait_physics_frames > 0):
		_did_last_wait_timeout = _elapsed_frames >= _wait_physics_frames
	elif(_wait_process_frames > 0):
		_did_last_wait_timeout = _elapsed_frames >= _wait_process_frames

	if(_signal_to_wait_on != null and \
	   is_instance_valid(_signal_to_wait_on.get_object()) and \
	   _signal_to_wait_on.is_connected(_signal_callback)):
		_signal_to_wait_on.disconnect(_signal_callback)

	_wait_process_frames = 0
	_wait_time = 0.0
	_wait_physics_frames = 0
	_signal_to_wait_on = null
	_predicate_method = null
	_elapsed_time = 0.0
	_elapsed_frames = 0
	timeout.emit()


const ARG_NOT_SET = '_*_argument_*_is_*_not_set_*_'
func _signal_callback(
		_arg1=ARG_NOT_SET, _arg2=ARG_NOT_SET, _arg3=ARG_NOT_SET,
		_arg4=ARG_NOT_SET, _arg5=ARG_NOT_SET, _arg6=ARG_NOT_SET,
		_arg7=ARG_NOT_SET, _arg8=ARG_NOT_SET, _arg9=ARG_NOT_SET):

	_signal_to_wait_on.disconnect(_signal_callback)
	# DO NOT _end_wait here.  For other parts of the test to get the signal that
	# was waited on, we have to wait for another frames.  For example, the
	# signal_watcher doesn't get the signal in time if we don't do this.
	_wait_process_frames = 1


func wait_seconds(x):
	_did_last_wait_timeout = false
	_wait_time = x
	wait_started.emit()


func wait_process_frames(x):
	_did_last_wait_timeout = false
	_wait_process_frames = x
	wait_started.emit()


func wait_physics_frames(x):
	_did_last_wait_timeout = false
	_wait_physics_frames = x
	wait_started.emit()


func wait_for_signal(the_signal, max_time):
	_did_last_wait_timeout = false
	the_signal.connect(_signal_callback)
	_signal_to_wait_on = the_signal
	_wait_time = max_time
	wait_started.emit()


func wait_until(predicate_function: Callable, max_time, time_between_calls:=0.0):
	_predicate_time_between = time_between_calls
	_predicate_method = predicate_function
	_wait_time = max_time

	_waiting_for_predicate_to_be = true
	_predicate_time_between_elpased = 0.0
	_did_last_wait_timeout = false

	wait_started.emit()


func wait_while(predicate_function: Callable, max_time, time_between_calls:=0.0):
	_predicate_time_between = time_between_calls
	_predicate_method = predicate_function
	_wait_time = max_time

	_waiting_for_predicate_to_be = false
	_predicate_time_between_elpased = 0.0
	_did_last_wait_timeout = false

	wait_started.emit()

func is_waiting():
	return _wait_time != 0.0 || _wait_physics_frames != 0
