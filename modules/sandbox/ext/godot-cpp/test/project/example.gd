extends Example

func _do_something_virtual(p_name, p_value):
	custom_signal.emit(p_name, p_value)
	return "Implemented"
