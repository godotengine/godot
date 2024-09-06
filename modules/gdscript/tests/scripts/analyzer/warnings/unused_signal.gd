# Doesn't produce the warning:
signal used_as_first_class_signal()
signal used_with_signal_constructor()
signal used_with_signal_emit()
signal used_with_object_emit_signal()
signal used_with_object_connect()
signal used_with_object_disconnect()
signal used_with_self_prefix()

# Produce the warning:
signal used_with_dynamic_name()
signal just_unused()
@warning_ignore("unused_signal")
signal unused_but_ignored()

func no_exec():
	print(used_as_first_class_signal)
	print(Signal(self, "used_with_signal_constructor"))
	used_with_signal_emit.emit()
	print(emit_signal("used_with_object_emit_signal"))
	print(connect("used_with_object_connect", Callable()))
	disconnect("used_with_object_disconnect", Callable())
	print(self.emit_signal("used_with_self_prefix"))

	var dynamic_name := "used_with_dynamic_name"
	print(emit_signal(dynamic_name))

func test():
	pass
