# TODO: This test is currently disabled since it triggers some complex memory leaks. Try enabling it again once GH-101830 is fixed.

signal finished

const scr: GDScript = preload("reload_suspended_function_helper.notest.gd")

func test():
	@warning_ignore("UNSAFE_METHOD_ACCESS")
	scr.test(self)
	@warning_ignore("RETURN_VALUE_DISCARDED")
	scr.reload(true)
	finished.emit()
