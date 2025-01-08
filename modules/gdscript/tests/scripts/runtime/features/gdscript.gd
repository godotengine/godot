func test():
	var gdscr: = GDScript.new()
	gdscr.source_code = '''
extends Resource

func test() -> void:
	prints("Outer")
	var inner = InnerClass.new()

class InnerClass:
	func _init() -> void:
		prints("Inner")
'''
	@warning_ignore("return_value_discarded")
	gdscr.reload()

	var inst = gdscr.new()

	@warning_ignore("unsafe_method_access")
	inst.test()
