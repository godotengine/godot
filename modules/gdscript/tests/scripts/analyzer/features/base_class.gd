const external_script = preload("gdscript_to_preload.notest.gd")

func virtual_method(_ext: external_script) -> String:
	return "Base"

func test():
	var external_instance = external_script.new()
	print("virtual_method call: " + virtual_method(external_instance))
