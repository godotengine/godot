extends RefCounted

const AbstractScript = preload("./construct_abstract_script.notest.gd")

@abstract class AbstractClass:
	pass

func test():
	var _a := AbstractScript.new()
	var _b := AbstractClass.new()
