# ------------------------------------------------------------------------------
# A sample script for illustrating multiple scripts and what it looks like
# when all tests pass.
# ------------------------------------------------------------------------------
extends "res://addons/gut/test.gd"

func test_auto_load_works():
	var g = get_node("/root/global")
	g.print_loaded()
	pass_test("should get here")
