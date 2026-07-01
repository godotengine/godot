const Node2D = preload("override_native_class_with_constant.notest.gd")
var member_test: Node2D = Node2D.new()

func test() -> void:
	member_test.do_something("member");

	const Node3D = preload("override_native_class_with_constant.notest.gd")
	var local_test: Node3D = Node3D.new()

	local_test.do_something("local");
