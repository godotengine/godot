class ThingA:
	pass

class ThingB extends ThingA:
	pass

class Sub extends TypedDictionarySubclassKeyBase:
	pass

class MySpecialNode extends Node2D:
	pass

# Verifies that a dictionary accepts a subclass (ThingB) as a key.
func testBasic() -> void:
	var dict: Dictionary[ThingA, String] = {}
	var a := ThingA.new()
	var b := ThingB.new()
	dict[a] = "yes"
	dict[b] = "no"
	print(dict[a])
	print(dict[b])
	print("testBasic: ok")

# Verifies that a dictionary accepts a subclass of another file.
func testMultiFile() -> void:
	var dict: Dictionary[TypedDictionarySubclassKeyBase, String] = {}
	var sub := Sub.new()
	dict[sub] = "subclass instance"
	var as_base: TypedDictionarySubclassKeyBase = Sub.new()
	dict[as_base] = "base-typed variable"
	print(dict[sub])
	print(dict[as_base])
	print(dict.size())
	print("testMultiFile: ok")

# Validates subclass key behavior with native (builtin) types.
func testNative() -> void:
	var dict: Dictionary[Node, String] = {}
	var n2d = Node2D.new()
	var n3d = Node3D.new()

	dict[n2d] = "node2d inst"
	dict[n3d] = "node3d inst"
	print(dict[n2d])
	print(dict[n3d])

	n2d.free()
	n3d.free()
	print("testNative: ok")

# Validates subclass key behavior between native base types
# and script extension types
func testNativeMixed() -> void:
	var dict: Dictionary[Node, String] = {}
	var special_node = MySpecialNode.new()

	dict[special_node] = "special node"
	print(dict[special_node])
	special_node.free()
	print("testNativeMixed: ok")

func test() -> void:
	testBasic()
	testMultiFile()
	testNative()
	testNativeMixed()
