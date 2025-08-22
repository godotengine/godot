extends GutTest

var Sandbox_TestsTests = load("res://tests/tests.elf")

func test_restrictions():
	var s = Sandbox.new()
	s.set_program(Sandbox_TestsTests)

	# Adding an allowed object changes the Sandbox from unrestricted to restricted
	# access_a_parent
	assert_eq(s.has_function("access_a_parent"), true)
	# Create a new node
	var n = Node.new()
	n.name = "Node"
	# Set the sandbox as the parent of the node, so it can be accessed
	s.add_child(n)
	# Add an allowed object
	s.add_allowed_object(n)
	# Now restrictions are in place
	var exceptions = s.get_exceptions()
	s.vmcall("access_a_parent", n)
	# The function should have thrown an exception, as we didn't allow the parent object
	assert_eq(s.get_exceptions(), exceptions + 1)

	# Allow the parent object
	s.add_allowed_object(n.get_parent())
	# Now restrictions are in place
	exceptions = s.get_exceptions()
	s.vmcall("access_a_parent", n)
	# The function should *NOT* have thrown an exception, as we allowed the parent object
	assert_eq(s.get_exceptions(), exceptions)

	# Allow the parent object using a callback
	s.remove_allowed_object(n.get_parent())
	s.set_object_allowed_callback(func(sandbox, obj): return obj == n.get_parent())
	# Now restrictions are in place
	exceptions = s.get_exceptions()
	s.vmcall("access_a_parent", n)
	# The function should *NOT* have thrown an exception, as we allowed the parent object
	assert_eq(s.get_exceptions(), exceptions)

	# Setting a callback for allowed classes changes ClassDB instantiation from unrestricted to restricted
	# creates_a_node
	assert_eq(s.has_function("creates_a_node"), true)
	# Add an allowed class (Node)
	s.set_class_allowed_callback(func(sandbox, name): return name == "Node")
	# Now restrictions are in place
	assert_true(s.is_allowed_class("Node"), "Node should be allowed")
	exceptions = s.get_exceptions()
	s.vmcall("creates_a_node")
	# The function should *NOT* have thrown an exception, as we allowed the Node class
	assert_eq(s.get_exceptions(), exceptions)

	# Now only allow the class Node2D
	s.set_class_allowed_callback(func(sandbox, name): return name == "Node2D")
	# Now restrictions are in place
	assert_true(s.is_allowed_class("Node2D"), "Node2D should be allowed")
	assert_false(s.is_allowed_class("Node"), "Node should not be allowed")
	exceptions = s.get_exceptions()
	s.vmcall("creates_a_node")
	# The function should have thrown an exception, as we only allowed the Node2D class
	assert_eq(s.get_exceptions(), exceptions + 1)

	# Disable all restrictions
	s.restrictions = false
	# Now restrictions are disabled
	exceptions = s.get_exceptions()
	s.vmcall("creates_a_node")
	# The function should *NOT* have thrown an exception, as we disabled all restrictions

	# Enable restrictions (by adding dummy values to allowed_classes and allowed_objects)
	s.restrictions = true
	# Now restrictions are enabled
	exceptions = s.get_exceptions()
	s.vmcall("creates_a_node")
	# The function should have thrown an exception, as we enabled restrictions
	assert_eq(s.get_exceptions(), exceptions + 1, "Should have thrown an exception")

	s.queue_free()

func test_restriction_callbacks():
	var s = Sandbox.new()
	s.set_program(Sandbox_TestsTests)

	s.set_object_allowed_callback(func(sandbox, obj): return obj.get_name() == "Test")
	var n = Node.new()
	n.name = "Test"
	assert_true(s.is_allowed_object(n), "Test node should be allowed")

	s.set_method_allowed_callback(func(sandbox, obj, method): return method != "free")
	assert_true(s.is_allowed_method(n, "queue_free"), "Node.queue_free() should be allowed")
	assert_false(s.is_allowed_method(n, "free"), "Node.free() should *NOT* be allowed")

	s.set_property_allowed_callback(func(sandbox, obj, property, is_set): return property != "owner")
	assert_true(s.is_allowed_property(n, "name"), "Node.get/set_name should be allowed")
	assert_false(s.is_allowed_property(n, "owner"), "Node.get/set_owner should *NOT* be allowed")

	s.set_class_allowed_callback(func(sandbox, name): return name == "Node")
	assert_true(s.is_allowed_class("Node"), "Node creation should be allowed")
	assert_false(s.is_allowed_class("Node2D"), "Node2D creation should *NOT* be allowed")

	s.set_resource_allowed_callback(func(sandbox, name): return name == "res://test.tscn")
	assert_true(s.is_allowed_resource("res://test.tscn"), "Resource should be allowed")
	assert_false(s.is_allowed_resource("res://other.tscn"), "Resource should *NOT* be allowed")

	s.queue_free()


func test_insanity():
	var s = Sandbox.new()
	s.set_program(Sandbox_TestsTests)

	assert_true(s.has_function("access_an_invalid_child_node"), "access_an_invalid_child_node should exist")

	s.restrictions = true
	s.set_class_allowed_callback(func(sandbox, name): return name == "Node")
	assert_true(s.is_allowed_class("Node"), "Node should be allowed")

	#s.set_object_allowed_callback(func(sandbox, obj): return obj.get_name() == "Node")
	#s.set_method_allowed_callback(func(sandbox, obj, method): return method == "get_name")

	var exceptions = s.get_exceptions()
	s.vmcall("access_an_invalid_child_node")

	assert_eq(s.get_exceptions(), exceptions + 1)


	# access_an_invalid_child_resource
	assert_true(s.has_function("access_an_invalid_child_resource"), "access_an_invalid_child_resource should exist")
	# allow instantiate method
	s.set_method_allowed_callback(func(sandbox, obj, method): return method == "instantiate")

	# allow a resource that can be loaded and instantiated
	s.set_resource_allowed_callback(func(sandbox, name): return name == "res://tests/test.elf")
	assert_true(s.is_allowed_resource("res://tests/test.elf"), "Resource should be allowed")

	exceptions = s.get_exceptions()
	var inst = s.vmcall("access_an_invalid_child_resource", "res://tests/test.elf")
	# The function should *NOT* have thrown an exception, as we allowed the resource
	assert_eq(s.get_exceptions(), exceptions)

	s.vmcall("access_an_invalid_child_resource", "res://other.tscn")
	# The function should have thrown an exception, as we didn't allow the resource
	assert_eq(s.get_exceptions(), exceptions + 1)

	if inst is Node:
		inst.queue_free()

	# disable_restrictions
	assert_true(s.has_function("disable_restrictions"), "disable_restrictions should exist")

	s.restrictions = true
	s.vmcall("disable_restrictions")
	# The function should have denied disabling restrictions, as it is forbidden
	# to disable restrictions from within the sandbox
	assert_eq(s.restrictions, true)

	s.queue_free()
