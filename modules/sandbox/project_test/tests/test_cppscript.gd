extends GutTest

var Sandbox_TestsTests = load("res://tests/tests.elf")
var cpp = load("res://tests/tests.cpp")

func test_set_script():
	var n = Node.new()
	n.set_script(cpp)
	# Create an ELFScript instance
	var nn = Sandbox.new()
	nn.set_script(Sandbox_TestsTests)
	# Sanity check that we can use the ELFScript
	assert_true(nn.execution_timeout > 0, "Can use property execution_timeout from ELFScript")
	assert_true(nn.is_allowed_object(n), "Can use is_allowed_object function from Node with ELFScript")
	# Attach n under nn
	nn.add_child(n)

	# Verify that we can call methods from the ELF program
	n.associated_script = nn
	assert_eq(n.test_int(1234), 1234, "Can call test_int function directly")
	assert_eq(n.get_tree_base_parent(), nn, "Verify node hierarchy is correct")

	# Verify that we also can call Sandbox-related functions through the Node
	assert_true(n.execution_timeout > 0, "Can use property execution_timeout from Node")
	assert_true(n.is_allowed_object(n), "Can use is_allowed_object function from Node")

	# Cleanup
	nn.queue_free()
	n.queue_free()

func test_associated_script():
	var n = Node.new()
	n.set_script(cpp)
	# Create an ELFScript instance
	var nn = Sandbox.new()
	nn.set_script(Sandbox_TestsTests)
	# Attach n under nn
	nn.add_child(n)

	n.associated_script = nn
	assert_eq(n.associated_script, nn, "Verify associated_script is set correctly")
	assert_eq(n.get_associated_script(), nn, "Verify get_associated_script returns the correct script")
	n.set_associated_script(null)
	n.set_associated_script(nn)
	assert_eq(n.associated_script, nn, "Verify associated_script can be set again")

	# Verify that we can call methods from the ELF program
	assert_eq(n.test_int(1234), 1234, "Can call test_int function directly")
	assert_eq(n.get_tree_base_parent(), nn, "Verify node hierarchy is correct")

	# Verify that we also can call Sandbox-related functions through the Node
	assert_true(n.execution_timeout > 0, "Can use property execution_timeout from Node")
	assert_true(n.is_allowed_object(n), "Can use is_allowed_object function from Node")

	# Cleanup
	nn.queue_free()
	n.queue_free()

func test_associated_elf_resource():
	var n = Node.new()
	n.set_script(cpp)
	# Attach n under nn
	var nn = Node.new()
	nn.add_child(n)

	assert_eq(n.associated_script, Sandbox_TestsTests, "Verify associated_script is set correctly")
	assert_eq(n.get_associated_script(), Sandbox_TestsTests, "Verify get_associated_script returns the correct script")

	# Verify that we can call methods from the ELF program
	assert_eq(n.test_int(1234), 1234, "Can call test_int function directly")
	assert_eq(n.get_tree_base_parent(), nn, "Verify node hierarchy is correct")
	for i in n.get_method_list():
		if i.name == "test_int":
			#print(i)
			assert_eq(n.test_int(1234), 1234, "Can call test_int function directly")
			assert_eq(n.call("test_int", 1234), 1234, "Can call test_int function through call()")
			break

	# Verify that we also can call Sandbox-related functions through the Node
	assert_true(n.execution_timeout > 0, "Can use property execution_timeout from Node")
	assert_true(n.is_allowed_object(n), "Can use is_allowed_object function from Node")
	assert_true(n.monitor_calls_made > 0, "We have made some calls in the Sandbox")

	# Cleanup
	nn.queue_free()
	n.queue_free()

func test_associated_elf_resource_on_sandbox():
	var n = Sandbox.new()
	n.set_script(cpp)
	assert_eq(n.associated_script, Sandbox_TestsTests, "Sandbox_TestsTests is already set as associated_script")
	# Attach n under nn
	var nn = Node.new()
	nn.add_child(n)

	# Verify that we can call methods from the ELF program
	assert_eq(n.test_int(1234), 1234, "Can call test_int function directly")
	assert_eq(n.get_tree_base_parent(), nn, "Verify node hierarchy is correct")
	for i in n.get_method_list():
		if i.name == "test_int":
			assert_eq(n.test_int(1234), 1234, "Can call test_int function directly")
			assert_eq(n.call("test_int", 1234), 1234, "Can call test_int function through call()")
			break

	# Verify that we also can call Sandbox-related functions through the Node
	assert_true(n.execution_timeout > 0, "Can use property execution_timeout from Node")
	assert_true(n.is_allowed_object(n), "Can use is_allowed_object function from Node")
	assert_true(n.monitor_calls_made > 0, "We have made some calls in the Sandbox")

	# Cleanup
	nn.queue_free()
	n.queue_free()
