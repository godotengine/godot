extends GutTest

var Sandbox_TestsTests = load("res://tests/tests.elf")

func test_shared_memory():
	var s = Sandbox.new()
	s.set_program(Sandbox_TestsTests)

	assert(s.has_function("test_shm"))

	# Create a packed array of floats
	var array = PackedFloat32Array([1.0, 2.0, 3.0, 4.0, 5.0])

	# Share the array with the node read+write
	var vaddr = s.share_float32_array(true, array)
	var result = s.vmcall("test_shm", vaddr, array.size())

	# The function should double the values in the array
	assert_eq_deep(result, PackedFloat32Array([2.0, 4.0, 6.0, 8.0, 10.0]))

	# After unsharing, the original array should also reflect the changes
	assert_true(s.unshare_array(vaddr), "Unsharing the array should succeed")
	assert_eq_deep(array, PackedFloat32Array([2.0, 4.0, 6.0, 8.0, 10.0]))

	s.queue_free()

func test_sharing_sandboxes():
	var s1 = Sandbox.new()
	s1.set_program(Sandbox_TestsTests)
	s1.restrictions = true # Enable full sandboxing
	var s2 = Sandbox.new()
	s2.set_program(Sandbox_TestsTests)
	s2.restrictions = true # Enable full sandboxing
	assert_true(s1.has_function("test_shm2"), "Sandbox 1 should have the test_shm2 function")
	assert_true(s1.has_function("verify_shm2"), "Sandbox 1 should have the verify_shm2 function")

	# Create a packed array of 100k floats
	var array = PackedFloat32Array()
	array.resize(100000)

	# Perform the test multiple times to ensure stability
	for i in range(10):
		# Share the array with the both sandboxes read+write
		var vaddr1 = s1.share_float32_array(true, array)
		var vaddr2 = s2.share_float32_array(true, array)

		# Call the test function in the first sandbox
		# The function should double the values in the array
		var result = s1.vmcall("test_shm2", vaddr1, array.size())
		# The function should already have modified the first 5 elements of the array
		# because the array is larger than a single page, and all elements within
		# those fully shared pages should be accessible without unsharing.
		for j in range(5):
			assert_eq(array[j], 2.0 * (j + 1), "First 5 elements should be doubled")
		# Call the verification function in the second sandbox
		assert_true(s2.vmcall("verify_shm2", vaddr2, array.size()), "Verification should succeed")

		# After unsharing, the original array should also reflect the changes
		assert_true(s1.unshare_array(vaddr1), "Unsharing the array should succeed")
		assert_true(s2.unshare_array(vaddr2), "Unsharing the array should succeed")

	s1.queue_free()
	s2.queue_free()

func test_sharing_sandboxes_after_mutation():
	var s1 = Sandbox.new()
	s1.set_program(Sandbox_TestsTests)
	s1.restrictions = true # Enable full sandboxing
	var s2 = Sandbox.new()
	s2.set_program(Sandbox_TestsTests)
	s2.restrictions = true # Enable full sandboxing
	assert_true(s1.has_function("test_shm2"), "Sandbox 1 should have the test_shm2 function")
	assert_true(s1.has_function("verify_shm2"), "Sandbox 1 should have the verify_shm2 function")

	# Create a packed array of 100k floats
	var array = PackedFloat32Array()
	array.resize(100000)

	# Pre-share the array with both sandboxes read+write
	var vaddr1 = s1.share_float32_array(true, array)
	var vaddr2 = s2.share_float32_array(true, array)

	# Duplicate and destroy the original array to find dragons
	var array2 = array.duplicate()
	array = null

	await Engine.get_main_loop().process_frame

	# Perform the test multiple times to ensure stability
	for i in range(10):
		# Call the test function in the first sandbox
		var result = s1.vmcall("test_shm2", vaddr1, array2.size())
		# Call the verification function in the second sandbox
		assert_true(s2.vmcall("verify_shm2", vaddr2, array2.size()), "Verification should succeed")

	# Unshare the arrays after the tests
	assert_true(s1.unshare_array(vaddr1), "Unsharing the array should succeed")
	assert_true(s2.unshare_array(vaddr2), "Unsharing the array should succeed")

	s1.queue_free()
	s2.queue_free()
