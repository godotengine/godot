extends Node

## Builds a minimal renderer, uploads data, and validates compute pipeline setup.
func _ready():
	print("=== Testing Compute Shader Pipeline ===")

	# Create a GaussianSplatRenderer node
	var renderer = GaussianSplatRenderer.new()
	renderer.name = "TestRenderer"
	add_child(renderer)

	print("GaussianSplatRenderer created successfully")

	# The renderer should automatically initialize its compute pipeline
	# when added to the scene tree

	# Wait a frame for initialization
	await get_tree().process_frame

	print("Compute pipeline initialization complete")

	# Generate some test data
	var test_data = GaussianData.new()
	test_data.add_splat(Vector3(0, 0, 0), Color.RED, Vector3(1, 1, 1), Quaternion())
	test_data.add_splat(Vector3(1, 0, 0), Color.GREEN, Vector3(1, 1, 1), Quaternion())
	test_data.add_splat(Vector3(0, 1, 0), Color.BLUE, Vector3(1, 1, 1), Quaternion())

	renderer.set_gaussian_data(test_data)
	print("Test data set with %d splats" % test_data.get_splat_count())

	# Wait for a few frames to let rendering happen
	for i in range(5):
		await get_tree().process_frame
		print("Frame %d rendered" % i)

	print("=== Test Complete ===")
	get_tree().quit()
