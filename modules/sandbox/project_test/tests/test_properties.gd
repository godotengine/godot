extends GutTest

var Sandbox_TestsTests = load("res://tests/tests.elf")

func validate_property(prop_list, name):
	for p in prop_list:
		if p.name == name:
			return true
	return false

func test_script_properties():
	var n = Node.new()
	n.name = "Node"
	n.set_script(Sandbox_TestsTests)

	# Verify properties exist
	var prop_list = n.get_property_list()
	assert_true(validate_property(prop_list, "player_speed"), "Property player_speed found")
	assert_true(validate_property(prop_list, "player_jump_vel"), "Property player_jump_vel found")
	assert_true(validate_property(prop_list, "player_name"), "Property player_name found")

	# Test property access
	assert_eq(n.get("player_speed"), 150.0, "Property player_speed has value 150")
	assert_eq(n.get("player_jump_vel"), -300.0, "Property player_jump_vel has value -300")
	assert_eq(n.get("player_name"), "Slide Knight", "Property player_name has value Slide Knight")

	# Test property set
	n.set("player_speed", 200.0)
	n.set("player_jump_vel", -400.0)
	n.set("player_name", "Jump Knight")

	assert_eq(n.get("player_speed"), 200.0, "Property player_speed has value 200")
	assert_eq(n.get("player_jump_vel"), -400.0, "Property player_jump_vel has value -400")
	assert_eq(n.get("player_name"), "Jump Knight", "Property player_name has value Jump Knight")

	n.queue_free()

func test_elfscript_properties():
	var n = Sandbox.new()
	n.set_program(Sandbox_TestsTests)

	# Verify properties exist
	var prop_list = n.get_property_list()
	assert_true(validate_property(prop_list, "player_speed"), "Property player_speed found")
	assert_true(validate_property(prop_list, "player_jump_vel"), "Property player_jump_vel found")
	assert_true(validate_property(prop_list, "player_name"), "Property player_name found")

	# Test property access
	assert_eq(n.get("player_speed"), 150.0, "Property player_speed has value 150")
	assert_eq(n.get("player_jump_vel"), -300.0, "Property player_jump_vel has value -300")
	assert_eq(n.get("player_name"), "Slide Knight", "Property player_name has value Slide Knight")

	# Test property set
	n.set("player_speed", 200.0)
	n.set("player_jump_vel", -400.0)
	n.set("player_name", "Jump Knight")

	assert_eq(n.get("player_speed"), 200.0, "Property player_speed has value 200")
	assert_eq(n.get("player_jump_vel"), -400.0, "Property player_jump_vel has value -400")
	assert_eq(n.get("player_name"), "Jump Knight", "Property player_name has value Jump Knight")

	n.queue_free()
