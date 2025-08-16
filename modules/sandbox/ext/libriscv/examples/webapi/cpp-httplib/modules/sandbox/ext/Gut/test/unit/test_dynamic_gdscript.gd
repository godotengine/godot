extends GutTest

var DynmaicGdScript = load("res://addons/gut/dynamic_gdscript.gd")

var _dyn_gd = null
var _created_instance_count = 0

func before_each():
	_dyn_gd = DynmaicGdScript.new()
	_dyn_gd.default_script_name_no_extension = 'test_create_script_from_source'
	# makes sure we don't repeat resources paths in tests
	_dyn_gd.default_script_resource_path = str('res://addons/tests/', _created_instance_count, '/')
	_created_instance_count += 1


func test_can_create_a_script_from_source():
	var DynScript = _dyn_gd.create_script_from_source('var a = 1')
	assert_not_null(DynScript)


func test_can_create_instance_of_script_from_source():
	var DynScript = _dyn_gd.create_script_from_source('var a = 1')
	var i = DynScript.new()
	assert_eq(i.a, 1)


func test_resource_path_is_in_addons_directory():
	var DynScript = _dyn_gd.create_script_from_source('var a = 1')
	var i = DynScript.new()
	assert_string_starts_with(DynScript.resource_path, 'res://addons/')


func test_each_script_gets_a_unique_resource_path():
	var DynScript1 = _dyn_gd.create_script_from_source('var a = 1')
	var DynScript2 = _dyn_gd.create_script_from_source('var a = 1')
	var DynScript3 = _dyn_gd.create_script_from_source('var a = 1')

	assert_ne(DynScript1.resource_path, DynScript2.resource_path, '1 - 2')
	assert_ne(DynScript2.resource_path, DynScript3.resource_path, '2 - 3')
	assert_ne(DynScript1.resource_path, DynScript3.resource_path, '1 - 3')


func test_when_override_path_specified_it_is_used_for_resource_path():
	var DynScript = _dyn_gd.create_script_from_source('var a = 1', 'res://foo/bar.gd')
	assert_eq(DynScript.resource_path, 'res://foo/bar.gd')


func test_when_script_source_invalid_the_error_code_is_returned():
	if(EngineDebugger.is_active()):
		pending("Test skipped, disable debug to run test")
		return

	var DynScript = _dyn_gd.create_script_from_source("asdf\n\n\nasdfasfd\n\nasdf")
	assert_eq(typeof(DynScript), TYPE_INT)
