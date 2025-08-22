extends GutTest

var ResultTree = load('res://addons/gut/gui/ResultsTree.tscn')

func make_json_top():
	return {
		"test_scripts": {
		"props": {
			"errors": 0,
			"failures": 0,
			"orphans": 0,
			"passing": 0,
			"pending": 0,
			"tests": 0,
			"time": 0,
			"warnings": 0
		},
		"scripts": {}
		}
	}

func add_script_to_json(script_name, add_to):
	var script_json = {
		"props": {
			"failures": 0,
			"pending": 0,
			"tests": 0
		},
		"tests": {}
	}
	add_to.test_scripts.scripts[script_name] = script_json
	return script_json

func add_test_to_json(test_name, add_to):
	var test_json ={
		"failing": [],
		"orphans": 0,
		"passing": [],
		"pending": [],
		"status": "pass"
	}
	add_to.tests["test_can_spy_on_built_ins_when_doing_a_full_double"] = test_json
	return test_json


func test_assert_can_create_one():
	var rt = autofree(ResultTree.instantiate())
	assert_not_null(rt);

func test_has_show_orphans_property():
	var rt = autofree(ResultTree.instantiate())
	assert_property(rt, 'show_orphans', true, false)

func test_has_hide_passing_property():
	var rt = autofree(ResultTree.instantiate())
	assert_property(rt, 'hide_passing', true, false)


func test_load_a_single_script():
	var rt = add_child_autofree(ResultTree.instantiate())
	var j = make_json_top()
	var s1 = add_script_to_json("res://some_script.gd", j)
	var t1 = add_test_to_json("test_foo", s1)
	rt.load_json_results(j)
	pass_test('made it here')
