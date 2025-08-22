extends GutTest
# ------------------------------------------------------------------------------
# This test script is used for spot checking orphans created by GUT.  If you
# run GUT with just this script, and you get engine warnings about orphans or
# resources still in use, then GUT has a problem.
# ------------------------------------------------------------------------------
func test_true():
    assert_true(true)

func test_false():
    assert_false(false)

func test_eq_n():
    assert_eq(1, 1)

func test_eq_s():
    assert_eq('a', 'a')

func test_with_double_of_script():
    var d = double(load('res://test/resources/doubler_test_objects/double_me.gd')).new()
    assert_not_null(d)

func test_with_double_of_scene():
    var d = double(load('res://test/resources/doubler_test_objects/double_me_scene.tscn')).instantiate()
    assert_not_null(d)

func test_with_script_partial_double_of_script():
    var d = partial_double(load('res://test/resources/doubler_test_objects/double_me.gd')).new()
    assert_not_null(d)

func test_with_partial_double_of_scene():
    var d = partial_double(load('res://test/resources/doubler_test_objects/double_me_scene.tscn')).instantiate()
    assert_not_null(d)

func test_with_spy_on_double_of_script():
    var d = double(load('res://test/resources/doubler_test_objects/double_me.gd')).new()
    d.get_value()
    assert_called(d, 'get_value')

func test_with_spy_on_partial_double_of_scene():
    var d = partial_double(load('res://test/resources/doubler_test_objects/double_me_scene.tscn')).instantiate()
    d.return_hello()
    assert_called(d, 'return_hello')
