extends "res://addons/gut/test.gd"

# This test file should only be detected when specifying a 'specific_prefix' and 'suffix.gd'.

func test_empty():
    assert_true(true, 'All is well.')
