extends "res://addons/gut/test.gd"

var counts = {
	before_all = 0,
	before_each = 0,
	after_all = 0,
	after_each = 0,

	prerun_setup = 0,
	setup = 0,
	postrun_teardown = 0,
	teardown = 0
}

func setup():
	counts.setup += 1

func teardown():
	counts.teardown += 1

func prerun_setup():
	counts.prerun_setup += 1

func postrun_teardown():
	counts.postrun_teardown += 1

func before_all():
	counts.before_all += 1

func before_each():
	counts.before_each += 1

func after_all():
	counts.after_all += 1

func after_each():
	counts.after_each += 1


func test_sample_one():
	assert_true(true)

func test_sample_two():
	assert_true(true)

func test_sample_three():
	assert_true(true)
