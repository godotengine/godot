extends GutTest

class DoubleThis:
	extends Node2D

	func get_something():
		return 'something'

var DoubleMe = load('res://test/resources/doubler_test_objects/double_me.gd')

func before_all():
	pass
	# gut.get_doubler()._print_source = true


func test_assert_true():
	assert_true(true, 'this is true')


# func test_making_double_of_string():
# 	var d = double('res://test/resources/doubler_test_objects/double_me.gd')
# 	assert_not_null(d)
# 	var d_inst = d.new()
# 	assert_not_null(d_inst)


# func test_making_partial_double_of_string():
# 	var d = partial_double('res://test/resources/doubler_test_objects/double_me.gd')
# 	assert_not_null(d)
# 	var d_inst = d.new()
# 	assert_not_null(d_inst)

# func  test_make_double_of_loaded_thing():
# 	var Dbl = double(DoubleMe)
# 	assert_not_null(Dbl)
# 	var d_inst = Dbl.new()
# 	assert_not_null(d_inst)

# func  test_make_partial_double_of_loaded_thing():
# 	var Dbl = partial_double(DoubleMe)
# 	assert_not_null(Dbl)
# 	var d_inst = Dbl.new()
# 	assert_not_null(d_inst)
# 	assert_not_null(d_inst.__gut_metadata_.path)

# func test_can_stub_a_method_making_double_of_string():
# 	var d = double('res://test/resources/doubler_test_objects/double_me.gd')
# 	assert_not_null(d)
# 	var d_inst = d.new()
# 	assert_not_null(d_inst)
# 	stub(d_inst, 'get_value').to_return('poop')
# 	assert_eq(d_inst.get_value(), 'poop')

# func test_can_watch_signals():
# 	var dbl_me = DoubleMe.new()
# 	watch_signals(dbl_me)
# 	dbl_me.emit_signal('signal_signal')
# 	assert_signal_emitted(dbl_me, 'signal_signal')

