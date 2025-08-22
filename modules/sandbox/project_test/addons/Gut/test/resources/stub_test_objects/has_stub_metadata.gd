# ------------------------------------------------------------------------------
# this class is used by test_stubber and represents a doubled object
# which is why we have __gutdbl in here.
# ------------------------------------------------------------------------------
var __gutdbl_values = {
	double = self,
	thepath = 'res://test/resources/stub_test_objects/to_stub.gd',
	subpath = '',
	stubber = -1,
	spy = -1,
	gut = -1,
	from_singleton = '',
	is_partial = false,
}
var __gutdbl = load('res://addons/gut/double_tools.gd').new(__gutdbl_values)

# Here so other things can check for a method to know if this is a double.
func __gutdbl_check_method__():
	pass