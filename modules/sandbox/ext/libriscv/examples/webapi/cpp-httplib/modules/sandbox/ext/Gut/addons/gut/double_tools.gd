var thepath = ''
var subpath = ''
var stubber = null
var spy = null
var gut = null
var from_singleton = null
var is_partial = null
var double = null

const NO_DEFAULT_VALUE = '!__gut__no__default__value__!'
func _init(values=null):
	if(values != null):
		double = values.double
		thepath = values.thepath
		subpath = values.subpath
		stubber = from_id(values.stubber)
		spy = from_id(values.spy)
		gut = from_id(values.gut)
		from_singleton = values.from_singleton
		is_partial = values.is_partial

	if(gut != null):
		gut.get_autofree().add_free(double)


func _get_stubbed_method_to_call(method_name, called_with):
	var method = stubber.get_call_this(double, method_name, called_with)
	if(method != null):
		method = method.bindv(called_with)
		return method
	return method


func from_id(inst_id):
	if(inst_id ==  -1):
		return null
	else:
		return instance_from_id(inst_id)


func is_stubbed_to_call_super(method_name, called_with):
	if(stubber != null):
		return stubber.should_call_super(double, method_name, called_with)
	else:
		return false


func handle_other_stubs(method_name, called_with):
	if(stubber == null):
		return

	var method = _get_stubbed_method_to_call(method_name, called_with)
	if(method != null):
		return await method.call()
	else:
		return stubber.get_return(double, method_name, called_with)


func spy_on(method_name, called_with):
	if(spy != null):
		spy.add_call(double, method_name, called_with)


func default_val(method_name, p_index, default_val=NO_DEFAULT_VALUE):
	if(stubber != null):
		return stubber.get_default_value(double, method_name, p_index)
	else:
		return null


func vararg_warning():
	if(gut != null):
		gut.get_logger().warn(
			"This method contains a vararg argument and the paramter count was not stubbed.  " + \
			"GUT adds extra parameters to this method which should fill most needs.  " + \
			"It is recommended that you stub param_count for this object's class to ensure " + \
			"that there are not any parameter count mismatch errors.")
