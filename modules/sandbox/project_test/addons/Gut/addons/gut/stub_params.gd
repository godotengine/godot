
var _lgr = GutUtils.get_logger()
var logger = _lgr :
	get: return _lgr
	set(val): _lgr = val

var return_val = null
var stub_target = null
# the parameter values to match method call on.
var parameters = null
var stub_method = null
var call_super = false
var call_this = null
# Whether this is a stub for default parameter values as they are defined in
# the script, and not an overridden default value.
var is_script_default = false

# -- Paramter Override --
# Parmater overrides are stored in here along with all the other stub info
# so that you can chain stubbing parameter overrides along with all the
# other stubbing.  This adds some complexity to the logic that tries to
# find the correct stub for a call by a double.  Since an instance of this
# class could be just a parameter override, or it could have been chained
# we have to have _paramter_override_only so that we know when to tell the
# difference.
var parameter_count = -1
var parameter_defaults = null
# Anything that would make this stub not just an override of paramters
# must set this flag to false.  This must be private bc the actual logic
# to determine if this stub is only an override is more complicated.
var _parameter_override_only = true
# --

const NOT_SET = '|_1_this_is_not_set_1_|'

func _init(target=null, method=null, _subpath=null):
	stub_target = target
	stub_method = method

	if(typeof(target) == TYPE_CALLABLE):
		stub_target = target.get_object()
		stub_method = target.get_method()
		parameters = target.get_bound_arguments()
		if(parameters.size() == 0):
			parameters = null
	elif(typeof(target) == TYPE_STRING):
		if(target.is_absolute_path()):
			stub_target = load(str(target))
		else:
			_lgr.warn(str(target, ' is not a valid path'))

	if(stub_target is PackedScene):
		stub_target = GutUtils.get_scene_script_object(stub_target)

	# this is used internally to stub default parameters for everything that is
	# doubled...or something.  Look for stub_defaults_from_meta for usage.  This
	# behavior is not to be used by end users.
	if(typeof(method) == TYPE_DICTIONARY):
		_load_defaults_from_metadata(method)


func _load_defaults_from_metadata(meta):
	stub_method = meta.name
	var values = meta.default_args.duplicate()
	while (values.size() < meta.args.size()):
		values.push_front(null)

	param_defaults(values)


func to_return(val):
	if(stub_method == '_init'):
		_lgr.error("You cannot stub _init to do nothing.  Super's _init is always called.")
	else:
		return_val = val
		call_super = false
		_parameter_override_only = false
	return self


func to_do_nothing():
	to_return(null)
	return self


func to_call_super():
	call_super = true
	_parameter_override_only = false
	return self


func to_call(callable : Callable):
	call_this = callable
	return self


func when_passed(p1=NOT_SET,p2=NOT_SET,p3=NOT_SET,p4=NOT_SET,p5=NOT_SET,p6=NOT_SET,p7=NOT_SET,p8=NOT_SET,p9=NOT_SET,p10=NOT_SET):
	parameters = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]
	var idx = 0
	while(idx < parameters.size()):
		if(str(parameters[idx]) == NOT_SET):
			parameters.remove_at(idx)
		else:
			idx += 1
	return self


func param_count(x):
	parameter_count = x
	return self


func param_defaults(values):
	parameter_count = values.size()
	parameter_defaults = values
	return self


func has_param_override():
	return parameter_count != -1


func is_param_override_only():
	var ret_val = false
	if(has_param_override()):
		ret_val = _parameter_override_only
	return ret_val


func to_s():
	var base_string = str(stub_target, '.', stub_method)

	if(has_param_override()):
		base_string += str(' (param count override=', parameter_count, ' defaults=', parameter_defaults)
		if(is_param_override_only()):
			base_string += " ONLY"
		if(is_script_default):
			base_string += " script default"
		base_string += ') '

	if(call_super):
		base_string += " to call SUPER"

	if(call_this != null):
		base_string += str(" to call ", call_this)

	if(parameters != null):
		base_string += str(' with params (', parameters, ') returns ', return_val)
	else:
		base_string += str(' returns ', return_val)

	return base_string
