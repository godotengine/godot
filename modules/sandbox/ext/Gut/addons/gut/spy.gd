# {
#   instance_id_or_path1:{
#       method1:[ [p1, p2], [p1, p2] ],
#       method2:[ [p1, p2], [p1, p2] ]
#   },
#   instance_id_or_path1:{
#       method1:[ [p1, p2], [p1, p2] ],
#       method2:[ [p1, p2], [p1, p2] ]
#   },
# }
var _calls = {}
var _lgr = GutUtils.get_logger()
var _compare = GutUtils.Comparator.new()

func _find_parameters(call_params, params_to_find):
	var found = false
	var idx = 0
	while(idx < call_params.size() and !found):
		var result = _compare.deep(call_params[idx], params_to_find)
		if(result.are_equal):
			found = true
		else:
			idx += 1
	return found


func _get_params_as_string(params):
	var to_return = ''
	if(params == null):
		return ''

	for i in range(params.size()):
		if(params[i] == null):
			to_return += 'null'
		else:
			if(typeof(params[i]) == TYPE_STRING):
				to_return += str('"', params[i], '"')
			else:
				to_return += str(params[i])
		if(i != params.size() -1):
			to_return += ', '
	return to_return


func add_call(variant, method_name, parameters=null):
	if(!_calls.has(variant)):
		_calls[variant] = {}

	if(!_calls[variant].has(method_name)):
		_calls[variant][method_name] = []

	_calls[variant][method_name].append(parameters)


func was_called(variant, method_name, parameters=null):
	var to_return = false
	if(_calls.has(variant) and _calls[variant].has(method_name)):
		if(parameters):
			to_return = _find_parameters(_calls[variant][method_name], parameters)
		else:
			to_return = true
	return to_return


func get_call_parameters(variant, method_name, index=-1):
	var to_return = null
	var get_index = -1

	if(_calls.has(variant) and _calls[variant].has(method_name)):
		var call_size = _calls[variant][method_name].size()
		if(index == -1):
			# get the most recent call by default
			get_index =  call_size -1
		else:
			get_index = index

		if(get_index < call_size):
			to_return = _calls[variant][method_name][get_index]
		else:
			_lgr.error(str('Specified index ', index, ' is outside range of the number of registered calls:  ', call_size))

	return to_return


func call_count(instance, method_name, parameters=null):
	var to_return = 0

	if(was_called(instance, method_name)):
		if(parameters):
			for i in range(_calls[instance][method_name].size()):
				if(_calls[instance][method_name][i] == parameters):
					to_return += 1
		else:
			to_return = _calls[instance][method_name].size()
	return to_return


func clear():
	_calls = {}


func get_call_list_as_string(instance):
	var to_return = ''
	if(_calls.has(instance)):
		for method in _calls[instance]:
			for i in range(_calls[instance][method].size()):
				to_return += str(method, '(', _get_params_as_string(_calls[instance][method][i]), ")\n")
	return to_return


func get_logger():
	return _lgr


func set_logger(logger):
	_lgr = logger
