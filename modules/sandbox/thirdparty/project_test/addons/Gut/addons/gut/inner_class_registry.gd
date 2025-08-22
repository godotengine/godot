var _registry = {}


func _create_reg_entry(base_path, subpath):
	var to_return = {
		"base_path":base_path,
		"subpath":subpath,
		"base_resource":load(base_path),
		"full_path":str("'", base_path, "'", subpath)
	}
	return to_return

func _register_inners(base_path, obj, prev_inner = ''):
	var const_map = obj.get_script_constant_map()
	var consts = const_map.keys()
	var const_idx = 0

	while(const_idx < consts.size()):
		var key = consts[const_idx]
		var thing = const_map[key]

		if(typeof(thing) == TYPE_OBJECT):
			var cur_inner = str(prev_inner, ".", key)
			_registry[thing] = _create_reg_entry(base_path, cur_inner)
			_register_inners(base_path, thing, cur_inner)

		const_idx += 1


func register(base_script):
	var base_path = base_script.resource_path
	_register_inners(base_path, base_script)


func get_extends_path(inner_class):
	if(_registry.has(inner_class)):
		return _registry[inner_class].full_path
	else:
		return null

# returns the subpath for the inner class.  This includes the leading "." in
# the path.
func get_subpath(inner_class):
	if(_registry.has(inner_class)):
		return _registry[inner_class].subpath
	else:
		return ''

func get_base_path(inner_class):
	if(_registry.has(inner_class)):
		return _registry[inner_class].base_path


func has(inner_class):
	return _registry.has(inner_class)


func get_base_resource(inner_class):
	if(_registry.has(inner_class)):
		return _registry[inner_class].base_resource


func to_s():
	var text = ""
	for key in _registry:
		text += str(key, ": ", _registry[key], "\n")
	return text
