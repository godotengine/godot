extends RefCounted


var _base_script_text = GutUtils.get_file_as_text('res://addons/gut/double_templates/script_template.txt')
var _script_collector = GutUtils.ScriptCollector.new()
# used by tests for debugging purposes.
var print_source = false
var inner_class_registry = GutUtils.InnerClassRegistry.new()

# ###############
# Properties
# ###############
var _stubber = GutUtils.Stubber.new()
func get_stubber():
	return _stubber
func set_stubber(stubber):
	_stubber = stubber

var _lgr = GutUtils.get_logger()
func get_logger():
	return _lgr
func set_logger(logger):
	_lgr = logger
	_method_maker.set_logger(logger)

var _spy = null
func get_spy():
	return _spy
func set_spy(spy):
	_spy = spy

var _gut = null
func get_gut():
	return _gut
func set_gut(gut):
	_gut = gut

var _strategy = null
func get_strategy():
	return _strategy
func set_strategy(strategy):
	if(GutUtils.DOUBLE_STRATEGY.values().has(strategy)):
		_strategy = strategy
	else:
		_lgr.error(str('doubler.gd:  invalid double strategy ', strategy))


var _method_maker = GutUtils.MethodMaker.new()
func get_method_maker():
	return _method_maker

var _ignored_methods = GutUtils.OneToMany.new()
func get_ignored_methods():
	return _ignored_methods

# ###############
# Private
# ###############
func _init(strategy=GutUtils.DOUBLE_STRATEGY.SCRIPT_ONLY):
	set_logger(GutUtils.get_logger())
	_strategy = strategy


func _get_indented_line(indents, text):
	var to_return = ''
	for _i in range(indents):
		to_return += "\t"
	return str(to_return, text, "\n")


func _stub_to_call_super(parsed, method_name):
	if(!parsed.get_method(method_name).is_eligible_for_doubling()):
		return

	var params = GutUtils.StubParams.new(parsed.script_path, method_name, parsed.subpath)
	params.to_call_super()
	_stubber.add_stub(params)


func _get_base_script_text(parsed, override_path, partial, included_methods):
	var path = parsed.script_path
	if(override_path != null):
		path = override_path

	var stubber_id = -1
	if(_stubber != null):
		stubber_id = _stubber.get_instance_id()

	var spy_id = -1
	if(_spy != null):
		spy_id = _spy.get_instance_id()

	var gut_id = -1
	if(_gut != null):
		gut_id = _gut.get_instance_id()

	var extends_text  = parsed.get_extends_text()

	var values = {
		# Top  sections
		"extends":extends_text,
		"constants":'',#obj_info.get_constants_text(),
		"properties":'',#obj_info.get_properties_text(),

		# metadata values
		"path":path,
		"subpath":GutUtils.nvl(parsed.subpath, ''),
		"stubber_id":stubber_id,
		"spy_id":spy_id,
		"gut_id":gut_id,
		"singleton_name":'',#GutUtils.nvl(obj_info.get_singleton_name(), ''),
		"is_partial":partial,
		"doubled_methods":included_methods,
	}

	return _base_script_text.format(values)


func _is_method_eligible_for_doubling(parsed_script, parsed_method):
	return !parsed_method.is_accessor() and \
		parsed_method.is_eligible_for_doubling() and \
		!_ignored_methods.has(parsed_script.resource, parsed_method.meta.name)


# Disable the native_method_override setting so that doubles do not generate
# errors or warnings when doubling with INCLUDE_NATIVE or when a method has
# been added because of param_count stub.
func _create_script_no_warnings(src):
	var prev_native_override_value = null
	var native_method_override = 'debug/gdscript/warnings/native_method_override'
	prev_native_override_value = ProjectSettings.get_setting(native_method_override)
	ProjectSettings.set_setting(native_method_override, 0)

	var DblClass = GutUtils.create_script_from_source(src)

	ProjectSettings.set_setting(native_method_override, prev_native_override_value)
	return DblClass


func _create_double(parsed, strategy, override_path, partial):
	var path = ""

	path = parsed.script_path
	var dbl_src = ""
	var included_methods = []

	for method in parsed.get_local_methods():
		if(_is_method_eligible_for_doubling(parsed, method)):
			included_methods.append(method.meta.name)
			var mthd = parsed.get_local_method(method.meta.name)
			if(parsed.is_native):
				dbl_src += _get_func_text(method.meta, parsed.resource)
			else:
				dbl_src += _get_func_text(method.meta, path)

	if(strategy == GutUtils.DOUBLE_STRATEGY.INCLUDE_NATIVE):
		for method in parsed.get_super_methods():
			if(_is_method_eligible_for_doubling(parsed, method)):
				included_methods.append(method.meta.name)
				_stub_to_call_super(parsed, method.meta.name)
				if(parsed.is_native):
					dbl_src += _get_func_text(method.meta, parsed.resource)
				else:
					dbl_src += _get_func_text(method.meta, path)

	var base_script = _get_base_script_text(parsed, override_path, partial, included_methods)
	dbl_src = base_script + "\n\n" + dbl_src


	if(print_source):
		print(GutUtils.add_line_numbers(dbl_src))

	var DblClass = _create_script_no_warnings(dbl_src)
	if(_stubber != null):
		_stub_method_default_values(DblClass, parsed, strategy)

	return DblClass


func _stub_method_default_values(which, parsed, strategy):
	for method in parsed.get_local_methods():
		if(method.is_eligible_for_doubling() and !_ignored_methods.has(parsed.resource, method.meta.name)):
			_stubber.stub_defaults_from_meta(parsed.script_path, method.meta)


func _double_scene_and_script(scene, strategy, partial):
	var dbl_bundle = scene._bundled.duplicate(true)
	var script_obj = GutUtils.get_scene_script_object(scene)
	# I'm not sure if the script object for the root node of a packed scene is
	# always the first entry in "variants" so this tries to find it.
	var script_index = dbl_bundle["variants"].find(script_obj)
	var script_dbl = null

	if(script_obj != null):
		if(partial):
			script_dbl = _partial_double(script_obj, strategy, scene.get_path())
		else:
			script_dbl = _double(script_obj, strategy, scene.get_path())

	if(script_index != -1):
		dbl_bundle["variants"][script_index] = script_dbl

	var doubled_scene = PackedScene.new()
	doubled_scene._set_bundled_scene(dbl_bundle)

	return doubled_scene


func _get_inst_id_ref_str(inst):
	var ref_str = 'null'
	if(inst):
		ref_str = str('instance_from_id(', inst.get_instance_id(),')')
	return ref_str


func _get_func_text(method_hash, path):
	var override_count = null;
	if(_stubber != null):
		override_count = _stubber.get_parameter_count(path, method_hash.name)

	var text = _method_maker.get_function_text(method_hash, override_count) + "\n"

	return text


func _parse_script(obj):
	var parsed = null

	if(GutUtils.is_inner_class(obj)):
		if(inner_class_registry.has(obj)):
			parsed = _script_collector.parse(inner_class_registry.get_base_resource(obj), obj)
		else:
			_lgr.error('Doubling Inner Classes requires you register them first.  Call register_inner_classes passing the script that contains the inner class.')
	else:
		parsed = _script_collector.parse(obj)

	return parsed


# Override path is used with scenes.
func _double(obj, strategy, override_path=null):
	var parsed = _parse_script(obj)
	if(parsed != null):
		return _create_double(parsed, strategy, override_path, false)


func _partial_double(obj, strategy, override_path=null):
	var parsed = _parse_script(obj)
	if(parsed != null):
		return _create_double(parsed, strategy, override_path, true)


# -------------------------
# Public
# -------------------------

# double a script/object
func double(obj, strategy=_strategy):
	return _double(obj, strategy)

func partial_double(obj, strategy=_strategy):
	return _partial_double(obj, strategy)


# double a scene
func double_scene(scene, strategy=_strategy):
	return _double_scene_and_script(scene, strategy, false)


func partial_double_scene(scene, strategy=_strategy):
	return _double_scene_and_script(scene, strategy, true)


func double_gdnative(which):
	return _double(which, GutUtils.DOUBLE_STRATEGY.INCLUDE_NATIVE)


func partial_double_gdnative(which):
	return _partial_double(which, GutUtils.DOUBLE_STRATEGY.INCLUDE_NATIVE)


func double_inner(parent, inner, strategy=_strategy):
	var parsed = _script_collector.parse(parent, inner)
	return _create_double(parsed, strategy, null, false)


func partial_double_inner(parent, inner, strategy=_strategy):
	var parsed = _script_collector.parse(parent, inner)
	return _create_double(parsed, strategy, null, true)


func add_ignored_method(obj, method_name):
	_ignored_methods.add(obj, method_name)



# ##############################################################################
#(G)odot (U)nit (T)est class
#
# ##############################################################################
# The MIT License (MIT)
# =====================
#
# Copyright (c) 2025 Tom "Butch" Wesley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##############################################################################
