# ##############################################################################
# This is a script I used to poke around script and method properties.  It's
# not supposed to be a "real" tool but it has a lot of examples in it and has
# come in handy numerous times.
#
# Feel free to use whatever you find in here for your own purposes.
# ##############################################################################
extends SceneTree

const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
var DoubleMe = load(DOUBLE_ME_PATH)
var DoubleMeScene = load('res://test/resources/doubler_test_objects/double_me_scene.tscn')
var SetGetTestNode = load('res://test/resources/test_assert_setget_test_objects/test_node.gd')
# const INNER_CLASSES_PATH = 'res://test/resources/doubler_test_objects/inner_classes.gd'
# var InnerClasses = load(INNER_CLASSES_PATH)

var json = JSON.new()
var _strutils = load('res://addons/gut/strutils.gd').new()

const DEFAULT_ARGS = 'default_args'
const NAME = 'name'
const ARGS = 'args'


var PROPERTY_USAGES = {
	'PROPERTY_USAGE_NONE' : PROPERTY_USAGE_NONE,
	'PROPERTY_USAGE_STORAGE' : PROPERTY_USAGE_STORAGE,
	'PROPERTY_USAGE_EDITOR' : PROPERTY_USAGE_EDITOR,
	'PROPERTY_USAGE_INTERNAL' : PROPERTY_USAGE_INTERNAL,
	'PROPERTY_USAGE_CHECKABLE' : PROPERTY_USAGE_CHECKABLE,
	'PROPERTY_USAGE_CHECKED' : PROPERTY_USAGE_CHECKED,
	'PROPERTY_USAGE_GROUP' : PROPERTY_USAGE_GROUP,
	'PROPERTY_USAGE_CATEGORY' : PROPERTY_USAGE_CATEGORY,
	'PROPERTY_USAGE_SUBGROUP' : PROPERTY_USAGE_SUBGROUP,
	'PROPERTY_USAGE_CLASS_IS_BITFIELD' : PROPERTY_USAGE_CLASS_IS_BITFIELD,
	'PROPERTY_USAGE_NO_INSTANCE_STATE' : PROPERTY_USAGE_NO_INSTANCE_STATE,
	'PROPERTY_USAGE_RESTART_IF_CHANGED' : PROPERTY_USAGE_RESTART_IF_CHANGED,
	'PROPERTY_USAGE_SCRIPT_VARIABLE' : PROPERTY_USAGE_SCRIPT_VARIABLE,
	'PROPERTY_USAGE_STORE_IF_NULL' : PROPERTY_USAGE_STORE_IF_NULL,
	'PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED' : PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED,
	'PROPERTY_USAGE_SCRIPT_DEFAULT_VALUE' : PROPERTY_USAGE_SCRIPT_DEFAULT_VALUE,
	'PROPERTY_USAGE_CLASS_IS_ENUM' : PROPERTY_USAGE_CLASS_IS_ENUM,
	'PROPERTY_USAGE_NIL_IS_VARIANT' : PROPERTY_USAGE_NIL_IS_VARIANT,
	'PROPERTY_USAGE_ARRAY' : PROPERTY_USAGE_ARRAY,
	'PROPERTY_USAGE_ALWAYS_DUPLICATE' : PROPERTY_USAGE_ALWAYS_DUPLICATE,
	'PROPERTY_USAGE_NEVER_DUPLICATE' : PROPERTY_USAGE_NEVER_DUPLICATE,
	'PROPERTY_USAGE_HIGH_END_GFX' : PROPERTY_USAGE_HIGH_END_GFX,
	'PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT' : PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT,
	'PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT' : PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT,
	'PROPERTY_USAGE_KEYING_INCREMENTS' : PROPERTY_USAGE_KEYING_INCREMENTS,
	'PROPERTY_USAGE_DEFERRED_SET_RESOURCE' : PROPERTY_USAGE_DEFERRED_SET_RESOURCE,
	'PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT' : PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT,
	'PROPERTY_USAGE_EDITOR_BASIC_SETTING' : PROPERTY_USAGE_EDITOR_BASIC_SETTING,
	'PROPERTY_USAGE_READ_ONLY' : PROPERTY_USAGE_READ_ONLY,
	'PROPERTY_USAGE_SECRET' : PROPERTY_USAGE_SECRET,
	'PROPERTY_USAGE_DEFAULT' : PROPERTY_USAGE_DEFAULT,
	'PROPERTY_USAGE_NO_EDITOR' : PROPERTY_USAGE_NO_EDITOR,
}

class HasSomeInners:
	signal look_at_me_now

	const WHATEVER = 'maaaaaan'

	class Inner1:
		extends 'res://addons/gut/test.gd'
		var a = 'b'

	class Inner2:
		var b = 'a'

		class Inner2_a:
			extends 'res://addons/gut/test.gd'

		class Inner2_b:
			var foo = 'bar'

	class Inner3:
		extends GutInternalTester

	class ExtendsInner1:
		extends Inner1

class ExtendsNode2D:
	extends Node2D

	static var static_foo := "static foo"

	static func a_static_func():
		return true

	var local_foo := "local foo"

	func my_function(some_string : String, defaulted_node : Node = null):
		return 7

	func get_position():
		return Vector2(0, 0)

	func _input(event):
		print(event)

class SimpleObject:
	var a = 'a'


class ExtendsAnInnerClassElsewhere:
	extends 'res://test/resources/doubler_test_objects/inner_classes.gd'.InnerB

	func foobar():
		return 'foobar'


func get_methods_by_flag(obj):
	var methods_by_flags = {}
	var methods = obj.get_method_list()

	for i in range(methods.size()):
		var flag = methods[i]['flags']
		var name = methods[i]['name']
		if(methods_by_flags.has(flag)):
			methods_by_flags[flag].append(name)
		else:
			methods_by_flags[flag] = [name]
	return methods_by_flags


func print_methods_by_flags(methods_by_flags):
	for flag in methods_by_flags:
		for i in range(methods_by_flags[flag].size()):
			print(flag, ":  ", methods_by_flags[flag][i])

	var total = 0
	for flag in methods_by_flags:
		if(methods_by_flags[flag].size() > 0):
			print("-- ", flag, " (", methods_by_flags[flag].size(), ") --")
			total += methods_by_flags[flag].size()
	print("Total:  ", total)


func print_methods_with_defaults(thing):
	print('------------------------------------------------------------------')
	print('--- Methods (object) ---')
	print('------------------------------------------------------------------')
	var methods = thing.get_method_list()
	for m in methods:
		if(m.default_args.size() > 0):
			print(m.name)
			pp(m, '  ')


	print('------------------------------------------------------------------')
	print('--- Methods (script) ---')
	print('------------------------------------------------------------------')
	methods  = thing.get_script_method_list()
	for m in methods:
		print(m.name)
		pp(m, '  ')


func print_methods_with_flags(obj, flags, print_all = true):
	var methods = obj.get_method_list()
	for i in range(methods.size()):
		var mflags = methods[i]['flags']
		if(is_flagged(mflags, flags)):
			print(methods[i]['name'], ':', mflags)
			print_method_flags(mflags, print_all)


func print_methods_named(obj, name):
	var methods = obj.get_method_list()
	for i in range(methods.size()):
		if(methods[i]['name'] == name):
			print(methods[i]['name'], ':')
			print_method_flags(methods[i]['flags'], false)


func subtract_dictionary(sub_this, from_this):
	var result = {}
	for key in sub_this:
		if(from_this.has(key)):
			result[key] = []

			for value in from_this[key]:
				var index = sub_this[key].find(value)
				if(index == -1):
					result[key].append(value)
	return result


func is_flagged(mask, index):
	return mask & index != 0


func print_flag(name, flags, flag, print_all=true):
	var is_set = is_flagged(flags, flag)
	if(print_all or is_set):
		print(name,'(', flag, ') = ', is_set)


func print_method_flags(flags, print_all=true):
	print_flag('  normal', flags, METHOD_FLAG_NORMAL, print_all)
	print_flag('  editor', flags, METHOD_FLAG_EDITOR, print_all)
	print_flag('  const', flags, METHOD_FLAG_CONST, print_all)
	print_flag('  virtual', flags, METHOD_FLAG_VIRTUAL, print_all)
	print_flag('  vararg', flags, METHOD_FLAG_VARARG, print_all)
	print_flag('  static', flags, METHOD_FLAG_STATIC, print_all)
	print_flag('  core', flags, METHOD_FLAG_OBJECT_CORE, print_all)
	print_flag('  default', flags, METHOD_FLAGS_DEFAULT, print_all)


func print_method_info(obj):
	var methods = obj.get_method_list()
	for i in range(methods.size()):
		print(methods[i]['name'], ' ', methods[i]['flags'])
		# if(methods[i]['default_args'].size() > 0):
		# 	print(" *** here be defaults ***")


		print_method_flags(methods[i]['flags'], false)
		# if(methods[i]['flags'] == 65):
		# 	for key in methods[i]:
		# 		if(key == 'args'):
		# 			print('  args:')
		# 			for argname in range(methods[i][key].size()):
		# 				print('    ',  methods[i][key][argname]['name'], ':  ', methods[i][key][argname])
		# 		else:
		# 			print('  ', key, ':  ', methods[i][key])


func get_defaults_and_types(method_meta):
	var text = ""
	text += method_meta[NAME] + "\n"
	for i in range(method_meta[DEFAULT_ARGS].size()):
		var arg_index = method_meta[ARGS].size() - (method_meta[DEFAULT_ARGS].size() - i)
		text += str('  ', method_meta[ARGS][arg_index][NAME])
		text += str('(type=', method_meta[ARGS][arg_index]['type'], ")")
		text += str(' = ', method_meta[DEFAULT_ARGS][i], "\n")
		# text += str('  ', method_meta[ARGS][arg_index]['usage'], "\n")
	return text


func class_db_stuff():
	print(ClassDB.class_exists('Node2D'))
	# print('category = ',  ClassDB.('Node2D'))
	# print(str(JSON.print(ClassDB.class_get_method_list('Node2D'), ' ')))
	# print(ClassDB.class_get_integer_constant_list('Node2D'))
	print(ClassDB.get_class_list())


func does_inherit_from_test(thing):
	var base_script = thing.get_base_script()
	var to_return = false
	print('  *base_script = ', base_script)
	if(base_script != null):
		var base_path = base_script.get_path()
		print('  *base_path = ', base_path)
		if(base_path == 'res://addons/gut/test.gd'):
			to_return = true
		else:
			to_return = does_inherit_from_test(base_script)
	return to_return


func print_other_info(loaded, msg = '', indent=''):
	print(indent, '--------------------- ', msg, ' ---------------------')
	print(indent, loaded)

	var base_script_path = 'NO base script'
	if(loaded.has_method('get_base_script')):
		if(loaded.get_base_script() != null):
			base_script_path = str('"', loaded.get_base_script().get_path(), '"')
		else:
			base_script_path = 'Null base script'

	print(indent, 'base_script path          ', base_script_path)
	print(indent, 'class                     ', loaded.get_class())
	print(indent, 'instance base type        ', loaded.get_instance_base_type())
	print(indent, 'instance_id               ', loaded.get_instance_id())
	print(indent, 'meta_list                 ', loaded.get_meta_list())
	print(indent, 'name                      ', loaded.get_name())
	print(indent, 'path                      ', loaded.get_path())
	print(indent, 'resource local to scene   ', loaded.resource_local_to_scene)
	print(indent, 'resource name             ', loaded.resource_name)
	print(indent, 'resource path             ', loaded.resource_path)
	print(indent, 'RID                       ', loaded.get_rid())
	print(indent, 'script                    ', loaded.get_script())
	print()
	print(loaded.get_script_property_list())
	print_properties(loaded.get_property_list(), loaded)

	var const_map = loaded.new().get_script().get_script_constant_map()
	# if(const_map.size() > 0):
	print(indent, '--- Constants ---')

	for key in const_map:
		var thing = const_map[key]
		print(indent, key, ' = ', thing)
		if(typeof(thing) == TYPE_OBJECT):
			print_other_info(thing, key, indent + '    ')
			# print('  ', 'meta         ', thing.get_meta_list())
			# print('  ', 'class        ', thing.get_class())
			# print('  ', 'path         ', thing.get_path())
			# var base_script = thing.get_base_script()
			# print('  ', 'base script  ', base_script)
			# if(base_script != null):
			# 	print('  ', 'base id      ', base_script.get_instance_id())
			# 	print('  ', 'base path    ', base_script.get_path() )
			# print('  ', 'base type    ', thing.get_instance_base_type())
			# print('  ', 'can instantiate ', thing.can_instantiate())
			# print('  ', 'id           ', thing.get_instance_id())
			# print('  ', 'is test      ', does_inherit_from_test(thing))



func print_inner_test_classes(loaded, from=null):
	# print('path = ', loaded.get_path())
	# if(loaded.get_base_script() != null):
	# 	print('base = ', loaded.get_base_script().get_path())
	# else:
	# 	print('base = ')

	var const_map = loaded.get_script_constant_map()
	for key in const_map:
		var thing = const_map[key]

		if(typeof(thing) == TYPE_OBJECT):
			print('Class ', key, ':')
			if(does_inherit_from_test(thing)):
				print('  is a test class')
			else:
				print('  noooooooooope')
			var next_from
			if(from == null):
				next_from = key
			else:
				next_from = str(from, '/', key)
			print_inner_test_classes(thing, next_from)

		else:
			print('CONST ', key, ' = ', thing)


func print_script_methods():
	var script = load('res://tests_4_0/test_print.gd')

	var methods = script.get_script_method_list()
	for i in range(methods.size()):
		print(methods[i]['name'])
		pp(methods[i])


func print_methods(methods, print_all_meta = false):
	var methods_by_name = {}
	var method_names = []
	for method in methods:
		methods_by_name[method.name] = method
		method_names.append(method.name)

	method_names.sort()

	for m_name in method_names:

		print(m_name)
		if(print_all_meta):
			pp(methods_by_name[m_name], '  ')


func print_properties(props, thing, print_all_meta=false):
	for i in range(props.size()):
		var prop_name = props[i].name
		var prop_value = thing.get(props[i].name)
		var print_value = str(prop_value)
		if(print_value.length() > 100):
			print_value = print_value.substr(0, 97) + '...'
		elif(print_value == ''):
			print_value = 'EMPTY'

		print(prop_name, ' = ', print_value)
		if(print_all_meta):
			print('  ', props[i])
			print_property_usage(props[i])


func print_property_usage(prop):
	for key in PROPERTY_USAGES:
		var flag = PROPERTY_USAGES[key]
		if(prop.usage & flag):
			print('- ', key, ' ', flag)

func print_script_info(thing):
	print('path = ', thing.get_path())

	print('--- Methods (script) ---')
	print_methods(thing.get_script_method_list(), true)

	print('--- Properties (script) ---')
	var props = thing.get_script_property_list()
	print_properties(props, thing, true)

	print('--- Constants ---')
	pp(thing.get_script_constant_map())

	print('--- Signals ---')
	var sigs = thing.get_signal_list()
	for sig in sigs:
		print(sig['name'])
		print('  ', sig)


func print_all_info(thing):
	print('path = ', thing.get_path())

	print('--- Methods (object) ---')
	print_methods(thing.get_method_list(), true)

	print('--- Methods (script) ---')
	print_methods(thing.get_script_method_list(), true)

	print('--- Properties (object) ---')
	var props = thing.get_property_list()
	print_properties(props, thing)

	print('--- Properties (script) ---')
	props = thing.get_script_property_list()
	print_properties(props, thing, true)

	print('--- Constants ---')
	print_inner_test_classes(thing)

	print('--- Signals ---')
	var sigs = thing.get_signal_list()
	for sig in sigs:
		print(sig['name'])
		print('  ', sig)




func print_class_db_class_list():
	var list = ClassDB.get_class_list()
	list.sort()
	print("\n".join(list))

func pp(dict, indent=''):
	var text = json.stringify(dict, '  ')
	text = _strutils.indent_text(text, 1, indent)
	print(text)

func has_script_method(Class):
	var methods = Class.get_script_method_list()

func print_inner_classes(loaded, parent=''):
	var const_map = loaded.get_script_constant_map()

	for key in const_map:
		var thing = const_map[key]

		if(typeof(thing) == TYPE_OBJECT):
			print(parent, key, ':  ', thing)
			print_inner_classes(thing, key + '.')

func print_inner_class_path(loaded):
	pass

func print_scene_info(scene):
	pp(scene._bundled)
	var state = scene.get_state()
	print('state = ', state)
	print('nodes = ', state.get_node_count())
	for i in range(state.get_node_count()):
		print(i, '. ', state.get_node_name(i))
		print(' type           ', state.get_node_type(i))
		print(' node index     ', state.get_node_index(i))
		print(' #props         ', state.get_node_property_count(i))
		print(' is placehodler ', state.is_node_instance_placeholder(i))
		print(' node path      ', state.get_node_path(i))
		print(' groups         ', state.get_node_groups(i))
		print(" properties:")
		for j in range(state.get_node_property_count(i)):
			var n = state.get_node_property_name(i, j)
			var v = state.get_node_property_value(i, j)
			print('       - ', n, ' = ', v)


func get_scene_script_object(scene):
	return GutUtils.get_scene_script_object(scene)


func _init():


	# var TestScene = load('res://test/resources/Issue436Scene.tscn')
	# print_scene_info(TestScene)
	# var inst = TestScene.instantiate()

	# print_all_info(inst.get_script())
	print_script_info(ExtendsNode2D)
	# print_methods_with_flags(ExtendsNode2D.new(), METHOD_FLAG_VARARG, false)
	# print_methods_with_flags(ExtendsNode2D.new(), METHOD_FLAG_OBJECT_CORE, false)
	# print_methods_named(ExtendsNode2D.new(), '_input')
	# print_methods_named(ExtendsNode2D.new(), '_notification')
	# print_methods_named(ExtendsNode2D.new(), '_to_string')
	# print_methods_named(ExtendsNode2D.new(), 'get_script')
	# print_methods_named(ExtendsNode2D.new(), 'has_method')
	# print_methods_named(ExtendsNode2D.new(), 'print_orphan_nodes')


	# pp(Button.new().get_method_list())
	# print_other_info(GutDoubleTestInnerClasses)
	# print("---------------\n\n\n\n-------------------")
	# print_other_info(GutDoubleTestInnerClasses.InnerA)
	# class_db_stuff()
	# print_all_info(DoubleMeScene)
	# var TestScene = load('res://test/resources/doubler_test_objects/double_me_scene.tscn')
	# print_scene_info(TestScene)
	# print(get_scene_script_object(TestScene))

	# var ThatInnerClassScript = load('res://test/resources/doubler_test_objects/inner_classes.gd')
	# print_other_info(HasSomeInners, 'HasSomeInners')
	# print_other_info(HasSomeInners.Inner2.Inner2_a, 'Inner2_a')
	# print_inner_classes(ThatInnerClassScript)
	# print_other_info(ThatInnerClassScript, 'ThatInnerClassScript')
	# print_other_info(ThatInnerClassScript.AnotherInnerA, 'AnotherInnerA')
	# print_all_info(ThatInnerClassScript.AnotherInnerA)
	# print_all_info(ExtendsAnInnerClassElsewhere)
	# print_all_info(DoubleMe)
	# print_class_db_class_list()

	# print_all_info(GutTest)
	# var WithDefaults = load('res://test/resources/doubler_test_objects/double_default_parameters.gd')
	# print_methods_with_defaults(WithDefaults)
	# print_other_info(GutTest)
	# print_all_info(SetGetTestNode)
	# print(SetGetTestNode.has_method('has_setter_setter'))
	# var inst = SetGetTestNode.new()
	# print(inst.has_method('has_setter_setter'))
	# print(inst.has_method('@has_setter_setter'))
	# print(inst.has_method('@has_getter_getter'))





	quit()
