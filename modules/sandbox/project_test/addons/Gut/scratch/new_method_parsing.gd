extends SceneTree


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class ParsedMethod:
	var _meta = {}
	var _parameters = []
	var is_local = false

	const NO_DEFAULT = '__no__default__'

	func _init(metadata):
		_meta = metadata
		var start_default = _meta.args.size() - _meta.default_args.size()
		for i in range(_meta.args.size()):
			var arg = _meta.args[i]
			# Add a "default" property to the metadata so we don't have to do
			# weird default position math again.
			if(i >= start_default):
				arg['default'] = _meta.default_args[start_default - i]
			else:
				arg['default'] = NO_DEFAULT
			_parameters.append(arg)

	func to_s():
		var s = _meta.name + "("

		for i in range(_meta.args.size()):
			var arg = _meta.args[i]
			if(str(arg.default) != NO_DEFAULT):
				var val = str(arg.default)
				if(val == ''):
					val = '""'
				s += str(arg.name, ' = ', val)
			else:
				s += str(arg.name)

			if(i != _meta.args.size() -1):
				s += ', '

		s += ")"
		return s


# ------------------------------------------------------------------------------
# Doesn't know if a method is local and in super, but not sure if that will
# ever matter.
# ------------------------------------------------------------------------------
class ParsedScript:

	# All methods indexed by name.
	var _methods_by_name = {}

	func _init(thing):

		print(thing.get_path())

		var methods = thing.get_method_list()
		for m in methods:
			var meth = ParsedMethod.new(m)
			_methods_by_name[m.name] = meth

		# This loop will overwrite all entries in _methods_by_name with the local
		# method object so there is only ever one listing for a function with
		# the right "is_local" flag.
		methods = thing.get_script_method_list()
		for m in methods:
			var meth = ParsedMethod.new(m)
			meth.is_local = true
			_methods_by_name[m.name] = meth

	func print_it():
		var names = _methods_by_name.keys()
		names.sort()
		for n in names:
			print(_methods_by_name[n].to_s())

	func print_super():
		var names = _methods_by_name.keys()
		names.sort()
		for n in names:
			if(!_methods_by_name[n].is_local):
				print(_methods_by_name[n].to_s())

	func print_local():
		var names = _methods_by_name.keys()
		names.sort()
		for n in names:
			if(_methods_by_name[n].is_local):
				print(_methods_by_name[n].to_s())

	func get_method(name):
		return _methods_by_name[name]

	func get_sorted_method_names():
		var keys = _methods_by_name.keys()
		keys.sort()
		return keys





# ------------------------------------------------------------------------------
# Issues
# * without typed parameters, I'm not sure you can know what type the defaults
# 	are.
# * When the default is set to a class variable, you get null.  So this might
#	mean we can never fully know what the default values are.  This is true for
#	class and instance metadata.
# * Appears you can get all info about a thing without having to make an
#	instance.
# ------------------------------------------------------------------------------
const DOUBLE_ME_PATH = 'res://test/resources/doubler_test_objects/double_me.gd'
var DoubleMe = load(DOUBLE_ME_PATH)
var json = JSON.new()

func pp(dict):
	print(json.stringify(dict, ' '))

func _init():
	var Thing = Node2D
	print(Thing)
	print(Thing.new().get_class())
	var id_str = str(Thing).replace("<", '').replace(">", '').split('#')[1]
	print(id_str)

	var by_id = instance_from_id(id_str.to_int())
	print(by_id)




	# var dbl_inst = DoubleMe.new()
	# print(dbl_inst.get_script().get_path())

	# # print(DoubleMe.get_method_list())
	# print(DoubleMe.get_script_method_list())
	# print('************************************************')
	# print(DoubleMe.get_method_list())


	# var ps = ParsedScript.new(DoubleMe)
	# print('************************************************')
	# ps.print_super()
	# print('************************************************')
	# ps.print_local()

	# var m_assert_eq = ps.get_method('default_is_value')
	# print(m_assert_eq.to_s())
	# pp(m_assert_eq._meta)

	quit()