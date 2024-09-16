extends Resource

var foo = preload("./foo.tres")
var foo_bar_null = preload("./foo.tres")


func test():
	# Sunny day
	var obj:Object = foo
	var dict:Dictionary = inst_to_dict(obj)
	var dict_ok = true
	dict_ok = dict_ok && dict.get("number") == 42
	dict_ok = dict_ok && dict.get("bar") is Resource
	if not dict_ok:
		printerr("Can't convert foo instance to dictionary properly")

	dict = inst_to_dict(obj, true)
	print(dict.keys())
	print(dict.values())

	var inst = dict_to_inst(dict, true)
	var equals = true
	equals = equals && foo.number == inst.number
	equals = equals && foo.bar.text == inst.bar.text
	equals = equals && foo.bar.qux.decimal == inst.bar.qux.decimal
	if not equals:
		printerr("Can't revert from foo instance to dictionary properly")

	# null in inner object
	foo_bar_null.bar = null
	obj = foo_bar_null
	dict = inst_to_dict(obj)
	dict_ok = true
	dict_ok = dict_ok && dict.get("number") == 42
	dict_ok = dict_ok && dict.get("bar") == null
	if not dict_ok:
		printerr("Can't convert foo_bar_null instance to dictionary properly")

	dict = inst_to_dict(obj, true)
	print(dict.keys())
	print(dict.values())

	inst = dict_to_inst(dict, true)
	equals = true
	equals = equals && foo.number == inst.number
	equals = equals && foo.bar == null
	if not equals:
		printerr("Can't revert from foo_bar_null instance to dictionary properly")

	var should_be_null = inst_to_dict(null)
	if should_be_null != null:
		printerr("It should return null")

	print('ok')
