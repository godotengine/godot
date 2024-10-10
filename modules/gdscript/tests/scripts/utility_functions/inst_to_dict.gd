extends Resource

var foo = preload("./foo.tres")


func test():
	var obj:Object = foo
	var dict:Dictionary = inst_to_dict(obj)
	var dict_ok = true
	dict_ok = dict_ok && dict.get("number") == 42
	dict_ok = dict_ok && dict.get("bar") is Resource
	if not dict_ok:
		printerr("Can't convert instance to dictionary properly")

	dict = inst_to_dict(obj, true)
	print(dict.keys())
	print(dict.values())

	var inst = dict_to_inst(dict, true)
	var equals = true
	equals = equals && foo.number == inst.number
	equals = equals && foo.bar.text == inst.bar.text
	equals = equals && foo.bar.qux.decimal == inst.bar.qux.decimal
	if not equals:
		printerr("Can't revert from instance to dictionary properly")

	print('ok')
