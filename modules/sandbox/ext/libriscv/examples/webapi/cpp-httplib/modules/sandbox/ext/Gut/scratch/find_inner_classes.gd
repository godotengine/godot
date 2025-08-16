extends SceneTree

var ThatInnerClassScript = load('res://test/resources/doubler_test_objects/inner_classes.gd')


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


func get_extends_text(inner, parent_script):
	if(parent_script.get_path() == ''):
		return null

	var to_return = null
	var inner_string = get_inner_class_string(inner, parent_script)
	if(inner_string != null):
		to_return = str("extends '", parent_script.get_path(), "'.", inner_string)

	return to_return


func get_inner_class_string(inner, parent_script):

	var const_map = parent_script.get_script_constant_map()
	var consts = const_map.keys()
	var const_idx = 0
	var found = false
	var to_return = null

	while(const_idx < consts.size() and !found):
		var key = consts[const_idx]
		var thing = const_map[key]

		if(typeof(thing) == TYPE_OBJECT):
			if(thing == inner):
				found = true
				to_return = key
			else:
				to_return = get_inner_class_string(inner, thing)
				if(to_return != null):
					to_return = str(key, '.', to_return)
					found = true

		const_idx += 1

	return to_return


func print_other_info(loaded, msg = '', indent=''):
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





func find_parent_script(InnerClass):
	var max_search = 20
	var start_id = InnerClass.get_instance_id()
	var search_id = start_id + 1
	var found = false

	while(search_id < start_id + max_search and !found):
		print(search_id)
		var search_obj = instance_from_id(search_id)
		print(search_obj)
		if(search_obj != null):
			print(search_obj)
		search_id += 1


func _init():
	print(GutDoubleTestInnerClasses)
	print(GutDoubleTestInnerClasses.InnerA)
	print(ThatInnerClassScript.get_instance_id() + ThatInnerClassScript.InnerA.get_instance_id())
	find_parent_script(GutDoubleTestInnerClasses.InnerA)
	quit()

	# var result = get_inner_class_string(HasSomeInners.Inner2.Inner2_b, self.get_script())
	# print(result)

	# print()
	# result = get_inner_class_string(ThatInnerClassScript.InnerWithSignals, ThatInnerClassScript)
	# print(result)

	# print(get_extends_text(HasSomeInners.Inner2.Inner2_b, self.get_script()))

	# print(get_extends_text(HasSomeInners.Inner2.Inner2_b, HasSomeInners))

	# print(get_extends_text(ThatInnerClassScript.InnerWithSignals, ThatInnerClassScript))

	# quit()