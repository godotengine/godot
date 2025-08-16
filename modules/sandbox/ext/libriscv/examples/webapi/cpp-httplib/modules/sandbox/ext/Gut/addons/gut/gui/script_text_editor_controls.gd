# Holds weakrefs to a ScriptTextEditor and related children nodes
# that might be useful.  Though the CodeEdit is really the only one, but
# since the tree may change, the first TextEdit under a CodeTextEditor is
# the one we use...so we hold a ref to the CodeTextEditor too.
class ScriptEditorControlRef:
	var _text_edit = null
	var _script_editor = null
	var _code_editor = null

	func _init(script_edit):
		_script_editor = weakref(script_edit)
		_populate_controls()
		# print("_script_editor = ", script_edit, ' vis = ', is_visible())


	func _populate_controls():
		# who knows if the tree will change so get the first instance of each
		# type of control we care about.  Chances are there won't be more than
		# one of these in the future, but their position in the tree may change.
		_code_editor = weakref(_get_first_child_named('CodeTextEditor', _script_editor.get_ref()))
		_text_edit = weakref(_get_first_child_named("CodeEdit", _code_editor.get_ref()))


	func _get_first_child_named(obj_name, parent_obj):
		if(parent_obj == null):
			return null

		var kids = parent_obj.get_children()
		var index = 0
		var to_return = null

		while(index < kids.size() and to_return == null):
			if(str(kids[index]).find(str("<", obj_name)) != -1):
				to_return = kids[index]
			else:
				to_return = _get_first_child_named(obj_name, kids[index])
				if(to_return == null):
					index += 1

		return to_return


	func get_script_text_edit():
		return _script_editor.get_ref()


	func get_text_edit():
		# ScriptTextEditors that are loaded when the project is opened
		# do not have their children populated yet.  So if we may have to
		# _populate_controls again if we don't have one.
		if(_text_edit == null):
			_populate_controls()
		return _text_edit.get_ref()


	func get_script_editor():
		return _script_editor


	func is_visible():
		var to_return = false
		if(_script_editor.get_ref()):
			to_return = _script_editor.get_ref().visible
		return to_return

# ##############################################################################
#
# ##############################################################################

# Used to make searching for the controls easier and faster.
var _script_editors_parent = null
# reference the ScriptEditor instance
var _script_editor = null
# Array of ScriptEditorControlRef containing all the opened ScriptTextEditors
# and related controls at the time of the last refresh.
var _script_editor_controls = []

var _method_prefix = 'test_'
var _inner_class_prefix = 'Test'

func _init(script_edit):
	_script_editor = script_edit
	refresh()


func _is_script_editor(obj):
	return str(obj).find('<ScriptTextEditor') != -1


# Find the first ScriptTextEditor and then get its parent.  Done this way
# because who knows if the parent object will change.  This is somewhat
# future proofed.
func _find_script_editors_parent():
	var _first_editor = _get_first_child_of_type_name("ScriptTextEditor", _script_editor)
	if(_first_editor != null):
		_script_editors_parent = _first_editor.get_parent()


func _populate_editors():
	if(_script_editors_parent == null):
		return

	_script_editor_controls.clear()
	for child in _script_editors_parent.get_children():
		if(_is_script_editor(child)):
			var ref = ScriptEditorControlRef.new(child)
			_script_editor_controls.append(ref)

# Yes, this is the same as the one above but with a different name.  This was
# easier than trying to find a place where it could be used by both.
func _get_first_child_of_type_name(obj_name, parent_obj):
	if(parent_obj == null):
		# print('aborting search for ', obj_name, ' parent is null')
		return null

	var kids = parent_obj.get_children()
	var index = 0
	var to_return = null

	var search_for = str("<", obj_name)
	# print('searching for ', search_for, ' in ', parent_obj, ' kids ', kids.size())
	while(index < kids.size() and to_return == null):
		var this_one = str(kids[index])
		# print(search_for, ' :: ', this_one)
		if(this_one.find(search_for) != -1):
			to_return = kids[index]
		else:
			to_return = _get_first_child_of_type_name(obj_name, kids[index])
			if(to_return == null):
				index += 1

	return to_return


func _get_func_name_from_line(text):
	text = text.strip_edges()
	var left = text.split("(")[0]
	var func_name = left.split(" ")[1]
	return func_name


func _get_class_name_from_line(text):
	text = text.strip_edges()
	var right = text.split(" ")[1]
	var the_name = right.rstrip(":")
	return the_name

func refresh():
	if(_script_editors_parent == null):
		_find_script_editors_parent()
		# print("script editors parent = ", _script_editors_parent)

	if(_script_editors_parent != null):
		_populate_editors()

	# print("script editor controls = ", _script_editor_controls)


func get_current_text_edit():
	var cur_script_editor = null
	var idx = 0

	while(idx < _script_editor_controls.size() and cur_script_editor == null):
		if(_script_editor_controls[idx].is_visible()):
			cur_script_editor = _script_editor_controls[idx]
		else:
			idx += 1

	var to_return = null
	if(cur_script_editor != null):
		to_return = cur_script_editor.get_text_edit()

	return to_return


func get_script_editor_controls():
	var to_return = []
	for ctrl_ref in _script_editor_controls:
		to_return.append(ctrl_ref.get_script_text_edit())

	return to_return


func get_line_info():
	var editor = get_current_text_edit()
	if(editor == null):
		return

	var info = {
		script = null,
		inner_class = null,
		test_method = null
	}

	var line = editor.get_caret_line()
	var done_func = false
	var done_inner = false
	while(line > 0 and (!done_func or !done_inner)):
		if(editor.can_fold_line(line)):
			var text = editor.get_line(line)
			var strip_text = text.strip_edges(true, false) # only left

			if(!done_func and strip_text.begins_with("func ")):
				var func_name = _get_func_name_from_line(text)
				if(func_name.begins_with(_method_prefix)):
					info.test_method = func_name
				done_func = true
				# If the func line is left justified then there won't be any
				# inner classes above it.
				if(strip_text == text):
					done_inner = true

			if(!done_inner and strip_text.begins_with("class")):
				var inner_name = _get_class_name_from_line(text)
				if(inner_name.begins_with(_inner_class_prefix)):
					info.inner_class = inner_name
					done_inner = true
					# if we found an inner class then we are already past
					# any test the cursor could be in.
					done_func = true
		line -= 1

	return info


func get_method_prefix():
	return _method_prefix


func set_method_prefix(method_prefix):
	_method_prefix = method_prefix


func get_inner_class_prefix():
	return _inner_class_prefix


func set_inner_class_prefix(inner_class_prefix):
	_inner_class_prefix = inner_class_prefix
