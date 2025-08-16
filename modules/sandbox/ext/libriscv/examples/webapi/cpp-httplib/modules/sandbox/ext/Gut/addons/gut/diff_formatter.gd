var _strutils = GutUtils.Strutils.new()
const INDENT = '    '
var _max_to_display = 30
const ABSOLUTE_MAX_DISPLAYED = 10000
const UNLIMITED = -1


func _single_diff(diff, depth=0):
	var to_return = ""
	var brackets = diff.get_brackets()

	if(brackets != null and !diff.are_equal):
		to_return = ''
		to_return += str(brackets.open, "\n",
			_strutils.indent_text(differences_to_s(diff.differences, depth), depth+1, INDENT), "\n",
			brackets.close)
	else:
		to_return = str(diff)

	return to_return


func make_it(diff):
	var to_return = ''
	if(diff.are_equal):
		to_return = diff.summary
	else:
		if(_max_to_display ==  ABSOLUTE_MAX_DISPLAYED):
			to_return = str(diff.get_value_1(), ' != ', diff.get_value_2())
		else:
			to_return = diff.get_short_summary()
		to_return +=  str("\n", _strutils.indent_text(_single_diff(diff, 0), 1, '  '))
	return to_return


func differences_to_s(differences, depth=0):
	var to_return = ''
	var keys = differences.keys()
	keys.sort()
	var limit = min(_max_to_display, differences.size())

	for i in range(limit):
		var key = keys[i]
		to_return += str(key, ":  ", _single_diff(differences[key], depth))

		if(i != limit -1):
			to_return += "\n"

	if(differences.size() > _max_to_display):
		to_return += str("\n\n... ", differences.size() - _max_to_display, " more.")

	return to_return


func get_max_to_display():
	return _max_to_display


func set_max_to_display(max_to_display):
	_max_to_display = max_to_display
	if(_max_to_display == UNLIMITED):
		_max_to_display = ABSOLUTE_MAX_DISPLAYED

