extends 'res://addons/gut/compare_result.gd'
const INDENT = '    '
enum {
	DEEP,
	SIMPLE
}

var _strutils = GutUtils.Strutils.new()
var _compare = GutUtils.Comparator.new()
var DiffTool = load('res://addons/gut/diff_tool.gd')

var _value_1 = null
var _value_2 = null
var _total_count = 0
var _diff_type = null
var _brackets = null
var _valid = true
var _desc_things = 'somethings'

# -------- comapre_result.gd "interface" ---------------------
func set_are_equal(val):
	_block_set('are_equal', val)

func get_are_equal():
	if(!_valid):
		return null
	else:
		return differences.size() == 0


func set_summary(val):
	_block_set('summary', val)

func get_summary():
	return summarize()

func get_different_count():
	return differences.size()

func  get_total_count():
	return _total_count

func get_short_summary():
	var text = str(_strutils.truncate_string(str(_value_1), 50),
		' ', _compare.get_compare_symbol(are_equal), ' ',
		_strutils.truncate_string(str(_value_2), 50))
	if(!are_equal):
		text += str('  ', get_different_count(), ' of ', get_total_count(),
			' ', _desc_things, ' do not match.')
	return text

func get_brackets():
	return _brackets
# -------- comapre_result.gd "interface" ---------------------


func _invalidate():
	_valid = false
	differences = null


func _init(v1,v2,diff_type=DEEP):
	_value_1 = v1
	_value_2 = v2
	_diff_type = diff_type
	_compare.set_should_compare_int_to_float(false)
	_find_differences(_value_1, _value_2)


func _find_differences(v1, v2):
	if(GutUtils.are_datatypes_same(v1, v2)):
		if(typeof(v1) == TYPE_ARRAY):
			_brackets = {'open':'[', 'close':']'}
			_desc_things = 'indexes'
			_diff_array(v1, v2)
		elif(typeof(v2) == TYPE_DICTIONARY):
			_brackets = {'open':'{', 'close':'}'}
			_desc_things = 'keys'
			_diff_dictionary(v1, v2)
		else:
			_invalidate()
			GutUtils.get_logger().error('Only Arrays and Dictionaries are supported.')
	else:
		_invalidate()
		GutUtils.get_logger().error('Only Arrays and Dictionaries are supported.')


func _diff_array(a1, a2):
	_total_count = max(a1.size(), a2.size())
	for i in range(a1.size()):
		var result = null
		if(i < a2.size()):
			if(_diff_type == DEEP):
				result = _compare.deep(a1[i], a2[i])
			else:
				result = _compare.simple(a1[i], a2[i])
		else:
			result = _compare.simple(a1[i], _compare.MISSING, 'index')

		if(!result.are_equal):
			differences[i] = result

	if(a1.size() < a2.size()):
		for i in range(a1.size(), a2.size()):
			differences[i] = _compare.simple(_compare.MISSING, a2[i], 'index')


func _diff_dictionary(d1, d2):
	var d1_keys = d1.keys()
	var d2_keys = d2.keys()

	# Process all the keys in d1
	_total_count += d1_keys.size()
	for key in d1_keys:
		if(!d2.has(key)):
			differences[key] = _compare.simple(d1[key], _compare.MISSING, 'key')
		else:
			d2_keys.remove_at(d2_keys.find(key))

			var result = null
			if(_diff_type == DEEP):
				result = _compare.deep(d1[key], d2[key])
			else:
				result = _compare.simple(d1[key], d2[key])

			if(!result.are_equal):
				differences[key] = result

	# Process all the keys in d2 that didn't exist in d1
	_total_count += d2_keys.size()
	for i in range(d2_keys.size()):
		differences[d2_keys[i]] = _compare.simple(_compare.MISSING, d2[d2_keys[i]], 'key')


func summarize():
	var summary = ''

	if(are_equal):
		summary = get_short_summary()
	else:
		var formatter = load('res://addons/gut/diff_formatter.gd').new()
		formatter.set_max_to_display(max_differences)
		summary = formatter.make_it(self)

	return summary


func get_diff_type():
	return _diff_type


func get_value_1():
	return _value_1


func get_value_2():
	return _value_2
