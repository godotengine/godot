const A1 = Array()
const A2 = Array(Array())
const A3 = Array([])
const A4 = [Array()]
const A5 = [[]]
const A6 = Array([1], TYPE_INT, &"", null)

const D1 = Dictionary()
const D2 = Dictionary(Dictionary())
const D3 = Dictionary({})
const D4 = { Dictionary(): Dictionary() }
const D5 = { {}: {} }
const D6 = Dictionary({ 1: 1 }, TYPE_INT, &"", null, TYPE_INT, &"", null)

var a1 = Array()
var a2 = Array(Array())
var a3 = Array([])
var a4 = [Array()]
var a5 = [[]]
var a6 = Array([1], TYPE_INT, &"", null)

var d1 = Dictionary()
var d2 = Dictionary(Dictionary())
var d3 = Dictionary({})
var d4 = { Dictionary(): Dictionary() }
var d5 = { {}: {} }
var d6 = Dictionary({ 1: 1 }, TYPE_INT, &"", null, TYPE_INT, &"", null)

func test_value(value: Variant) -> void:
	@warning_ignore("unsafe_method_access")
	prints(value.is_read_only(), var_to_str(value).replace("\n", " "))

func test():
	print('---')
	test_value(A1)
	test_value(A2)
	test_value(A3)
	test_value(A4)
	test_value(A5)
	test_value(A6)

	print('---')
	test_value(D1)
	test_value(D2)
	test_value(D3)
	test_value(D4)
	test_value(D5)
	test_value(D6)

	print('---')
	test_value(a1)
	test_value(a2)
	test_value(a3)
	test_value(a4)
	test_value(a5)
	test_value(a6)

	print('---')
	test_value(d1)
	test_value(d2)
	test_value(d3)
	test_value(d4)
	test_value(d5)
	test_value(d6)
