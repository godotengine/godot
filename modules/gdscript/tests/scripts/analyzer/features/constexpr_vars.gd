const LITERAL_1 = 10
const LITERAL_2 = 10.3

const UNARY_OP_1 = -17
const UNARY_OP_2 = ~16912
const UNARY_OP_3 = ~0x4210
const UNARY_OP_4 = ~0b0100001000010000
const UNARY_OP_5 = not true
const UNARY_OP_6 = not not true

const BINARY_OP_1 = 11 + 12
const BINARY_OP_2 = 15 * 3
const BINARY_OP_3 = 21 / 2.0
const BINARY_OP_4 = 11**2
const BINARY_OP_5 = BINARY_OP_4 - 1
const BINARY_OP_6 = UNARY_OP_5 or (LITERAL_1 == 11)
const BINARY_OP_7 = UNARY_OP_5 or (LITERAL_1 == 10)

const ARRAY_1 = ["a", "b", "c"]
const ARRAY_2 = ["a" + "b", "b" + "d", "c"]
const ARRAY_3 = [ARRAY_1[2], ARRAY_2[0], ARRAY_1[2]]
const ARRAY_4 = [ARRAY_1, ARRAY_3, ARRAY_3]
const ARRAY_5 = ARRAY_1 + [ARRAY_1, "d", "e"] + ["f"]

const DICT_1 = {"a": "b", "c": 2}
const DICT_2 = {"a": "b", "c": 2, "d": ["e", "f"]}
const DICT_3 = {"a": "b", "c": 2, "d" + "l": ["e", "f"], DICT_1["a"]: "a"}

# TODO constexpr casts
# const CAST_1 = 17.8 as int
# const CAST_2 = 18 as String
# const CAST_3 = "193" as int

func test():
	print(LITERAL_1)
	print(LITERAL_2)
	print(UNARY_OP_1)
	print(UNARY_OP_2)
	print(UNARY_OP_3)
	print(UNARY_OP_4)
	print(UNARY_OP_5)
	print(UNARY_OP_6)
	print(BINARY_OP_1)
	print(BINARY_OP_2)
	print(BINARY_OP_3)
	print(BINARY_OP_4)
	print(BINARY_OP_5)
	print(BINARY_OP_6)
	print(BINARY_OP_7)
	print(ARRAY_1)
	print(ARRAY_2)
	print(ARRAY_3)
	print(ARRAY_4)
	print(ARRAY_5)
	print(DICT_1)
	print(DICT_2)
	print(DICT_3)
	# print(CAST_1)
	# print(CAST_2)
	# print(CAST_3)
