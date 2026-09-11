const TEST_TYPE: PackedByteArray = []
const TEST_TYPE_ARRAY: Array[PackedByteArray] = []
const TEST_TYPE_DICT_KEY: Dictionary[PackedByteArray, int] = {}
const TEST_TYPE_DICT_VALUE: Dictionary[int, PackedByteArray] = {}

const TEST_INITIALIZER = [] as PackedByteArray
const TEST_INITIALIZER_INFER := [] as PackedByteArray
const TEST_INITIALIZER_NESTED = [[] as PackedByteArray]
const TEST_INITIALIZER_NESTED_INFER := [[] as PackedByteArray]

const TEST_INITIALIZER_ARRAY = [] as Array[PackedByteArray]
const TEST_INITIALIZER_INFER_ARRAY := [] as Array[PackedByteArray]
const TEST_INITIALIZER_NESTED_ARRAY = [[] as Array[PackedByteArray]]
const TEST_INITIALIZER_NESTED_INFER_ARRAY := [[] as Array[PackedByteArray]]

const TEST_INITIALIZER_ADICT_KEY = {} as Dictionary[PackedByteArray, int]
const TEST_INITIALIZER_INFER_DICT_KEY := {} as Dictionary[PackedByteArray, int]
const TEST_INITIALIZER_NESTED_DICT_KEY = [{} as Dictionary[PackedByteArray, int]]
const TEST_INITIALIZER_NESTED_INFER_DICT_KEY := [{} as Dictionary[PackedByteArray, int]]

const TEST_INITIALIZER_ADICT_VALUE = {} as Dictionary[int, PackedByteArray]
const TEST_INITIALIZER_INFER_DICT_VALUE := {} as Dictionary[int, PackedByteArray]
const TEST_INITIALIZER_NESTED_DICT_VALUE = [{} as Dictionary[int, PackedByteArray]]
const TEST_INITIALIZER_NESTED_INFER_DICT_VALUE := [{} as Dictionary[int, PackedByteArray]]

const TEST_EXPLICIT_CONSTRUCTOR = PackedByteArray()
const TEST_EXPLICIT_CONSTRUCTOR_ARRAY = Array(
	[],
	TYPE_PACKED_BYTE_ARRAY, &"", null,
)
const TEST_EXPLICIT_CONSTRUCTOR_DICT_KEY = Dictionary(
	{},
	TYPE_PACKED_BYTE_ARRAY, &"", null,
	TYPE_INT, &"", null,
)
const TEST_EXPLICIT_CONSTRUCTOR_DICT_VALUE = Dictionary(
	{},
	TYPE_INT, &"", null,
	TYPE_PACKED_BYTE_ARRAY, &"", null,
)

func test():
	pass
