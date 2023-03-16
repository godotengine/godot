extends Node

@onready var later_inferred := [1]
@onready var later_static: Array
@onready var later_static_with_init: Array = [1]
@onready var later_untyped = [1]

func test():
	assert(typeof(later_inferred) == TYPE_ARRAY)
	assert(later_inferred.size() == 0)

	assert(typeof(later_static) == TYPE_ARRAY)
	assert(later_static.size() == 0)

	assert(typeof(later_static_with_init) == TYPE_ARRAY)
	assert(later_static_with_init.size() == 0)

	assert(typeof(later_untyped) == TYPE_NIL)

	print("ok")
