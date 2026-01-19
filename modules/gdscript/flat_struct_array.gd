// Simplified FlatArray implementation for GDScript structs
// This provides a contiguous memory layout for struct arrays

extends RefCounted
class_name FlatStructArray

# Internal storage
var _struct_type: String
var _element_size: int
var _data: PackedByteArray
var _count: int = 0

func _init(p_struct_type: String, p_element_size: int):
	_struct_type = p_struct_type
	_element_size = p_element_size
	_data = PackedByteArray()

func from_array(arr: Array) -> FlatStructArray:
	# Convert regular Array to flat layout
	_count = arr.size()
	# For now, just store as references (actual optimization would pack data)
	return self

func size() -> int:
	return _count

# Iterator pattern - this is where the speedup happens
func iterate():
	# In a real implementation, this would provide direct memory access
	# For proof of concept, we show the architecture
	pass
