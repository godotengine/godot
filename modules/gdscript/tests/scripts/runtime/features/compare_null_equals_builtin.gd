func test():
	var value

	# null
	value = null
	print(null == value)

	# bool
	value = false
	print(null == value)

	# int
	value = 0
	print(null == value)

	# float
	value = 0.0
	print(null == value)

	# String
	value = ""
	print(null == value)

	# Vector2
	value = Vector2()
	print(null == value)

	# Vector2i
	value = Vector2i()
	print(null == value)

	# Rect2
	value = Rect2()
	print(null == value)

	# Rect2i
	value = Rect2i()
	print(null == value)

	# Vector3
	value = Vector3()
	print(null == value)

	# Vector3i
	value = Vector3i()
	print(null == value)

	# Transform2D
	value = Transform2D()
	print(null == value)

	# Plane
	value = Plane()
	print(null == value)

	# Quaternion
	value = Quaternion()
	print(null == value)

	# AABB
	value = AABB()
	print(null == value)

	# Basis
	value = Basis()
	print(null == value)

	# Transform3D
	value = Transform3D()
	print(null == value)

	# Color
	value = Color()
	print(null == value)

	# StringName
	value = &""
	print(null == value)

	# NodePath
	value = ^""
	print(null == value)

	# RID
	value = RID()
	print(null == value)

	# Callable
	value = Callable()
	print(null == value)

	# Signal
	value = Signal()
	print(null == value)

	# Dictionary
	value = {}
	print(null == value)

	# Array
	value = []
	print(null == value)

	# PackedByteArray
	value = PackedByteArray()
	print(null == value)

	# PackedInt32Array
	value = PackedInt32Array()
	print(null == value)

	# PackedInt64Array
	value = PackedInt64Array()
	print(null == value)

	# PackedFloat32Array
	value = PackedFloat32Array()
	print(null == value)

	# PackedFloat64Array
	value = PackedFloat64Array()
	print(null == value)

	# PackedStringArray
	value = PackedStringArray()
	print(null == value)

	# PackedVector2Array
	value = PackedVector2Array()
	print(null == value)

	# PackedVector3Array
	value = PackedVector3Array()
	print(null == value)

	# PackedColorArray
	value = PackedColorArray()
	print(null == value)

	# PackedVector4Array
	value = PackedVector4Array()
	print(null == value)
