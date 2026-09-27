class TestNativeProperty extends Line2D:
	func _ready() -> void:
		points = PackedVector2Array([Vector2()])
		points[0] = Vector2.ONE # Warn.
		points[0].x = 2.0 # Warn.

		var _size := points.size()
		points.clear() # Warn.
		var _size_callable := points.size
		var _clear_callable := points.clear # Warn.

		var base: TestNativeProperty = self

		base.points = PackedVector2Array([Vector2()])
		base.points[0] = Vector2.ONE # Warn.
		base.points[0].x = 2.0 # Warn.

		var _base_size := base.points.size()
		base.points.clear() # Warn.
		var _base_size_callable := base.points.size
		var _base_clear_callable := base.points.clear # Warn.

# No warnings for a custom property.
class TestCustomProperty extends Node:
	var points: PackedVector2Array

	func _ready() -> void:
		points = PackedVector2Array([Vector2()])
		points[0] = Vector2.ONE
		points[0].x = 2.0

		var _size := points.size()
		points.clear()
		var _size_callable := points.size
		var _clear_callable := points.clear

		var base: TestCustomProperty = self

		base.points = PackedVector2Array([Vector2()])
		base.points[0] = Vector2.ONE
		base.points[0].x = 2.0

		var _base_size := base.points.size()
		base.points.clear()
		var _base_size_callable := base.points.size
		var _base_clear_callable := base.points.clear

func test():
	pass
