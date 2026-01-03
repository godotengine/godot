# GH-94667

class Inner:
	var subprop: Vector2:
		set(value):
			prints("subprop setter", value)
			subprop = value
		get:
			print("subprop getter")
			return subprop

	func _to_string() -> String:
		return "<Inner>"

var prop1:
	set(value):
		prints("prop1 setter", value)
		prop1 = value

var prop2: Inner:
	set(value):
		prints("prop2 setter", value)
		prop2 = value

var prop3:
	set(value):
		prints("prop3 setter", value)
		prop3 = value
	get:
		print("prop3 getter")
		return prop3

var prop4: Inner:
	set(value):
		prints("prop4 setter", value)
		prop4 = value
	get:
		print("prop4 getter")
		return prop4

func test():
	print("===")
	prop1 = Vector2()
	prop1.x = 1.0
	print("---")
	prop1 = Inner.new()
	prop1.subprop.x = 1.0

	print("===")
	prop2 = Inner.new()
	prop2.subprop.x = 1.0

	print("===")
	prop3 = Vector2()
	prop3.x = 1.0
	print("---")
	prop3 = Inner.new()
	prop3.subprop.x = 1.0

	print("===")
	prop4 = Inner.new()
	prop4.subprop.x = 1.0
