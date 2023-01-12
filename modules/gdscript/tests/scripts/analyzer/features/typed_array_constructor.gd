var int_arr: Array[int] = [0, 200, 30];
var vec:Array[Vector2] = [Vector2(1.5, 0.2), Vector2(2.25, 300.4)];
var strings:Array[String] = ["Hey!", "Make me a sname please!"];

# Array[Type](base) is an alias for Array(base, type, class_name, script);
func test():
	var float_arr: Array[float] = Array[float](int_arr);
	print(float_arr); # 0.0, 200.0, 30.0

	var non_aliased_conversion: Array[float] = Array(int_arr, TYPE_FLOAT, &"", null);
	print(float_arr == non_aliased_conversion); # true

	# Other various conversions
	var veci := Array[Vector2i](vec);
	print(veci); # [(1, 0), (2, 300)]

	var snames := Array[StringName](strings);
	print(snames); # [&"Hey!", &"Make me a sname please!"]
