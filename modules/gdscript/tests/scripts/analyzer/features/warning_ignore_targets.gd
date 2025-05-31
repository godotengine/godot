@warning_ignore("confusable_identifier")
class MyClАss:
	var my_vАr

@warning_ignore("narrowing_conversion")
var i: int = f:
	get:
		return f

var f: float

@warning_ignore("narrowing_conversion")
func test_func(_i: int = f):
	i = f

func test():
	@warning_ignore("narrowing_conversion")
	if signi(f): # TODO: Allow `@warning_ignore` before `elif`?
		i = f

	@warning_ignore("narrowing_conversion")
	match signi(f):
		1:
			i = f
		@warning_ignore("confusable_identifier")
		var my_vАr:
			var _my_vАr: Variant = my_vАr

	@warning_ignore("narrowing_conversion")
	for j in signi(f):
		i = f

	@warning_ignore("narrowing_conversion")
	while signi(f):
		i = f
