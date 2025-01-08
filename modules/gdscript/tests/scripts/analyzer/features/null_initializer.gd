func check(input: int) -> bool:
	return input == 1

var recur = null
var prop = null

func check_arg(arg = null) -> void:
	if arg != null:
		@warning_ignore("unsafe_call_argument")
		print(check(arg))

func check_recur() -> void:
	if recur != null:
		@warning_ignore("unsafe_call_argument")
		print(check(recur))
	else:
		recur = 1
		check_recur()

func test() -> void:
	check_arg(1)

	check_recur()

	if prop == null:
		set('prop', 1)
		@warning_ignore("unsafe_call_argument")
		print(check(prop))
		set('prop', null)

	var loop = null
	while loop != 2:
		if loop != null:
			@warning_ignore("unsafe_call_argument")
			print(check(loop))
		loop = 1 if loop == null else 2
