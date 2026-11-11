class Test:
	var member := 0


func test_variable():
	var bar := Test.new()
	var member = 0
	var _discard = bar.member

func test_constant():
	var bar := Test.new()
	const member := 0
	var _discard = bar.member

func test_parameter(member: int) -> void:
	var bar := Test.new()
	var _discard = bar.member

func test_pattern_bind():
	var bar := Test.new()
	match 0:
		var member:
			var _discard = bar.member


func test():
	pass
