# GH-77666
func test_exit_if():
	var ref := RefCounted.new()
	print(ref.get_reference_count())

	if true:
		var _temp := ref

	print(ref.get_reference_count())

# GH-94654
func test_exit_while():
	var slots_data := []

	while true:
		@warning_ignore("confusable_local_declaration")
		var slot = 42
		slots_data.append(slot)
		break

	var slot: int = slots_data[0]
	print(slot)

func test():
	test_exit_if()
	test_exit_while()
