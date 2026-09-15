func run(param):
	# Validated native static call with return value.
	print(FileAccess.file_exists("some_file"))

	# Validated native static call without return value.
	Node.print_orphan_nodes()

	# Not validated native static call with return value.
	@warning_ignore("unsafe_call_argument")
	print(FileAccess.file_exists(param))

	# Not validated native static call without return value.
	FileDialog.set_favorite_list([param])
	print(FileDialog.get_favorite_list())
	FileDialog.set_favorite_list([])

func test():
	run("some_file")
