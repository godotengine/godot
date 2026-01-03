func test():
	print(FileAccess.file_exists("no file here")); # Original.

	var FileAccess = Shadow.new()
	print(FileAccess.file_exists("no file here")); # Shadowed by local variable.

	var member_shadow = MemberShadow.new()
	member_shadow.test()

class Shadow:
	func file_exists(path: String) -> String:
		return "Called shadows file_exists with path: " + path

class MemberShadow:
	var FileAccess = Shadow.new()

	func test():
		print(FileAccess.file_exists("no file here")); # Shadowed by member variable.
