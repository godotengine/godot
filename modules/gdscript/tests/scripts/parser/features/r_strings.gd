func test():
	print(r"test ' \' \" \\ \n \t \u2023 test")
	print(r"\n\\[\t ]*(\w+)")
	print(r"")
	print(r"\"")
	print(r"\\\"")
	print(r"\\")
	print(r"\" \\\" \\\\\"")
	print(r"\ \\ \\\ \\\\ \\\\\ \\")
	print(r'"')
	print(r'"(?:\\.|[^"])*"')
	print(r"""""")
	print(r"""test \t "test"="" " \" \\\" \ \\ \\\ test""")
	print(r'''r"""test \t "test"="" " \" \\\" \ \\ \\\ test"""''')
	print(r"\t
			\t")
	print(r"\t \
			\t")
	print(r"""\t
			\t""")
	print(r"""\t \
			\t""")
