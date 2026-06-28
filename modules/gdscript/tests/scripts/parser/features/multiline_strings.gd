func test():
	var __ = """
	This is a standalone string, not a multiline comment.
	Writing both "double" quotes and 'simple' quotes is fine as
	long as there is only ""one"" or ''two'' of those in a row, not more.

	If you have more quotes, they need to be escaped like this: \"\"\"
	"""
	__ = '''
	Another standalone string, this time with single quotes.
	Writing both "double" quotes and 'simple' quotes is fine as
	long as there is only ""one"" or ''two'' of those in a row, not more.

	If you have more quotes, they need to be escaped like this: \'\'\'
	'''
