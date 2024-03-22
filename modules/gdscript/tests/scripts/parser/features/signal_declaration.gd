# No parentheses.
signal a

# No parameters.
signal b()

# With parameters.
signal c(a, b, c)

# With parameters multiline.
signal d(
	a,
	b,
	c,
)

# With type hints.
signal e(a: int, b: Variant, c: Node)

func no_exec():
	a.emit()
	b.emit()
	c.emit()
	d.emit()
	e.emit()

func test():
	print("Ok")
