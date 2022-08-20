#GDTEST_OK

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

func test():
	print("Ok")
