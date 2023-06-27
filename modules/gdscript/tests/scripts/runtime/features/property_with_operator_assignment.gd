#GDTEST_OK
var prop: int = 0:
	get:
		return prop
	set(value):
		prop = value % 7

func test():
	for i in 7:
		prop += 1
		print(prop)
