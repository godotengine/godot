class Base:
	func describe():
		return "base"


func test():
	# Inline single-line body with one func.
	var obj = Base.new(): func describe(): return "inline"
	print(obj.describe())
