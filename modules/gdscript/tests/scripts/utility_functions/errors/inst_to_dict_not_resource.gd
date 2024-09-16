extends Resource


class NotResource:
	@export var number: int

func test():
	var obj:NotResource = NotResource.new()
	inst_to_dict(obj)
	print('not ok')
