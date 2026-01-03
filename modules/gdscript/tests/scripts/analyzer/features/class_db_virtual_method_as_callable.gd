func test() -> void:
	var o: Object = Object.new.call()
	var free_callable: Callable = o.free
	print(free_callable)
	o.free()
