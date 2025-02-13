extends Node

func test():
	process_priority = 10
	var change = 20

	print(process_priority)
	print(change)

	process_priority += change

	print(process_priority)
	print(change)
