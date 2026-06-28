# GH-80157

extends Node

func f():
	pass

signal s()

func test():
	print(f)
	print(s)
	print(get_child)
	print(ready)
