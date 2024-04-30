# https://github.com/godotengine/godot/issues/89439
extends Node

signal my_signal

func async_func():
	await my_signal
	my_signal.emit()

func test():
	async_func()
	my_signal.emit()
