func test() -> void:
	Inner \
			.new("test") \
			._accepting_lambdas(
				func() -> void:
					match 123:
						123:
							var _hello: = "world",
			)


class Inner extends Node:
	func _init(_p_test: String) -> void:
		pass


	func _accepting_lambdas(_p_lambda: Callable) -> void:
		pass
