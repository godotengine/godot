func test():
	match 0:
		var bind:
			pass

	match 0:
		var bind when 1 == 1:
			pass

	match 0:
		var bind when bind == 0:
			pass

	match 0:
		var bind when false:
			var _discard = bind
