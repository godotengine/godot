# GH-89403

func test():
	var x := 1
	if x == 0 \
			# Comment.
			# Comment.
			and (x < 1 or x > 2) \
			# Comment.
			and x != 3:
		pass
	print("Ok")
