func test():
	var variant_int: Variant = 1
	var weak_int = 1

	for x in variant_int:
		if x is String:
			print('never')
		print(x)

	for x in weak_int:
		if x is String:
			print('never')
		print(x)

	print('ok')
