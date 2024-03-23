func test():
	var i_string := ''
	for i in 3:
		if i == 1: continue
		var lambda := func():
			var j_string := ''
			for j in 3:
				if j == 1: continue
				j_string += str(j)
			return j_string
		i_string += lambda.call()
	assert(i_string == '0202')
	print('ok')
