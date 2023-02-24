enum E { E0 = 0, E3 = 3 }

func test():
	var total := 0
	for value in range(E.E0, E.E3):
		var inferable := value
		total += inferable
	assert(total == 0 + 1 + 2)
	print('ok')
