func test():
	var dictionary = {1: {2: {3: {4: {5: {6: {7: {8: {"key": "value"}}}}}}}}}
	print(dictionary[1][2][3][4][5][6][7])
	print(dictionary[1][2][3][4][5][6][7][8])
	print(dictionary[1][2][3][4][5][6][7][8].key)
	print(dictionary[1][2][3][4][5][6][7][8]["key"])
