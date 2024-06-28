var member := 1

func test():
	var number := 1
	var string := "1"
	var vector := Vector2i(1, 0)
	var array_assign := [1]
	var array_append := [1]
	var f := func ():
		member = 2
		number = 2
		string += "2"
		vector.x = 2
		array_assign = [2]
		array_append.append(2)
		var g := func ():
			member = 3
			number = 3
			string += "3"
			vector.x = 3
			array_assign = [3]
			array_append.append(3)
			prints("g", member, number, string, vector, array_assign, array_append)
		g.call()
		prints("f", member, number, string, vector, array_assign, array_append)
	f.call()
	prints("test", member, number, string, vector, array_assign, array_append)
