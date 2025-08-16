extends SceneTree

func dec2bistr(decimal_value, max_bits = 31):
	var binary_string = ""
	var temp
	var count = max_bits

	while(count >= 0):
		temp = decimal_value >> count
		if(temp & 1):
			binary_string = binary_string + "1"
		else:
			binary_string = binary_string + "0"
		count -= 1

	return binary_string

func print_binary(i):
	print(str(i).rpad(5), dec2bistr(i, 10))

func print_has_index_set(i, index):
	var s = str(str(i).rpad(10), dec2bistr(i, 10))
	var result = i & (1 << index) != 0
	s += str(' has ', index, ' = ', result)
	print(s)

func _init():
	print_has_index_set(5, 0)
	print_has_index_set(5, 1)
	print_has_index_set(5, 2)
	print()
	print_has_index_set(17, 4)
	print()
	print_has_index_set(73, 3)
	print_has_index_set(73, 6)


	print(1<< 2)
	print(1 << 3)
	print(1 << 6)
	quit()