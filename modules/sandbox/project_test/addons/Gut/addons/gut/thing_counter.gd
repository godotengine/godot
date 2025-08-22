var things = {}

func get_unique_count():
	return things.size()


func add_thing_to_count(thing):
	if(!things.has(thing)):
		things[thing] = 0


func add(thing):
	if(things.has(thing)):
		things[thing] += 1
	else:
		things[thing] = 1


func has(thing):
	return things.has(thing)


func count(thing):
	var to_return = 0
	if(things.has(thing)):
		to_return = things[thing]
	return to_return


func sum():
	var to_return = 0
	for key in things:
		to_return += things[key]
	return to_return


func to_s():
	var to_return = ""
	for key in things:
		to_return += str(key, ":  ", things[key], "\n")
	to_return += str("sum: ", sum())
	return to_return


func get_max_count():
	var max_val = null
	for key in things:
		if(max_val == null or things[key] > max_val):
			max_val = things[key]
	return max_val


func add_array_items(array):
	for i in range(array.size()):
		add(array[i])
