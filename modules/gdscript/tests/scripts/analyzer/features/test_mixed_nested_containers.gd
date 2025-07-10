#Test complex mixed nested container types

func test():
#Array of dictionaries with array values
	var users: Array[Dictionary[String, Array[int]]] = [
		{"scores": [95, 87, 92], "years": [2020, 2021, 2022]},
		{"scores": [88, 90, 85], "years": [2020, 2021, 2022]}
	]
	Utils.check(users[0]["scores"][0] == 95)
	users[0]["scores"][1] = 88
	Utils.check(users[0]["scores"][1] == 88)

#Dictionary of arrays with dictionary elements
	var data: Dictionary[String, Array[Dictionary[String, int]]] = {
		"morning": [{"temp": 15, "humidity": 60}, {"temp": 16, "humidity": 65}],
		"evening": [{"temp": 22, "humidity": 50}, {"temp": 21, "humidity": 55}]
	}
	Utils.check(data["morning"][0]["temp"] == 15)
	data["morning"][0]["temp"] = 14
	Utils.check(data["morning"][0]["temp"] == 14)

#Function parameter and return type tests
	var result = process_nested_data(users)
	Utils.check(result["0"] == 95)
	var matrix = create_matrix(3, 3)
	Utils.check(matrix[0][0] == 0)
	Utils.check(matrix[2][2] == 8)

#Edge cases
	var empty_nested: Array[Array[int]] = [[], [], []]
	Utils.check(empty_nested.size() == 3)
	Utils.check(empty_nested[0].size() == 0)
	var partial_nested: Array[Dictionary[String, int]] = [{}, {"a": 1}, {}]
	Utils.check(partial_nested[1]["a"] == 1)

	print("ok")

func process_nested_data(data: Array[Dictionary[String, Array[int]]]) -> Dictionary[String, int]:
	var result: Dictionary[String, int] = {}
	for i in data.size():
		result[str(i)] = data[i]["scores"][0] if data[i].has("scores") else 0
	return result

func create_matrix(rows: int, cols: int) -> Array[Array[int]]:
	var matrix: Array[Array[int]] = []
	for i in rows:
		var row: Array[int] = []
		for j in cols:
			row.append(i * cols + j)
		matrix.append(row)
	return matrix
