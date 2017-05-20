extends Node2D

var path_finder
var path
var points = Vector2Array()
var connections = IntArray()

var obstacle
var point1_btn
var point2_btn
var mouse_position_label

func _ready():
	set_fixed_process(true)
	
	path_finder = PolygonPathFinder.new()
	obstacle = get_node("obstacle/CollisionPolygon2D")
	point1_btn = get_node("point1")
	point2_btn = get_node("point2")
	points.push_back(Vector2(0, 0))
	points.push_back(Vector2(1024, 0))
	points.push_back(Vector2(1024, 600))
	points.push_back(Vector2(0, 600))
	var tmp = Vector2Array()
	tmp = obstacle.get_polygon()
	for i in range(tmp.size()):
		tmp[i] += obstacle.get_parent().get_pos()
		print("point: ", tmp[i])
	points += tmp

	connections.push_back(0) # connect vertex 0 ...
	connections.push_back(1) # ... to 1
	connections.push_back(1) # connect vertex 1 ...
	connections.push_back(2) # ... to 2
	connections.push_back(2) # etc.
	connections.push_back(3)
	connections.push_back(3) # connect vertex 3 ...
	connections.push_back(0) # back to vertex 0, to close the polygon
	connections.push_back(4)
	connections.push_back(5)
	connections.push_back(5)
	connections.push_back(6)
	connections.push_back(6)
	connections.push_back(7)
	connections.push_back(7)
	connections.push_back(4)
	for i in points:
		print("points: ", i)
	print("connections: ", connections)
	
	path_finder.setup(points, connections)
	path = path_finder.find_path(point1_btn.get_pos(), point2_btn.get_pos())

func _draw():
	var last_step = null
	for step in path:
		if (last_step != null):
			draw_line(last_step, step, Color(1, 0, 0), 7)
		last_step = step

func _fixed_process(delta):
	var tmp = obstacle.get_polygon()
	for i in range(tmp.size()):
		tmp[i] += obstacle.get_parent().get_pos()
	points.resize(4)
	points += tmp
	path_finder.setup(points, connections)
	path = path_finder.find_path(point1_btn.get_pos(), point2_btn.get_pos())
	update()