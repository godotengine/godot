
extends Spatial

func _ready():
	var pf = PolygonPathFinder.new()
	
	var points = Vector2Array()
	var connections = IntArray()
	
	# poly 1
	points.push_back(Vector2(0, 0))		#0
	points.push_back(Vector2(10, 0))	#1
	points.push_back(Vector2(10, 10))	#2
	points.push_back(Vector2(0, 10))	#3

	connections.push_back(0) # connect vertex 0 ...
	connections.push_back(1) # ... to 1
	drawLine(points[0], points[1], get_node("/root/Spatial/Polys"))
	connections.push_back(1) # connect vertex 1 ...
	connections.push_back(2) # ... to 2
	drawLine(points[1], points[2], get_node("/root/Spatial/Polys"))
	connections.push_back(2) # etc.
	connections.push_back(3)
	drawLine(points[2], points[3], get_node("/root/Spatial/Polys"))
	connections.push_back(3) # connect vertex 3 ...
	connections.push_back(0) # back to vertex 0, to close the polygon
	drawLine(points[3], points[0], get_node("/root/Spatial/Polys"))

	# poly 2, as obstacle inside poly 1
	points.push_back(Vector2(2, 0.5))	#4
	points.push_back(Vector2(4, 0.5))	#5
	points.push_back(Vector2(4, 9.5))	#6
	points.push_back(Vector2(2, 9.5))	#7

	connections.push_back(4)
	connections.push_back(5)
	drawLine(points[4], points[5], get_node("/root/Spatial/Polys"))
	connections.push_back(5)
	connections.push_back(6)
	drawLine(points[5], points[6], get_node("/root/Spatial/Polys"))
	connections.push_back(6)
	connections.push_back(7)
	drawLine(points[6], points[7], get_node("/root/Spatial/Polys"))
	connections.push_back(7)
	connections.push_back(4)
	drawLine(points[7], points[4], get_node("/root/Spatial/Polys"))

	
	print("points: ",points)
	print("connections: ",connections)
	
	pf.setup(points, connections)
	
	var path = pf.find_path(Vector2(1, 5), Vector2(8, 5))
	
	var lastStep = null
	print("path: ",path)
	for step in path:
		print("step: ",step)
		if (lastStep != null):
			var currPathSegment = Vector2Array()
			drawLine(lastStep, step, get_node("/root/Spatial/Path"))
		lastStep = step
		


func drawLine(pointA, pointB, immediateGeo):
	var drawPosY = 0.1
	var im = immediateGeo
	
	im.begin(Mesh.PRIMITIVE_POINTS, null)
	im.add_vertex(Vector3(pointA.x, drawPosY, pointA.y))
	im.add_vertex(Vector3(pointB.x, drawPosY, pointB.y))
	im.end()
	im.begin(Mesh.PRIMITIVE_LINE_STRIP, null)
	im.add_vertex(Vector3(pointA.x, drawPosY, pointA.y))
	im.add_vertex(Vector3(pointB.x, drawPosY, pointB.y))
	im.end()
	
	
