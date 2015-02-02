
extends TileMap

# member variables here, example:
# var a=2
# var b="textvar"

# boundarys for the fog rectangle
var x_min = -20 # left start tile
var x_max = 20 # right end tile
var y_min = -20 # top start tile
var y_max = 20 # bottom end tile

var position # players position

# iteration variables
var x
var y

# variable to check if player moved
var x_old
var y_old

# array to build up the visible area like a square
# first value determines the width/height of the tip
# here it would be 2*2 + 1 = 5 tiles wide/high
# second value determines the total squares size
# here it would be 5*2 + 1 = 10 tiles wide/high
var l = range(2,5)

# process that runs in realtime
func _fixed_process(delta):
	position = get_node("../troll").get_pos()
	
	# calculate the corresponding tile
	# from the players position
	x = int(position.x/get_cell_size().x)
	# switching from positive to negative tile positions
	# causes problems because of rounding problems
	if position.x < 0:
		x -= 1 # correct negative values
	
	y = int(position.y/get_cell_size().y)
	if position.y < 0:
		y -= 1
		
	# check if the player moved one tile further
	if (x_old != x) or (y_old != y):
		
		# create the transparent part (visited area)
		var end = l.size()-1
		var start = 0
		for steps in range(l.size()):
			for m in range(x-l[end]-1,x+l[end]+2):
				for n in range(y-l[start]-1,y+l[start]+2):
					if get_cell(m,n) != 0:
						set_cell(m,n,1,0,0)
			end -= 1
			start += 1
	
		# create the actual and active visible part
		var end = l.size()-1
		var start = 0
		for steps in range(l.size()):
			for m in range(x-l[end],x+l[end]+1):
				for n in range(y-l[start],y+l[start]+1):
					set_cell(m,n,-1)
			end -= 1
			start += 1
		
	x_old = x
	y_old = y
	
	pass

func _ready():
	# Initalization here
	
	# create a square filled with the 100% opaque fog
	for x in range(x_min,x_max):
		for y in range(y_min,y_max):
			set_cell(x,y,0,0,0)
	set_fixed_process(true)
	pass


