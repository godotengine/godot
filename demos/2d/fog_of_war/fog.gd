
extends TileMap

# Member variables

# Boundaries for the fog rectangle
var x_min = -20 # Left start tile
var x_max = 20 # Right end tile
var y_min = -20 # Top start tile
var y_max = 20 # Bottom end tile

var position # Player's position

# Iteration variables
var x
var y

# Variables to check if the player moved
var x_old
var y_old

# Array to build up the visible area like a square.
# First value determines the width/height of the tip.
# Here it would be 2*2 + 1 = 5 tiles wide/high.
# Second value determines the total squares size.
# Here it would be 5*2 + 1 = 10 tiles wide/high.
var l = range(2, 5)


# Process that runs in realtime
func _fixed_process(delta):
	position = get_node("../troll").get_pos()
	
	# Calculate the corresponding tile
	# from the players position
	x = int(position.x/get_cell_size().x)
	# Switching from positive to negative tile positions
	# causes problems because of rounding problems
	if position.x < 0:
		x -= 1 # Correct negative values
	
	y = int(position.y/get_cell_size().y)
	if (position.y < 0):
		y -= 1
	
	# Check if the player moved one tile further
	if ((x_old != x) or (y_old != y)):
		# Create the transparent part (visited area)
		var end = l.size() - 1
		var start = 0
		for steps in range(l.size()):
			for m in range(x - l[end] - 1, x + l[end] + 2):
				for n in range(y - l[start] - 1, y + l[start] + 2):
					if (get_cell(m, n) != 0):
						set_cell(m, n, 1, 0, 0)
			end -= 1
			start += 1
		
		# Create the actual and active visible part
		var end = l.size() - 1
		var start = 0
		for steps in range(l.size()):
			for m in range(x - l[end], x + l[end] + 1):
				for n in range(y - l[start], y + l[start] + 1):
					set_cell(m, n, -1)
			end -= 1
			start += 1
	
	x_old = x
	y_old = y


func _ready():
	# Initalization here
	# Create a square filled with the 100% opaque fog
	for x in range(x_min, x_max):
		for y in range(y_min, y_max):
			set_cell(x, y, 0, 0, 0)
	set_fixed_process(true)
