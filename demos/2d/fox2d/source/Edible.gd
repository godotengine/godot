extends KinematicBody2D

# Generic class of all edible objects. 
# a class or object extending this class must not override _process, or must call its parent, because it contains the code to make the object being sucked by the player.
# Because sometime the player stops to suck before the edible object is eaten, a timeout is added as a workaround to avoid the object being stuck.
# !Important! The class or node extending this class must be the root class of the whole object being edible. And it must always be a KinematicBody2D.

# Constants ---------------------------------------------------------------
const EAT_SPEED=500   #speed of the object to where it's sucked
const EAT_TIMEOUT=0.5 #maximum time in [s] before the object can get away from the vacuum.

# Variables ---------------------------------------------------------------
var _target=null  # node of the origin of the vacuum
var age=0 # time left before the object get away from the vacuum


# Functions ---------------------------------------------------------------

# start the vacuum mode.
# Vector2 target : position of the origin of the vacuum (usually from the root node of the Cone)
func startMove(target):
	_target=target
	set_process(true)
	age=EAT_TIMEOUT

# moves the object in the direction of the vacuum origin point. It's automatically called when set_process(true). 
# It decrease and check also the timeout timer. If the timeout is reached, the vacuum mode is ended (and usually the object extending Edible reactivates its default behavior).
func _process(delta):
	# check the timout
	age-=delta
	if(age<=0):
		moveTimeOut()
	else:
		# get the edible object's position, calculate its new position and set its new position.
		# The node is not moved with move() but teleported with set_pos(). This avoids collision problems with KinematicBody2D physics.
		var current_pos=get_pos()
		var movement=(_target-current_pos)
		movement=movement.normalized()*delta*EAT_SPEED
		set_pos(current_pos+movement)

# ends the vacuum mode.
func moveTimeOut():
	set_process(false)
	_target=null