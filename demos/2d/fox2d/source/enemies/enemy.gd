extends "res://Edible.gd"

# Generic class for common enemies for Node Enemy1
# Since those enemies have lot in common, and change only in the sprite they use, the idea of this class and node was to use a AnimationTreePlayer to manage which sprite animation to play.
# It's okay for simple enemies with only one or two kind of animation, but it won't be a good approach for a more complex kind of enemy.
# This generic class contains a basic behavior, but usually the node created in the scene will extend this class or a class that extend this class and specify which behavior to use.

# To prevent the enemy to wake up and move too soon, meaning leaving the area before the player reaches it, the enemy is disabled until the player arrives.

# properties --------------------------------------------------------
export(int,4) var enemy_type=0                # kind of enemy
export(int,1) var anim_state=0                # enemy's current animation
export (int,"left","right") var direction = 0 # tell which direction it's facing. Default left.

# Constants ------------------------------------------------------------
const FLOOR_ANGLE_TOLERANCE = 40 # for physics, angle in °
const ACCELERATION=1000          # acceleration of the enemy, in [px/s²]. Not used?
const GRAVITY = 700.0            # gravity applied to the enemy, in [px/s²]
const WALK_FORCE = 400           # Force added to speedm in [px/s]
const WALK_MAX_SPEED = 200       # Max speed the enemy can go, in [px/s]
const WALK_SPEED=150        


# Variables -------------------------------------------------------------------
var sub_animations=[null,"enemy2Transition","enemy3Transition",null,null]

var rc_left=null          # left sensor for collision detection
var rc_right=null         # right sensor for collision detection
var velocity= Vector2()   # current velocity of the enemy
var alive=true            # flag for if the enemy is still not death. Used by player to check if enemy can still hurt.
var is_on_floor=false     # flag for if it's on floor. Can be used by some implementations of enemies, like to jump whenever the enemy touches the ground.
var active=false          # flag for if the enemy is still not activated and wait for the player to come before it moves. Default: false=not active

# Functions -----------------------------------------------------------------------

# override from Edible
# called when sucked by player
func startMove(target):
	alive=false # the enemy is being sucked. It's considered to be already dead, even if it's not. But the player must not be hurt by an enemy he's been sucking.
	set_fixed_process(false)
	.startMove(target) # call super

# override from Edible
# called when escaping the vacuum of the player
func moveTimeOut():
	.moveTimeOut() # call super
	alive=true     # since the enemy escaped the vacuum, he can be considered to be alive again.
	set_fixed_process(true)

# called by animation
func _die():
	queue_free()

# starts explode animation
func explode():
	alive=false # enemy is considered dead and won't hurt anymore.
	get_node("AnimationPlayer").play("explode")

# flip the direction of the node, and its sprite
func _flip_direction():
	direction=-direction
	get_node("sprite").set_scale( Vector2(direction,1))

# Initializer
func _ready():
	# initialize the animation based on the parameters of the node instance
	get_node("animTreePlayer").transition_node_set_current("charType",enemy_type)
	set_sub_animation(anim_state)

	# initialize sensors
	rc_left=get_node("raycast_left")
	rc_right=get_node("raycast_right")
	
	# flip the sprite if needed
	if(direction==0):
		direction=1
		_flip_direction()
	
	# start moving
	set_fixed_process(true)

# Main loop
func _fixed_process(delta):
	if(alive and active):
		move_process(delta)

# function to override to define the enemy's behavior.
func move_process(delta):
	# call default behavior, running in circle by collision
	run_and_turn(delta)

# function behavior to run in circle, meaning it run straight forward until it collides with something. And then it goes the other direction.
func run_and_turn(delta):
	var force = Vector2(0,GRAVITY)
	
	#if not at full speed, accelerate in the same direction
	if(abs(velocity.x)<WALK_MAX_SPEED):
		force.x+=direction*WALK_FORCE
	
	slide(delta,force)
	
	# check if colliding with something. If it's the case, flip the direction to the other side
	if(direction>0 and rc_right.is_colliding() or direction<0 and rc_left.is_colliding()):
		_flip_direction()

# generic function than can be called by all implementation of enemies.
# add a force to current velocity and make the actor move. If collision is detected, the actor slides against the wall.
# If the enemy is on a moving plateform, he moves with it.
#
# Note : this kind of script is used by a lot of actors. TODO: put this function in a common script library and make it with weak coupling (like passing the node in parameter).
func slide(delta,force):
	velocity+=force*delta
	var motion = velocity * delta
	motion = move(motion)
	
	is_on_floor=false
	
	var floor_velocity=Vector2()
	if (is_colliding()):
		var n = get_collision_normal()
		
		if ( rad2deg(acos(n.dot( Vector2(0,-1)))) < FLOOR_ANGLE_TOLERANCE ):
			floor_velocity=get_collider_velocity()
			is_on_floor=true
		
		motion = n.slide(motion)
		velocity = n.slide(velocity)
		
		move(motion)
	
	if (floor_velocity!=Vector2()):
		move(floor_velocity*delta)

# setter for current animation in the animationtreeplayer
func set_sub_animation(pos):
	anim_state=pos
	var name=sub_animations[enemy_type]
	if(name!=null):
		get_node("animTreePlayer").transition_node_set_current(name,pos)

# activate the enemy
func activate():
	if(OS.get_ticks_msec()>2000): # tweak for a bug at the time the demo was developped. It's still not perfect.
		active=true
