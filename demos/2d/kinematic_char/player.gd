
extends KinematicBody2D

#This is a simple collision demo showing how
#the kinematic cotroller works.
#move() will allow to move the node, and will
#always move it to a non-colliding spot, 
#as long as it starts from a non-colliding spot too.


#pixels / second
const GRAVITY = 500.0

#Angle in degrees towards either side that the player can 
#consider "floor".
const FLOOR_ANGLE_TOLERANCE = 40
const WALK_FORCE = 600
const WALK_MIN_SPEED=10
const WALK_MAX_SPEED = 200
const STOP_FORCE = 1300
const JUMP_SPEED = 200
const JUMP_MAX_AIRBORNE_TIME=0.2

var velocity = Vector2()
var on_air_time=100
var jumping=false

var prev_jump_pressed=false

func _fixed_process(delta):

	#create forces
	var force = Vector2(0,GRAVITY)

	var stop = velocity.x!=0.0
	
	var walk_left = Input.is_action_pressed("move_left")
	var walk_right = Input.is_action_pressed("move_right")
	var jump = Input.is_action_pressed("jump")

	var stop=true
	
	if (walk_left):
		if (velocity.x<=WALK_MIN_SPEED and velocity.x > -WALK_MAX_SPEED):
			force.x-=WALK_FORCE			
			stop=false
		
	elif (walk_right):
		if (velocity.x>=-WALK_MIN_SPEED and velocity.x < WALK_MAX_SPEED):
			force.x+=WALK_FORCE
			stop=false
	
	if (stop):
		var vsign = sign(velocity.x)
		var vlen = abs(velocity.x)
		
		vlen -= STOP_FORCE * delta
		if (vlen<0):
			vlen=0
			
		velocity.x=vlen*vsign
		

		
	#integrate forces to velocity
	velocity += force * delta
	
	#integrate velocity into motion and move
	var motion = velocity * delta

	#move and consume motion
	motion = move(motion)


	var floor_velocity=Vector2()

	if (is_colliding()):
		# you can check which tile was collision against with this
		# print(get_collider_metadata())

		#ran against something, is it the floor? get normal
		var n = get_collision_normal()

		if ( rad2deg(acos(n.dot( Vector2(0,-1)))) < FLOOR_ANGLE_TOLERANCE ):
			#if angle to the "up" vectors is < angle tolerance
			#char is on floor
			on_air_time=0
			floor_velocity=get_collider_velocity()
			#velocity.y=0 
			
		#But we were moving and our motion was interrupted, 
		#so try to complete the motion by "sliding"
		#by the normal
		motion = n.slide(motion)
		velocity = n.slide(velocity)
		
		#then move again
		move(motion)

	if (floor_velocity!=Vector2()):
		#if floor moves, move with floor
		move(floor_velocity*delta)

	if (jumping and velocity.y>0):
		jumping=false
		
	if (on_air_time<JUMP_MAX_AIRBORNE_TIME and jump and not prev_jump_pressed and not jumping):	
		velocity.y=-JUMP_SPEED	
		jumping=true
		
	on_air_time+=delta
	prev_jump_pressed=jump	

func _ready():
	#Initalization here
	set_fixed_process(true)
	pass


