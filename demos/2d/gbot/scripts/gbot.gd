extends RigidBody2D

### import the input helper class
var input_states = preload("input_states.gd")

### export variables for ui customization
export var speed = 700
export var jump_force = 1100
export var gravity = 0

### speed calculation variables
var ground_acceleration = 10
var air_acceleration = 3
var current_speed = .1
var current_rot = 0

### animations and blendtimes here
var sprite_anim = ""
var sprite_anim_new = ""
var blendtime = -1
var anim_speed = 1.0

### create input states classes
var move_left = input_states.new("ui_left")
var move_right = input_states.new("ui_right")
var jump = input_states.new("ui_jump")
var ui_accept = input_states.new("ui_accept")

var btn_jump = null
var btn_left = null
var btn_right = null


### animation player
var animation_player = null

var delta = null

### create raycasts
var raycast_left = null
var raycast_center = null
var raycast_right = null
var raycast_front = null
var raycast_shadow = null
var raycast_border_check = null
var raycast_border_climb = null

### create global timer variables
var wallslide_timer = null
var controls_lock_timer = null
var jump_pad_timer = null

### create states that the character uses
var JUMPSTATE = 0
var PLAYERSTATE = "ground"
var PLAYERSTATE_PREV = "ground"
var PLAYERSTATE_NEXT = "ground"
var WALLSTICK = false
var WALLSTICK_HIST = false
var ORIENTATION = "right"
var ORIENTATION_PREV = "right"

### methode for the shadow. projects it on the ground beneath
func drop_shadow():
	var shadow = get_node("body_tiles/shadow")
	var raycast_shadow = get_node("body_tiles/raycast_shadow")
	
	var hit_position = raycast_shadow.get_collision_point()
	
	var world_hit_pos = hit_position.y - shadow.get_parent().get_parent().get_pos().y
	var ray_length = abs(raycast_shadow.get_pos().y + shadow.get_parent().get_parent().get_pos().y - hit_position.y + 12.7)/300
	
	if ray_length <= 0:
		ray_length = 0.0
	elif ray_length >= 1:
		ray_length = 1.0	
	var ray_angle = atan2(raycast_shadow.get_collision_normal().x,raycast_shadow.get_collision_normal().y)
	
	
	if ORIENTATION == "right":
		shadow.set_rot(ray_angle)
	elif ORIENTATION == "left":
		shadow.set_rot(-1*ray_angle)
	
	if hit_position != null:
		shadow.set_pos(Vector2(0,world_hit_pos))
	
	if raycast_shadow.is_colliding():
		shadow.set_opacity((1-ray_length)*0.7)
	else:
		shadow.set_opacity(0.0)


### rotates the sprite in given direction
func rotate_sprite(direction):
	var body = get_node("body_tiles")
	if direction == "left":
		body.set_scale(Vector2(-1,1))
		ORIENTATION = "left"
	elif direction == "right":
		body.set_scale(Vector2(1,1))
		ORIENTATION = "right"


func move(move_speed, acceleration,delta):
	#current_speed = current_speed * acceleration - move_speed * (1-acceleration)
	current_speed = lerp(current_speed,-move_speed,acceleration*delta)
	set_linear_velocity(Vector2(current_speed,get_linear_velocity().y))

func rotation():
	current_rot = lerp(current_rot,0,3.0*delta)	
	get_node("body_tiles/rotation").set_rot(current_rot)

func movement(acceleration):
	if btn_left == 2 and controls_lock_timer.get_time_left() == 0:
		rotate_sprite("left")
		move(speed,acceleration,delta)
	elif btn_right == 2 and controls_lock_timer.get_time_left() == 0:
		rotate_sprite("right")
		move(-speed,acceleration,delta)
	elif (btn_left == 0 and btn_right == 0) or controls_lock_timer.get_time_left() > 0:
		move(0,acceleration,delta)

func jump(force=1100,radians=0):
	var jump_vector = Vector2(0,-force)
	jump_vector = jump_vector.rotated(radians)
	set_axis_velocity(jump_vector)
	
func jump2(force=1100,radians=0):
	var jump_vector = Vector2(0,-force)
	jump_vector = jump_vector.rotated(radians)
	set_linear_velocity(jump_vector)
		
func walljump(force):
	set_applied_force(Vector2(0,-force))	

func die():
	var world = get_scene().get_root().get_node("world")
	self.set_pos(world.checkpoint.get_pos())
	self.set_linear_velocity(Vector2(0,0))

func fb_hit(fb):
	if fb.lifetime > 0.05:
		if get_pos().x < fb.get_pos().x:
			rotate_sprite("right")
		else:
			rotate_sprite("left")

		if fb.get_node("AnimationPlayer 2").get_current_animation() != "explode":
			PLAYERSTATE_NEXT = "die"



func _ready():
	# Initalization here
	### camera zoom
	var viewport_scale = get_node("/root/global").viewport_scale
	get_node("Camera2D").set_zoom(Vector2(1.3*viewport_scale,1.3*viewport_scale))
	
################################################################################################################################### initiliaze raycasts
	raycast_left = get_node("body_tiles/raycast_left")
	raycast_center = get_node("body_tiles/raycast_right")
	raycast_right = get_node("body_tiles/raycast_right")
	raycast_front = get_node("body_tiles/raycast_front")
	raycast_shadow = get_node("body_tiles/raycast_shadow")
	raycast_border_check = get_node("body_tiles/raycast_border_check")
	raycast_border_climb = get_node("body_tiles/raycast_border_climb")
	
	raycast_left.add_exception(self)
	raycast_center.add_exception(self)
	raycast_right.add_exception(self)
	raycast_left.add_exception(get_node("/root/world/ForeGround/chest"))
	raycast_center.add_exception(get_node("/root/world/ForeGround/chest"))
	raycast_right.add_exception(get_node("/root/world/ForeGround/chest"))
	raycast_front.add_exception(self)
	raycast_shadow.add_exception(self)
	raycast_shadow.add_exception(get_node("/root/world/ForeGround/chest"))
	raycast_border_check.add_exception(self)
	raycast_border_climb.add_exception(self)

################################################################################################################################### initiliaze timer	
	wallslide_timer = get_node("wallslide_timer")
	controls_lock_timer = get_node("controls_lock_timer")
	jump_pad_timer = get_node("jump_pad_timer")
	
	
	connect("body_enter",self,"on_body_enter")
	
	set_fixed_process(true)	
	
	
func _fixed_process(delta2):	
	delta = delta2
	animation_player = get_node("sprite_anims")
################################################################################################################################### update button inputs
	btn_jump = jump.check()
	btn_left = move_left.check()
	btn_right = move_right.check()
	
################################################################################################################################### state independent
	ORIENTATION_PREV = ORIENTATION
	### drop shadow onto the ground
	drop_shadow()
	
	
	### apply extra gravity
	set_applied_force(Vector2(0,gravity))

	
	PLAYERSTATE_PREV = PLAYERSTATE
	PLAYERSTATE = PLAYERSTATE_NEXT




########################################################################### State Handling
	if( PLAYERSTATE == "ground"):
		ground_state()
	elif( PLAYERSTATE == "air"):
		air_state()
	elif( PLAYERSTATE == "wall"):
		wall_state()
	elif( PLAYERSTATE == "jump"):
		jump_state()
	elif( PLAYERSTATE == "die"):
		die_state()
	elif( PLAYERSTATE == "respawn"):
		respawn_state()
	elif( PLAYERSTATE == "jump_pad"):
		jump_pad_state()
		
########################################################################### Animation Player
	if (sprite_anim_new!=sprite_anim):
		sprite_anim_new = sprite_anim
		animation_player.play(sprite_anim,blendtime,anim_speed)	
		animation_player.seek(0.0)
		

################################################################################################################################### respawn state function
func respawn_state():
	set_linear_velocity(Vector2(0,0))
	if PLAYERSTATE_PREV != "respawn":
		set_linear_velocity(Vector2(0,0))
		controls_lock_timer.start()
	var world = get_scene().get_root().get_node("world")
	set_pos(world.checkpoint.get_pos())
	
	if raycast_left.is_colliding() or raycast_right.is_colliding():
		PLAYERSTATE_NEXT = "ground"
	else:
		PLAYERSTATE_NEXT = "air"
	


################################################################################################################################### die state function
func die_state():
	sprite_anim = "die"
	blendtime = .1
	anim_speed = 1.5
	
	if (not raycast_left.is_colliding() and not raycast_right.is_colliding()):
		if animation_player.get_current_animation_pos() >= .8:
			animation_player.seek(.8,true)
	
	if animation_player.get_current_animation() == "die":
		if animation_player.get_current_animation_pos() <= .8:
			if ORIENTATION == "left":
				set_axis_velocity(Vector2(600,0))
			if ORIENTATION == "right":
				set_axis_velocity(Vector2(-600,0))
		else:
			set_linear_velocity(Vector2(0,get_linear_velocity().y))
		
		if animation_player.get_current_animation_pos() >= 4:
			PLAYERSTATE_NEXT = "respawn"
	
################################################################################################################################### state damage

func jump_state():
	jump(jump_force)
	PLAYERSTATE_NEXT = "air"

func damage_state():
	sprite_anim = "damage"
	blendtime = .1
	
	if animation_player.get_current_animation() == "damage":
		if animation_player.get_current_animation_pos() <= .3:
			if ORIENTATION == "left":
				set_axis_velocity(Vector2(500,0))
			if ORIENTATION == "right":
				set_axis_velocity(Vector2(-500,0))
		else:
			set_linear_velocity(Vector2(0,get_linear_velocity().y))

		
		if animation_player.get_current_animation_pos() >= .5:		
			if(raycast_left.is_colliding() or raycast_right.is_colliding()):
				PLAYERSTATE_NEXT = "ground"
			else:
				PLAYERSTATE_NEXT = "air"



################################################################################################################################### state ground

func jump_pad_state():
	rotation()
	
	if controls_lock_timer.get_time_left() == 0:
		if btn_left == 2 or btn_right == 2:
			PLAYERSTATE_NEXT = "air"

	
	if btn_jump == 1 and JUMPSTATE == 1:
		jump(jump_force)
		JUMPSTATE += 1

		anim_speed = 1.3
		sprite_anim = "salto"
		animation_player.play(sprite_anim)
	
	
	if (self.get_linear_velocity().y < -1.0) and sprite_anim != "salto":
		sprite_anim = "jump_up"
		blendtime = .1
		anim_speed = 1.0
	if (self.get_linear_velocity().y > 1.0) and sprite_anim != "salto":
		sprite_anim = "jump_down"
		blendtime = .5
		anim_speed = 1.0

	if jump_pad_timer.get_time_left() == 0:
		PLAYERSTATE_NEXT = "air"

################################################################################################################################### state ground

func ground_state():
	rotation()
	
	if PLAYERSTATE_PREV != "ground":
		JUMPSTATE = 0
		if animation_player.get_current_animation() == "salto" and get_linear_velocity().y > 0:
			animation_player.seek(0.0)

		
	movement(ground_acceleration)
	if btn_left == 2:
		sprite_anim = "run"
		anim_speed = 1.0
		blendtime = .2
	elif btn_right == 2:
		sprite_anim = "run"
		blendtime = .2
		anim_speed = 1.0
	elif (btn_left == 0 and btn_right == 0):
		
		if animation_player.get_current_animation() != "salto":
			if raycast_border_check.is_colliding():
				sprite_anim = "idle"
			else:
				sprite_anim = "border"
		anim_speed = 1.0
		if PLAYERSTATE_PREV == "air":
			blendtime = 0.1
		else:	
			blendtime = 0.4


	if btn_jump == 1:
		jump(jump_force)
		JUMPSTATE = 1
			
	if not raycast_left.is_colliding() and not raycast_right.is_colliding():	
		PLAYERSTATE_NEXT = "air"
	

################################################################################################################################### state air
func air_state():
	rotation()
	
	if PLAYERSTATE_PREV != "air":
		JUMPSTATE = 1
	
	movement(air_acceleration)	
	
	if btn_jump == 1 and JUMPSTATE == 1 and controls_lock_timer.get_time_left() < 0.2:
		jump(jump_force)
		JUMPSTATE += 1

		anim_speed = 1.3
		sprite_anim = "salto"
		animation_player.play(sprite_anim)
	
	
	if animation_player.get_current_animation_pos() >= 1.5 and animation_player.get_current_animation() == "salto":
		sprite_anim = "jump_up"
		anim_speed = 1.0

	if (self.get_linear_velocity().y < -1.0) and sprite_anim != "salto":
		sprite_anim = "jump_up"
		blendtime = .1
		anim_speed = 1.0
	if (self.get_linear_velocity().y > 1.0) and sprite_anim != "salto":
		sprite_anim = "jump_down"
		blendtime = .5
		anim_speed = 1.0
		
	if raycast_left.is_colliding() and raycast_right.is_colliding():
		PLAYERSTATE_NEXT = "ground"
	elif raycast_front.is_colliding():
		if raycast_front.get_collider().is_in_group("wall"):
			PLAYERSTATE_NEXT = "wall"
			blendtime = 0.0
			
		
################################################################################################################################### state wall
func wall_state():
	anim_speed = 1.0
	blendtime = .05
	#sprite_anim = "wallstick"
	if (PLAYERSTATE == "wall" and PLAYERSTATE_PREV != "wall"):
		if animation_player.get_current_animation() == "salto":
			animation_player.seek(0.1)
		wallslide_timer.start()
	
	### if player touches the border he climb
	if not raycast_border_climb.is_colliding():
		blendtime = .1
		anim_speed = 1.0
		sprite_anim = "wallstick"
		self.set_linear_velocity(Vector2(0,-42))
		if ORIENTATION == "left":
			if wallslide_timer.get_time_left() > 0:
				set_axis_velocity(Vector2(-200,0))
		if ORIENTATION == "right":
			if wallslide_timer.get_time_left() > 0:
				set_axis_velocity(Vector2(200,0))
	
	
	if raycast_border_climb.is_colliding() and get_linear_velocity().y > -500:
		sprite_anim = "wallstick"
		anim_speed = 1.0
		blendtime = .1

		if get_linear_velocity().y < 0:
			wallslide_timer.start()
			
		if ORIENTATION == "left":
			if wallslide_timer.get_time_left() > 0:
				set_linear_velocity(Vector2(-200,0))
		if ORIENTATION == "right":
			if wallslide_timer.get_time_left() > 0:
				set_linear_velocity(Vector2(200,0))

	elif raycast_border_climb.is_colliding():
		sprite_anim = "jump_up"	
		anim_speed = 1.0
	
	
	
	
	if ORIENTATION == "left" and controls_lock_timer.get_time_left() < 0.1:
		### make a walljump
		if btn_jump == 1:
			controls_lock_timer.start()
			rotate_sprite("right")
	elif ORIENTATION == "right" and controls_lock_timer.get_time_left() < 0.1:
		### make a walljump
		if btn_jump == 1:
			controls_lock_timer.start()
			rotate_sprite("left")
			
	if ORIENTATION_PREV == ORIENTATION and not raycast_front.is_colliding():
		PLAYERSTATE_NEXT = "air"
		if jump.state_old == 1:
			if ORIENTATION == "left":
				move(speed*2.0,50,delta)
				jump(jump_force*1.2)
			elif ORIENTATION == "right":
				move(-speed*2.0,50,delta)
				jump(jump_force*1.2)	
			
	elif raycast_left.is_colliding() or raycast_right.is_colliding():
		PLAYERSTATE_NEXT = "ground"


func on_body_enter( body ):
	pass
	