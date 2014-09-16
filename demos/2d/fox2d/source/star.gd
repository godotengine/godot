extends KinematicBody2D

# class for the bullet in shape of a star.
# It's a bullet that the player can shoot when charged. It hurt enemies and destructible objects.
# It has a simple pattern of going straight forward without falling. But it collides with the terrain too.
# When it collides with terrain or a destructible object, a specific animation is played, and then it dies.

# Classes ----------------------------------------------------------
const enemy_class = preload("enemies/enemy.gd")
const MAXIMUM_LIFESPAN=10  # maximum lifespan, in [s]

# Variables ---------------------------------------------------
var age=0       # age of the object, in [s]. If going out of screen, it must die after maximum lifespan.
var velocity=0

# Functions ----------------------------------------------------------
# for animation
func _die():
	queue_free()

# Initializer
func _ready():
	set_fixed_process(true)

# move and check lifespan
func _fixed_process(delta):
	age=age+delta
	if(age>MAXIMUM_LIFESPAN):
		queue_free()
	else:
		var motion = velocity * delta
		motion = move(motion)
		if (is_colliding()):
			velocity=Vector2(0,0)
			get_node("anim").play("die")
			set_fixed_process(false) # disable fixed process to avoid infinite dying loop

# setter for speed
# Vector2 vel : velocity of the star
func set_linear_velocity(vel):
	velocity=vel

# detect collision with destructible objects
func _on_Area2D_body_enter( body ):
	if(body extends enemy_class):
		body._die()
		velocity=Vector2(0,0)
		get_node("anim").play("explode")
		get_node("/root/soundMgr").play_sfx("enemy_hit")
	if("is_enemy" in body):
		body.hit()
		velocity=Vector2(0,0)
		get_node("anim").play("explode")
		get_node("/root/soundMgr").play_sfx("enemy_hit")

