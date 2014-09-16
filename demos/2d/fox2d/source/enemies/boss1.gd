extends Node2D

# Class for boss of level 1
# This class is a special kind of enemy, because its movement pattern is based on multiple animations, it can't be sucked and it has a life bar.
# A boss is typically killed by star bullets and he moves in a random sequence of animations, without considering where the player is.
# a boss can spawn missiles too, like bombs, common enemies and others stuffs. Each boss is unique, and this one is jumping from places to places and throws bombs.

# This boss has a default, idling, animation that he plays a couple of times in a row before he does another movement. After one movement, he goes back to idling a few times. And so on.

# Constants -------------------------------------------------------------------------
const BOMB_SPEED=400 # initial speed which the bomb is thrown

# Classes --------------------------------------------------------------------------
var bomb_class = preload("res://bomb.res")

# Variables --------------------------------------------------------------------------------
var _current_seq=0                                                 # current movement animation
var _waiting_time=0                                                # waiting time before next animation
var animations=["Idle","BombAttack1","BombAttack2","TripleJump"]   # list of available movement animations
var lastTimeHit=0                                                  # timeout for invulnerability when hit by a star

# Functions -----------------------------------------------------------------------------

# Initializer
func _ready():
	get_node("/root/gamedata").set_boss_life(3)                   # set boss' life to 3 points
	get_node("/root/gamedata").set_boss_bar_visibility(true)      # show his life bar
	_waiting_time=2+randi()%3                                     # define an initial waiting time for idling
	set_sequence()                                                # starts his movement
	

# event when the boss get hit
func hit():
	# check if boss still invulnerable from last hit (1[s] invulnerability)
	var timeHit=OS.get_ticks_msec()
	if(timeHit-lastTimeHit>1000):
	    # if not, hit the boss and make him invulnerable
		lastTimeHit=timeHit
		get_node("/root/gamedata").dec_boss_life()
	# check if boss is dead
	if(get_node("/root/gamedata").get_boss_life()<=0):
		# if so, make him explode
		get_node("anim").play("explosion")
		get_node("/root/soundMgr").play_sfx("boss_hit1")

# Die event called by animation. Also tells the level to change active camera and update the HUD
func _die():
	get_scene().get_nodes_in_group("player")[0].focusCamera()
	get_node("/root/gamedata").set_boss_bar_visibility(false)
	queue_free()

# play the next movement animation sequence
func next_sequence():
	# check if boss was idling
	if(_current_seq>0):
		# if no, start idling for a random time
		_waiting_time=2+randi()%3
		_current_seq=0
	else:
		# if yes, check if idling time is over
		_waiting_time-=1
		if(_waiting_time<=0):
			# if it is, start a movement animation randomly
			_current_seq=(randi() % 3)+1
	set_sequence() # start the finally decided sequence

# setter for the movement animation
func set_sequence():
	get_node("anim").play(animations[_current_seq])

# spawn a bomb
# Vector2 velocity : initial velocity of the bomb. It specify the direction and the speed.
func create_bomb(velocity):
	var bomb_instance=bomb_class.instance()
	var new_pos=get_node("actor_boss/bombSpawnPos").get_global_pos()
	bomb_instance.set_pos(new_pos)
	bomb_instance.set_velocity(velocity)
	get_parent().add_child(bomb_instance)

# spawn a bomb horizontally, sliding along the ground
func launch_bomb_ground():
	create_bomb(Vector2(-1*BOMB_SPEED,0))

# spawn a bomb in diagonal, when the boss is in the middle of a jump
func launch_bomb_air():
	create_bomb(Vector2(-1*BOMB_SPEED,BOMB_SPEED))
