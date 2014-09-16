extends Area2D

# Class managing the vacuum when the player starts to suck edible objects.

# Classes ---------------------------------------------------------------
const bonus_class = preload("res://bonus.gd")
const edible_class = preload("res://Edible.gd")

# Variables ---------------------------------------------------------------
var _player=null    # owner of the vacuum.
var count_in_cone=0 # number of objects actually being sucked. Used to avoid interrupting the vacuum when more than one edible object is being sucked at the same time.
var count_bloc=0    # number of being sucked objects that will charge the player.

# Functions ---------------------------------------------------------------

# Initializer
func _ready():
	# plays the sound, which automatically loops. 
	# Because the node sfx is part of the cone, when the cone is queue_freed, the sound stops at the same time.
	get_node("sfx").play("inhale")

# setter for the owner.
# Player player : owner of the vacuum.
func set_player(player):
	_player=player

# flips the root node to the right direction.
# boolean siding_left : true if the cone must side to the left, false if the right.
func set_direction(siding_left):
	if(siding_left):
		set_scale( Vector2(-1,1))

# trigger event when a KinematicBody2D node enters the vacuum area. If its edible, the object will be sucked.
# Node2D body : detected node.
func _on_root_body_enter( body ):
	if(body extends edible_class):
		# set the edible object in vacuum mode
		body.startMove(get_parent().get_pos())
		count_in_cone+=1
	

# trigger event when a KinematicBody2D node enters the mouth of the player. If edible, which was probably being sucked in the cone, 
# the object is "eaten", meaning it's destroyed and the player get either a charge or a bonus effect if it was a bonus.
# Node2D body : detected node.
func _on_eatArea_body_enter( body ):
	if(body extends edible_class):
		# if it's a bonus, give the bonus effect to the owner. Otherwise it's an enemy or something else and the player will get a charge.
		if(body extends bonus_class):
			_player.get_bonus(body.bonus_type)
		else:
			count_bloc+=1
		
		# kill the object
		body.queue_free()
		
		# check if it was the last object to eat. If so, interrupt the vacuum and charge the player if there was at least one non-bonus object.
		count_in_cone-=1
		if(count_in_cone<=0):
			if(count_bloc>0):
				_player.set_charged() # gives the player a charge. The player will interrupt the vacuum at the same time.
			count_in_cone=0 # the player might still suck after getting a bonus. better be sure that the counter has a valid value.
