
extends Node2D

var animator
var timer
var timer_label
var first_time = true
func _ready():
	if(first_time):
		animator = get_node("AnimationPlayer")
		#animator.queue("spawn")
		#animator.queue("loop")
		timer = get_node("Timer")
		#timer.start()
		timer_label = get_node("labelnode/timerLabel")
		first_time = false
		set_fixed_process(true)
		set_process(true)

func _notification(what):
	if(what != NOTIFICATION_FIXED_PROCESS && what != NOTIFICATION_PROCESS):
		if(what == NOTIFICATION_ENTER_TREE):
			print("NOTIFICATION_ENTER_TREE")
		if(what == NOTIFICATION_EXIT_TREE):
			print("NOTIFICATION_EXIT_TREE")
		if(what == NOTIFICATION_READY):
			print("NOTIFICATION_READY")
		#if(what == NOTIFICATION_REPARENTING):
		#	print("NOTIFICATION_REPARENTING")
		#if(what == NOTIFICATION_REPARENTED):
		#	print("NOTIFICATION_REPARENTED")
		if(what == NOTIFICATION_PARENTED):
			print("NOTIFICATION_PARENTED")
		if(what == NOTIFICATION_POSTINITIALIZE):
			print("NOTIFICATION_POSTINITIALIZE")
		if(what == NOTIFICATION_UNPARENTED):
			print("NOTIFICATION_UNPARENTED")

func _fixed_process(delta):
	timer_label.set_text(str("TIMER:") + str(round(timer.get_time_left())))

func _process(delta):
	pass
