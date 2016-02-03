# Demo for reproduce the bad behaviour of animation player when moving to another node, executes his ready again internally and stop the animation
extends Node2D
var traveler
var containerA
var containerB
var pauseButton
var addRemoveButton
var travelerScn
var collisionerScn
var collisioner
func _ready():
	travelerScn = preload("res://traveler.scn")
	collisionerScn = preload("res://collisioner.scn")
	pauseButton = get_node("hud/pause")
	addRemoveButton = get_node("hud/addRemove")
	containerA = get_node("containerA")
	containerB = get_node("containerB")
	var travelers = get_tree().get_nodes_in_group("traveler")
	if(travelers.size() > 0):
		traveler = travelers[0]
		addRemoveButton.set_text("REMOVE")
	else:
		traveler = null
		addRemoveButton.set_text("ADD")
	set_process_input(true)
#	containerB.get_parent().print_tree()
#	print("------------------------------")
	OS.set_window_size(Vector2(960,540))
	
func _input(event):
	if(traveler != null):
		if(event.type == InputEvent.MOUSE_BUTTON):
			if(event.button_index == 2 && event.is_pressed()):
				var pos = event.pos
				if(pos.y > 50):
					if(!collisioner):
						collisioner = collisionerScn.instance()
						traveler.add_child(collisioner)
					collisioner.set_pos(pos-traveler.get_parent().get_pos())
				
			
func _on_move_pressed():
	if(traveler != null):
		if(traveler.get_parent() == containerA):
			traveler.get_parent().remove_child(traveler)
			containerB.add_child(traveler)
	#		traveler.set_rot(traveler.get_rot())
	#		traveler.get_node("Camera2D").force_update_scroll()
	#		traveler.get_node("Camera2D").make_current()
	#		traveler.reparent(containerB)
	#		containerB.get_parent().print_tree()
	#		print("name: " + traveler.get_name())
	#		print("------------------------------")
		else:
			traveler.get_parent().remove_child(traveler)
			containerA.add_child(traveler)
	#		traveler.set_rot(traveler.get_rot())
	#		traveler.get_node("Camera2D").force_update_scroll()
	#		traveler.get_node("Camera2D").make_current()
	#		traveler.reparent(containerA)
	#		containerA.get_parent().print_tree()
	#		print("name: " + traveler.get_name())
	#		print("------------------------------")

func _on_addRemove_pressed():
	if(traveler == null):
		traveler = travelerScn.instance()
		containerA.add_child(traveler)
		addRemoveButton.set_text("REMOVE")
	else:
		traveler.queue_free()
		traveler = null
		collisioner = null
		addRemoveButton.set_text("ADD")


func _on_pause_pressed():
	if(get_tree().is_paused()):
		get_tree().set_pause(false)
		pauseButton.set_text("PAUSE")
	else:
		get_tree().set_pause(true)
		pauseButton.set_text("RESUME")
		
