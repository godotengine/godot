tool # Always declare as Tool, if it's meant to run in the editor.
extends EditorPlugin

var timer = null
var label = null

func _timeout():
	if (label):
		var time = OS.get_time()
		label.set_text(str(time.hour).pad_zeros(2)+":"+str(time.minute).pad_zeros(2)+":"+str(time.second).pad_zeros(2))

func get_name(): 
	return "The Time"


func _init():
	print("PLUGIN INIT")
	timer = Timer.new()
	add_child(timer)
	timer.set_wait_time(0.5)
	timer.set_one_shot(false)
	timer.connect("timeout",self,"_timeout")
 
func _enter_tree():
	label = Label.new()
	add_custom_control(CONTAINER_TOOLBAR,label)
	timer.start()
	
func _exit_tree():
	timer.stop()
	label.free()
	label=null
