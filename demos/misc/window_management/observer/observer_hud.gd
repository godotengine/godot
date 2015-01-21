var parent

func printdebug():
	
	var s
	
	if(parent.state == parent.STATE_GAME):
		s = str( "TIME_FPS: ", Performance.get_monitor(Performance.TIME_FPS), "\n")
		s += str("OBJECT_COUNT: ", Performance.get_monitor(Performance.OBJECT_COUNT), "\n")
		s += str("OBJECT_RESOURCE_COUNT : ", Performance.get_monitor(Performance.OBJECT_RESOURCE_COUNT), "\n")
		s += str("OBJECT_NODE_COUNT : ", Performance.get_monitor(Performance.OBJECT_NODE_COUNT), "\n")
		s += str("RENDER_OBJECTS_IN_FRAME : ", Performance.get_monitor(Performance.RENDER_OBJECTS_IN_FRAME), "\n")
		s += str("RENDER_VERTICES_IN_FRAME : ", Performance.get_monitor(Performance.RENDER_VERTICES_IN_FRAME), "\n")
		s += str("RENDER_DRAW_CALLS_IN_FRAME : ", Performance.get_monitor(Performance.RENDER_DRAW_CALLS_IN_FRAME), "\n")
		s += str("RENDER_VERTICES_IN_FRAME : ", Performance.get_monitor(Performance.RENDER_VERTICES_IN_FRAME), "\n")
	#	s += str("RENDER_USAGE_VIDEO_MEM_TOTAL  : ", Performance.get_monitor(Performance.RENDER_USAGE_VIDEO_MEM_TOTAL), "\n")
	#	s += str("RENDER_VIDEO_MEM_USED : ", Performance.get_monitor(Performance.RENDER_VIDEO_MEM_USED), "\n")
	#	s += str("RENDER_TEXTURE_MEM_USED : ", Performance.get_monitor(Performance.RENDER_TEXTURE_MEM_USED), "\n")
	#	s += str("RENDER_VERTEX_MEM_USED : ", Performance.get_monitor(Performance.RENDER_VERTEX_MEM_USED), "\n")
		s += str("CUBES: ", get_node("/root/World").world.size(), "\n")
	else:
		s = ""
	
	get_node("Label_Debug").set_text(s)


func _fixed_process(delta):
	parent = get_parent()
	
	printdebug()
	
	if( parent.state == parent.STATE_MENU ):
		get_node("Menu").show()
	else:
		get_node("Menu").hide()
	


func _ready():
	set_fixed_process(true)


func _on_Fullscreen_toggled( pressed ):
	if( pressed ):
		OS.set_fullscreen(true)
	else:
		OS.set_fullscreen(false)


func _on_DebugInfo_toggled( pressed ):
	if( pressed ):
		get_node("Label_Debug").show()
	else:
		get_node("Label_Debug").hide()


func _on_Save_pressed():
	var file_dialog = get_node("Menu/SaveDialog")
	file_dialog.clear_filters()
	file_dialog.add_filter("*.json")
	file_dialog.set_mode(3)
	file_dialog.show()
	file_dialog._update_file_list()


func _on_SaveDialog_file_selected( path ):
	get_node("/root/World").save_world( path )


func _on_Load_pressed():
	var file_dialog = get_node("Menu/LoadDialog")
	file_dialog.clear_filters()
	file_dialog.add_filter("*.json")
	file_dialog.set_mode(0)
	file_dialog.show()
	file_dialog._update_file_list()


func _on_LoadDialog_file_selected( path ):
	get_node("/root/World").load_world( path )


func _on_Server_toggled( pressed ):
	if pressed:
		get_node("/root/World/Server").start()
		get_node("Menu/Client").hide()
	else:
		get_node("/root/World/Server").stop()
		get_node("Menu/Client").show()


func _on_Client_toggled( pressed ):
	if pressed:
		get_node("/root/World/Client").start()
		get_node("Menu/Server").hide()
	else:
		get_node("/root/World/Client").stop()
		get_node("Menu/Server").show()