extends Button

var feed:CameraFeed

# Called when the node enters the scene tree for the first time.
func _ready():	
	var feeds = CameraServer.feeds()		
	if (feeds.is_empty()):
		text = "NO CAMERA"
	else:
		text = "START"
		_on_texture_rect_property_list_changed()
	

func _toggled(toggled_on):
	feed.feed_is_active = toggled_on
	if (toggled_on):
		text = "STOP"
	else:
		text = "START"


func _on_texture_rect_property_list_changed():
	if (feed != null && feed.feed_is_active): _toggled(false)
	
	var texture_rect = get_node("/root/Control/TextureRect")
	var camera_id = texture_rect.texture.get_camera_feed_id()
	feed = CameraServer.get_feed_by_id(camera_id)
