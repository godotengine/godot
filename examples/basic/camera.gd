extends Control

# Called when the node enters the scene tree for the first time.
func _ready():
	var label: Label = $Label
	
	var feeds = CameraServer.feeds()
	if feeds.is_empty():
		label.text = "No cameras found"
		return
	
	label.text = "%d cameras found!" % feeds.size()
	
	for feed in CameraServer.feeds():
		print(feed.get_id(), ": ", feed.get_name())
	
	var feed = CameraServer.get_feed(0);
	feed.feed_is_active = true
