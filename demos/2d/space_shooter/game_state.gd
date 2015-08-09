extends Node


var points = 0
var max_points = 0


func _ready():
	var f = File.new()
	#load high score
	
	if (f.open("user://highscore",File.READ)==OK):
		
		max_points=f.get_var()


func game_over():
	if (points>max_points):
		max_points=points
		#save high score
		var f = File.new()
		f.open("user://highscore",File.WRITE)
		f.store_var(max_points)
		