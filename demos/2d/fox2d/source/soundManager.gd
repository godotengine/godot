extends StreamPlayer

# Manager for sound and music.
# It plays a background music in loop and sound effects.
# Its main purpose is to be able to not interrupt the music or the sound effect when changing scenes.
# And it's a singleton, so it's centralized.

# list of musics
const tracks={
	"greens":"Winds Of Stories.ogg",
	"boss":"battle1.ogg"
}

# currently played music
var current=""

# load play a new music.
# String track : name of the music to play. If null, empty or invalid, the music stop.
func play_track(track):
	if(current!=track):
		stop()
		current=track
		if(track!=null and track!=""):
			var filename=tracks[track]
			if(filename!=null):
				var stream=load("res://audio/"+filename)
				set_stream(stream)
				play()

# play a sound effect from the sound catalog.
# String name : name of the sound effect
func play_sfx(name):
	var id=get_node("sfx").play(name)