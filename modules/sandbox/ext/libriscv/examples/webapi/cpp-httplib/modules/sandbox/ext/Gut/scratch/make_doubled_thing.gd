# Used with doubled_thing.gd.  This will create an instance of that to try and
# find issues.  doubled_thing.gd has source from a doubled object.
extends SceneTree

func _init():
    var DoubledThing = load('res://scratch/doubled_thing.gd')
    print(DoubledThing)
    var inst = DoubledThing.new()
    print(inst)
    quit()