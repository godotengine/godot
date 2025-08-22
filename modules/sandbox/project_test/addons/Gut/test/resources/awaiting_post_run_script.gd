extends 'res://addons/gut/hook_script.gd'

var awaited = false

func run():
	print('!! --- post-run script awaiting --- !!')
	
	var awaiter = GutUtils.Awaiter.new()
	gut.add_child(awaiter)
	awaiter.wait_seconds(1)
	await awaiter.timeout
	awaited = true
	awaiter.queue_free()
	
	print('!! --- post-run script has successfully awaited --- !!')
