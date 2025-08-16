extends  'res://addons/gut/hook_script.gd'

var run_called = false

func run():
	run_called = true
	print('!! --- post-run script ran --- !!')
