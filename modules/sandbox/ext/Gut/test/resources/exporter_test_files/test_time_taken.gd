extends 'res://addons/gut/test.gd'

func more_accurate_wait_time(time_to_wait_msec : int)->void:
	gut._lgr.yield_msg(str('-- Awaiting ', time_to_wait_msec / 1000.0, ' second(s) -- '))
	var start := Time.get_ticks_msec()
	while (Time.get_ticks_msec() - start < time_to_wait_msec):
		await get_tree().process_frame

func test_pass_time_taken_about_half_s():
	await more_accurate_wait_time(500)
	assert_eq('one', 'one')

func test_fail_time_taken_about_half_s():
	await more_accurate_wait_time(500)
	assert_eq(1, 'two')

func test_pending_time_taken_about_half_s():
	await more_accurate_wait_time(500)
	pending('this has text')

func test_pass_time_taken_about_2s():
	await more_accurate_wait_time(2000)
	assert_eq('one', 'one')
