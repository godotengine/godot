extends Node

var lambda_member1 := func(alpha: int, beta): return alpha + beta
#   |            |         |   |       |  |          |   |   ^^^^ -> \1:beta
#   |            |         |   |       |  |          ^^^^^ -> \1:alpha
#   |            |         |   |       ^^^^ \1:beta -> \1:beta
#   |            |         ^^^^^ \1:alpha -> \1:alpha
#   ^^^^^^^^^^^^^^ \1 -> \1

var lambda_member2 := func(alpha, beta: int) -> int: 
#   |            |         |   |  |  |
#   |            |         |   |  |  |
#   |            |         |   |  ^^^^ \2:beta -> \2:beta
#   |            |         ^^^^^ \2:alpha -> \2:alpha
#   ^^^^^^^^^^^^^^ \2 -> \2
	return alpha + beta
	#      |   |   ^^^^ -> \2:beta
	#      ^^^^^ -> \2:alpha

var lambda_member3 := func add_values(alpha, beta): return alpha + beta
#   |            |                    |   |  |  |          |   |   ^^^^ -> \3:beta
#   |            |                    |   |  |  |          ^^^^^ -> \3:alpha
#   |            |                    |   |  ^^^^ \3:beta -> \3:beta
#   |            |                    ^^^^^ \3:alpha -> \3:alpha
#   ^^^^^^^^^^^^^^ \3 -> \3

var lambda_multiline = func(alpha: int, beta: int) -> int:
#   |              |        |   |       |  |
#   |              |        |   |       |  |
#   |              |        |   |       ^^^^ \multi:beta -> \multi:beta
#   |              |        ^^^^^ \multi:alpha -> \multi:alpha
#   ^^^^^^^^^^^^^^^^ \multi -> \multi
	print(alpha + beta)
	#     |   |   ^^^^ -> \multi:beta
	#     ^^^^^ -> \multi:alpha
	var tmp = alpha + beta + 42
	#   | |   |   |   ^^^^ -> \multi:beta
	#   | |   ^^^^^ -> \multi:alpha
	#   ^^^ \multi:tmp -> \multi:tmp
	print(tmp)
	#     ^^^ -> \multi:tmp
	if tmp > 50:
	#  ^^^ -> \multi:tmp
		tmp += alpha
		# |    ^^^^^ -> \multi:alpha
		#<^ -> \multi:tmp
	else:
		tmp -= beta
		# |    ^^^^ -> \multi:beta
		#<^ -> \multi:tmp
	print(tmp)
	#     ^^^ -> \multi:tmp
	return beta + tmp + alpha
	#      |  |   | |   ^^^^^ -> \multi:alpha
	#      |  |   ^^^ -> \multi:tmp
	#      ^^^^ -> \multi:beta


var some_name := "foo bar"
#   ^^^^^^^^^ member:some_name -> member:some_name

func _ready() -> void:
	var a = lambda_member1.call(1,2)
	#       ^^^^^^^^^^^^^^ -> \1
	var b = lambda_member2.call(1,2)
	#       ^^^^^^^^^^^^^^ -> \2
	var c = lambda_member3.call(1,2)
	#       ^^^^^^^^^^^^^^ -> \3
	var d = lambda_multiline.call(1,2)
	#       ^^^^^^^^^^^^^^^^ -> \multi
	print(a,b,c,d)

	var lambda_local = func(alpha, beta): return alpha + beta
	#   |          |        |   |  |  |          |   |   ^^^^ -> \local:beta
	#   |          |        |   |  |  |          ^^^^^ -> \local:alpha
	#   |          |        |   |  ^^^^ \local:beta -> \local:beta
	#   |          |        ^^^^^ \local:alpha -> \local:alpha
	#   ^^^^^^^^^^^^ \local -> \local
	
	var value := 42
	#   ^^^^^ local:value -> local:value
	var lambda_capture = func(): return value + some_name.length()
	#   |            |                  |   |   ^^^^^^^^^ -> member:some_name
	#   |            |                  ^^^^^ -> local:value
	#   ^^^^^^^^^^^^^^ \capture -> \capture

	var z = lambda_local.call(1,2)
	#       ^^^^^^^^^^^^ -> \local
	var x = lambda_capture.call()
	#       ^^^^^^^^^^^^^^ -> \capture
	print(z,x)
