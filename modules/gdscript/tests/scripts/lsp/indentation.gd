extends Node

var root = 0
#   ^^^^ 0_indent -> 0_indent

func a():
	var alpha: int = root + 42
	#   |    |       ^^^^ -> 0_indent
	#   ^^^^^ 1_indent -> 1_indent
	if alpha > 42:
	#  ^^^^^ -> 1_indent
		var beta := alpha + 13
		#   |  |    ^^^^ -> 1_indent
		#   ^^^^ 2_indent -> 2_indent
		if beta > alpha:
		#  |  |   ^^^^^ -> 1_indent
		#  ^^^^ -> 2_indent
			var gamma = beta + 1
			#   |   |   ^^^^ -> 2_indent
			#   ^^^^^ 3_indent -> 3_indent
			print(gamma)
			#     ^^^^^ -> 3_indent
		print(beta)
		#     ^^^^ -> 2_indent
	print(alpha)
	#     ^^^^^ -> 1_indent
	print(root)
	#     ^^^^ -> 0_indent
