
extends Control

# Simple Tetris-like demo, (c) 2012 Juan Linietsky
# Implemented by using a regular Control and drawing on it during the _draw() callback.
# The drawing surface is updated only when changes happen (by calling update())

# Member variables
var score = 0
var score_label = null

const MAX_SHAPES = 7

var block = preload("block.png")

var block_colors = [
	Color(1, 0.5, 0.5),
	Color(0.5, 1, 0.5),
	Color(0.5, 0.5, 1),
	Color(0.8, 0.4, 0.8),
	Color(0.8, 0.8, 0.4),
	Color(0.4, 0.8, 0.8),
	Color(0.7, 0.7, 0.7)]

var block_shapes = [
	[ Vector2(0, -1), Vector2(0, 0), Vector2(0, 1), Vector2(0, 2) ], # I
	[ Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1) ], # O
	[ Vector2(-1, 1), Vector2(0, 1), Vector2(0, 0), Vector2(1, 0) ], # S
	[ Vector2(1, 1), Vector2(0, 1), Vector2(0, 0), Vector2(-1, 0) ], # Z
	[ Vector2(-1, 1), Vector2(-1, 0), Vector2(0, 0), Vector2(1, 0) ], # L
	[ Vector2(1, 1), Vector2(1, 0), Vector2(0, 0), Vector2(-1, 0) ], # J
	[ Vector2(0, 1), Vector2(1, 0), Vector2(0, 0), Vector2(-1, 0) ]] # T

var block_rotations = [
	Matrix32(Vector2(1, 0), Vector2(0, 1), Vector2()),
	Matrix32(Vector2(0, 1), Vector2(-1, 0), Vector2()),
	Matrix32(Vector2(-1, 0), Vector2(0, -1), Vector2()),
	Matrix32(Vector2(0, -1), Vector2(1, 0), Vector2())]

var width = 0
var height = 0

var cells = {}

var piece_active = false
var piece_shape = 0
var piece_pos = Vector2()
var piece_rot = 0


func piece_cell_xform(p, er = 0):
	var r = (4 + er + piece_rot) % 4
	return piece_pos + block_rotations[r].xform(p)


func _draw():
	var sb = get_stylebox("bg", "Tree") # Use line edit bg
	draw_style_box(sb, Rect2(Vector2(), get_size()).grow(3))
	
	var bs = block.get_size()
	for y in range(height):
		for x in range(width):
			if (Vector2(x, y) in cells):
				draw_texture_rect(block, Rect2(Vector2(x, y)*bs, bs), false, block_colors[cells[Vector2(x, y)]])
	
	if (piece_active):
		for c in block_shapes[piece_shape]:
			draw_texture_rect(block, Rect2(piece_cell_xform(c)*bs, bs), false, block_colors[piece_shape])


func piece_check_fit(ofs, er = 0):
	for c in block_shapes[piece_shape]:
		var pos = piece_cell_xform(c, er) + ofs
		if (pos.x < 0):
			return false
		if (pos.y < 0):
			return false
		if (pos.x >= width):
			return false
		if (pos.y >= height):
			return false
		if (pos in cells):
			return false
	
	return true


func new_piece():
	piece_shape = randi() % MAX_SHAPES
	piece_pos = Vector2(width/2, 0)
	piece_active = true
	piece_rot = 0
	if (piece_shape == 0):
		piece_pos.y += 1
	
	if (not piece_check_fit(Vector2())):
		# Game over
		game_over()
	
	update()


func test_collapse_rows():
	var accum_down = 0
	for i in range(height):
		var y = height - i - 1
		var collapse = true
		for x in range(width):
			if (Vector2(x, y) in cells):
				if (accum_down):
					cells[Vector2(x, y + accum_down)] = cells[Vector2(x, y)]
			else:
				collapse = false
				if (accum_down):
					cells.erase(Vector2(x, y + accum_down))
		
		if (collapse):
			accum_down += 1
	
	score += accum_down*100
	score_label.set_text(str(score))


func game_over():
	piece_active = false
	get_node("gameover").set_text("Game over!")
	update()


func restart_pressed():
	score = 0
	score_label.set_text("0")
	cells.clear()
	get_node("gameover").set_text("")
	piece_active = true
	get_node("../restart").release_focus()
	update()


func piece_move_down():
	if (!piece_active):
		return
	if (piece_check_fit(Vector2(0, 1))):
		piece_pos.y += 1
		update()
	else:
		for c in block_shapes[piece_shape]:
			var pos = piece_cell_xform(c)
			cells[pos] = piece_shape
		test_collapse_rows()
		new_piece()


func piece_rotate():
	var adv = 1
	if (not piece_check_fit(Vector2(), 1)):
		return
	piece_rot = (piece_rot + adv) % 4
	update()


func _input(ie):
	if (not piece_active):
		return
	if (!ie.is_pressed()):
		return

	if (ie.is_action("move_left")):
		if (piece_check_fit(Vector2(-1, 0))):
			piece_pos.x -= 1
			update()
	elif (ie.is_action("move_right")):
		if (piece_check_fit(Vector2(1, 0))):
			piece_pos.x += 1
			update()
	elif (ie.is_action("move_down")):
		piece_move_down()
	elif (ie.is_action("rotate")):
		piece_rotate()


func setup(w, h):
	width = w
	height = h
	set_size(Vector2(w, h)*block.get_size())
	new_piece()
	get_node("timer").start()


func _ready():
	setup(10, 20)
	score_label = get_node("../score")
	
	set_process_input(true)
