extends KinematicBody2D

const is_enemy=true

func hit():
	get_parent().hit()