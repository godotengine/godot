#*
#* example_waypoints.gd
#* =============================================================================
#* Copyright 2021-2023 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*

extends Node2D

@onready var agent1: CharacterBody2D = $Agent
@onready var agent2: CharacterBody2D = $Agent2


func _ready() -> void:
	var waypoints: Array[Node] = $Waypoints.get_children()

	for wp in waypoints:
		agent1.add_waypoint(wp.global_position)

	waypoints.reverse()
	for wp in waypoints:
		agent2.add_waypoint(wp.global_position)


