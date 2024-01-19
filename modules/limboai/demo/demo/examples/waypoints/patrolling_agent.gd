#*
#* patrolling_agent.gd
#* =============================================================================
#* Copyright 2021-2023 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*

extends CharacterBody2D


@onready var bt_player: BTPlayer = $BTPlayer


func _ready() -> void:
	bt_player.blackboard.set_var("waypoints", [])


func add_waypoint(p_waypoint: Vector2) -> void:
	(bt_player.blackboard.get_var("waypoints") as Array).append(p_waypoint)
