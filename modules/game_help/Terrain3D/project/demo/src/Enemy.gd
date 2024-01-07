extends CharacterBody3D

const RETARGET_COOLDOWN: float = 1.0

@export var MOVE_SPEED: float = 50.0
@export var target: Node3D

@onready var nav_agent: NavigationAgent3D = $NavigationAgent3D

var _retarget_timer: float = 1.0


func _ready() -> void:
	nav_agent.velocity_computed.connect(_on_velocity_computed)


func _process(p_delta: float) -> void:
	_retarget_timer += p_delta
	if _retarget_timer > RETARGET_COOLDOWN and target:
		# Don't reset the target position every frame. It triggers an A* search, which is expensive.
		_retarget_timer = 0.0
		nav_agent.set_target_position(target.global_position)


func is_on_nav_mesh() -> bool:
	var closest_point := NavigationServer3D.map_get_closest_point(nav_agent.get_navigation_map(), global_position)
	return global_position.distance_squared_to(closest_point) < nav_agent.path_max_distance ** 2


func _physics_process(p_delta: float) -> void:
	if nav_agent.is_navigation_finished():
		velocity.x = 0.0
		velocity.z = 0.0
	else:
		var next_path_position: Vector3 = nav_agent.get_next_path_position()
		var current_agent_position: Vector3 = global_position
		var velocity_xz := (next_path_position - current_agent_position).normalized() * MOVE_SPEED
		velocity.x = velocity_xz.x
		velocity.z = velocity_xz.z
	
	velocity.y -= 40 * p_delta
	
	if nav_agent.avoidance_enabled:
		nav_agent.set_velocity(velocity)
	else:
		_on_velocity_computed(velocity)


func _on_velocity_computed(p_safe_velocity: Vector3) -> void:
	velocity.x = p_safe_velocity.x
	velocity.z = p_safe_velocity.z
	move_and_slide()
