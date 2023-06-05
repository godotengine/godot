# GH-74188

@export_enum("A", "B", "C") var v_enum
@export_file var v_file
@export_dir var v_dir
@export_global_file var v_global_file
@export_global_dir var v_global_dir
@export_multiline var v_multiline
@export_placeholder("Test") var v_placeholder
@export_range(1, 10) var v_range
@export_exp_easing var v_exp_easing
@export_color_no_alpha var v_color_no_alpha
@export_node_path var v_node_path
@export_flags("A", "B", "C") var v_flags
@export_flags_2d_render var v_flags_2d_render
@export_flags_2d_physics var v_flags_2d_physics
@export_flags_2d_navigation var v_flags_2d_navigation
@export_flags_3d_render var v_flags_3d_render
@export_flags_3d_physics var v_flags_3d_physics
@export_flags_3d_navigation var v_flags_3d_navigation
@export_flags_avoidance var v_flags_avoidance

# Variables are untyped, so null assignment is allowed.
@export_enum("A", "B", "C") var n_enum = null
@export_file var n_file = null
@export_dir var n_dir = null
@export_global_file var n_global_file = null
@export_global_dir var n_global_dir = null
@export_multiline var n_multiline = null
@export_placeholder("Test") var n_placeholder = null
@export_range(1, 10) var n_range = null
@export_exp_easing var n_exp_easing = null
@export_color_no_alpha var n_color_no_alpha = null
@export_node_path var n_node_path = null
@export_flags("A", "B", "C") var n_flags = null
@export_flags_2d_render var n_flags_2d_render = null
@export_flags_2d_physics var n_flags_2d_physics = null
@export_flags_2d_navigation var n_flags_2d_navigation = null
@export_flags_3d_render var n_flags_3d_render = null
@export_flags_3d_physics var n_flags_3d_physics = null
@export_flags_3d_navigation var n_flags_3d_navigation = null
@export_flags_avoidance var n_flags_avoidance = null

func test():
	print(var_to_str(v_enum))
	print(var_to_str(v_file))
	print(var_to_str(v_dir))
	print(var_to_str(v_global_file))
	print(var_to_str(v_global_dir))
	print(var_to_str(v_multiline))
	print(var_to_str(v_placeholder))
	print(var_to_str(v_range))
	print(var_to_str(v_exp_easing))
	print(var_to_str(v_color_no_alpha))
	print(var_to_str(v_node_path))
	print(var_to_str(v_flags))
	print(var_to_str(v_flags_2d_render))
	print(var_to_str(v_flags_2d_physics))
	print(var_to_str(v_flags_2d_navigation))
	print(var_to_str(v_flags_3d_render))
	print(var_to_str(v_flags_3d_physics))
	print(var_to_str(v_flags_3d_navigation))
	print(var_to_str(v_flags_avoidance))

	print(var_to_str(n_enum))
	print(var_to_str(n_file))
	print(var_to_str(n_dir))
	print(var_to_str(n_global_file))
	print(var_to_str(n_global_dir))
	print(var_to_str(n_multiline))
	print(var_to_str(n_placeholder))
	print(var_to_str(n_range))
	print(var_to_str(n_exp_easing))
	print(var_to_str(n_color_no_alpha))
	print(var_to_str(n_node_path))
	print(var_to_str(n_flags))
	print(var_to_str(n_flags_2d_render))
	print(var_to_str(n_flags_2d_physics))
	print(var_to_str(n_flags_2d_navigation))
	print(var_to_str(n_flags_3d_render))
	print(var_to_str(n_flags_3d_physics))
	print(var_to_str(n_flags_3d_navigation))
	print(var_to_str(n_flags_avoidance))
