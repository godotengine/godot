extends Node3D
## Demo script showing how to use the Hand Tracking module
##
## This script demonstrates accessing hand tracking data from the
## HandTrackingServer singleton and visualizing hand joints in 3D space.

# References to hand trackers
var left_hand_tracker: XRHandTracker
var right_hand_tracker: XRHandTracker

# Visual debugging: spheres to visualize joints
var left_hand_joint_meshes: Array[MeshInstance3D] = []
var right_hand_joint_meshes: Array[MeshInstance3D] = []

func _ready():
	var hand_tracking = HandTrackingServer

	if not hand_tracking:
		push_error("HandTrackingServer not available!")
		return

	print("Hand Tracking Demo initialized")

	# Create visual markers for hand joints
	_create_joint_visualizers()

func _create_joint_visualizers():
	# Create small spheres to visualize each hand joint
	var sphere_mesh = SphereMesh.new()
	sphere_mesh.radius = 0.01
	sphere_mesh.height = 0.02

	var material_left = StandardMaterial3D.new()
	material_left.albedo_color = Color(0.2, 0.8, 1.0) # Blue for left hand

	var material_right = StandardMaterial3D.new()
	material_right.albedo_color = Color(1.0, 0.8, 0.2) # Orange for right hand

	# Create joint markers for each hand joint
	for i in range(XRHandTracker.HAND_JOINT_MAX):
		# Left hand
		var left_joint = MeshInstance3D.new()
		left_joint.mesh = sphere_mesh
		left_joint.material_override = material_left
		left_joint.visible = false
		add_child(left_joint)
		left_hand_joint_meshes.append(left_joint)

		# Right hand
		var right_joint = MeshInstance3D.new()
		right_joint.mesh = sphere_mesh
		right_joint.material_override = material_right
		right_joint.visible = false
		add_child(right_joint)
		right_hand_joint_meshes.append(right_joint)

func _process(_delta):
	var hand_tracking = HandTrackingServer

	if not hand_tracking:
		return

	# Update hand tracking data
	hand_tracking.update_hand_tracking()

	# Get hand trackers
	left_hand_tracker = hand_tracking.get_left_hand_tracker()
	right_hand_tracker = hand_tracking.get_right_hand_tracker()

	# Update visualizations
	_update_hand_visualization(left_hand_tracker, left_hand_joint_meshes)
	_update_hand_visualization(right_hand_tracker, right_hand_joint_meshes)

func _update_hand_visualization(tracker: XRHandTracker, joint_meshes: Array[MeshInstance3D]):
	if not tracker or not tracker.get_has_tracking_data():
		# Hide all joints if tracking is lost
		for mesh in joint_meshes:
			mesh.visible = false
		return

	# Update each joint position
	for joint_id in range(XRHandTracker.HAND_JOINT_MAX):
		var flags = tracker.get_hand_joint_flags(joint_id)

		# Check if joint position is valid
		if flags.has_flag(XRHandTracker.HAND_JOINT_FLAG_POSITION_VALID):
			var transform = tracker.get_hand_joint_transform(joint_id)
			joint_meshes[joint_id].global_transform = transform
			joint_meshes[joint_id].visible = true
		else:
			joint_meshes[joint_id].visible = false

func _print_debug_info():
	"""Helper function to print hand tracking debug information"""
	var hand_tracking = HandTrackingServer

	if not hand_tracking:
		return

	print("=== Hand Tracking Debug Info ===")
	print("Hand tracking available: ", hand_tracking.is_hand_tracking_available())

	if left_hand_tracker and left_hand_tracker.get_has_tracking_data():
		print("Left hand is being tracked")
		var wrist_pos = left_hand_tracker.get_hand_joint_transform(XRHandTracker.HAND_JOINT_WRIST).origin
		print("  Left wrist position: ", wrist_pos)
	else:
		print("Left hand not tracked")

	if right_hand_tracker and right_hand_tracker.get_has_tracking_data():
		print("Right hand is being tracked")
		var wrist_pos = right_hand_tracker.get_hand_joint_transform(XRHandTracker.HAND_JOINT_WRIST).origin
		print("  Right wrist position: ", wrist_pos)
	else:
		print("Right hand not tracked")
