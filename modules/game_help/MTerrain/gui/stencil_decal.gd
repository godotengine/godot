@tool
extends Decal

var active_terrain:MTerrain=null

var is_fix:=false
## Initial current size must be in power of two
var current_size:float = 8
var angle_offset:Vector3
var desire_position:Vector3

var original_tex:Texture2D
var active_tex:Texture2D
var orignal_image:Image
var active_image:Image

var is_dirty:=true

var image_rotation:int=0

var wpx:Texture2D

func _ready():
	size.x = current_size
	size.z = current_size
	angle_offset = Vector3(size.x/2,0,size.z/2)
	var img = Image.create(1,1,false,Image.FORMAT_R8)
	img.fill(Color(1,1,1,1))
	wpx = ImageTexture.create_from_image(img)
	

func set_absolute_terrain_pos(pos:Vector3,terrain:MTerrain):
	active_terrain = terrain
	if is_dirty:
		update_active_image()
	if orignal_image:
		terrain.set_brush_mask(active_image)
	else:
		terrain.disable_brush_mask()
	if not is_fix:
		var angle_pos = pos - angle_offset
		angle_pos -= terrain.offset
		angle_pos = angle_pos/terrain.get_h_scale()
		angle_pos = Vector3(round(angle_pos.x),pos.y,round(angle_pos.z))
		var px_pos = Vector2i(angle_pos.x,angle_pos.z)
		angle_pos.x *= terrain.get_h_scale()
		angle_pos.z *= terrain.get_h_scale()
		angle_pos += terrain.offset
		desire_position = angle_pos + angle_offset
		terrain.set_brush_mask_px_pos(px_pos)
		set_process(true)

func toggle_fix():
	is_fix = not is_fix

func increase_size(terrain:MTerrain,amount:int=1):
	change_size(amount,terrain)

func change_size(amount:float,terrain:MTerrain):
	if is_fix: return
	if current_size < terrain.get_h_scale():
		current_size = terrain.min_h_scale
	current_size += amount * terrain.get_h_scale()
	if current_size < terrain.get_h_scale():
		current_size = terrain.get_h_scale()
	size.x = current_size
	size.z = current_size
	var last_pos = position
	angle_offset = Vector3(size.x/2,0,size.z/2)
	is_dirty = true
	set_absolute_terrain_pos(position,terrain)

func _process(delta):
	if (desire_position - position).length() < 0.01:
		set_process(false)
	position = position.lerp(desire_position,0.15)


func set_mask(img:Image,tex:Texture2D):
	image_rotation = 0
	active_tex = tex
	orignal_image = img
	original_tex = tex
	is_dirty = true
	if tex:
		texture_albedo = tex
		albedo_mix = 0.8
	else:
		texture_albedo = wpx
		albedo_mix = 0.0
	

func reset_image_rotation():
	image_rotation = 0
	is_dirty = true
	update_active_image()

func rotate_image(amount:int):
	image_rotation += amount
	image_rotation = image_rotation % 4
	is_dirty = true
	update_active_image()

func update_active_image():
	if not active_terrain:return
	is_dirty = false
	if orignal_image == null:
		return
	active_image = orignal_image.duplicate()
	if image_rotation != 0:
		if image_rotation == 1 or image_rotation == -3:
			active_image.rotate_90(CLOCKWISE)
			rotation_degrees.y = -90
		if image_rotation == 3 or image_rotation == -1:
			active_image.rotate_90(COUNTERCLOCKWISE)
			rotation_degrees.y = 90
		elif image_rotation == 2 or image_rotation == -2:
			active_image.rotate_180()
			rotation_degrees.y = 180
		active_tex = ImageTexture.create_from_image(active_image)
	else:
		rotation_degrees.y = 0
	
	var new_size = current_size/active_terrain.get_h_scale()
	active_image.resize(new_size,new_size)
	



