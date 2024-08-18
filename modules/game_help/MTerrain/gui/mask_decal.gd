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

var is_being_edited = false

func _ready():	
	size.x = current_size
	size.z = current_size
	angle_offset = Vector3(size.x/2,0,size.z/2)
	var img = Image.create(1,1,false,Image.FORMAT_R8)
	img.fill(Color(1,1,1,1))
	wpx = ImageTexture.create_from_image(img)
	

func set_absolute_terrain_pos(pos:Vector3, force =false):
	if is_dirty:
		update_active_image()
	if orignal_image:
		active_terrain.set_brush_mask(active_image)		
	else:
		active_terrain.disable_brush_mask()		
	if is_being_edited or force:
		var angle_pos = pos - angle_offset
		angle_pos -= active_terrain.offset
		angle_pos = angle_pos/active_terrain.get_h_scale()
		angle_pos = Vector3(round(angle_pos.x),pos.y,round(angle_pos.z))
		var px_pos = Vector2i(angle_pos.x,angle_pos.z)
		angle_pos.x *= active_terrain.get_h_scale()
		angle_pos.z *= active_terrain.get_h_scale()
		angle_pos += active_terrain.offset
		desire_position = angle_pos + angle_offset
		active_terrain.set_brush_mask_px_pos(px_pos)
		set_process(true)

func increase_size(amount:int=1):
	if active_terrain is MTerrain:
		change_size(amount)
	else:
		push_error("trying to set mterrain mask size, but mask has no active_terrain")

func change_size(amount:float):
	if is_fix: return
	if current_size < active_terrain.get_h_scale():
		current_size = active_terrain.min_h_scale
	current_size += amount * active_terrain.get_h_scale()
	if current_size < active_terrain.get_h_scale():
		current_size = active_terrain.get_h_scale()
	size.x = current_size
	size.z = current_size
	var last_pos = position
	angle_offset = Vector3(size.x/2,0,size.z/2)
	is_dirty = true
	set_absolute_terrain_pos(position)

func set_size(value):	
	if current_size < active_terrain.get_h_scale():
		current_size = active_terrain.min_h_scale
	current_size = value * active_terrain.get_h_scale()
	if current_size < active_terrain.get_h_scale():
		current_size = active_terrain.get_h_scale()
	size.x = current_size
	size.z = current_size
	var last_pos = position
	angle_offset = Vector3(size.x/2,0,size.z/2)
	is_dirty = true
	update_active_image()
	is_being_edited = true

func _process(delta):
	if (desire_position - position).length() < 0.01:
		set_process(false)
	position = position.lerp(desire_position,0.15)


func set_mask(img:Image,tex:Texture2D):
	#image_rotation = 0
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
	if active_terrain:
		update_active_image()	
		if orignal_image:
			active_terrain.set_brush_mask(active_image)
		else:
			active_terrain.disable_brush_mask()		

func reset_image_rotation():
	image_rotation = 0
	is_dirty = true
	update_active_image()

func rotate_image(amount:int):
	image_rotation += amount
	image_rotation = image_rotation % 4
	is_dirty = true
	update_active_image()

func set_image_rotation(value:int):
	image_rotation = value % 4
	is_dirty = true
	update_active_image()
	is_being_edited = true

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
	
