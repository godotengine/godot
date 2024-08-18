@tool
extends ItemList

const brush_masks_dir:String = "res://addons/m_terrain/brush_masks/"
const allowed_extension:PackedStringArray = ["jpg","jpeg","png","exr","bmp","dds","hdr","tga","svg","webp"]

var mask = null

var is_loaded := false
var current_selected_index:int=-1

var images:Array
var textures:Array

func _ready():
	clear()

func load_images(mask_decal):
	mask = mask_decal
	if is_loaded: return
	clear()
	add_icon_item( preload("res://addons/m_terrain/icons/no_mask_icon.svg"))
	#add_item("NULL")
	var dir = DirAccess.open(brush_masks_dir)
	if not dir:
		printerr("Can not open brush masks directory")
		return
	var files:Array
	dir.list_dir_begin()
	## finding files inside mask directory
	while true:
		var f = dir.get_next()
		if f == "":
			break
		files.push_back(f)
	var files_path:Array
	## Validating files path
	for f in files:
		if allowed_extension.has(f.get_extension()):
			files_path.push_back(brush_masks_dir.path_join(f))
	## Creating image and texture
	for p in files_path:
		var img = Image.load_from_file(p)
		img.convert(Image.FORMAT_RF)
		if img:
			images.push_back(img)
			textures.push_back(ImageTexture.create_from_image(img))
	## Adding items
	for tex in textures:
		add_item("",tex)
	is_loaded = true
	if images.size() > 0:
		mask.set_mask(null,null)

func get_image():
	if images.size() == 0 : return -1
	return images[current_selected_index]

func get_texture():
	if images.size() == 0 : return -1
	return textures[current_selected_index]

func _on_item_selected(index):
	if index == 0:
		mask.set_mask(null,null)
		current_selected_index = -1
		return
	current_selected_index = index - 1
	mask.set_mask(images[current_selected_index],textures[current_selected_index])

func invert_selected_image():
	if current_selected_index == -1:return
	var img:Image= images[current_selected_index]
	for j in range(img.get_height()):
		for i in range(img.get_width()):
			var val:float= img.get_pixel(i,j).r
			val = 1.0 - val;
			img.set_pixel(i,j,Color(val,0,0,1))
	textures[current_selected_index] = ImageTexture.create_from_image(img)
	set_item_icon(current_selected_index+1,textures[current_selected_index])
	mask.set_mask(images[current_selected_index],textures[current_selected_index])
