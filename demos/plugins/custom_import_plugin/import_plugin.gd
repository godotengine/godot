tool

extends EditorImportPlugin


# Simple plugin that imports a text file with extension .mtxt
# which contains 3 integers in format R,G,B (0-255)
# (see example .mtxt in this folder)
# Imported file is converted to a material

var dialog = null

func get_name():
	return "silly_material"

func get_visible_name():
	return "Silly Material"

func import_dialog(path):
	var md = null
	if (path!=""):
		md = ResourceLoader.load_import_metadata(path)
	dialog.configure(self,path,md)
	dialog.popup_centered()

func import(path,metadata):

	assert(metadata.get_source_count() == 1)

	var source = metadata.get_source_path(0)
	var use_red_anyway = metadata.get_option("use_red_anyway")

	var f = File.new()
	var err = f.open(source,File.READ)
	if (err!=OK):
		return ERR_CANT_OPEN

	var l = f.get_line()

	f.close()

	var channels = l.split(",")
	if (channels.size()!=3):
		return ERR_PARSE_ERROR

	var color = Color8(int(channels[0]),int(channels[1]),int(channels[2]))

	var material

	if (ResourceLoader.has(path)):
		# Material is in use, update it
		material = ResourceLoader.load(path)
	else:
		# Material not in use, create
		material = FixedMaterial.new()

	if (use_red_anyway):
		color=Color8(255,0,0)
	
	material.set_parameter(FixedMaterial.PARAM_DIFFUSE,color)	

	# Make sure import metadata links to this plugin
	
	metadata.set_editor("silly_material")

	# Update the import metadata

	material.set_import_metadata(metadata)
	

	# Save
	err = ResourceSaver.save(path,material)

	return err


func config(base_control):

	dialog = preload("res://addons/custom_import_plugin/material_dialog.tscn").instance()
	base_control.add_child(dialog)

