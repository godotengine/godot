
def tw(f,t,st):
	for x in range(t):
		f.write("\t")
	nl = True
	if st[-1] == "#":
		nl = False
		st = st[:-1]
	f.write(st)
	if nl:
		f.write("\n")


def write_property_lua(f, tab, name, value, pref = ""):

	tw(f, tab, '%s{ name = "%s",' % (pref, name))
	tab = tab + 1

	if (type(value)==str):

		tw(f, tab, 'value = "%s",' % value)
		tw(f, t, 'type = "string",')

	elif (type(value)==bool):


		if (value):
			tw(f, tab, 'value = true,')
		else:
			tw(f, tab, 'value = false,')
	
		tw(f, t, 'type = "bool",')

	elif (type(value)==int):

		tw(f, t, 'type = "int",')
		tw(f, tab, 'value = %d,' % value)

	elif (type(value)==float):

		tw(f, t, 'type = "real",')
		tw(f, tab, 'value = %f,' % value)

	elif (type(value)==dict):
		
		tw(f, t, 'type = "dictionary",')
		for x in value:
			write_property_lua(f,tab,x,value[x])

	elif (isinstance(value,ObjectTree)):
		if (not value._resource):
			print("ERROR: Not a resource!!")
			tw(f, tab-1, "},")
			return

		tw(f, tab, 'type = "resource",')
		tw(f, tab, 'resource_type = "%s",' % value._type)

		if (value._res_path!=""):
	
			tw(f, tab, 'path = "%s",' % value._res_path)
	
		else:

			tw(f, tab, "value = {")
			tab = tab + 1
			tw(f, tab, 'type = "%s",' % value._type)
			
			for x in value._properties:
				write_property_lua(f,tab,x[0],x[1])
			
			tab = tab - 1
			tw(f, tab, "},")
			
	elif (isinstance(value,Color)):
		
		tw(f, tab, 'type = "color",')
		tw(f, tab, 'value = { %.20f, %.20f, %.20f, %.20f },' % (value.r, value.g, value.b, value.a))
		
	elif (isinstance(value,Vector3)):

		tw(f, tab, 'type = "vector3",')
		tw(f, tab, 'value = { %.20f, %.20f, %.20f },' % (value.x, value.y, value.z))

	elif (isinstance(value,Quat)):

		tw(f, tab, 'type = "quaternion",')
		tw(f, tab, 'value = { %.20f, %.20f, %.20f, %.20f },' % (value.x, value.y, value.z, value.w))

	elif (isinstance(value,Matrix4x3)): # wtf, blender matrix?
	
		tw(f, tab, 'type = "transform",')
		tw(f, tab, 'value = { #')
		for i in range(3):
			for j in range(3):
				f.write("%.20f, " % value.m[j][i])
		
		for i in range(3):
			f.write("%.20f, " % value.m[i][3])
		
		f.write("},\n")
	
	elif (type(value)==list):
		if (len(value)==0):
			tw(f, tab-1, "},")
			return
		first=value[0]
		if (type(first)==int):

			tw(f, tab, 'type = "int_array",')
			tw(f, tab, 'value = #')
			for i in range(len(value)):
				f.write("%d, " % value[i])
			f.write("},\n")

		elif (type(first)==float):

			tw(f, tab, 'type = "real_array",')
			tw(f, tab, 'value = #')
			for i in range(len(value)):
				f.write("%.20f, " % value[i])
			f.write("},\n")


		elif (type(first)==str):
			
			tw(f, tab, 'type = "string_array",')
			tw(f, tab, 'value = #')
			for i in range(len(value)):
				f.write('"%s", ' % value[i])
			f.write("},\n")

		elif (isinstance(first,Vector3)):

			tw(f, tab, 'type = "vector3_array",')
			tw(f, tab, 'value = #')
			for i in range(len(value)):
				f.write("{ %.20f, %.20f, %.20f }, " % (value[i].x, value[i].y, value[i].z))
			f.write("},\n")

		elif (isinstance(first,Color)):

			tw(f, tab, 'type = "color_array",')
			tw(f, tab, 'value = #')
			for i in range(len(value)):
				f.write("{ %.20f, %.20f, %.20f, %.20f }, " % (value[i].r, value[i].g, value[i].b, value[i].a))
			f.write("},\n")

		elif (type(first)==dict):
			
			tw(f, tab, 'type = "dict_array",')
			tw(f, tab, 'value = {')
			
			for i in range(len(value)):
				write_property_lua(f,tab+1,str(i+1),value[i])
			
			tw(f, tab, '},')
			

	tw(f, tab-1, "},")



def write_node_lua(f,tab,tree,path):

	tw(f, tab, '{ type = "%s",')
	
	if not tree._resource:
		tw(f, tab+1, 'meta = {')
		write_property_lua(f, tab+3, "name", tree._name)
		if path != "":
			write_property_lua(f, tab+3, "path", path)
		tw(f, tab+1, '},')
	
	tw(f, tab+1, "properties = {,")
	for x in tree._properties:
		write_property_lua(f,tab+2,x[0],x[1])
	tw(f, tab+1, "},")
	
	tw(f, tab, '},')


	if (path==""):
		path="."
	else:
		if (path=="."):
			path=tree._name
		else:
			path=path+"/"+tree._name
	#path="."
	for x in tree._children:
		write_node_lua(f,tab,x,path)

def write(tree,fname):
	f=open(fname,"wb")
	f.write("return = {\n")

	f.write('\tmagic = "SCENE",\n')
	tab = 1

	write_node_lua(f,tab,tree,"")

	f.write("}\n\n")
	





