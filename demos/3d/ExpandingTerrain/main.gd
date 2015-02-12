#
#  Script by Brian Jack
#  gau_veldt@hotmail.com
#  CC-BY-SA
#

extends Spatial

# member variables here, example:
# var a=2
# var b="textvar"
var magic_random_max=pow(2,31)-1
var rs

var terrainMesh
var terrain
var surf
var chunkMap

var hm_data=[]
var hm_size
var hm_min
var hm_max
var hm_curseed

# needs to be a mutliple of two (even)
export(int) var chunk_size=16
export(float) var seed=1337
export(float) var height_scale=8.0
export(float) var smoothing=-1.5

func get_rand():
	if rs==null:
		rs=rand_seed(seed)
	else:
		rs=rand_seed(rs[1])
	return float(rs[0])/float(magic_random_max)

func init_hm():
	hm_size=2
	hm_min=0
	hm_max=1
	hm_data=[
		get_rand()*height_scale,
		get_rand()*height_scale,
		get_rand()*height_scale,
		get_rand()*height_scale,
	]
	hm_curseed=rs[1]

func get_hmdata(x,y):
	return hm_data[y*hm_size+x]

func getChunk(x,y):
	# initialize terrain chunk at (x,y)
	var lo=min(x,y)
	var hi=max(x+1,y+1)

	# Expand the heightmap square
	# one layer at a time until chunk's
	# until chunk's points are contained
	#
	# for landscapes you want a consistent generation loop
	# that can replay from the seed to any chunk of the heightmap
	# consistently.  same seed should yield same terrain regardless
	# of the order chunks are generated
	#
	var newsize
	var dpos
	var hrz
	var vrt
	var new_data
	rs[1]=hm_curseed
	while hm_min>lo or hm_max<hi:
		# Expand the heightmap square
		# one layer at a time until chunk's
		# until chunk's points are contained
		newsize=hm_size+2
		dpos=0
		# new top row
		new_data=[hm_data[dpos]+get_rand()*height_scale]
		for hrz in range(hm_size):
			new_data.append(hm_data[dpos]+get_rand()*height_scale)
			dpos+=1
		new_data.append(hm_data[dpos-1]+get_rand()*height_scale)
		# rewind inner data back to start
		dpos=0
		# append each row of old data expanded with new data before and after
		for vrt in range(hm_size):
			# append new left cell
			new_data.append(hm_data[dpos]+get_rand()*height_scale)
			# append existing row
			for hrz in range(hm_size):
				new_data.append(hm_data[dpos])
				dpos+=1
			# append new right cell
			new_data.append(hm_data[dpos-1]+get_rand()*height_scale)
		# rewind inner data to repeat last row
		dpos-=hm_size
		# new bottom row
		new_data.append(hm_data[dpos]+get_rand()*height_scale)
		for k in range(hm_size):
			new_data.append(hm_data[dpos]+get_rand()*height_scale)
			dpos+=1
		new_data.append(hm_data[dpos-1]+get_rand()*height_scale)
		# recalc data properties
		hm_size+=2
		hm_min-=1
		hm_max+=1
		hm_data=new_data
	hm_curseed=rs[1]
	
	var ax=x+(hm_size/2)-1
	var ay=y+(hm_size/2)-1
	return [
		get_hmdata(ax  ,ay  ),
		get_hmdata(ax+1,ay  ),
		get_hmdata(ax  ,ay+1),
		get_hmdata(ax+1,ay+1)
	]

func genChunkNode(x,y):
	# Set up new node and mesh
	terrain=MeshInstance.new()
	add_child(terrain)
	terrainMesh=Mesh.new()
	terrain.set_mesh(terrainMesh)
	surf=SurfaceTool.new()
	# origin is half the chunk size
	var org=chunk_size/2
	# convert to a float for lerp calcs
	var fCellSz=float(chunk_size)
	# get height points for the chunk corners
	var cpoints=getChunk(x,y)
	var ul=cpoints[0]
	var ur=cpoints[1]
	var bl=cpoints[2]
	var br=cpoints[3]
	# interpolate height over the chunk
	# using lerp and ease to round the transitions
	var il
	var ir
	var elv
	var hmap=[]
	hmap.resize(1+chunk_size)
	for hmap_v in range(1+chunk_size):
		il=lerp(bl,ul,ease(hmap_v/fCellSz,smoothing))
		ir=lerp(br,ur,ease(hmap_v/fCellSz,smoothing))
		hmap[hmap_v]=[]
		hmap[hmap_v].resize(1+chunk_size)
		for hmap_h in range(1+chunk_size):
			elv=lerp(il,ir,ease(hmap_h/fCellSz,smoothing))
			hmap[hmap_v][hmap_h]=elv
	# generates mesh:
	# paired triangles (quads) for each cell of the chunk
	surf.begin(Mesh.PRIMITIVE_TRIANGLES)
	var vs
	var hs
	for v in range(chunk_size):
		vs=v-org
		for h in range(chunk_size):
			hs=h-org
			surf.add_color(Color(1,1,1))
			surf.add_uv(Vector2(0,0))
			surf.add_vertex(Vector3(hs,hmap[v][h],vs))
			surf.add_color(Color(1,1,1))
			surf.add_uv(Vector2(1,0))
			surf.add_vertex(Vector3(hs+1,hmap[v][h+1],vs))
			surf.add_color(Color(1,1,1))
			surf.add_uv(Vector2(0,1))
			surf.add_vertex(Vector3(hs,hmap[v+1][h],vs+1))
			surf.add_color(Color(1,1,1))
			surf.add_uv(Vector2(0,1))
			surf.add_vertex(Vector3(hs,hmap[v+1][h],vs+1))
			surf.add_color(Color(1,1,1))
			surf.add_uv(Vector2(1,0))
			surf.add_vertex(Vector3(hs+1,hmap[v][h+1],vs))
			surf.add_color(Color(1,1,1))
			surf.add_uv(Vector2(1,1))
			surf.add_vertex(Vector3(hs+1,hmap[v+1][h+1],vs+1))

	# generate normals based on vertex ordering
	surf.generate_normals()
	# applies changes to the mesh object
	surf.commit(terrainMesh)
	# position chunk appropriately in world
	terrain.set_name("chunk_"+str(x)+"_"+str(y))
	terrain.set_translation(Vector3(x*chunk_size,0,y*-chunk_size))
	return terrain

func registerChunk(n,x,y):
	if not chunkMap.has(y):
		chunkMap[y]={}
	chunkMap[y][x]=n

func _ready():
	# Initalization here
	chunkMap={}
	chunkMap[0]={}
	rs=rand_seed(seed)
	init_hm()

	# generate some chunks nodes for the scene
	var v
	var h
	for v in range(-4,5):
		for h in range(-4,5):
			registerChunk(genChunkNode(h,v),h,v)
