# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

# Script copyright (C) Juan Linietsky
# Contact Info: juan@codenix.com

"""
This script is an exporter to the Khronos Collada file format.

http://www.khronos.org/collada/
"""

# TODO:
# Materials & Textures
# Optionally export Vertex Colors
# Morph Targets
# Control bone removal
# Copy textures
# Export Keyframe Optimization
# --
# Morph Targets
# Blender native material? (?)

import os
import time
import math  # math.pi
import shutil
import bpy
from mathutils import Vector, Matrix

#according to collada spec, order matters
S_ASSET=0
S_IMGS=1
S_FX=2
S_MATS=3
S_GEOM=4
S_CONT=5
S_CAMS=6
S_LAMPS=7
S_ANIM_CLIPS=8
S_NODES=9
S_ANIM=10

CMP_EPSILON=0.0001

def snap_tup(tup):
	ret=()
	for x in tup:
		ret+=( x-math.fmod(x,0.0001), )

	return tup


def strmtx(mtx):
	s=" "
	for x in range(4):
		for y in range(4):
			s+=str(mtx[x][y])
			s+=" "
	s+=" "
	return s

def numarr(a,mult=1.0):
	s=" "
	for x in a:
		s+=" "+str(x*mult)
	s+=" "
	return s

def strarr(arr):
	s=" "
	for x in arr:
		s+=" "+str(x)
	s+=" "
	return s



class DaeExporter:

	def validate_id(self,d):
		if (d.find("id-")==0):
			return "z"+d
		return d


	def new_id(self,t):
		self.last_id+=1
		return "id-"+t+"-"+str(self.last_id)

	class Vertex:

		def close_to(v):
			if ( (self.vertex-v.vertex).length() > CMP_EPSILON ):
				return False
			if ( (self.normal-v.normal).length() > CMP_EPSILON ):
				return False
			if ( (self.uv-v.uv).length() > CMP_EPSILON ):
				return False
			if ( (self.uv2-v.uv2).length() > CMP_EPSILON ):
				return False

			return True

		def get_tup(self):
			tup = (self.vertex.x,self.vertex.y,self.vertex.z,self.normal.x,self.normal.y,self.normal.z)
			for t in self.uv:
				tup = tup + (t.x,t.y)
			return tup

		def __init__(self):
			self.vertex = Vector( (0.0,0.0,0.0) )
			self.normal = Vector( (0.0,0.0,0.0) )
			self.color = Vector( (0.0,0.0,0.0) )
			self.uv = []
			self.uv2 = Vector( (0.0,0.0) )
			self.bones=[]
			self.weights=[]


	def writel(self,section,indent,text):
		if (not (section in self.sections)):
			self.sections[section]=[]
		line=""
		for x in range(indent):
			line+="\t"
		line+=text
		self.sections[section].append(line)


	def export_image(self,image):

		if (image in self.image_cache):
			return self.image_cache[image]

		imgpath = image.filepath
		if (imgpath.find("//")==0 or imgpath.find("\\\\")==0):
			#if relative, convert to absolute
			imgpath = bpy.path.abspath(imgpath)

		#path is absolute, now do something!

		if (self.config["use_copy_images"]):
			#copy image
			basedir = os.path.dirname(self.path)+"/images"
			if (not os.path.isdir(basedir)):
				os.makedirs(basedir)
			dstfile=basedir+"/"+os.path.basename(imgpath)
			if (not os.path.isfile(dstfile)):
				shutil.copy(imgpath,dstfile)
			imgpath="images/"+os.path.basename(imgpath)

		else:
			#export relative, always, no one wants absolute paths.
			imgpath = os.path.relpath(imgpath,os.path.dirname(self.path)).replace("\\","/") # export unix compatible always


		imgid = self.new_id("image")
		self.writel(S_IMGS,1,'<image id="'+imgid+'" name="'+image.name+'">')
		self.writel(S_IMGS,2,'<init_from>'+imgpath+'</init_from>"/>')
		self.writel(S_IMGS,1,'</image>')
		self.image_cache[image]=imgid
		return imgid

	def export_material(self,material,double_sided_hint=True):

		if (material in self.material_cache):
			return self.material_cache[material]

		fxid = self.new_id("fx")
		self.writel(S_FX,1,'<effect id="'+fxid+'" name="'+material.name+'-fx">')
		self.writel(S_FX,2,'<profile_COMMON>')

		#Find and fetch the textures and create sources
		sampler_table={}
		diffuse_tex=None
		specular_tex=None
		emission_tex=None
		normal_tex=None
		for i in range(len(material.texture_slots)):
			ts=material.texture_slots[i]
			if (not ts):
				continue
			if (not ts.use):
				continue
			if (not ts.texture):
				continue
			if (ts.texture.type!="IMAGE"):
				continue

			if (ts.texture.image==None):
				continue

			#image
			imgid = self.export_image(ts.texture.image)

			#surface
			surface_sid = self.new_id("fx_surf")
			self.writel(S_FX,3,'<newparam sid="'+surface_sid+'">')
			self.writel(S_FX,4,'<surface type="2D">')
			self.writel(S_FX,5,'<init_from>'+imgid+'</init_from>') #this is sooo weird
			self.writel(S_FX,5,'<format>A8R8G8B8</format>')
			self.writel(S_FX,4,'</surface>')
			self.writel(S_FX,3,'</newparam>')
			#sampler, collada sure likes it difficult
			sampler_sid = self.new_id("fx_sampler")
			self.writel(S_FX,3,'<newparam sid="'+sampler_sid+'">')
			self.writel(S_FX,4,'<sampler2D>')
			self.writel(S_FX,5,'<source>'+surface_sid+'</source>')
			self.writel(S_FX,4,'</sampler2D>')
			self.writel(S_FX,3,'</newparam>')
			sampler_table[i]=sampler_sid

			if (ts.use_map_color_diffuse and diffuse_tex==None):
				diffuse_tex=sampler_sid
			if (ts.use_map_color_spec and specular_tex==None):
				specular_tex=sampler_sid
			if (ts.use_map_emit and emission_tex==None):
				emission_tex=sampler_sid
			if (ts.use_map_normal and normal_tex==None):
				normal_tex=sampler_sid

		self.writel(S_FX,3,'<technique sid="common">')
		shtype="blinn"
		self.writel(S_FX,4,'<'+shtype+'>')
		#ambient? from where?

		self.writel(S_FX,5,'<emission>')
		if (emission_tex!=None):
			self.writel(S_FX,6,'<texture texture="'+emission_tex+'" texcoord="CHANNEL1"/>')
		else:
			self.writel(S_FX,6,'<color>'+numarr(material.diffuse_color,material.emit)+' </color>') # not totally right but good enough
		self.writel(S_FX,5,'</emission>')

		self.writel(S_FX,5,'<ambient>')
		self.writel(S_FX,6,'<color>'+numarr(self.scene.world.ambient_color,material.ambient)+' </color>')
		self.writel(S_FX,5,'</ambient>')

		self.writel(S_FX,5,'<diffuse>')
		if (diffuse_tex!=None):
			self.writel(S_FX,6,'<texture texture="'+diffuse_tex+'" texcoord="CHANNEL1"/>')
		else:
			self.writel(S_FX,6,'<color>'+numarr(material.diffuse_color,material.diffuse_intensity)+'</color>')
		self.writel(S_FX,5,'</diffuse>')

		self.writel(S_FX,5,'<specular>')
		if (specular_tex!=None):
			self.writel(S_FX,6,'<texture texture="'+specular_tex+'" texcoord="CHANNEL1"/>')
		else:
			self.writel(S_FX,6,'<color>'+numarr(material.specular_color,material.specular_intensity)+'</color>')
		self.writel(S_FX,5,'</specular>')

		self.writel(S_FX,5,'<shininess>')
		self.writel(S_FX,6,'<float>'+str(material.specular_hardness)+'</float>')
		self.writel(S_FX,5,'</shininess>')

		self.writel(S_FX,5,'<reflective>')
		self.writel(S_FX,6,'<color>'+strarr(material.mirror_color)+'</color>')
		self.writel(S_FX,5,'</reflective>')

		if (material.use_transparency):
			self.writel(S_FX,5,'<transparency>')
			self.writel(S_FX,6,'<float>'+str(material.alpha)+'</float>')
			self.writel(S_FX,5,'</transparency>')


		self.writel(S_FX,5,'<index_of_refraction>'+str(material.specular_ior)+'</index_of_refraction>')

		self.writel(S_FX,4,'</'+shtype+'>')

		self.writel(S_FX,4,'<extra>')
		self.writel(S_FX,5,'<technique profile="FCOLLADA">')
		if (normal_tex):
			self.writel(S_FX,6,'<bump bumptype="NORMALMAP">')
			self.writel(S_FX,7,'<texture texture="'+normal_tex+'" texcoord="CHANNEL1"/>')
			self.writel(S_FX,6,'</bump>')

		self.writel(S_FX,5,'</technique>')
		self.writel(S_FX,5,'<technique profile="GOOGLEEARTH">')
		self.writel(S_FX,6,'<double_sided>'+["0","1"][double_sided_hint]+"</double_sided>")
		self.writel(S_FX,5,'</technique>')
		self.writel(S_FX,4,'</extra>')

		self.writel(S_FX,3,'</technique>')
		self.writel(S_FX,2,'</profile_COMMON>')
		self.writel(S_FX,1,'</effect>')

		# Also export blender material in all it's glory (if set as active)


		#Material
		matid = self.new_id("material")
		self.writel(S_MATS,1,'<material id="'+matid+'" name="'+material.name+'">')
		self.writel(S_MATS,2,'<instance_effect url="#'+fxid+'"/>')
		self.writel(S_MATS,1,'</material>')

		self.material_cache[material]=matid
		return matid


	def export_mesh(self,node,armature=None,shapename=None):

		if (node.data in self.mesh_cache) and shapename==None:
			return self.mesh_cache[mesh]

		if (len(node.modifiers) and self.config["use_mesh_modifiers"]) or shapename!=None:
			mesh=node.to_mesh(self.scene,True,"RENDER") #is this allright?
		else:
			mesh=node.data

		mesh.update(calc_tessface=True)
		vertices=[]
		vertex_map={}
		surface_indices={}
		materials={}

		materials={}

		si=None
		if (armature!=None):
			si=self.skeleton_info[armature]

		has_uv=False
		has_uv2=False
		has_weights=armature!=None
		has_colors=False
		mat_assign=[]

		uv_layer_count=len(mesh.uv_textures)

		for fi in range(len(mesh.tessfaces)):
			f=mesh.tessfaces[fi]


			if (not (f.material_index in surface_indices)):
				surface_indices[f.material_index]=[]
				print("Type: "+str(type(f.material_index)))
				print("IDX: "+str(f.material_index)+"/"+str(len(mesh.materials)))


				try:
					#Bizarre blender behavior i don't understand, so catching exception
					mat = mesh.materials[f.material_index]
				except:
					mat= None

				if (mat!=None):
					materials[f.material_index]=self.export_material( mat,mesh.show_double_sided )
				else:
					materials[f.material_index]=None #weird, has no material?

			indices = surface_indices[f.material_index]
			vi=[]
			#make triangles always
			if (len(f.vertices)==3):
				vi.append(0)
				vi.append(1)
				vi.append(2)
			elif (len(f.vertices)==4):
				vi.append(0)
				vi.append(1)
				vi.append(2)
				vi.append(0)
				vi.append(2)
				vi.append(3)

			for x in vi:
				mv = mesh.vertices[f.vertices[x]]

				v = self.Vertex()
				v.vertex = Vector( mv.co )

				for xt in mesh.tessface_uv_textures:
					d = xt.data[fi]
					uvsrc = [d.uv1,d.uv2,d.uv3,d.uv4]
					v.uv.append( Vector( uvsrc[x] ) )


				if (f.use_smooth):
					v.normal=Vector( mv.normal )
				else:
					v.normal=Vector( f.normal )

			       # if (armature):
			       #         v.vertex = node.matrix_world * v.vertex

				#v.color=Vertex(mv. ???

				if (armature!=None):
					wsum=0.0
					for vg in mv.groups:
						if vg.group >= len(node.vertex_groups):
							continue;
						name = node.vertex_groups[vg.group].name
						if (name in si["bone_index"]):
							if (vg.weight>0.001): #blender has a lot of zero weight stuff
								v.bones.append(si["bone_index"][name])
								v.weights.append(vg.weight)
								wsum+=vg.weight


				tup = v.get_tup()
				idx = 0
				if (tup in vertex_map):
					idx = vertex_map[tup]
				else:
					idx = len(vertices)
					vertices.append(v)
					vertex_map[tup]=idx

				indices.append(idx)

		if shapename != None:
			meshid = self.new_id("mesh_"+shapename)
			self.writel(S_GEOM,1,'<geometry id="'+meshid+'" name="'+mesh.name+'_'+shapename+'">')
		else:
			meshid = self.new_id("mesh")
			self.writel(S_GEOM,1,'<geometry id="'+meshid+'" name="'+mesh.name+'">')

		self.writel(S_GEOM,2,'<mesh>')


		# Vertex Array
		self.writel(S_GEOM,3,'<source id="'+meshid+'-positions">')
		float_values=""
		for v in vertices:
			 float_values+=" "+str(v.vertex.x)+" "+str(v.vertex.y)+" "+str(v.vertex.z)
		self.writel(S_GEOM,4,'<float_array id="'+meshid+'-positions-array" count="'+str(len(vertices)*3)+'">'+float_values+'</float_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+meshid+'-positions-array" count="'+str(len(vertices))+'" stride="3">')
		self.writel(S_GEOM,5,'<param name="X" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Y" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Z" type="float"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,4,'</technique_common>')
		self.writel(S_GEOM,3,'</source>')

		# Normal Array

		self.writel(S_GEOM,3,'<source id="'+meshid+'-normals">')
		float_values=""
		for v in vertices:
			 float_values+=" "+str(v.normal.x)+" "+str(v.normal.y)+" "+str(v.normal.z)
		self.writel(S_GEOM,4,'<float_array id="'+meshid+'-normals-array" count="'+str(len(vertices)*3)+'">'+float_values+'</float_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+meshid+'-normals-array" count="'+str(len(vertices))+'" stride="3">')
		self.writel(S_GEOM,5,'<param name="X" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Y" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Z" type="float"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,4,'</technique_common>')
		self.writel(S_GEOM,3,'</source>')

		# UV Arrays

		for uvi in range(uv_layer_count):

			self.writel(S_GEOM,3,'<source id="'+meshid+'-texcoord-'+str(uvi)+'">')
			float_values=""
			for v in vertices:
				float_values+=" "+str(v.uv[uvi].x)+" "+str(v.uv[uvi].y)
			self.writel(S_GEOM,4,'<float_array id="'+meshid+'-texcoord-'+str(uvi)+'-array" count="'+str(len(vertices)*2)+'">'+float_values+'</float_array>')
			self.writel(S_GEOM,4,'<technique_common>')
			self.writel(S_GEOM,4,'<accessor source="#'+meshid+'-texcoord-'+str(uvi)+'-array" count="'+str(len(vertices))+'" stride="2">')
			self.writel(S_GEOM,5,'<param name="S" type="float"/>')
			self.writel(S_GEOM,5,'<param name="T" type="float"/>')
			self.writel(S_GEOM,4,'</accessor>')
			self.writel(S_GEOM,4,'</technique_common>')
			self.writel(S_GEOM,3,'</source>')

		# Triangle Lists
		self.writel(S_GEOM,3,'<vertices id="'+meshid+'-vertices">')
		self.writel(S_GEOM,4,'<input semantic="POSITION" source="#'+meshid+'-positions"/>')
		self.writel(S_GEOM,3,'</vertices>')

		for m in surface_indices:
			indices = surface_indices[m]
			mat = materials[m]
			if (mat!=None):
				matref = self.new_id("trimat")
				self.writel(S_GEOM,3,'<triangles count="'+str(int(len(indices)/3))+'" material="'+matref+'">') # todo material
				mat_assign.append( (mat,matref) )
			else:
				self.writel(S_GEOM,3,'<triangles count="'+str(int(len(indices)/3))+'">') # todo material
			self.writel(S_GEOM,4,'<input semantic="VERTEX" source="#'+meshid+'-vertices" offset="0"/>')
			self.writel(S_GEOM,4,'<input semantic="NORMAL" source="#'+meshid+'-normals" offset="1"/>')
			extra_indices=0
			for uvi in range(uv_layer_count):
				self.writel(S_GEOM,4,'<input semantic="TEXCOORD" source="#'+meshid+'-texcoord-'+str(uvi)+'" offset="'+str(2+uvi)+'" set="'+str(uvi)+'"/>')
				extra_indices+=1

			int_values="<p>"
			for i in range(len(indices)):
				int_values+=" "+str(indices[i]) # vertex index
				int_values+=" "+str(indices[i]) # normal index
				for e in range(extra_indices):
					int_values+=" "+str(indices[i]) # normal index
			int_values+="</p>"
			self.writel(S_GEOM,4,int_values)
			self.writel(S_GEOM,3,'</triangles>')


		self.writel(S_GEOM,2,'</mesh>')
		self.writel(S_GEOM,1,'</geometry>')

		meshdata={}
		meshdata["id"]=meshid
		meshdata["material_assign"]=mat_assign
		self.mesh_cache[node.data]=meshdata


		# Export armature data (if armature exists)

		if (armature!=None):

			contid = self.new_id("controller")

			self.writel(S_CONT,1,'<controller id="'+contid+'">')
			self.writel(S_CONT,2,'<skin source="'+meshid+'">')
			self.writel(S_CONT,3,'<bind_shape_matrix>'+strmtx(node.matrix_world)+'</bind_shape_matrix>')
			#Joint Names
			self.writel(S_CONT,3,'<source id="'+contid+'-joints">')
			name_values=""
			for v in si["bone_names"]:
				name_values+=" "+v

			self.writel(S_CONT,4,'<Name_array id="'+contid+'-joints-array" count="'+str(len(si["bone_names"]))+'">'+name_values+'</Name_array>')
			self.writel(S_CONT,4,'<technique_common>')
			self.writel(S_CONT,4,'<accessor source="#'+contid+'-joints-array" count="'+str(len(si["bone_names"]))+'" stride="1">')
			self.writel(S_CONT,5,'<param name="JOINT" type="Name"/>')
			self.writel(S_CONT,4,'</accessor>')
			self.writel(S_CONT,4,'</technique_common>')
			self.writel(S_CONT,3,'</source>')
			#Pose Matrices!
			self.writel(S_CONT,3,'<source id="'+contid+'-bind_poses">')
			pose_values=""
			for v in si["bone_bind_poses"]:
				pose_values+=" "+strmtx(v)

			self.writel(S_CONT,4,'<float_array id="'+contid+'-bind_poses-array" count="'+str(len(si["bone_bind_poses"])*16)+'">'+pose_values+'</float_array>')
			self.writel(S_CONT,4,'<technique_common>')
			self.writel(S_CONT,4,'<accessor source="#'+contid+'-bind_poses-array" count="'+str(len(si["bone_bind_poses"]))+'" stride="16">')
			self.writel(S_CONT,5,'<param name="TRANSFORM" type="float4x4"/>')
			self.writel(S_CONT,4,'</accessor>')
			self.writel(S_CONT,4,'</technique_common>')
			self.writel(S_CONT,3,'</source>')
			#Skin Weights!
			self.writel(S_CONT,3,'<source id="'+contid+'-skin_weights">')
			skin_weights=""
			skin_weights_total=0
			for v in vertices:
				skin_weights_total+=len(v.weights)
				for w in v.weights:
					skin_weights+=" "+str(w)

			self.writel(S_CONT,4,'<float_array id="'+contid+'-skin_weights-array" count="'+str(skin_weights_total)+'">'+skin_weights+'</float_array>')
			self.writel(S_CONT,4,'<technique_common>')
			self.writel(S_CONT,4,'<accessor source="#'+contid+'-skin_weights-array" count="'+str(skin_weights_total)+'" stride="1">')
			self.writel(S_CONT,5,'<param name="WEIGHT" type="float"/>')
			self.writel(S_CONT,4,'</accessor>')
			self.writel(S_CONT,4,'</technique_common>')
			self.writel(S_CONT,3,'</source>')


			self.writel(S_CONT,3,'<joints>')
			self.writel(S_CONT,4,'<input semantic="JOINT" source="#'+contid+'-joints"/>')
			self.writel(S_CONT,4,'<input semantic="INV_BIND_MATRIX" source="#'+contid+'-bind_poses"/>')
			self.writel(S_CONT,3,'</joints>')
			self.writel(S_CONT,3,'<vertex_weights count="'+str(len(vertices))+'">')
			self.writel(S_CONT,4,'<input semantic="JOINT" source="#'+contid+'-joints" offset="0"/>')
			self.writel(S_CONT,4,'<input semantic="WEIGHT" source="#'+contid+'-skin_weights" offset="1"/>')
			vcounts=""
			vs=""
			vcount=0
			for v in vertices:
				vcounts+=" "+str(len(v.weights))
				for b in v.bones:
					vs+=" "+str(b)
					vs+=" "+str(vcount)
					vcount+=1
			self.writel(S_CONT,4,'<vcount>'+vcounts+'</vcount>')
			self.writel(S_CONT,4,'<v>'+vs+'</v>')
			self.writel(S_CONT,3,'</vertex_weights>')


			self.writel(S_CONT,2,'</skin>')
			self.writel(S_CONT,1,'</controller>')
			meshdata["skin_id"]=contid


		return meshdata


	def export_mesh_node(self,node,il,shapename=None):

		if (node.data==None):
			return
		armature=None

		if (node.parent!=None):
			if (node.parent.type=="ARMATURE"):
				armature=node.parent

		meshdata = self.export_mesh(node,armature,shapename)

		if (armature==None):
			self.writel(S_NODES,il,'<instance_geometry url="#'+meshdata["id"]+'">')
		else:
			self.writel(S_NODES,il,'<instance_controller url="#'+meshdata["skin_id"]+'">')
			for sn in self.skeleton_info[armature]["skeleton_nodes"]:
				self.writel(S_NODES,il+1,'<skeleton>#'+sn+'</skeleton>')


		if (len(meshdata["material_assign"])>0):

			self.writel(S_NODES,il+1,'<bind_material>')
			self.writel(S_NODES,il+2,'<technique_common>')
			for m in meshdata["material_assign"]:
				self.writel(S_NODES,il+3,'<instance_material symbol="'+m[1]+'" target="#'+m[0]+'"/>')

			self.writel(S_NODES,il+2,'</technique_common>')
			self.writel(S_NODES,il+1,'</bind_material>')

		if (armature==None):
			self.writel(S_NODES,il,'</instance_geometry>')
		else:
			self.writel(S_NODES,il,'</instance_controller>')


	def export_armature_bone(self,bone,il,si):
		boneid = self.new_id("bone")
		boneidx = si["bone_count"]
		si["bone_count"]+=1
		bonesid = si["name"]+"-"+str(boneidx)
		si["bone_index"][bone.name]=boneidx
		si["bone_ids"][bone]=boneid
		si["bone_names"].append(bonesid)
		self.writel(S_NODES,il,'<node id="'+boneid+'" sid="'+bonesid+'" name="'+bone.name+'" type="JOINT">')
		il+=1
		xform = bone.matrix_local
		si["bone_bind_poses"].append((si["armature_xform"] * xform).inverted())

		if (bone.parent!=None):
			xform = bone.parent.matrix_local.inverted() * xform
		else:
			si["skeleton_nodes"].append(boneid)

		self.writel(S_NODES,il,'<matrix sid="transform">'+strmtx(xform)+'</matrix>')
		for c in bone.children:
			self.export_armature_bone(c,il,si)
		il-=1
		self.writel(S_NODES,il,'</node>')


	def export_armature_node(self,node,il):

		if (node.data==None):
			return

		self.skeletons.append(node)

		armature = node.data
		self.skeleton_info[node]={ "bone_count":0, "name":node.name, "bone_index":{},"bone_ids":{},"bone_names":[],"bone_bind_poses":[],"skeleton_nodes":[],"armature_xform":node.matrix_world }



		for b in armature.bones:
			if (b.parent!=None):
				continue
			self.export_armature_bone(b,il,self.skeleton_info[node])

		if (node.pose):
			for b in node.pose.bones:
				for x in b.constraints:
					if (x.type=='ACTION'):
						self.action_constraints.append(x.action)


	def export_camera_node(self,node,il):

		if (node.data==None):
			return

		camera=node.data
		camid=self.new_id("camera")
		self.writel(S_CAMS,1,'<camera id="'+camid+'" name="'+camera.name+'">')
		self.writel(S_CAMS,2,'<optics>')
		self.writel(S_CAMS,3,'<technique_common>')
		if (camera.type=="PERSP"):
			self.writel(S_CAMS,4,'<perspective>')
			self.writel(S_CAMS,5,'<yfov> '+str(math.degrees(camera.angle))+' </yfov>') # I think?
			self.writel(S_CAMS,5,'<aspect_ratio> '+str(self.scene.render.resolution_x / self.scene.render.resolution_y)+' </aspect_ratio>')
			self.writel(S_CAMS,5,'<znear> '+str(camera.clip_start)+' </znear>')
			self.writel(S_CAMS,5,'<zfar> '+str(camera.clip_end)+' </zfar>')
			self.writel(S_CAMS,4,'</perspective>')
		else:
			self.writel(S_CAMS,4,'<orthografic>')
			self.writel(S_CAMS,5,'<xmag> '+str(camera.ortho_scale)+' </xmag>') # I think?
			self.writel(S_CAMS,5,'<aspect_ratio> '+str(self.scene.render.resolution_x / self.scene.render.resolution_y)+' </aspect_ratio>')
			self.writel(S_CAMS,5,'<znear> '+str(camera.clip_start)+' </znear>')
			self.writel(S_CAMS,5,'<zfar> '+str(camera.clip_end)+' </zfar>')
			self.writel(S_CAMS,4,'</orthografic>')

		self.writel(S_CAMS,3,'</technique_common>')
		self.writel(S_CAMS,2,'</optics>')
		self.writel(S_CAMS,1,'</camera>')


		self.writel(S_NODES,il,'<instance_camera url="#'+camid+'"/>')

	def export_lamp_node(self,node,il):

		if (node.data==None):
			return

		light=node.data
		lightid=self.new_id("light")
		self.writel(S_LAMPS,1,'<light id="'+lightid+'" name="'+light.name+'">')
		self.writel(S_LAMPS,2,'<optics>')
		self.writel(S_LAMPS,3,'<technique_common>')

		if (light.type=="POINT"):
			self.writel(S_LAMPS,4,'<point>')
			self.writel(S_LAMPS,5,'<color>'+strarr(light.color)+'</color>')
			att_by_distance = 2.0 / light.distance # convert to linear attenuation
			self.writel(S_LAMPS,5,'<linear_attenuation>'+str(att_by_distance)+'</linear_attenuation>')
			if (light.use_sphere):
				self.writel(S_LAMPS,5,'<zfar>'+str(light.distance)+'</zfar>')

			self.writel(S_LAMPS,4,'</point>')
		elif (light.type=="SPOT"):
			self.writel(S_LAMPS,4,'<spot>')
			self.writel(S_LAMPS,5,'<color>'+strarr(light.color)+'</color>')
			att_by_distance = 2.0 / light.distance # convert to linear attenuation
			self.writel(S_LAMPS,5,'<linear_attenuation>'+str(att_by_distance)+'</linear_attenuation>')
			self.writel(S_LAMPS,5,'<falloff_angle>'+str(math.degrees(light.spot_size))+'</falloff_angle>')
			self.writel(S_LAMPS,4,'</spot>')


		else: #write a sun lamp for everything else (not supported)
			self.writel(S_LAMPS,4,'<directional>')
			self.writel(S_LAMPS,5,'<color>'+strarr(light.color)+'</color>')
			self.writel(S_LAMPS,4,'</directional>')


		self.writel(S_LAMPS,3,'</technique_common>')
		self.writel(S_LAMPS,2,'</optics>')
		self.writel(S_LAMPS,1,'</light>')


		self.writel(S_NODES,il,'<instance_light url="#'+lightid+'"/>')


	def export_curve(self,curve):

		splineid = self.new_id("spline")

		self.writel(S_GEOM,1,'<geometry id="'+splineid+'" name="'+curve.name+'">')
		self.writel(S_GEOM,2,'<spline closed="0">')

		points=[]
		interps=[]
		handles_in=[]
		handles_out=[]
		tilts=[]

		for cs in curve.splines:

			if (cs.type=="BEZIER"):
				for s in cs.bezier_points:
					points.append(s.co[0])
					points.append(s.co[1])
					points.append(s.co[2])


					handles_in.append(s.handle_left[0])
					handles_in.append(s.handle_left[1])
					handles_in.append(s.handle_left[2])

					handles_out.append(s.handle_right[0])
					handles_out.append(s.handle_right[1])
					handles_out.append(s.handle_right[2])


					tilts.append(s.tilt)
					interps.append("BEZIER")
			else:

				for s in cs.points:
					points.append(s.co[0])
					points.append(s.co[1])
					points.append(s.co[2])
					handles_in.append(s.co[0])
					handles_in.append(s.co[1])
					handles_in.append(s.co[2])
					handles_out.append(s.co[0])
					handles_out.append(s.co[1])
					handles_out.append(s.co[2])
					tilts.append(s.tilt)
					interps.append("LINEAR")




		self.writel(S_GEOM,3,'<source id="'+splineid+'-positions">')
		position_values=""
		for x in points:
			position_values+=" "+str(x)
		self.writel(S_GEOM,4,'<float_array id="'+splineid+'-positions-array" count="'+str(len(points))+'">'+position_values+'</float_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+splineid+'-positions-array" count="'+str(len(points)/3)+'" stride="3">')
		self.writel(S_GEOM,5,'<param name="X" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Y" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Z" type="float"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,3,'</source>')

		self.writel(S_GEOM,3,'<source id="'+splineid+'-intangents">')
		intangent_values=""
		for x in handles_in:
			intangent_values+=" "+str(x)
		self.writel(S_GEOM,4,'<float_array id="'+splineid+'-intangents-array" count="'+str(len(points))+'">'+intangent_values+'</float_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+splineid+'-intangents-array" count="'+str(len(points)/3)+'" stride="3">')
		self.writel(S_GEOM,5,'<param name="X" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Y" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Z" type="float"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,3,'</source>')

		self.writel(S_GEOM,3,'<source id="'+splineid+'-outtangents">')
		outtangent_values=""
		for x in handles_out:
			outtangent_values+=" "+str(x)
		self.writel(S_GEOM,4,'<float_array id="'+splineid+'-outtangents-array" count="'+str(len(points))+'">'+outtangent_values+'</float_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+splineid+'-outtangents-array" count="'+str(len(points)/3)+'" stride="3">')
		self.writel(S_GEOM,5,'<param name="X" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Y" type="float"/>')
		self.writel(S_GEOM,5,'<param name="Z" type="float"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,3,'</source>')

		self.writel(S_GEOM,3,'<source id="'+splineid+'-interpolations">')
		interpolation_values=""
		for x in interps:
			interpolation_values+=" "+x
		self.writel(S_GEOM,4,'<Name_array id="'+splineid+'-interpolations-array" count="'+str(len(interps))+'">'+interpolation_values+'</Name_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+splineid+'-interpolations-array" count="'+str(len(interps))+'" stride="1">')
		self.writel(S_GEOM,5,'<param name="INTERPOLATION" type="name"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,3,'</source>')


		self.writel(S_GEOM,3,'<source id="'+splineid+'-tilts">')
		tilt_values=""
		for x in tilts:
			tilt_values+=" "+str(x)
		self.writel(S_GEOM,4,'<float_array id="'+splineid+'-tilts-array" count="'+str(len(tilts))+'">'+tilt_values+'</float_array>')
		self.writel(S_GEOM,4,'<technique_common>')
		self.writel(S_GEOM,4,'<accessor source="#'+splineid+'-tilts-array" count="'+str(len(tilts))+'" stride="1">')
		self.writel(S_GEOM,5,'<param name="TILT" type="float"/>')
		self.writel(S_GEOM,4,'</accessor>')
		self.writel(S_GEOM,3,'</source>')

		self.writel(S_GEOM,3,'<control_vertices>')
		self.writel(S_GEOM,4,'<input semantic="POSITION" source="#'+splineid+'-positions"/>')
		self.writel(S_GEOM,4,'<input semantic="IN_TANGENT" source="#'+splineid+'-intangents"/>')
		self.writel(S_GEOM,4,'<input semantic="OUT_TANGENT" source="#'+splineid+'-outtangents"/>')
		self.writel(S_GEOM,4,'<input semantic="INTERPOLATION" source="#'+splineid+'-interpolations"/>')
		self.writel(S_GEOM,4,'<input semantic="TILT" source="#'+splineid+'-tilts"/>')
		self.writel(S_GEOM,3,'</control_vertices>')


		self.writel(S_GEOM,2,'</spline>')
		self.writel(S_GEOM,1,'</geometry>')

		return splineid

	def export_curve_node(self,node,il):

		if (node.data==None):
			return
		curveid = self.export_curve(node.data)

		self.writel(S_NODES,il,'<instance_geometry url="#'+curveid+'">')
		self.writel(S_NODES,il,'</instance_geometry>')



	def export_node(self,node,il,shapename=None):
		if (not self.is_node_valid(node)):
			return
		bpy.context.scene.objects.active = node

		if shapename != None:
			self.writel(S_NODES,il,'<node id="'+self.validate_id(node.name + '_' + shapename)+'" name="'+node.name+'_'+shapename+'" type="NODE">')
		else:
			self.writel(S_NODES,il,'<node id="'+self.validate_id(node.name)+'" name="'+node.name+'" type="NODE">')
		il+=1

		self.writel(S_NODES,il,'<matrix sid="transform">'+strmtx(node.matrix_local)+'</matrix>')
		print("NODE TYPE: "+node.type+" NAME: "+node.name)
		if (node.type=="MESH"):
			self.export_mesh_node(node,il,shapename)
		elif (node.type=="CURVE"):
			self.export_curve_node(node,il)
		elif (node.type=="ARMATURE"):
			self.export_armature_node(node,il)
		elif (node.type=="CAMERA"):
			self.export_camera_node(node,il)
		elif (node.type=="LAMP"):
			self.export_lamp_node(node,il)

		self.valid_nodes.append(node)
		if shapename==None:
			for x in node.children:
				self.export_node(x,il)
			if node.type=="MESH":
				for k in range(0,len(node.data.shape_keys.key_blocks)):
					shape = node.data.shape_keys.key_blocks[k]
					shape.value = 1.0
					node.active_shape_key_index = k
					p = node.data
					v = node.to_mesh(bpy.context.scene, True, "RENDER")
					node.data = v
					self.export_node(node,il,shape.name)
					node.data = p
					node.data.update()
		il-=1
		self.writel(S_NODES,il,'</node>')

	def is_node_valid(self,node):
		if (not node.type in self.config["object_types"]):
			return False
		if (self.config["use_active_layers"]):
			valid=False
			for i in range(20):
				if (node.layers[i] and  self.scene.layers[i]):
					valid=True
					break
			if (not valid):
				return False

		if (self.config["use_export_selected"] and not node.select):
			return False

		return True


	def export_scene(self):


		self.writel(S_NODES,0,'<library_visual_scenes>')
		self.writel(S_NODES,1,'<visual_scene id="'+self.scene_name+'" name="scene">')

		for obj in self.scene.objects:
			if (obj.parent==None):
				self.export_node(obj,2)

		self.writel(S_NODES,1,'</visual_scene>')
		self.writel(S_NODES,0,'</library_visual_scenes>')

	def export_asset(self):


		self.writel(S_ASSET,0,'<asset>')
		# Why is this time stuff mandatory?, no one could care less...
		self.writel(S_ASSET,1,'<contributor>')
		self.writel(S_ASSET,2,'<author> Anonymous </author>') #Who made Collada, the FBI ?
		self.writel(S_ASSET,2,'<authoring_tool> Collada Exporter for Blender 2.6+, by Juan Linietsky (juan@codenix.com) </authoring_tool>') #Who made Collada, the FBI ?
		self.writel(S_ASSET,1,'</contributor>')
		self.writel(S_ASSET,1,'<created>'+time.strftime("%Y-%m-%dT%H:%M:%SZ     ")+'</created>')
		self.writel(S_ASSET,1,'<modified>'+time.strftime("%Y-%m-%dT%H:%M:%SZ")+'</modified>')
		self.writel(S_ASSET,1,'<unit meter="1.0" name="meter"/>')
		self.writel(S_ASSET,1,'<up_axis>Z_UP</up_axis>')
		self.writel(S_ASSET,0,'</asset>')


	def export_animation_transform_channel(self,target,transform_keys):

		frame_total=len(transform_keys)
		anim_id=self.new_id("anim")
		self.writel(S_ANIM,1,'<animation id="'+anim_id+'">')
		source_frames = ""
		source_transforms = ""
		source_interps = ""

		for k in transform_keys:
			source_frames += " "+str(k[0])
			source_transforms += " "+strmtx(k[1])
			source_interps +=" LINEAR"


		# Time Source
		self.writel(S_ANIM,2,'<source id="'+anim_id+'-input">')
		self.writel(S_ANIM,3,'<float_array id="'+anim_id+'-input-array" count="'+str(frame_total)+'">'+source_frames+'</float_array>')
		self.writel(S_ANIM,3,'<technique_common>')
		self.writel(S_ANIM,4,'<accessor source="'+anim_id+'-input-array" count="'+str(frame_total)+'" stride="1">')
		self.writel(S_ANIM,5,'<param name="TIME" type="float"/>')
		self.writel(S_ANIM,4,'</accessor>')
		self.writel(S_ANIM,3,'</technique_common>')
		self.writel(S_ANIM,2,'</source>')

		# Transform Source
		self.writel(S_ANIM,2,'<source id="'+anim_id+'-transform-output">')
		self.writel(S_ANIM,3,'<float_array id="'+anim_id+'-transform-output-array" count="'+str(frame_total*16)+'">'+source_transforms+'</float_array>')
		self.writel(S_ANIM,3,'<technique_common>')
		self.writel(S_ANIM,4,'<accessor source="'+anim_id+'-transform-output-array" count="'+str(frame_total)+'" stride="16">')
		self.writel(S_ANIM,5,'<param name="TRANSFORM" type="float4x4"/>')
		self.writel(S_ANIM,4,'</accessor>')
		self.writel(S_ANIM,3,'</technique_common>')
		self.writel(S_ANIM,2,'</source>')

		# Interpolation Source
		self.writel(S_ANIM,2,'<source id="'+anim_id+'-interpolation-output">')
		self.writel(S_ANIM,3,'<Name_array id="'+anim_id+'-interpolation-output-array" count="'+str(frame_total)+'">'+source_interps+'</Name_array>')
		self.writel(S_ANIM,3,'<technique_common>')
		self.writel(S_ANIM,4,'<accessor source="'+anim_id+'-interpolation-output-array" count="'+str(frame_total)+'" stride="1">')
		self.writel(S_ANIM,5,'<param name="INTERPOLATION" type="Name"/>')
		self.writel(S_ANIM,4,'</accessor>')
		self.writel(S_ANIM,3,'</technique_common>')
		self.writel(S_ANIM,2,'</source>')

		self.writel(S_ANIM,2,'<sampler id="'+anim_id+'-sampler">')
		self.writel(S_ANIM,3,'<input semantic="INPUT" source="#'+anim_id+'-input"/>')
		self.writel(S_ANIM,3,'<input semantic="OUTPUT" source="#'+anim_id+'-transform-output"/>')
		self.writel(S_ANIM,3,'<input semantic="INTERPOLATION" source="#'+anim_id+'-interpolation-output"/>')
		self.writel(S_ANIM,2,'</sampler>')
		self.writel(S_ANIM,2,'<channel source="#'+anim_id+'-sampler" target="'+target+'/transform"/>')
		self.writel(S_ANIM,1,'</animation>')

		return [anim_id]


	def export_animation(self,start,end,allowed=None):

		#Blender -> Collada frames needs a little work
		#Collada starts from 0, blender usually from 1
		#The last frame must be included also

		frame_len = 1.0 / self.scene.render.fps
		frame_total = end - start + 1
		frame_sub = 0
		if (start>0):
			frame_sub=start*frame_len

		tcn = []
		xform_cache={}
		# Change frames first, export objects last
		# This improves performance enormously

		print("anim from: "+str(start)+" to "+str(end)+" allowed: "+str(allowed))
		for t in range(start,end+1):
			self.scene.frame_set(t)
			key = t * frame_len - frame_sub
#			print("Export Anim Frame "+str(t)+"/"+str(self.scene.frame_end+1))

			for node in self.scene.objects:

				if (not node in self.valid_nodes):
					continue
				if (allowed!=None and not (node in allowed)):
					continue

				if (node.type=="MESH" and node.parent and node.parent.type=="ARMATURE"):
					continue #In Collada, nodes that have skin modifier must not export animation, animate the skin instead.

				if (len(node.constraints)>0 or node.animation_data!=None):
					#If the node has constraints, or animation data, then export a sampled animation track
					name=self.validate_id(node.name)
					if (not (name in xform_cache)):
						xform_cache[name]=[]

					mtx = node.matrix_world.copy()
					if (node.parent):
						mtx = node.parent.matrix_world.inverted() * mtx

					xform_cache[name].append( (key,mtx) )

				if (node.type=="ARMATURE"):
					#All bones exported for now
					for bone in node.data.bones:

						bone_name=self.skeleton_info[node]["bone_ids"][bone]

						if (not (bone_name in xform_cache)):
							print("has bone: "+bone_name)
							xform_cache[bone_name]=[]

						posebone = node.pose.bones[bone.name]
						parent_posebone=None

						mtx = posebone.matrix.copy()
						if (bone.parent):
							parent_posebone=node.pose.bones[bone.parent.name]
							parent_invisible=False

							for i in range(3):
								if (parent_posebone.scale[i]==0.0):
								    parent_invisible=True

							if (not parent_invisible):
								mtx = parent_posebone.matrix.inverted() * mtx


						xform_cache[bone_name].append( (key,mtx) )


		#export animation xml
		for nid in xform_cache:
			tcn+=self.export_animation_transform_channel(nid,xform_cache[nid])

		return tcn

	def export_animations(self):

		self.writel(S_ANIM,0,'<library_animations>')



		if (self.config["use_anim_action_all"] and len(self.skeletons)):

			self.writel(S_ANIM_CLIPS,0,'<library_animation_clips>')

			for x in bpy.data.actions[:]:
				if x in self.action_constraints:
					continue

				bones=[]
				#find bones used
				for p in x.fcurves:
					dp = str(p.data_path)
					base = "pose.bones[\""
					if (dp.find(base)==0):
						dp=dp[len(base):]
						if (dp.find('"')!=-1):
							dp=dp[:dp.find('"')]
							if (not dp in bones):
								bones.append(dp)

				allowed_skeletons=[]
				for y in self.skeletons:
					if (y.animation_data):
						for z in y.pose.bones:
							if (z.bone.name in bones):
								if (not y in allowed_skeletons):
									allowed_skeletons.append(y)
						y.animation_data.action=x;



				print(str(x))

				tcn = self.export_animation(int(x.frame_range[0]),int(x.frame_range[1]),allowed_skeletons)
				framelen=(1.0/self.scene.render.fps)
				start = x.frame_range[0]*framelen
				end = x.frame_range[1]*framelen
				print("Export anim: "+x.name)
				self.writel(S_ANIM_CLIPS,1,'<animation_clip name="'+x.name+'" start="'+str(start)+'" end="'+str(end)+'">')
				for z in tcn:
					self.writel(S_ANIM_CLIPS,2,'<instance_animation url="#'+z+'">')
				self.writel(S_ANIM_CLIPS,1,'</animation_clip>')


			self.writel(S_ANIM_CLIPS,0,'</library_animation_clips>')

		else:
			self.export_animation(self.scene.frame_start,self.scene.frame_end)

		self.writel(S_ANIM,0,'</library_animations>')

	def export(self):

		self.writel(S_GEOM,0,'<library_geometries>')
		self.writel(S_CONT,0,'<library_controllers>')
		self.writel(S_CAMS,0,'<library_cameras>')
		self.writel(S_LAMPS,0,'<library_lights>')
		self.writel(S_IMGS,0,'<library_images>')
		self.writel(S_MATS,0,'<library_materials>')
		self.writel(S_FX,0,'<library_effects>')


		self.skeletons=[]
		self.action_constraints=[]
		self.export_asset()
		self.export_scene()

		self.writel(S_GEOM,0,'</library_geometries>')
		self.writel(S_CONT,0,'</library_controllers>')
		self.writel(S_CAMS,0,'</library_cameras>')
		self.writel(S_LAMPS,0,'</library_lights>')
		self.writel(S_IMGS,0,'</library_images>')
		self.writel(S_MATS,0,'</library_materials>')
		self.writel(S_FX,0,'</library_effects>')

		if (self.config["use_anim"]):
		    self.export_animations()

		try:
			f = open(self.path,"wb")
		except:
			return False

		f.write(bytes('<?xml version="1.0" encoding="utf-8"?>\n',"UTF-8"))
		f.write(bytes('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">\n',"UTF-8"))


		s=[]
		for x in self.sections.keys():
			s.append(x)
		s.sort()
		for x in s:
			for l in self.sections[x]:
				f.write(bytes(l+"\n","UTF-8"))

		f.write(bytes('<scene>\n',"UTF-8"))
		f.write(bytes('\t<instance_visual_scene url="#'+self.scene_name+'" />\n',"UTF-8"))
		f.write(bytes('</scene>\n',"UTF-8"))
		f.write(bytes('</COLLADA>\n',"UTF-8"))
		return True

	def __init__(self,path,kwargs):
		self.scene=bpy.context.scene
		self.last_id=0
		self.scene_name=self.new_id("scene")
		self.sections={}
		self.path=path
		self.mesh_cache={}
		self.curve_cache={}
		self.material_cache={}
		self.image_cache={}
		self.skeleton_info={}
		self.config=kwargs
		self.valid_nodes=[]




def save(operator, context,
	filepath="",
	use_selection=False,
	**kwargs
	):

	exp = DaeExporter(filepath,kwargs)
	exp.export()

	return {'FINISHED'}  # so the script wont run after we have batch exported.


