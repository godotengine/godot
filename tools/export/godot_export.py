#!BPY
# -*- coding: utf-8 -*-
"""
Name: 'godot export (.xml)...'
Blender: 241
Group: 'Export'
Tooltip: 'Godot exporter'
"""


godot_revision="$Rev: 2068 $"

VERSION = "1.0"

import os
import Blender
import math
from Blender.BGL import *

MAX_WEIGHTS_PER_VERTEX = 4

class ExporterData:
	def __init__(self, fname):

		self.resource_list=[]
		self.mesh_caches={}
		self.material_caches={}
		self.filename = fname

class ObjectTree:
	

	def add(self,p_prop,p_val):
		self._properties+=[(p_prop,p_val)]
		
	def __init__(self,p_parent,p_type,p_name=""):
		self._parent=p_parent
		self._name=p_name
		self._type=p_type
		self._properties=[]
		self._children=[]
		self._resource=False
		self._res_path=""
		self._bone_map=None


def get_root_objects(scene):
	objs=[]
	for x in list(scene.objects):
		parent = x.getParent()
		if (parent==None):
			objs+=[x]
	return objs

def get_children_objects(scene,node):
	objs=[]
	for x in list(scene.objects):
		if (x.getParent()==None or x.getParent().getName()!=node.getName()):
			continue
		objs+=[x]
			
	return objs


def convert_matrix(m):

	mat = m.copy()
	

# Invert Z by Y, including position, but leave [2][2] alone, which is done by mirroring
# 

	for col in range(4):
		tmp = mat[col][1]
		mat[col][1] = mat[col][2]
		mat[col][2] = tmp

	for row in range(4):
		tmp = mat[1][row]
		mat[1][row] = mat[2][row]
		mat[2][row] = tmp


	mat[2][0]=-mat[2][0]
	mat[2][1]=-mat[2][1]
	mat[2][3]=-mat[2][3]
	mat[0][2]=-mat[0][2]
	mat[1][2]=-mat[1][2]
	mat[3][2]=-mat[3][2]

	return mat


def eq_vec(a,b):
	return (a.distance_to(b)<0.0001)

def eq_uv(a,b):
	return (a.distance_to(b)<0.0001)

def add_vec(a,b):
	return Vector3( (a.x+b.x, a.y+b.y, a.z+b.z ) )
	
def sub_vec(a,b):
	return Vector3( (a.x-b.x, a.y-b.y, a.z-b.z ) )

def mul_vec(a,b):
	return Vector3( (a.x*b.x, a.y*b.y, a.z*b.z ) )

def dot_vec(a,b):
	return a.x*b.x + a.y*b.y + a.z*b.z

def cross_vec(a,b):
       	x =  (a.y * b.z) - (a.z * b.y);
	y =  (a.z * b.x) - (a.x * b.z);
	z =  (a.x * b.y) - (a.y * b.x);
	return Vector3( (x,y,z) )

def mul_vecs(a,s):
	return Vector3( (a.x*s, a.y*s, a.z*s) )


def div_vecs(a,s):
	return Vector3( (a.x/s, a.y/s, a.z/s) )


class Color:
	def average(self):
		return (self.r+self.g+self.b)/3.0
		
	def __init__(self,tup):

		self.r=0
		self.g=0
		self.b=0
		self.a=1.0
		
		if (len(tup)>=1):
			self.r=tup[0]
		if (len(tup)>=2):
			self.g=tup[1]
		if (len(tup)>=3):
			self.b=tup[2]
		if (len(tup)>=4):
			self.a=tup[3]




class Vector3:
	
	def distance_to(self,v):
		return math.sqrt( (self.x-v.x)**2 + (self.y-v.y)**2 +  (self.z-v.z)**2 );
	def length(self):
		return math.sqrt( self.x**2 + self.y**2 + self.z**2 )
	def normalize(self):
		l=self.length()
		if (l==0.0):
			return
		self.x/=l
		self.y/=l
		self.z/=l
	
	def __init__(self,tup):
		self.x=0
		self.y=0
		self.z=0
		
		if (len(tup)>=1):
			self.x=tup[0]
		if (len(tup)>=2):
			self.y=tup[1]
		if (len(tup)>=3):
			self.z=tup[2]


class Matrix4x3:

	def invert(self):
		
		self.m[0][1], self.m[1][0]=self.m[1][0], self.m[0][1]
		self.m[0][2], self.m[2][0]=self.m[2][0], self.m[0][2]
		self.m[1][2], self.m[2][1]=self.m[2][1], self.m[1][2]

		x= -self.m[0][3];
		y= -self.m[1][3];
		z= -self.m[2][3];

		self.m[0][3]= (self.m[0][0]*x ) + ( self.m[1][0]*y ) + ( self.m[2][0]*z );
		self.m[1][3]= (self.m[0][1]*x ) + ( self.m[1][1]*y ) + ( self.m[2][1]*z );
		self.m[2][3]= (self.m[0][2]*x ) + ( self.m[1][2]*y ) + ( self.m[2][2]*z );
		
	def mult_by(self,mat):
		
		new_m=Matrix4x3()
		for j in range(4):
			for i in range(3):
				ab = 0;
				for k in range(3):
					ab += self.m[i][k] * mat.m[k][j];
					
				new_m.m[i][j]=ab;
		self.m=new_m.m
	"""
	def mult_by(mat):
		res=Matrix4x3()
		res.elements[0][0] =solf.m[0][0] * self.m[0][0] +solf.m[0][1] * self.m[1][0] +solf.m[0][2] * self.m[2][0];
		res.elements[0][1] =solf.m[0][0] * self.m[0][1] +solf.m[0][1] * self.m[1][1] +solf.m[0][2] * self.m[2][1];
		res.elements[0][2] =solf.m[0][0] * self.m[0][2] +solf.m[0][1] * self.m[1][2] +solf.m[0][2] * self.m[2][2];

		res.elements[1][0] =solf.m[1][0] * self.m[0][0] +solf.m[1][1] * self.m[1][0] +solf.m[1][2] * self.m[2][0];
		res.elements[1][1] =solf.m[1][0] * self.m[0][1] +solf.m[1][1] * self.m[1][1] +solf.m[1][2] * self.m[2][1];
		res.elements[1][2] =solf.m[1][0] * self.m[0][2] +solf.m[1][1] * self.m[1][2] +solf.m[1][2] * self.m[2][2];

		res.elements[2][0] =solf.m[2][0] * self.m[0][0] +solf.m[2][1] * self.m[1][0] +solf.m[2][2] * self.m[2][0];
		res.elements[2][1] =solf.m[2][0] * self.m[0][1] +solf.m[2][1] * self.m[1][1] +solf.m[2][2] * self.m[2][1];
		res.elements[2][2] =solf.m[2][0] * self.m[0][2] +solf.m[2][1] * self.m[1][2] +solf.m[2][2] * self.m[2][2];
	"""
	def xform(self,vec):
		
		x=self.m[0][0] * vec.x + self.m[0][1] * vec.y + self.m[0][2] * vec.z + self.m[0][3]
		y=self.m[1][0] * vec.x + self.m[1][1] * vec.y + self.m[1][2] * vec.z + self.m[1][3]
		z=self.m[2][0] * vec.x + self.m[2][1] * vec.y + self.m[2][2] * vec.z + self.m[2][3]
		return Vector3( (x,y,z ) )

	def xform_basis(self,vec):
		
		x=self.m[0][0] * vec.x + self.m[0][1] * vec.y + self.m[0][2] * vec.z
		y=self.m[1][0] * vec.x + self.m[1][1] * vec.y + self.m[1][2] * vec.z
		z=self.m[2][0] * vec.x + self.m[2][1] * vec.y + self.m[2][2] * vec.z
		return Vector3( (x,y,z ) )

	def copy(self):
		ret=Matrix4x3();
		for i in range(3):
			for j in range(4):
				ret.m[i][j]=self.m[i][j]
		return ret;
		
	def setBlenderMatrix(self,bm):
		for i in range(3):
			for j in range(3):
				self.m[i][j]=bm[i][j]
				
			self.m[i][3]=bm[3][i] #weird

	def getBlenderMatrix(self):
		bm=Blender.Mathutils.Matrix([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1])
		for i in range(3):
			for j in range(3):
				bm[i][j]=self.m[i][j]
				
			bm[3][i]=self.m[i][3] #weird
				
		return bm;

	def getPos(self):
		return Vector3( (self.m[0][3], self.m[1][3], self.m[2][3]) )
	def getScale(self):
		norm=((\
			Vector3((self.m[0][0], self.m[0][1], self.m[0][2])).length(),\
			Vector3((self.m[1][0], self.m[1][1], self.m[1][2])).length(),\
			Vector3((self.m[2][0], self.m[2][1], self.m[2][2])).length()\
			)) 
		return Vector3(norm)

	def scale(self,s):
		self.m[0][0]*=s.x;
		self.m[0][1]*=s.x;
		self.m[0][2]*=s.x;
		self.m[0][3]*=s.x;
		self.m[1][0]*=s.y;
		self.m[1][1]*=s.y;
		self.m[1][2]*=s.y;
		self.m[1][3]*=s.y;
		self.m[2][0]*=s.z;
		self.m[2][1]*=s.z;
		self.m[2][2]*=s.z;
		self.m[2][3]*=s.z;

	def scale3x3(self,s):
		self.m[0][0]*=s.x;
		self.m[0][1]*=s.x;
		self.m[0][2]*=s.x;
		self.m[1][0]*=s.y;
		self.m[1][1]*=s.y;
		self.m[1][2]*=s.y;
		self.m[2][0]*=s.z;
		self.m[2][1]*=s.z;
		self.m[2][2]*=s.z;

	def clearScale(self):
		s=self.getScale();
		s.x=1.0/s.x
		s.y=1.0/s.y
		s.z=1.0/s.z
		self.scale3x3(s)
	def set_rotation( self, p_axis, p_phi ):
		axis_sq = Vector3([p_axis.x*p_axis.x,p_axis.y*p_axis.y,p_axis.z*p_axis.z])

		cosine= math.cos(p_phi);
		sine= math.sin(p_phi);

		self.m[0][0] = axis_sq.x + cosine * ( 1.0 - axis_sq.x );
		self.m[0][1] = p_axis.x * p_axis.y *  ( 1.0 - cosine ) + p_axis.z * sine;
		self.m[0][2] = p_axis.z * p_axis.x * ( 1.0 - cosine ) - p_axis.y * sine;

		self.m[1][0] = p_axis.x * p_axis.y * ( 1.0 - cosine ) - p_axis.z * sine;
		self.m[1][1] = axis_sq.y + cosine  * ( 1.0 - axis_sq.y );
		self.m[1][2] = p_axis.y * p_axis.z * ( 1.0 - cosine ) + p_axis.x * sine;

		self.m[2][0] = p_axis.z * p_axis.x * ( 1.0 - cosine ) + p_axis.y * sine;
		self.m[2][1] = p_axis.y * p_axis.z * ( 1.0 - cosine ) - p_axis.x * sine;
		self.m[2][2] = axis_sq.z + cosine  * ( 1.0 - axis_sq.z );

	def __init__(self):			
		self.m=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]


class Quat:

	def distance_to(self,v):
		return math.sqrt( (self.x-v.x)**2 + (self.y-v.y)**2 +  (self.z-v.z)**2+  (self.w-v.w)**2 );

	def __init__(self,p_mat):
		"""			
		q=mat.getBlenderMatrix().toQuat();
		self.x=q.x;
		self.y=q.y;
		self.z=q.z;
		self.w=q.w;		
		"""	
		
		mat=p_mat.copy()
#create quaternion from 4x3 matrix

		trace = mat.m[0][0] + mat.m[1][1] + mat.m[2][2];
		temp=[0,0,0,0];
	
		if (trace > 0) :

			s =math.sqrt(trace + 1.0);
			temp[3]=(s * 0.5);
			s = 0.5 / s;
			
			temp[0]=((mat.m[2][1] - mat.m[1][2]) * s);
			temp[1]=((mat.m[0][2] - mat.m[2][0]) * s);
			temp[2]=((mat.m[1][0] - mat.m[0][1]) * s);

		else :
		
			i=int()
			if (mat.m[0][0] < mat.m[1][1]):
				if (mat.m[1][1] < mat.m[2][2]):
					i=2
				else:
					i=1
			else:
				if (mat.m[0][0] < mat.m[2][2]):
					i=2
				else:
					i=0
				
			j = (i + 1) % 3;  
			k = (i + 2) % 3;
			
			s = math.sqrt(mat.m[i][i] - mat.m[j][j] - mat.m[k][k] + 1.0);
			temp[i] = s * 0.5;
			s = 0.5 / s;
			
			temp[3] = (mat.m[k][j] - mat.m[j][k]) * s;
			temp[j] = (mat.m[j][i] + mat.m[i][j]) * s;
			temp[k] = (mat.m[k][i] + mat.m[i][k]) * s;
			
		self.x=temp[0]
		self.y=temp[1]
		self.z=temp[2]
		self.w=temp[3]

def snap_vec(vec):
	ret=()
	for x in vec:
		ret+=( x-math.fmod(x,0.0001), )
		
	return vec
	
class Surface:
	
	
	def write_to_res(self,res,i):
		prep="surfaces/"+str(i)+"/"
		format={}
		format["primitive"]=4 # triangles
		format["array_len"]=len(self._verts)
		format["index_array_len"]=len(self._indices)

		res.add(prep+"format",format)

		if (self._material!=None):
			res.add(prep+"material",self._material)
		res.add(prep+"vertex_array",self._verts)
		res.add(prep+"normal_array",self._normals)
		res.add(prep+"index_array",self._indices)
		format_str="vin"
		
		if (len(self._tangents)):
			res.add(prep+"tangent_array",self._tangents)
			format_str+="t"
		
		if (len(self._colors)):
			res.add(prep+"color_array",self._colors)
			format_str+="c"

		if (len(self._uvs)):
			res.add(prep+"tex_uv_array",self._uvs)
			format_str+="u"

		if (len(self._bone_indices)):
			res.add(prep+"bone_array",self._bone_indices)
			format_str+="b"
			
		if (len(self._weights)):
			res.add(prep+"weights_array",self._weights)
			format_str+="w"
			
		# binormals....
		format["format"]=format_str
			
# convert vertices to be compatile with Y_UP

	def fix_vertex_axis(self,v):

		return Vector3( (v.x, v.z, -v.y) );

	def convert(self,applymatrix=None):

		# STEP 1 fix coordinates
		for i in range(len(self._verts)):
			self._verts[i]=self.fix_vertex_axis(self._verts[i])
			self._normals[i]=self.fix_vertex_axis(self._normals[i])
		if (applymatrix):
			for i in range(len(self._verts)):
				self._verts[i]=applymatrix.xform( self._verts[i] )
				self._normals[i]=applymatrix.xform_basis( self._normals[i] )

		# STEP 2 fix indices

		for i in range(len(self._indices)/3):
			aux=self._indices[i*3+1]
			self._indices[i*3+1]=self._indices[i*3+2]
			self._indices[i*3+2]=aux
						
		# STEP 4 compute binormals
		if (len(self._uvs)):
			
			tangents=[ Vector3( (0,0,0 ) ) ] * len(self._verts)
			binormals=[ Vector3( (0,0,0 ) ) ] * len(self._verts)
			for i in range(len(self._indices)/3):

				v1 = self._verts[ self._indices[i*3+0] ]
				v2 = self._verts[ self._indices[i*3+1] ]
				v3 = self._verts[ self._indices[i*3+2] ]
				
				w1 = self._uvs[ self._indices[i*3+0] ] 
				w2 = self._uvs[ self._indices[i*3+1] ]
				w3 = self._uvs[ self._indices[i*3+2] ]
				
     
				x1 = v2.x - v1.x
				x2 = v3.x - v1.x
				y1 = v2.y - v1.y
				y2 = v3.y - v1.y
				z1 = v2.z - v1.z
				z2 = v3.z - v1.z
				
				s1 = w2.x - w1.x
				s2 = w3.x - w1.x
				t1 = w2.y - w1.y
				t2 = w3.y - w1.y
				
				r  = (s1 * t2 - s2 * t1);
				if (r==0):
					binormal=Vector3((0,0,0))
					tangent=Vector3((0,0,0))
				else:
					tangent = Vector3(((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
					(t2 * z1 - t1 * z2) * r))
					binormal = Vector3(((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
					(s1 * z2 - s2 * z1) * r))							
					
				tangents[ self._indices[i*3+0] ] = add_vec( tangents[ self._indices[i*3+0] ], tangent )
				binormals[ self._indices[i*3+0] ] = add_vec( binormals[ self._indices[i*3+0] ], binormal )
				tangents[ self._indices[i*3+1] ] = add_vec( tangents[ self._indices[i*3+1] ], tangent )
				binormals[ self._indices[i*3+1] ] = add_vec( binormals[ self._indices[i*3+1] ], binormal )
				tangents[ self._indices[i*3+2] ] = add_vec( tangents[ self._indices[i*3+2] ], tangent )
				binormals[ self._indices[i*3+2] ] = add_vec( binormals[ self._indices[i*3+2] ], binormal )
				
				
			for i in range(len(tangents)):
				
				T = tangents[i]
				T.normalize()
				B = binormals[i]
				B.normalize()
				N=self._normals[i]
				Tp = T #sub_vec( T, mul_vecs( N, dot_vec( N, T ) ) )
				#Tp.normalize()
				Bx = cross_vec( N, Tp )
				if (dot_vec( Bx, B )<0):
					Bw=-1.0
				else:
					Bw=1.0
					
				self._tangents.append(float(Tp.x))
				self._tangents.append(float(Tp.y))
				self._tangents.append(float(Tp.z))
				self._tangents.append(float(Bw))
		
		
		
	def _insertVertex(self,face,i):
		
		index_key=snap_vec((face.v[i].co.x,face.v[i].co.z,face.v[i].co.y))
		v=Vector3(face.v[i].co)

		if (face.smooth):
			index_key+=snap_vec((face.v[i].no[0],face.v[i].no[1],face.v[i].no[2]))
		else:
			index_key+=snap_vec((face.no[0],face.no[1],face.no[2]))


		uv=None
		if (self._has_uv):
			uv=face.uv[i]
			uv=Vector3((uv[0],1.0-uv[1],0)) #save as vector3
			index_key+=snap_vec((uv.x,uv.y))
			
		index=-1
		if (face.smooth and index_key in self._index_cache):
			index=self._index_cache[index_key]
			
						
		if (index==-1):
#no similar vertex exists, so create a new one
			self._verts+=[v]
			if (face.smooth):
				self._normals+=[Vector3(face.v[i].no)]
			else:
				self._normals+=[Vector3(face.no)]
			if (self._has_uv):
				self._uvs+=[uv]
			if (self._has_color):
				self._colors+=[Color((face.col[i].r/255.0,face.col[i].g/255.0,face.col[i].b/255.0,face.col[i].a/255.0))]
				
			if (self._vertex_weights!=None):
				for j in xrange(4):
					self._bone_indices.append( self._vertex_weights[face.v[i].index][j*2+0] )
					self._weights.append( self._vertex_weights[face.v[i].index][j*2+1] )
				
			index=len(self._verts)-1
			self._index_cache[index_key]=index
			
		self._indices+=[index]
			
			
		
		
		
	
	def insertFace(self,face):

		if (len(face.v)>=3):
			self._insertVertex(face,0)
			self._insertVertex(face,1)
			self._insertVertex(face,2)
		if (len(face.v)>=4):
			self._insertVertex(face,2)
			self._insertVertex(face,3)
			self._insertVertex(face,0)
		
	def __init__(self):
		self._mat=0
		self._verts=[]
		self._normals=[]
		self._tangents=[]
		self._colors=[]
		self._indices=[]
		self._uvs=[]
		self._has_uv=False
		self._bone_indices=[]
		self._weights=[]
		self._vertex_weights=[]
		self._has_color=False
		self._material=None
		self._index_cache={}


def make_material(mat,twosided_hint,exporter):


	if (mat.getName() in exporter.material_caches):
		# todo, find twosided and add it
		#if (twosided_hint):
		#	material_caches[mat.getName()]._two_sided=True
		return exporter.material_caches[mat.getName()]


	print("doesn't have it")
		
	res=ObjectTree(None,"FixedMaterial")
	res._resource=True
	res.add("resource/name",mat.getName())
#color
	diffuse_col = Color(mat.getRGBCol())
	diffuse_col.a = 1.0 # mat.getAlpha() this doesn't work..
	res.add("params/diffuse",diffuse_col)
	spec_col = Color(mat.getSpecCol())
	spec_col.r *= mat.getSpec()
	spec_col.g *= mat.getSpec()
	spec_col.b *= mat.getSpec()
	res.add("params/specular",spec_col)
	
	res.add("params/specular_exp",mat.getHardness())
	res.add("params/emission",Color([mat.getEmit(),mat.getEmit(),mat.getEmit()]))
#flags	
	res.add("flags/unshaded",bool(mat.getMode()&Blender.Material.Modes['SHADELESS']))
	res.add("flags/wireframe",bool(mat.getMode()&Blender.Material.Modes['WIRE']))
	res.add("flags/double_sided",bool(twosided_hint))
#textures

	have_primary=False
	have_detail=False
	detail_mix=1.0
	default_diffuse = Color((1,1,1,1))
	default_spec = Color((1,1,1,1))
	gen_mode=0

	for tx in mat.textures:
		if (tx==None):
			continue
		if (tx.tex.image==None):
			continue
		#gen_mode=0
		coord_mode=0
		
		if (tx.texco&Blender.Texture.TexCo['REFL']):
			gen_mode=1 # reflection
			coord_mode=3
		elif (tx.texco&Blender.Texture.TexCo['WIN']):
			gen_mode=2 # reflection
			coord_mode=3
		layer=""
		
		if (tx.mtCol and not have_primary):
			layer="textures/diffuse"
			have_primary=True
		elif (tx.mtCol and have_primary and not have_detail):
			layer="textures/detail"
			detail_mix = tx.colfac
			print("colfac: "+str(tx.colfac));
			have_detail=True
		elif (tx.mtNor):
			layer="textures/normal"
		elif (tx.mtSpec):
			layer="textures/specular"

		if (layer==""):
			continue

		img_file = tx.tex.image.getFilename()

		#Agregado por Ariel, trajo muchos problemas, lo saco.
		#img_file = Blender.sys.expandpath(tx.tex.image.getFilename())
		#exp_dir = os.path.dirname(exporter.filename)
		#img_file = os.path.relpath(os.path.abspath(img_file), exp_dir)

		img_file = img_file.replace("\\", "/")

		res.add(layer+"_tc",coord_mode)
		tex_res = ObjectTree(None,"Texture")
		tex_res._resource=True
		tex_res._res_path=img_file
		
		res.add(layer,tex_res)
		

	if (have_detail):
		res.add("params/detail_mix",detail_mix)
	if (gen_mode!=0):
		res.add("tex_gen",gen_mode)

	res._res_path="local://"+str(len(exporter.resource_list))

	exporter.resource_list.append(res)

	res_ref = ObjectTree(None,"Material")
	res_ref._resource=True
	res_ref._res_path=res._res_path


	exporter.material_caches[mat.getName()]=res_ref
	return res

def make_mesh_vertex_weights(node,skeleton):
	
	mesh = node.getData()
	verts=[]
	
	groups=mesh.getVertGroupNames()
	if (len(groups)==0):
		return None
		
	idx=0
	for x in mesh.verts:
		influences = mesh.getVertexInfluences(idx)
		inflist=[]
		for inf in influences:
			name=inf[0]
			if (not name in skeleton._bone_map):
				continue # no bone for group, ignore
			bone_idx=skeleton._bone_map[name]
			inflist.append( float(bone_idx) )
			inflist.append( inf[1] )
		
		verts.append(inflist)
		idx+=1


	for i in xrange(len(verts)):
		
		swaps=1
		while( swaps > 0 ):

			swaps=0
			for j in xrange(len(verts[i])/2-1):
				#small dirty bubblesort
				if (verts[i][j*2+1] < verts[i][(j+1)*2+1]):

					verts[i][j*2],verts[i][(j+1)*2]=verts[i][(j+1)*2],verts[i][j*2]

					verts[i][j*2+1],verts[i][(j+1)*2+1]=verts[i][(j+1)*2+1],verts[i][j*2+1]

					swaps+=1

		if ((len(verts[i])/2)>MAX_WEIGHTS_PER_VERTEX):
			#more than 4 weights, sort by most significant to least significant
			new_arr=[]
			
			for j in xrange(MAX_WEIGHTS_PER_VERTEX*2):
				new_arr+=[verts[i][j]]
			
			verts[i]=new_arr
			
		#make all the weights add up to 1
		max_w=0.0
		count=len(verts[i])/2
		
		for j in range(count):
			#small dirty bubblesort
			max_w+=verts[i][j*2+1]

		if (max_w>0.0):
			mult=1/max_w
			for j in range(count):
				verts[i][j*2+1]*=mult		
		#fill up empty slots
		while ((len(verts[i])/2)<MAX_WEIGHTS_PER_VERTEX):
			verts[i]+=[0,0] # add empty index
		
	return verts
	
	

def make_mesh(node,mesh,skeleton,exporter,applymatrix):


	mesh_res=ObjectTree(None,"Mesh")
	mesh_res._resource=True
	mesh_res.add("resource/name",mesh.name)


	#bake faces and surfaces

	weights=None

	if (skeleton!=None):
		weights=make_mesh_vertex_weights(node,skeleton)

	surfaces={}

	for f in mesh.faces:
		if (not f.mat in surfaces):
			surfaces[f.mat]=Surface()
			surfaces[f.mat]._vertex_weights=weights
			surfaces[f.mat]._has_uv=mesh.hasFaceUV()
			surfaces[f.mat]._has_color=mesh.hasVertexColours()
			surfaces[f.mat]._mat=f.mat

		surfaces[f.mat].insertFace(f)
	#bake materials

	for s in surfaces.values():
		if (s._mat<0 or s._mat>=len(mesh.materials)):
			continue
		s._material=make_material(mesh.materials[s._mat],(mesh.mode&Blender.Mesh.Modes['TWOSIDED'])!=0,exporter)

	#write surfaces
	surf_idx=1
	for x in surfaces.values():
		x.convert(applymatrix)
		x.write_to_res(mesh_res,surf_idx)
		surf_idx+=1


	mesh_res._res_path="local://"+str(len(exporter.resource_list))

	exporter.resource_list.append(mesh_res)

	res_ref = ObjectTree(None,"Mesh")
	res_ref._resource=True
	res_ref._res_path=mesh_res._res_path

	return mesh_res

def write_mesh(scene, node, tree,exporter):


	mesh = node.getData()
	tree._type="MeshInstance"

	skeleton=tree

	#find a skeleton

	while( skeleton!=None and skeleton._type!="Skeleton" ):
		skeleton=skeleton._parent

	mat=get_local_matrix(node)

	applymatrix=None

	if (skeleton):
		applymatrix=mat
	else:
		tree.add("transform/local",mat)

	#is mesh cached
	if (skeleton==None and mesh.name in exporter.mesh_caches):

		global last_local

		tree.add("mesh/mesh",exporter.mesh_caches[mesh.name])
		return tree

	#make mesh

	mesh_res = make_mesh(node,mesh,skeleton,exporter,applymatrix)
	tree.add("mesh/mesh",mesh_res)

	if (skeleton==None):
		exporter.mesh_caches[mesh.name]=mesh_res


	
	return tree

def write_armature_bone(bone,tree):
	
	idx=len(tree._bone_map)
	parent_idx=-1
	if (bone.parent != None):
		parent_idx = tree._bone_map[ bone.parent.name ]
	
	prop="bones/"+str(idx)+"/"
	mat = Matrix4x3()

	mat.setBlenderMatrix( convert_matrix(bone.matrix['ARMATURESPACE']) )
	if (bone.parent!=None):
		mat_parent=Matrix4x3()
		#mat_parent.scale(scale)
		mat_parent.setBlenderMatrix( convert_matrix( bone.parent.matrix['ARMATURESPACE'] ))
		mat_parent.invert()

		mat.setBlenderMatrix( mat.getBlenderMatrix() * mat_parent.getBlenderMatrix() )

	else:

		pass; #mat.scale(scale)
		
	tree.add(prop+"name",bone.name)
	tree.add(prop+"parent",parent_idx)
	tree.add(prop+"rest",mat)
	
	tree._bone_map[ bone.name ] = idx # map bone to idx
	
	for x in bone.children:
		
		write_armature_bone(x,tree)
	
	

def write_armature(scene, node, tree,exporter):

	mat=get_local_matrix(node)
	tree.add("transform/local",mat)

	mesh = node.getData()
	tree._type="Skeleton"
	tree._bone_map={}
	bone_map={}

	for x in node.data.bones.values():
		
		if (x.parent != None):
			continue
		
		write_armature_bone(x,tree)
	return tree


def write_camera(scene, node, tree,exporter):


	mesh = node.getData()
	tree._type="Camera"

	mat=get_local_matrix(node)
	tree.add("transform/local",mat)

	return tree

def write_empty(scene, node, tree,exporter):

	mat=get_local_matrix(node)
	tree.add("transform/local",mat)
	tree._type="Spatial"

	return tree



writers = {"Mesh": write_mesh, "Armature":write_armature, "Empty":write_empty, "Camera":write_camera }


def get_local_matrix(node):

	mat_bm=node.getMatrix('worldspace').copy()

	if (node.getParent()!=None):
		mat_parent_bm=node.getParent().getMatrix('worldspace').copy()
		mat_parent_bm.invert()

		mat_bm = mat_bm * mat_parent_bm


	if (node.getType()=="Camera"):
		mat2=Matrix4x3()
		mat2.set_rotation(Vector3([1,0,0]),-math.pi/2.0)
		mat2bm = mat2.getBlenderMatrix()
		mat_bm = mat2bm * mat_bm

	mat=Matrix4x3()
	mat.setBlenderMatrix(convert_matrix(mat_bm))

	return mat


def get_unscaled_matrix(node):

	mat_bm=convert_matrix(node.getMatrix('worldspace'))
	mat=Matrix4x3()
	mat.setBlenderMatrix(mat_bm)
	scale=mat.getScale()
#	print("--"+node.getName()+"  "+str(scale.x)+","+str(scale.y)+","+str(scale.z))
#	print(mat.m)
	mat.clearScale()
#	print(mat.getBlenderMatrix().determinant());
	
	if (node.getParent()!=None):
		mat_parent_bm=convert_matrix(node.getParent().getMatrix('worldspace'))
		
		mat_parent=Matrix4x3()
		mat_parent.setBlenderMatrix(mat_parent_bm)
		mat_parent.clearScale()
		mat_parent.invert()
		mat_scale=mat.getScale()
		mat_parent_scale=mat.getScale()
		
		if (False and node.getName()=="Cylinder.002"):
			
			print("Morth1? "+str(mat.m[0][0]*mat.m[1][0]+mat.m[0][1]*mat.m[1][1]+mat.m[0][2]*mat.m[1][2]))
			print("Morth2? "+str(mat.m[0][0]*mat.m[2][0]+mat.m[0][1]*mat.m[2][1]+mat.m[0][2]*mat.m[2][2]))
			print("Morth3? "+str(mat.m[1][0]*mat.m[2][0]+mat.m[1][1]*mat.m[2][1]+mat.m[1][2]*mat.m[2][2]))			
			print("North1? "+str(mat_parent.m[0][0]*mat_parent.m[1][0]+mat_parent.m[0][1]*mat_parent.m[1][1]+mat_parent.m[0][2]*mat_parent.m[1][2]))
			print("North2? "+str(mat_parent.m[0][0]*mat_parent.m[2][0]+mat_parent.m[0][1]*mat_parent.m[2][1]+mat_parent.m[0][2]*mat_parent.m[2][2]))
			print("North3? "+str(mat_parent.m[1][0]*mat_parent.m[2][0]+mat_parent.m[1][1]*mat_parent.m[2][1]+mat_parent.m[1][2]*mat_parent.m[2][2]))
			print(mat_parent.getBlenderMatrix().determinant());

		#print(m			
		#print(m
		
		mat_bm = mat.getBlenderMatrix();
		mat_parent_bm = mat_parent.getBlenderMatrix();
		mat_bm = mat_bm * mat_parent_bm
		mat.setBlenderMatrix(mat_bm)
		"""
		mat_parent.mult_by(mat)
		mat=mat_parent
		"""
		"""
		print("scale_mat "+str(mat_scale.x)+","+str(mat_scale.y)+","+str(mat_scale.z))
		print("scale_mat_parent "+str(mat_parent_scale.x)+","+str(mat_parent_scale.y)+","+str(mat_parent_scale.z))

		print("orth1? "+str(mat.m[0][0]*mat.m[1][0]+mat.m[0][1]*mat.m[1][1]+mat.m[0][2]*mat.m[1][2]))
		print("orth2? "+str(mat.m[0][0]*mat.m[2][0]+mat.m[0][1]*mat.m[2][1]+mat.m[0][2]*mat.m[2][2]))
		print("orth3? "+str(mat.m[1][0]*mat.m[2][0]+mat.m[1][1]*mat.m[2][1]+mat.m[1][2]*mat.m[2][2]))
		"""
	#print(m
	#print(mat.m)
	wscale=mat.getScale()		
			
	return mat,scale
	
def write_object(scene,node,tree,exporter):
	
	tree_node=ObjectTree(tree,"",node.getName())
		
	if writers.has_key(node.getType()):
		tree_node=writers[node.getType()](scene,node, tree_node,exporter)
	else:
		tree_node=None#write_dummy(node,tree)	

	if (tree_node != None):

		for node in get_children_objects(scene,node):
			write_object(scene, node, tree_node,exporter)

		tree._children+=[tree_node]


def export_scene(filename):
	
	exporter = ExporterData(filename)
	scene = None
	object = None

	scene = Blender.Scene.GetCurrent()
	if not scene:
		return
	tree = ObjectTree(None,"Spatial","Scene")
	write_scene(scene, tree,exporter)
	
	if widget_values["export_lua"]:
		write_godot_lua(tree, filename)
	else:
		write_godot_xml(tree,filename,exporter)

def write_scene(scene, tree,exporter):

	tree._name=scene.getName()
	for node in get_root_objects(scene):
		write_object(scene,node, tree,exporter )

""" --------- """
""" ANIMATION """
""" --------- """

class Animation:
	class Track:
    
		def insertKey(self,time,mat):	

			ofs = mat.getPos()
			rot = Quat(mat)
			scale = mat.getScale();

			self.xform_keys.append( time )
			self.xform_keys.append( 1.0 ) # transition
			self.xform_keys.append( ofs.x )
			self.xform_keys.append( ofs.y )
			self.xform_keys.append( ofs.z )

			self.xform_keys.append( -rot.x )
			self.xform_keys.append( -rot.y )
			self.xform_keys.append( -rot.z )
			self.xform_keys.append( rot.w )

			self.xform_keys.append( scale.x )
			self.xform_keys.append( scale.y )
			self.xform_keys.append( scale.z )


		def _optimized(self,arr):
			_new=[]
			#remove irrelevant keys
			for i in range( len(arr) ):
				if (i>0 and i<(len(arr)-1) and eq_vec(arr[i]["value"],arr[i+1]["value"]) and eq_vec(arr[i]["value"],arr[i-1]["value"])):
					continue
				_new.append(arr[i])
				
			return _new
		def optimize(self):
			#self.loc_keys=self._optimized(self.loc_keys)
			#self.rot_keys=self._optimized(self.rot_keys)
			#self.scale_keys=self._optimized(self.scale_keys)
			 pass

		def _get_track_array3(self,keys):
			_arr=[]
			for x in keys:
				_arr.append(x["time"])
				v=x["value"]
				_arr.append(v.x)
				_arr.append(v.y)
				_arr.append(v.z)
			return _arr;

		def _get_track_array4(self,keys):
			_arr=[]
			for x in keys:
				_arr.append(x["time"])
				v=x["value"]
				_arr.append(-v.x)
				_arr.append(-v.y)
				_arr.append(-v.z)
				_arr.append(v.w)
			return _arr;



		def write_to_res(self,res,i):
			prep="tracks/"+str(i)+"/"
			res.add(prep+"type","transform")	
			res.add(prep+"path",self.path)

			res.add(prep+"keys",self.xform_keys)

		
		def __init__(self):
			self.xform_keys=[]
			self.path=""
            
	def make_res(self):
		
		res = ObjectTree(None,"Animation")
		res._resource=True
		res.add("length",self.length);
		res.add("loop",self.loop);
		idx=0
		for t in self.tracks.values():
			t.optimize()
			t.write_to_res(res,idx)
			idx=idx+1
		return res

	def __init__(self):
		self.tracks={}
		self.fps=30
		self.length=0
        

def  write_animation_bone(scene,node,anim,path,bone,frame):

	rest = convert_matrix(bone.matrix['ARMATURESPACE'])


	if (bone.parent!=None):

		rest_parent = 	convert_matrix( bone.parent.matrix['ARMATURESPACE'] )
		rest_parent.invert()
		rest = rest * rest_parent

	bone_path = path+":"+bone.name;

	if (bone_path not in anim.tracks):
		t = Animation.Track()
		t.path=bone_path
		anim.tracks[bone_path] = t
	else:
		t=anim.tracks[bone_path]
	
	pose_bone = node.getPose().bones[bone.name]
	

	pose = convert_matrix(pose_bone.poseMatrix)

	if (bone.parent!=None):

		mat_parent=convert_matrix( pose_bone.parent.poseMatrix )
		mat_parent.invert()

		pose = pose * mat_parent

	# pose should actually be the transform from pose to rest

	rest.invert()
	pose = pose * rest

	mat43 = Matrix4x3()
	mat43.setBlenderMatrix(pose)

	t.insertKey(frame/float(anim.fps),mat43)

			

def write_animation_armature(scene,node,anim,path,frame):

	for x in node.data.bones.values():
				
		write_animation_bone(scene,node,anim,path,x,frame)


def write_animation_object(scene,node,anim,path,frame,parent_type):

	if not writers.has_key(node.getType()):
		return


	new_path=path+node.getName()

	if (path!=""):
		path=path+"/"+node.getName()
	else:
		path=node.getName()

	if (node.getType()=="Armature" or node.getIpo()!=None):
		#only export if it has animation
		if (path not in anim.tracks):
			t = Animation.Track()
			t.path=path
			anim.tracks[path] = t
		else:
			t=anim.tracks[path]


		if (parent_type!="Armature"):
			t.insertKey(frame/float(anim.fps),get_local_matrix(node))

		if (node.getType()=="Armature"):
			write_animation_armature(scene,node,anim,path,frame)
			return # children of armature will not be animated
		

	for node in get_children_objects(scene,node):
		write_animation_object(scene, node, anim, path, frame, node.getType())


def write_animation(scene, anim, frame):

	for node in get_root_objects(scene):
		write_animation_object(scene,node, anim,"",frame,"")

def export_animation(filename, end_frame = -1, loop = None):

	anim = Animation()
	anim.fps=Blender.Scene.GetCurrent().getRenderingContext().fps

	if loop == None:
		anim.loop=widget_values["anim_loop"]
	else:
		anim.loop = loop

	print("end_frame param: %d"%end_frame)
	if end_frame == -1:
		end_frame = Blender.Get("endframe")

	anim.length=(end_frame-Blender.Get("staframe")+1)/float(anim.fps)
	print("frames "+str((end_frame-Blender.Get("staframe")+1)))
	print("start: %d, end %d, fps %d, length %f" % (Blender.Get("staframe"), end_frame, anim.fps, anim.length));

	scene = Blender.Scene.GetCurrent()
	if not scene:
		return
		
	for frame in range( Blender.Get('staframe'), end_frame+1):
		Blender.Set("curframe",frame)
		write_animation(scene,anim,frame)

	anim_res = anim.make_res()
	res_name = filename

	if(res_name.rfind(".")!=-1):
		res_name=res_name[:res_name.rfind(".")]
	if(res_name.rfind("/")!=-1):
		res_name=res_name[res_name.rfind("/")+1:]
	if(res_name.rfind("\\")!=-1):
		res_name=res_name[res_name.rfind("\\")+1:]

	anim_res.add("resource/name",res_name)

	if widget_values['export_lua']:
		write_godot_lua(anim_res,filename)
	else:
		write_godot_xml(anim_res,filename,None)

""" -------------- """
""" SERIALIZATION """
""" ------------- """

def tw(f,t,st):
	for x in range(t):
		f.write("\t")
	nl = True
	if len(st) > 0 and st[-1] == "#":
		nl = False
		st = st[:-1]
	f.write(st)
	if nl:
		f.write("\n")

def write_property_godot(f,tab,name,value):
	
#	print(str(value))
#	print(type(value))
	if (type(value)==str):
		
		tw(f,tab,'<string name="'+name+'">')
		value=value.replace('"','\\&quot;')
		tw(f,tab+1,'"'+value+'"');
		tw(f,tab,'</string>')
	elif (type(value)==bool):
		tw(f,tab,'<bool name="'+name+'">')
		if (value):
			tw(f,tab+1,'True');
		else:
			tw(f,tab+1,'False');
		tw(f,tab,'</bool>')
	elif (type(value)==int):
		tw(f,tab,'<int name="'+name+'">')
		tw(f,tab+1,str(value));
		tw(f,tab,'</int>')
	elif (type(value)==float):
		tw(f,tab,'<real name="'+name+'">')
		tw(f,tab+1,str(value));
		tw(f,tab,'</real>')
	elif (type(value)==dict):
		tw(f,tab,'<dictionary name="'+name+'">')
		for x in value:
			write_property_godot(f,tab+1,"key",x)
			write_property_godot(f,tab+1,"value",value[x])
		tw(f,tab,'</dictionary>')
	elif (isinstance(value,ObjectTree)):
		if (not value._resource):
			print("ERROR: Not a resource!!")
			return
		if (value._res_path!=""):
			
			tw(f,tab,'<resource name="'+name+'" resource_type="'+value._type+'" path="'+value._res_path+'">')
			tw(f,tab,'</resource>')
		else:
			tw(f,tab,'<resource name="'+name+'" resource_type="'+value._type+'">')
			tw(f,tab+1,'<object type="'+value._type+'">')
			tw(f,tab+2,'<resource>')
			
			for x in value._properties:
				write_property_godot(f,tab+3,x[0],x[1])
			
			tw(f,tab+2,'</resource>')
			tw(f,tab+1,'</object>')			
			tw(f,tab,'</resource>')
	elif (isinstance(value,Color)):
		tw(f,tab,'<color name="'+name+'">')
		tw(f,tab+1,str(value.r)+", "+str(value.g)+", "+str(value.b)+", "+str(value.a));
		tw(f,tab,'</color>')
	elif (isinstance(value,Vector3)):
		tw(f,tab,'<vector3 name="'+name+'">')
		tw(f,tab+1,str(value.x)+", "+str(value.y)+", "+str(value.z));
		tw(f,tab,'</vector3>')
	elif (isinstance(value,Quat)):
		tw(f,tab,'<quaternion name="'+name+'">')
		tw(f,tab+1,str(-value.x)+", "+str(-value.y)+", "+str(-value.z)+", "+str(value.w));
		tw(f,tab,'</quaternion>')
	elif (isinstance(value,Matrix4x3)): # wtf, blender matrix?
		tw(f,tab,'<transform name="'+name+'" >')
		s=""
		for i in range(3):
			for j in range(3):
				s+=", "+str(value.m[j][i])
	
		for i in range(3):
			s+=", "+str(value.m[i][3])
		s=s[1:]
		tw(f,tab+1,s);
		tw(f,tab,'</transform>')
		
	elif (type(value)==list):
		if (len(value)==0):
			return
		first=value[0]
		if (type(first)==int):
			
			tw(f,tab,'<int_array name="'+name+'" len="'+str(len(value))+'">')
			arr=""
			for i in range(len(value)):
				if (i>0):
					arr+=", "
				arr+=str(value[i])
			tw(f,tab+1,arr)
			tw(f,tab,'</int_array>')
		elif (type(first)==float):
			
			tw(f,tab,'<real_array name="'+name+'" len="'+str(len(value))+'">')
			arr=""
			for i in range(len(value)):
				if (i>0):
					arr+=", "
				arr+=str(value[i])
			tw(f,tab+1,arr)
			tw(f,tab,'</real_array>')
		elif (type(first)==str):
			
			tw(f,tab,'<string_array name="'+name+'" len="'+str(len(value))+'">')
			arr=""
			for i in range(len(value)):
				if (i>0):
					arr+=", "
				arr+=str('"'+value[i]+'"')
			tw(f,tab+1,arr)
			tw(f,tab,'</string_array>')
		elif (isinstance(first,Vector3)):
			
			tw(f,tab,'<vector3_array name="'+name+'" len="'+str(len(value))+'">')
			arr=""
			for i in range(len(value)):
				if (i>0):
					arr+=", "
				arr+=str(str(value[i].x)+','+str(value[i].y)+','+str(value[i].z))
			tw(f,tab+1,arr)
			tw(f,tab,'</vector3_array>')
		elif (isinstance(first,Color)):
			
			tw(f,tab,'<color_array name="'+name+'" len="'+str(len(value))+'">')
			arr=""
			for i in range(len(value)):
				if (i>0):
					arr+=", "
				arr+=str(str(value[i].r)+','+str(value[i].g)+','+str(value[i].b)+','+str(value[i].a))
			tw(f,tab+1,arr)
			tw(f,tab,'</color_array>')
		elif (type(first)==dict):
			
			tw(f,tab,'<array name="'+name+'" len="'+str(len(value))+'">')
			for i in range(len(value)):
				write_property_godot(f,tab+1,str(i+1),value[i])
			tw(f,tab,'</array>')
				
	

def write_node_godot(f,tab,tree,path,root=False):

	if (root or not tree._resource):
		tw(f,tab,'<object type="'+tree._type+'">')
		tw(f,tab+1,'<dictionary name="__xml_meta__" type="dictionary">')
		write_property_godot(f,tab+3,"key","name")
		write_property_godot(f,tab+3,"value",tree._name)
		if (path!=""):
			write_property_godot(f,tab+3,"key","path")
			write_property_godot(f,tab+3,"value",path)

		tw(f,tab+1,'</dictionary>')
	else:
		if (tree._res_path!=""):
			tw(f,tab,'<resource type="'+tree._type+'" path="'+tree._res_path+'">')
		else:
			tw(f,tab,'<resource type="'+tree._type+'">')


	for x in tree._properties:
		write_property_godot(f,tab+1,x[0],x[1])

	if (root or not tree._resource):
		tw(f,tab,'</object>')
	else:
		tw(f,tab,'</resource>')

	if (path==""):
		path="."
	else:
		if (path=="."):
			path=tree._name
		else:
			path=path+"/"+tree._name
	#path="."

	for x in tree._children:
		write_node_godot(f,tab,x,path)

def write_godot_xml(tree,fname,exporter):

	f=open(fname,"wb")
	f.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
	if (not tree._resource):
		f.write('<object_file magic="SCENE" version="0.99">\n')
	else:
		f.write('<object_file magic="RESOURCE" version="0.99">\n')
		
	tab=1

	if (exporter!=None):
		for x in exporter.resource_list:
			write_node_godot(f,tab,x,"")

	write_node_godot(f,tab,tree,"",True)
	f.write('</object_file>\n')


def write_property_lua(f, tab, name, value, pref = ""):

	tw(f, tab, '%s{ name = "%s",' % (pref, name))
	tab = tab + 1

	if (type(value)==str):

		tw(f, tab, 'value = "%s",' % value)
		tw(f, tab, 'type = "string",')

	elif (type(value)==bool):


		if (value):
			tw(f, tab, 'value = true,')
		else:
			tw(f, tab, 'value = false,')
	
		tw(f, tab, 'type = "bool",')

	elif (type(value)==int):

		tw(f, tab, 'type = "int",')
		tw(f, tab, 'value = %d,' % value)

	elif (type(value)==float):

		tw(f, tab, 'type = "real",')
		tw(f, tab, 'value = %f,' % value)

	elif (type(value)==dict):
		
		tw(f, tab, 'type = "dictionary",')
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
		tw(f, tab, 'value = { %.20f, %.20f, %.20f, %.20f },' % (-value.x, -value.y, -value.z, value.w))

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
			tw(f, tab, 'value = { #')
			for i in range(len(value)):
				f.write("%d, " % value[i])
			f.write(" },\n")

		elif (type(first)==float):

			tw(f, tab, 'type = "real_array",')
			tw(f, tab, 'value = { #')
			for i in range(len(value)):
				f.write("%.20f, " % value[i])
			f.write(" },\n")


		elif (type(first)==str):
			
			tw(f, tab, 'type = "string_array",')
			tw(f, tab, 'value = { #')
			for i in range(len(value)):
				f.write('"%s", ' % value[i])
			f.write(" },\n")

		elif (isinstance(first,Vector3)):

			tw(f, tab, 'type = "vector3_array",')
			tw(f, tab, 'value = { #')
			for i in range(len(value)):
				f.write("{ %.20f, %.20f, %.20f }, " % (value[i].x, value[i].y, value[i].z))
			f.write(" },\n")

		elif (isinstance(first,Color)):

			tw(f, tab, 'type = "color_array",')
			tw(f, tab, 'value = { #')
			for i in range(len(value)):
				f.write("{ %.20f, %.20f, %.20f, %.20f }, " % (value[i].r, value[i].g, value[i].b, value[i].a))
			f.write(" },\n")

		elif (type(first)==dict):
			
			tw(f, tab, 'type = "dict_array",')
			tw(f, tab, 'value = {')
			
			for i in range(len(value)):
				write_property_lua(f,tab+1,str(i+1),value[i])
			
			tw(f, tab, '},')
			

	tw(f, tab-1, "},")


""" -------------- """
""" SERIALIZATION LUA """
""" ------------- """

def write_node_lua(f,tab,tree,path):

	tw(f, tab, '{ type = "%s",' % tree._type)
	
	if not tree._resource:
		tw(f, tab+1, 'meta = {')
		write_property_lua(f, tab+3, "name", tree._name)
		if path != "":
			write_property_lua(f, tab+3, "path", path)
		tw(f, tab+1, '},')
	
	tw(f, tab+1, "properties = {")
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

def write_godot_lua(tree,fname):
	f=open(fname,"wb")
	f.write("return {\n")

	f.write('\tmagic = "SCENE",\n')
	tab = 1

	write_node_lua(f,tab,tree,"")

	f.write("}\n\n")

		
widget_values={}

def action_path_change_callback(event, val):

	def callback(fname):
		widget_values["actions_scheme"] = fname
	Blender.Window.FileSelector(callback, "Save Action Scheme Name", widget_values["actions_scheme"])

def scene_path_change_callback(event,val):
	
	def callback(fname):
		widget_values["scene_path"]=fname
		
	Blender.Window.FileSelector(callback, "Save Scene XML",widget_values["scene_path"])

def scene_export_callback(event,val):
	export_scene( widget_values["scene_path"] )

def scene_lamps_cameras_changed(event,val):

	widget_values["scene_lamps_cameras"]=val

def anim_path_change_callback(event,val):
	
	def callback(fname):
		widget_values["anim_path"]=fname
		
	Blender.Window.FileSelector(callback, "Save Anim XML",widget_values["anim_path"])

def is_number(n):

	try:
		int(n)
	except:
		return False
	return True

def action_export_callback(event, val):

	import string

	idx = widget_values["actions_scheme"].rfind(".")
	if idx == -1:
		pref = widget_values["actions_scheme"]
		ext = ".xml"
	else:
		pref = widget_values["actions_scheme"][:idx]
		ext = widget_values["actions_scheme"][idx:]

	print("scheme is ", pref, ext)

	actions = Blender.Armature.NLA.GetActions()
	for k in actions.keys():

		l = string.split(k, "$");
		if len(l) <= 1:
			continue

		loop = 1
		endf = 0
		for v in l:
			if v == "nl":
				loop = 0
			if is_number(v):
				endf = int(v)

		if endf == 0:
			continue

		fname = pref + l[0] + ext
		print("fname is "+fname)

		objects = Blender.Object.Get()
		for o in objects:
			if o.getType() == "Armature":
				actions[k].setActive(o)

		print("writing with duration "+str(endf))
		export_animation(fname, endf, loop)

def anim_export_callback(event,val):
	export_animation( widget_values["anim_path"] )

def anim_fps_changed(event,val):

	widget_values["anim_fps"]=val

def anim_selected_changed(event,val):

	widget_values["anim_selected"]=val

def anim_loop_changed(event,val):

	widget_values["anim_loop"]=val

def export_lua_changed(event, val):
	widget_values["export_lua"] = val
	
def close_script(event,val):
#force a bug, because otherwise blender won't unload the script
	Blender.Draw.Exit()

def draw():
	Blender.Draw.Label("Godot Export v."+VERSION+"."+godot_revision+" (c) 2008 Juan Linietsky, Ariel Manzur.", 10,260,400,10);
	Blender.Draw.Label("Export Scene", 20,200,150,10);
	Blender.Draw.String(widget_values["scene_path"], 10,40, 170, 300, 20, "",398)
	Blender.Draw.Button("Choose", 0,340, 170, 70, 20, "",scene_path_change_callback)
	Blender.Draw.Button("Export", 0,410, 170, 70, 20, "",scene_export_callback)
	Blender.Draw.Toggle("Lamps & Cameras", 0,40, 140, 140, 20, widget_values["scene_lamps_cameras"],"",scene_lamps_cameras_changed)

	Blender.Draw.Label("Export Animation", 20,120,150,10);
	Blender.Draw.String(widget_values["anim_path"], 11, 40, 90, 300, 20, "",398)
	Blender.Draw.Button("Choose", 0,340, 90, 70, 20, "",anim_path_change_callback)
	Blender.Draw.Button("Export", 0,410, 90, 70, 20, "",anim_export_callback)
	Blender.Draw.Slider("FPS: ", 0, 40, 60, 120, 20,widget_values["anim_fps"],1,60,0,"",anim_fps_changed)
	Blender.Draw.Toggle("Only Selected", 0,180, 60, 120, 20, widget_values["anim_selected"],"",anim_selected_changed)
	Blender.Draw.Toggle("Loop", 0,320, 60, 60, 20, widget_values["anim_loop"],"",anim_loop_changed)
	Blender.Draw.Toggle("Export Lua", 0, 400, 60, 60, 20, widget_values["export_lua"], "", export_lua_changed)

	Blender.Draw.Label("Export Actions", 20,45,150,10);
	Blender.Draw.Label("Prefix", 40, 20, 50, 10)
	Blender.Draw.String(widget_values["actions_scheme"], 0, 40, 20, 300, 20, "",398)
	Blender.Draw.Button("Choose", 0,340, 20, 70, 20, "",action_path_change_callback)
	Blender.Draw.Button("Export", 0,410, 20, 70, 20, "",action_export_callback)
#	#	Blender.Draw.Button("Close", 0,410, 20, 70, 20, "",close_script)

widget_values["scene_path"]="scene.xml"
widget_values["anim_path"]="animation.xres"
widget_values["anim_fps"]=25
widget_values["anim_selected"]=0
widget_values["anim_loop"]=1
widget_values["scene_lamps_cameras"]=0
widget_values["export_lua"]=0
widget_values["actions_scheme"] = "action_.xml"

def event(ev, val):
	return None
	
def button_event(ev):
	return None

Blender.Draw.Register(draw, event, button_event)
