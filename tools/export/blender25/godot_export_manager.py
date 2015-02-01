# ##### BEGIN GPL LICENSE BLOCK #####
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# Script copyright (c) Andreas Esau

bl_info = {
    "name": "Godot Export Manager",
    "author": "Andreas Esau",
    "version": (1, 0),
    "blender": (2, 7, 0),
    "location": "Scene Properties > Godot Export Manager",
    "description": "Godot Export Manager uses the Better Collada Exporter to manage Export Groups and automatically export the objects groups to Collada Files.",
    "warning": "",
    "wiki_url": ("http://www.godotengine.org"),
    "tracker_url": "",
    "category": "Import-Export"}

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, CollectionProperty, PointerProperty
import os
from bpy.app.handlers import persistent
from mathutils import Vector, Matrix

class godot_export_manager(bpy.types.Panel):
    bl_label = "Godot Export Manager"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"
    
    bpy.types.Scene.godot_export_on_save = BoolProperty(default=False)
        
    ### draw function for all ui elements
    def draw(self, context):
        layout = self.layout
        split = self.layout.split()
        scene = bpy.data.scenes[0]
        ob = context.object
        scene = context.scene
        
        row = layout.row()
        col = row.column()
        col.prop(scene,"godot_export_on_save",text="Export Groups on save")
        
        row = layout.row()
        col = row.column(align=True)
        op = col.operator("scene.godot_add_objects_to_group",text="Add selected objects to Group",icon="COPYDOWN")
        
        op = col.operator("scene.godot_delete_objects_from_group",text="Delete selected objects from Group",icon="PASTEDOWN")
        
        
        
        row = layout.row()
        col = row.column()
        col.label(text="Export Groups:")
        
        
        row = layout.row()
        col = row.column()
        
        col.template_list("UI_List_Godot","dummy",scene, "godot_export_groups", scene, "godot_export_groups_index",rows=1,maxrows=10,type='DEFAULT')
        
        col = row.column(align=True)
        col.operator("scene.godot_add_export_group",text="",icon="ZOOMIN")
        col.operator("scene.godot_delete_export_group",text="",icon="ZOOMOUT")
        col.operator("scene.godot_export_all_groups",text="",icon="EXPORT")
        
        if len(scene.godot_export_groups) > 0:        
            row = layout.row()
            col = row.column()
            group = scene.godot_export_groups[scene.godot_export_groups_index]
            col.prop(group,"name",text="Group Name")
            col.prop(group,"export_name",text="Export Name")
            col.prop(group,"export_path",text="Export Filepath")
            
            row = layout.row()
            col = row.column()
            row = layout.row()
            col = row.column()
            col.label(text="Export Settings:")
            
            col = col.row(align=True)
            col.prop(group,"apply_loc",toggle=True,icon="MAN_TRANS")
            col.prop(group,"apply_rot",toggle=True,icon="MAN_ROT")
            col.prop(group,"apply_scale",toggle=True,icon="MAN_SCALE")
            
            row = layout.row()
            col = row.column()
            
            col.prop(group,"use_include_particle_duplicates")
            col.prop(group,"use_mesh_modifiers")
            col.prop(group,"use_tangent_arrays")
            col.prop(group,"use_triangles")
            col.prop(group,"use_copy_images")
            col.prop(group,"use_active_layers")
            col.prop(group,"use_exclude_ctrl_bones")
            col.prop(group,"use_anim")
            col.prop(group,"use_anim_action_all")
            col.prop(group,"use_anim_skip_noexp")
            col.prop(group,"use_anim_optimize")
            col.prop(group,"anim_optimize_precision")
            col.prop(group,"use_metadata")

### Custom template_list look
class UI_List_Godot(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        ob = data
        slot = item
        col = layout.row(align=True)
        
        col.label(text=item.name,icon="GROUP")
        col.prop(item,"active",text="")
        
        op = col.operator("scene.godot_select_group_objects",text="",emboss=False,icon="RESTRICT_SELECT_OFF")
        op.idx = index
        op = col.operator("scene.godot_export_group",text="",emboss=False,icon="EXPORT")
        op.idx = index
              
class add_objects_to_group(bpy.types.Operator):
    bl_idname = "scene.godot_add_objects_to_group"
    bl_label = "Add Objects to Group"
    bl_description = "Adds the selected Objects to the active group below."
    
    undo = BoolProperty(default=True)
    
    def execute(self,context):
        scene = context.scene
        
        objects_str = ""
        if len(scene.godot_export_groups) > 0:
            for i,object in enumerate(context.selected_objects):
                if object.name not in scene.godot_export_groups[scene.godot_export_groups_index].nodes:
                    node = scene.godot_export_groups[scene.godot_export_groups_index].nodes.add()
                    node.name = object.name
                    if i == 0:
                        objects_str += object.name
                    else:
                        objects_str += ", "+object.name    
        
        
            self.report({'INFO'}, objects_str + " added to group." )
            if self.undo:
                bpy.ops.ed.undo_push(message="Objects added to group")
        else:
            self.report({'WARNING'}, "Create a group first." )   
        return{'FINISHED'}

class del_objects_from_group(bpy.types.Operator):
    bl_idname = "scene.godot_delete_objects_from_group"
    bl_label = "Delete Objects from Group"
    bl_description = "Delets the selected Objects from the active group below."
    
    def execute(self,context):
        scene = context.scene
        
        if len(scene.godot_export_groups) > 0:
                
            selected_objects = []
            for object in context.selected_objects:
                selected_objects.append(object.name)
            
            objects_str = ""
            j = 0
            for i,node in enumerate(scene.godot_export_groups[scene.godot_export_groups_index].nodes):
                if node.name in selected_objects:
                    scene.godot_export_groups[scene.godot_export_groups_index].nodes.remove(i)

                    
                    if j == 0:
                            objects_str += object.name
                    else:
                        objects_str += ", "+object.name 
                    j+=1
                    
                    
            self.report({'INFO'}, objects_str + " deleted from group." )                    
            bpy.ops.ed.undo_push(message="Objects deleted from group")
        else:
            self.report({'WARNING'}, "There is no group to delete from." ) 
        return{'FINISHED'}

class select_group_objects(bpy.types.Operator):
    bl_idname = "scene.godot_select_group_objects"
    bl_label = "Select Group Objects"
    bl_description = "Will select all group Objects in the scene."
    
    idx = IntProperty()
    
    def execute(self,context):
        scene = context.scene
        for object in context.scene.objects:
            object.select = False
        for node in scene.godot_export_groups[self.idx].nodes:
            if node.name in bpy.data.objects:
                bpy.data.objects[node.name].select = True
                context.scene.objects.active = bpy.data.objects[node.name]
        return{'FINISHED'}
      
class export_groups_autosave(bpy.types.Operator):
    bl_idname = "scene.godot_export_groups_autosave"
    bl_label = "Export All Groups"
    bl_description = "Exports all groups to Collada."
    
    def execute(self,context):
        scene = context.scene
        if scene.godot_export_on_save:
            for i in range(len(scene.godot_export_groups)):
                if scene.godot_export_groups[i].active:
                    bpy.ops.scene.godot_export_group(idx=i)
        self.report({'INFO'}, "All Groups exported." )
        bpy.ops.ed.undo_push(message="Export all Groups")   
        return{'FINISHED'}
    
class export_all_groups(bpy.types.Operator):
    bl_idname = "scene.godot_export_all_groups"
    bl_label = "Export All Groups"
    bl_description = "Exports all groups to Collada."
    
    def execute(self,context):
        scene = context.scene
        
        for i in range(0,len(scene.godot_export_groups)):
            bpy.ops.scene.godot_export_group(idx=i,export_all=True)
                    
        self.report({'INFO'}, "All Groups exported." )
        return{'FINISHED'}    
        
        
class export_group(bpy.types.Operator):
    bl_idname = "scene.godot_export_group"
    bl_label = "Export Group"
    bl_description = "Exports the active group to destination folder as Collada file."
    
    idx = IntProperty(default=0)
    export_all = BoolProperty(default=False)

    
    def copy_object_recursive(self,ob,parent,single_user = True):
        new_ob = bpy.data.objects[ob.name].copy()
        if single_user or ob.type=="ARMATURE":
            new_mesh_data = new_ob.data.copy()
            new_ob.data = new_mesh_data
        bpy.context.scene.objects.link(new_ob)
        
        if ob != parent:
           new_ob.parent = parent
        else:
            new_ob.parent = None   
             
        for child in ob.children:        
            self.copy_object_recursive(child,new_ob,single_user)
        new_ob.select = True    
        return new_ob
    
    def delete_object(self,ob):
        if ob != None:
            for child in ob.children:
                self.delete_object(child)
            bpy.context.scene.objects.unlink(ob)
            bpy.data.objects.remove(ob) 
    
    def convert_group_to_node(self,group):
        if group.dupli_group != None:
            for object in group.dupli_group.objects:
                if object.parent == None:
                    object = self.copy_object_recursive(object,object,True)
                    matrix = Matrix(object.matrix_local)
                    object.matrix_local = Matrix()
                    object.matrix_local *= group.matrix_local
                    object.matrix_local *= matrix
        
            self.delete_object(group)   
    
    def execute(self,context):
        
        scene = context.scene
        group = context.scene.godot_export_groups
        
        if not group[self.idx].active and self.export_all:
            return{'FINISHED'}
        
        for i,object in enumerate(group[self.idx].nodes):
            if object.name in bpy.data.objects:
                pass
            else:
                group[self.idx].nodes.remove(i)
        bpy.ops.ed.undo_push(message="Clear not existent Group Nodes.")
        
        path = group[self.idx].export_path
        if (path.find("//")==0 or path.find("\\\\")==0):
            #if relative, convert to absolute
            path = bpy.path.abspath(path)
            path = path.replace("\\","/")
        
        ### if path exists and group export name is set the group will be exported  
        if os.path.exists(path) and  group[self.idx].export_name != "":
            
            context.scene.layers = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
            
            
            if group[self.idx].export_name.endswith(".dae"):
                path = os.path.join(path,group[self.idx].export_name)
            else:    
                path = os.path.join(path,group[self.idx].export_name+".dae")
            
            hide_select = []    
            for object in context.scene.objects:
                hide_select.append(object.hide_select)
                object.hide_select = False
                object.select = False
            context.scene.objects.active = None
            
            ### make particle duplicates, parent and select them
            nodes_to_be_added = []
            if group[self.idx].use_include_particle_duplicates:
                for i,object in enumerate(group[self.idx].nodes):
                    if bpy.data.objects[object.name].type != "EMPTY":
                        context.scene.objects.active = bpy.data.objects[object.name]
                        bpy.data.objects[object.name].select = True
                        bpy.ops.object.duplicates_make_real()
                        for object in context.selected_objects:
                            nodes_to_be_added.append(object)
                        bpy.ops.object.parent_set(type="OBJECT", keep_transform=False)

                        for object in context.selected_objects:
                            object.select = False
                        bpy.data.objects[object.name].select = False
                        context.scene.objects.active = None
            for object in nodes_to_be_added:
                object.select = True
            
            ### select all other nodes from the group
            for i,object in enumerate(group[self.idx].nodes):
                if bpy.data.objects[object.name].type == "EMPTY":
                    self.convert_group_to_node(bpy.data.objects[object.name])
                else:    
                    bpy.data.objects[object.name].select = True
                    
            bpy.ops.object.transform_apply(location=group[self.idx].apply_loc, rotation=group[self.idx].apply_rot, scale=group[self.idx].apply_scale)
            bpy.ops.export_scene.dae(check_existing=True, filepath=path, filter_glob="*.dae", object_types=group[self.idx].object_types, use_export_selected=group[self.idx].use_export_selected, use_mesh_modifiers=group[self.idx].use_mesh_modifiers, use_tangent_arrays=group[self.idx].use_tangent_arrays, use_triangles=group[self.idx].use_triangles, use_copy_images=group[self.idx].use_copy_images, use_active_layers=group[self.idx].use_active_layers, use_exclude_ctrl_bones=group[self.idx].use_exclude_ctrl_bones, use_anim=group[self.idx].use_anim, use_anim_action_all=group[self.idx].use_anim_action_all, use_anim_skip_noexp=group[self.idx].use_anim_skip_noexp, use_anim_optimize=group[self.idx].use_anim_optimize, anim_optimize_precision=group[self.idx].anim_optimize_precision, use_metadata=group[self.idx].use_metadata)    
          
            self.report({'INFO'}, '"'+group[self.idx].name+'"' + " Group exported." )  
            msg = "Export Group "+group[self.idx].name
            
            bpy.ops.ed.undo_push(message="")
            bpy.ops.ed.undo()
            bpy.ops.ed.undo_push(message=msg)
                
        else:
            self.report({'INFO'}, "Define Export Name and Export Path." )  
        return{'FINISHED'}

class add_export_group(bpy.types.Operator):
    bl_idname = "scene.godot_add_export_group"
    bl_label = "Adds a new export Group"
    bl_description = "Creates a new Export Group with the selected Objects assigned to it."
    
    def execute(self,context):
        scene = context.scene
        
        item = scene.godot_export_groups.add()
        item.name = "New Group"
        for object in context.selected_objects:
            node = item.nodes.add()
            node.name = object.name
        scene.godot_export_groups_index = len(scene.godot_export_groups)-1    
        bpy.ops.ed.undo_push(message="Create New Export Group")
        return{'FINISHED'}
    
class del_export_group(bpy.types.Operator):
    bl_idname = "scene.godot_delete_export_group"
    bl_label = "Delets the selected export Group"
    bl_description = "Delets the active Export Group."
    
    def invoke(self, context, event):
        wm = context.window_manager 
        return wm.invoke_confirm(self,event)
    
    def execute(self,context):
        scene = context.scene
        
        scene.godot_export_groups.remove(scene.godot_export_groups_index)
        if scene.godot_export_groups_index > 0:
            scene.godot_export_groups_index -= 1
        bpy.ops.ed.undo_push(message="Delete Export Group")
        return{'FINISHED'}    

class godot_node_list(bpy.types.PropertyGroup):
    name = StringProperty()
    
class godot_export_groups(bpy.types.PropertyGroup):
    name = StringProperty(name="Group Name")
    export_name = StringProperty(name="scene_name")
    nodes = CollectionProperty(type=godot_node_list)
    export_path = StringProperty(subtype="DIR_PATH")
    active = BoolProperty(default=True,description="Export Group")
    
    object_types = EnumProperty(name="Object Types",options={'ENUM_FLAG'},items=(('EMPTY', "Empty", ""),('CAMERA', "Camera", ""),('LAMP', "Lamp", ""),('ARMATURE', "Armature", ""),('MESH', "Mesh", ""),('CURVE', "Curve", ""),),default={'EMPTY', 'CAMERA', 'LAMP', 'ARMATURE', 'MESH','CURVE'})
    
    apply_scale = BoolProperty(name="Apply Scale",description="Apply Scale before export.",default=False)
    apply_rot = BoolProperty(name="Apply Rotation",description="Apply Rotation before export.",default=False)
    apply_loc = BoolProperty(name="Apply Location",description="Apply Location before export.",default=False)
    
    use_export_selected = BoolProperty(name="Selected Objects",description="Export only selected objects (and visible in active layers if that applies).",default=True)
    use_mesh_modifiers = BoolProperty(name="Apply Modifiers",description="Apply modifiers to mesh objects (on a copy!).",default=True)
    use_tangent_arrays = BoolProperty(name="Tangent Arrays",description="Export Tangent and Binormal arrays (for normalmapping).",default=False)
    use_triangles = BoolProperty(name="Triangulate",description="Export Triangles instead of Polygons.",default=False)

    use_copy_images = BoolProperty(name="Copy Images",description="Copy Images (create images/ subfolder)",default=False)
    use_active_layers = BoolProperty(name="Active Layers",description="Export only objects on the active layers.",default=True)
    use_exclude_ctrl_bones = BoolProperty(name="Exclude Control Bones",description="Exclude skeleton bones with names that begin with 'ctrl'.",default=True)
    use_anim = BoolProperty(name="Export Animation",description="Export keyframe animation",default=False)
    use_anim_action_all = BoolProperty(name="All Actions",description=("Export all actions for the first armature found in separate DAE files"),default=False)
    use_anim_skip_noexp = BoolProperty(name="Skip (-noexp) Actions",description="Skip exporting of actions whose name end in (-noexp). Useful to skip control animations.",default=True)
    use_anim_optimize = BoolProperty(name="Optimize Keyframes",description="Remove double keyframes",default=True)

    anim_optimize_precision = FloatProperty(name="Precision",description=("Tolerence for comparing double keyframes (higher for greater accuracy)"),min=1, max=16,soft_min=1, soft_max=16,default=6.0)

    use_metadata = BoolProperty(name="Use Metadata",default=True,options={'HIDDEN'})
    use_include_particle_duplicates = BoolProperty(name="Include Particle Duplicates",default=True)

def register():        
    bpy.utils.register_class(godot_export_manager)
    bpy.utils.register_class(godot_node_list)
    bpy.utils.register_class(godot_export_groups)
    bpy.utils.register_class(add_export_group)
    bpy.utils.register_class(del_export_group)
    bpy.utils.register_class(export_all_groups)
    bpy.utils.register_class(export_groups_autosave)
    bpy.utils.register_class(export_group)
    bpy.utils.register_class(add_objects_to_group)
    bpy.utils.register_class(del_objects_from_group)
    bpy.utils.register_class(select_group_objects)
    bpy.utils.register_class(UI_List_Godot)

    bpy.types.Scene.godot_export_groups = CollectionProperty(type=godot_export_groups)
    bpy.types.Scene.godot_export_groups_index = IntProperty(default=0,min=0)

def unregister():
    bpy.utils.unregister_class(godot_export_manager)
    bpy.utils.unregister_class(godot_node_list)
    bpy.utils.unregister_class(godot_export_groups)
    bpy.utils.unregister_class(export_groups_autosave)
    bpy.utils.unregister_class(add_export_group)
    bpy.utils.unregister_class(del_export_group)
    bpy.utils.unregister_class(export_all_groups)
    bpy.utils.unregister_class(export_group)
    bpy.utils.unregister_class(add_objects_to_group)
    bpy.utils.unregister_class(del_objects_from_group)
    bpy.utils.unregister_class(select_group_objects)
    bpy.utils.unregister_class(UI_List_Godot)

@persistent
def auto_export(dummy):
    bpy.ops.scene.godot_export_groups_autosave()
        
bpy.app.handlers.save_post.append(auto_export)

if __name__ == "__main__":
    register()
