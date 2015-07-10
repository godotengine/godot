local node = extends "Node"

local current_scene = nil

function node:goto_scene(scene)
	--load new scene
    local s = ResourceLoader:load(scene)
	--queue erasing old (don't use free because that scene is calling this method)
	current_scene:queue_free()
	--instance the new scene
	current_scene = s:instance()
	--add it to the active scene, as child of root
	self:get_tree():get_root():add_child(current_scene)
end

function node:_ready()
	-- get the current scene
	-- it is always the last child of root,
	-- after the autoloaded nodes
	local root = self:get_tree():get_root()
	current_scene = root:get_child( root:get_child_count() -1 )
end
