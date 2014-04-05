local node = extends "Spatial"


function node:_on_pause_pressed()
	self:get_node("pause_popup"):set_exclusive(true)
	self:get_node("pause_popup"):popup()
	self:get_scene():set_pause(true)
end

function node:_on_unpause_pressed()
	self:get_node("pause_popup"):hide()
	self:get_scene():set_pause(false)
end
