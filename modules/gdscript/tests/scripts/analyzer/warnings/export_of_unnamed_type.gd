extends Node

class InnerResource extends Resource:
	pass

class InnerNode extends Node:
	pass

const UnnamedResource = preload("./export_of_unnamed_type_resource.notest.gd")
const UnnamedNode = preload("./export_of_unnamed_type_node.notest.gd")

@export var inner_resource: InnerResource
@export var inner_node: InnerNode
@export var unnamed_resource: UnnamedResource
@export var unnamed_node: UnnamedNode

func test():
	pass
