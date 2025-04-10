extends Node

func test():
	var port = 0 # Only latin characters.
	var pοrt = 1 # The "ο" is Greek omicron.

	prints(port, pοrt)

# Do not call this since nodes aren't in the tree. It is just a parser check.
func nodes():
	var _node1 = $port # Only latin characters.
	var _node2 = $pοrt # The "ο" is Greek omicron.
