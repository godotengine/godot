struct Enemy:
	var health: int = 100
	var damage: float = 15.5
	var name: String = "Goblin"

func test():
	var e1 = Enemy()
	print(e1.health)
	print(e1.damage)
	print(e1.name)
	
	var e2 = Enemy(200, 25.0, "Dragon")
	print(e2.health)
	print(e2.damage)
	print(e2.name)
