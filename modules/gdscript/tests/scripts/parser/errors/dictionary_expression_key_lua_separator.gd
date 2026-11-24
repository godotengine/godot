extends Node

enum Status { BURN, FREEZE }

var statuses: Dictionary[Status, int] = {
	Status.BURN = 0,
	Status.FREEZE: 0,
}
