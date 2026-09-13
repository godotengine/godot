# https://github.com/godotengine/godot/issues/123465
class Reward:
	pass


class GoldReward extends Reward:
	var value: int = 2

	func calc() -> int:
		return 2 * value


func do_stuff(reward: Reward) -> void:
	if reward is GoldReward:
		prints("value", reward.value)
		prints("calc", reward.calc())
	else:
		print("not gold")


func test():
	do_stuff(GoldReward.new())
	do_stuff(Reward.new())
