# We don't want to execute it because of errors, just analyze.
class Reward:
	pass


class GoldReward extends Reward:
	var value: int = 2

	func calc() -> int:
		return 2 * value


func no_exec_test():
	var variant: Variant = null

	if variant is GoldReward:
		print(variant.value) # No warning: the type is narrowed to GoldReward.
		print(variant.calc()) # No warning.
	print(variant.value) # No warning: unsafe property access on "Variant" is not reported.
	print(variant.calc()) # Warning.

	if variant is GoldReward:
		variant = Reward.new() # The assignment invalidates the narrowed type.
		print(variant.value) # No warning: unsafe property access on "Variant" is not reported.
		print(variant.calc()) # Warning.

	var reward: Reward = GoldReward.new()
	if reward is GoldReward:
		print(reward.value) # No warning: the type is narrowed to GoldReward.
		print(reward.calc()) # No warning.
	print(reward.value) # Warning.
	print(reward.calc()) # Warning.


func param_no_exec_test(reward: Reward):
	if reward is GoldReward and reward is Reward:
		print(reward.value) # Warning: the type is narrowed to "Reward" by the last type test.
	print(reward.value) # Warning.
	print(reward.calc()) # Warning.


func test():
	pass
