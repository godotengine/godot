extends Node

# Singleton class containing all common data of the player and bosses
# It is also a controller of the main scene for displaying the change of values. But it's mostly getters and setters.


# Variables ---------------------------------------------------

# Contains data from the player, that needs to be common to the whole game
var playerData={
	life=6,
	score=0,
	continues=4
}

# name of the current default music. Used when a temporar music, like a boss theme or invulnerability, ends.
# It should be updated when a new chapter is loaded, since each chapter in the original game has its default theme.
var map_track="greens"

# Life of the boss, if it exists. Otherwise -1 when there's no boss.
# The life of the current boss is stored here because it needs to be displayed in the HUD.
var bossLife=-1

# Functions ----------------------------------------------------------

# --- for the player 
# removes an amount of points to the player's life and check if life is player has no more life.
func remove_player_life(amount):
	playerData.life-=amount
	if(playerData.life<=0):
		# player has no more life. Consume one continue and restore all his life.
		# Normally an animation should be played and the player restarts the level.
		restore_player_life()
		dec_player_continue()
	else:
		# player loses a bit of his life. The change must be displayed.
		_update_player_life()

# add life points to the player's life.
# Usually when the player eats a bonus item
func increase_player_life(amount):
	playerData.life+=amount
	# cap the life to the allowed maximum
	if(playerData.life>6):
		playerData.life=6
	# displays the change
	_update_player_life()

# Restore all the life points to the player
func restore_player_life():
	playerData.life=6
	# displays the change
	_update_player_life()

# tells the HUD to display the new value of player's life
func _update_player_life():
	var lifeBar=get_scene().get_nodes_in_group("lifeBar")[0]
	lifeBar.set_life(playerData.life)

# changes the player's score and display it
func add_player_score(amount):
	playerData.score+=amount
	_update_player_score()

# reset the player's score, like for a new game, and display it
func reset_player_score():
	playerData.score=0
	_update_player_score()

# tells the HUD to display the new value of player's score
func _update_player_score():
	var scoreLabel=get_scene().get_nodes_in_group("score")[0]
	scoreLabel.set_text(str(playerData.score))

# removes one continue to the player.
# normally if the player has no more continues, the game should go to a game over scene.
func dec_player_continue():
	playerData.continues-=1
	if(playerData.continues<0):
		playerData.continues=0 # game over
	_update_player_continues()

# adds a continue to the player, like when he eats a 1up element or when his score reaches a certain value.
func add_player_continue():
	playerData.continues+=1
	_update_player_continues()

# resets the continues of the player, like for a new game
func reset_player_continue():
	playerData.continues=4
	_update_player_continues()

# displays the change of the player's continues
func _update_player_continues():
	var continuesLabel=get_scene().get_nodes_in_group("continues")[0]
	var strContinues=str(playerData.continues)
	if(playerData.continues<10):
		strContinues="0"+strContinues
	continuesLabel.set_text(strContinues)


# --- for the boss

# setter for boss life
func set_boss_life(life):
	bossLife=life
	var lifeBar=get_scene().get_nodes_in_group("bossLifeBar")[0]
	lifeBar.set_life(bossLife)

# remove 1 point of life to boss' life
func dec_boss_life():
	bossLife-=1
	var lifeBar=get_scene().get_nodes_in_group("bossLifeBar")[0]
	lifeBar.set_life(bossLife)

# getter for boss' life
func get_boss_life():
	return bossLife

# display or hide the life of the boss in the HUD. When hidden, the score of the player is displayed instead.
func set_boss_bar_visibility(visible):
	var bossBar=get_scene().get_nodes_in_group("bossBar")[0]
	var scoreBar=get_scene().get_nodes_in_group("scoreBar")[0]
	if(visible):
		bossBar.show()
		scoreBar.hide()
	else:
		bossBar.hide()
		scoreBar.show()

# --- Various

# getter for the current default music name
func get_map_track():
	return map_track