extends Node

# Variables
var db;
var highscore = 0;
var row_id = 0;
@onready
var open = false;


func _ready():
	# Create SQLite instance
	db = SQLite.new()

	# Open the database
	if not db.open("user://player_stats.sqlite"):
		return

	open = true
	var query: SQLiteQuery = db.create_query("CREATE TABLE IF NOT EXISTS highscore (id INTEGER PRIMARY KEY, score INTEGER NOT NULL);")

	# Create table
	if not query.execute([]).is_empty():
		return
	query = db.create_query("SELECT id, score FROM highscore LIMIT 1;")
	# Retrieve current highscore
	var rows = query.execute()
	if (rows and not rows.is_empty()):
		row_id = rows[0][0];
		highscore = rows[0][1];

	# Test
	set_highscore(1000)
	set_highscore(2000)
	set_highscore(10000)
	set_highscore(50000)
	print("High score: ", get_highscore())


func _exit_tree():
	if db:
		# Close database
		db.close()


func set_highscore(score):
	if not open:
		return

	# Update highscore
	highscore = score

	# Execute sql syntax
	if row_id > 0:
		db.create_query("UPDATE highscore SET score=? WHERE id=?;").execute([highscore, row_id])
	else:
		db.create_query("INSERT INTO highscore (score) VALUES (?);").execute([row_id])
		var query = db.create_query("SELECT last_insert_rowid()")
		row_id = query.execute([])[0][query.get_columns().find("last_insert_rowid()")]


func get_highscore():
	if not open:
		return
	var query: SQLiteQuery = db.create_query("SELECT score FROM highscore WHERE id=? LIMIT 1;");
	# Retrieve highscore from database
	var rows = query.execute([row_id])
	if (rows and not rows.is_empty()):
		highscore = rows[0][0]

	# Return the highscore
	return highscore
