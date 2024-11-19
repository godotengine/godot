extends Node

func _ready() -> void:
	# Create new gdsqlite instance
	var db: SQLite = SQLite.new();

	# Open item database
	if (!db.open("items.db")):
		print("Failed opening database.");
		return;
	var query: String = "SELECT
				name
			FROM
				sqlite_schema
			WHERE
				type ='table' AND
				name NOT LIKE 'sqlite_%';"
	var result: Array = db.create_query(query).batch_execute([])
	print(result)
