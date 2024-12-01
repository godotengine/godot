extends Node


func _ready():
	# Create gdsqlite instance
	var db = SQLite.new()

	# Open database
	if !db.open("user://godot.sqlite"):
		return

	var query = ""
	var result = null

	# Create table
	query = "CREATE TABLE IF NOT EXISTS users ("
	query += "id integer PRIMARY KEY,"
	query += "first_name text NOT NULL,"
	query += "last_name text NOT NULL,"
	query += "email text NOT NULL"
	query += ");"
	result = db.create_query(query).execute()

	# Fetch rows
	query = "SELECT * FROM users;"
	result = db.create_query(query).execute()

	if (result == null || result.size() <= 0):
		# Insert new row
		query = "INSERT INTO users (first_name, last_name, email) VALUES ('godot', 'engine', 'user@users.org');"
		result = db.create_query(query).execute()

		if result.is_empty():
			print("Cannot insert data!")
		else:
			print("Data inserted into table.")

	else:
		# Print rows
		for i in result:
			print(i)

	# Close database
	db.close()
