import sqlite3

# Connect to the database
conn = sqlite3.connect("engagement.db")
cursor = conn.cursor()

# Fetch everything
cursor.execute("SELECT * FROM engagement_log")
rows = cursor.fetchall()

# Print results
for row in rows:
    print(row)

conn.close()
