#!/bin/bash

# Stop any running MongoDB instance
mongod --shutdown

# Wait a moment for the shutdown to complete
sleep 2

# Start MongoDB with our configuration
mongod --config "$(pwd)/mongod.conf"

# Restore the database if it doesn't exist
if ! mongosh --eval "db.getMongo().getDBs().databases.some(db => db.name === 'flask_ui_db')" --quiet | grep -q "true"; then
    echo "Restoring database from backup..."
    mongorestore --db flask_ui_db "$(pwd)/dump/flask_ui_db"
fi 