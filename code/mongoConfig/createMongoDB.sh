#!/bin/sh
docker-compose up
# configure our config servers replica set
docker exec -it mongocfg1 bash -c "echo 'rs.initiate({_id: \"mongors1conf\",configsvr: true, members: [{ _id : 0, host : \"mongocfg1\" },{ _id : 1, host : \"mongocfg2\" }, { _id : 2, host : \"mongocfg3\" }]})' | mongo"


# building replica shard
docker exec -it mongors1n1 bash -c "echo 'rs.initiate({_id : \"mongors1\", members: [{ _id : 0, host : \"mongors1n1\" },{ _id : 1, host : \"mongors1n2\" },{ _id : 2, host : \"mongors1n3\" }]})' | mongo"

# we add shard to the routers
docker exec -it mongos1 bash -c "echo 'sh.addShard(\"mongors1/mongors1n1\")' | mongo "

# create the database 
docker exec -it mongors1n1 bash -c "echo 'use maadbProjectDB' | mongo"

# enable sharding in database
docker exec -it mongos1 bash -c "echo 'sh.enableSharding(\"maadbProjectDB\")' | mongo "

# create the collection 
docker exec -it mongors1n1 bash -c "echo 'db.createCollection(\"maadbProjectDB.Anger_words\")' | mongo "

# shard the collection selecting a sharding key
docker exec -it mongos1 bash -c "echo 'sh.shardCollection(\"maadbProjectDB.Anger_words\", {name : 1})' | mongo "


