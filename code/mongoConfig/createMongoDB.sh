#!/bin/sh
docker-compose up
# configure our config servers replica set
docker exec -it mongocfg1 bash -c "echo 'rs.initiate({_id: \"mongors1conf\",configsvr: true, members: [{ _id : 0, host : \"mongocfg1\" },{ _id : 1, host : \"mongocfg2\" }, { _id : 2, host : \"mongocfg3\" }]})' | mongo"

# building replica shard
docker exec -it mongors1n1 bash -c "echo 'rs.initiate({_id : \"mongors1\", members: [{ _id : 0, host : \"mongors1n1\" },{ _id : 1, host : \"mongors1n2\" },{ _id : 2, host : \"mongors1n3\" }]})' | mongo"
docker exec -it mongors2n1 bash -c "echo 'rs.initiate({_id : \"mongors2\", members: [{ _id : 0, host : \"mongors2n1\" },{ _id : 1, host : \"mongors2n2\" },{ _id : 2, host : \"mongors2n3\" }]})' | mongo"


# we add shard to the routers
docker exec -it mongos1 bash -c "echo 'sh.addShard(\"mongors1/mongors1n1\")' | mongo "
docker exec -it mongos1 bash -c "echo 'sh.addShard(\"mongors2/mongors2n1\")' | mongo "

# open mongos
use maadbProject
sh.enableSharding("maadbProject")
db.createCollection("words")
sh.shardCollection("maadbProject.words", { name : "hashed" })
sh.status()
