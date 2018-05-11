import csv

from neo4j.v1 import GraphDatabase, basic_auth

# host = "bolt://localhost"
# password = "neo"

host = "bolt://54.197.87.95:32952"
password = "pennant-automation-pacific"

driver = GraphDatabase.driver(host, auth=basic_auth("neo4j", password))

with open("movies.emb", "r") as movies_file, driver.session() as session:
    next(movies_file)
    reader = csv.reader(movies_file, delimiter=" ")

    params = []

    for row in reader:
        movie_id = row[0]

        params.append({
            "id": int(movie_id),
            "embedding": [float(item) for item in row[1:]]
        })

    session.run("""\
    UNWIND {params} AS param
    MATCH (m:Movie) WHERE id(m) = param.id
    SET m.embedding = param.embedding
    """, {"params": params})
