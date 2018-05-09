import csv

from neo4j.v1 import GraphDatabase, basic_auth

driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "neo"))

with driver.session() as session, open("genres.txt", "w") as edges_file:
    result = session.run("""\
    MATCH (m:Movie)-[:IN_GENRE]-(genre)
    RETURN id(m) AS source, collect(id(genre)) AS target    
    """)

    writer = csv.writer(edges_file, delimiter=" ")

    for row in result:
        writer.writerow([row["source"], row["target"]])

