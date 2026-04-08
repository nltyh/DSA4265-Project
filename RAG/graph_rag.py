from collections import defaultdict

class GraphRAG:
    def __init__(self):
        """
        graph structure:
        node → list of connected nodes
        """
        self.graph = defaultdict(list)

    def add_edge(self, node1, node2):
        self.graph[node1].append(node2)
        self.graph[node2].append(node1)

    def build_graph(self, documents):
        """
        documents: list of dicts with extracted entities
        {
            "text": str,
            "entities": ["copper", "Chile", "port"]
        }
        """
        for doc in documents:
            entities = doc["entities"]
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    self.add_edge(entities[i], entities[j])

    def retrieve_subgraph(self, query_entities, depth=1):
        """
        Get connected nodes (risk propagation)
        """
        visited = set()
        results = set()

        def dfs(node, d):
            if d > depth or node in visited:
                return
            visited.add(node)
            results.add(node)

            for neighbor in self.graph[node]:
                dfs(neighbor, d + 1)

        for entity in query_entities:
            dfs(entity, 0)

        return list(results)