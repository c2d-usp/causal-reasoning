import networkx as nx
from causal_reasoning.graph.moral_node import MoralNode
from causal_reasoning.graph.node import Node


class Graph:
    def __init__(
        self,
        numberOfNodes: int,
        currNodes: list[str],
        visited: list[bool],
        cardinalities: dict[str, int],
        parents: dict[str, list[str]],
        adj: dict[str, list[str]],
        dagComponents: dict[str, list[str]],
        exogenous: list[str],
        endogenous: list[str],
        topologicalOrder: list[str],
        DAG: nx.DiGraph,
        cComponentToUnob: dict[int, int], #??
        graphNodes: list[Node],
        moralGraphNodes: list[MoralNode],
        node_set: set[str]
    ):
        self.numberOfNodes = numberOfNodes
        self.currNodes = currNodes
        self.visited = visited
        self.cardinalities = cardinalities
        self.parents = parents
        self.adj = adj
        self.dagComponents = dagComponents
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.topologicalOrder = topologicalOrder
        self.DAG = DAG
        self.cComponentToUnob = cComponentToUnob
        self.graphNodes = graphNodes
        self.moralGraphNodes = moralGraphNodes
        self.node_set = node_set

    def visit_nodes_in_same_cComponent(self, node: str):
        self.visited[node] = True
        self.currNodes.append(node)
        is_observable = self.cardinalities[node] > 1

        if not is_observable:
            for adj_node in self.adj[node]:
                if not self.visited[adj_node]:
                    self.visit_nodes_in_same_cComponent(adj_node)
        else:
            for parent_node in self.parents[node]:
                if (
                    not self.visited[parent_node]
                    and self.cardinalities[parent_node] < 1
                ):
                    self.visit_nodes_in_same_cComponent(parent_node)

    def find_cComponents(self):
        for node in self.node_set:
            if not self.visited[node] and self.cardinalities[node] < 1:
                self.currNodes.clear()
                self.visit_nodes_in_same_cComponent(node)
                self.dagComponents.append(self.currNodes[:])
                self.cComponentToUnob[len(self.dagComponents) - 1] = node

    def base_dfs(self, node: str):
        self.visited[node] = True
        for adj_node in self.graphNodes[node].children:
            if not self.visited[adj_node]:
                self.base_dfs(adj_node)

    def is_descendant(self, ancestor: str, descendant:str) -> bool:
        self.clear_visited()
        print(self.visited)
        self.base_dfs(node=ancestor)
        return self.visited[descendant]
    
    def clear_visited(self):
        for node in self.node_set:
            self.visited[node] = False

    def build_moral(
        self,
        consideredNodes: list[int],
        conditionedNodes: list[int],
        flag=False,
        intervention=-1,
    ):
        """
        Builds the moral graph, considering only part of the nodes.
        flag: if true, the outgoing edges of the intervention should not be considered.
        """
        self.moralGraphNodes = [
            MoralNode(adjacent=[]) for _ in range(self.numberOfNodes)
        ]
        for node in range(self.numberOfNodes):
            if node not in consideredNodes:
                continue

            if node in conditionedNodes:
                for parent1 in self.graphNodes[node].parents:
                    if flag and parent1 == intervention:
                        continue
                    for parent2 in self.graphNodes[node].parents:
                        if flag and parent2 == intervention:
                            continue

                        if parent1 in conditionedNodes and parent2 in consideredNodes:
                            if parent2 not in self.moralGraphNodes[parent1].adjacent:
                                self.moralGraphNodes[parent1].adjacent.append(parent2)
                            if parent1 not in self.moralGraphNodes[parent2].adjacent:
                                self.moralGraphNodes[parent2].adjacent.append(parent1)
            else:
                if flag and node == intervention:
                    continue

                for ch in self.graphNodes[node].children:
                    if ch in consideredNodes and ch not in conditionedNodes:
                        if node not in self.moralGraphNodes[ch].adjacent:
                            self.moralGraphNodes[ch].adjacent.append(node)
                        if ch not in self.moralGraphNodes[node].adjacent:
                            self.moralGraphNodes[node].adjacent.append(ch)

    def find_ancestors(self, node: int):
        self.currNodes.clear()
        self.visited = [False] * self.numberOfNodes
        self.dfs_ancestor(node)
        ancestors: list[int] = []
        for i in range(0, self.numberOfNodes):
            if self.visited[i]:
                ancestors.append(i)
        return ancestors

    def dfs_ancestor(self, node):
        self.visited[node] = True

        for parent in self.graphNodes[node].parents:
            if not self.visited[parent]:
                self.dfs_ancestor(parent)

    def independency_moral(self, node1: int, node2: int):
        self.visited = [False] * self.numberOfNodes
        self.dfs_moral(node1)

        return not self.visited[node2]

    def dfs_moral(self, node):
        self.visited[node] = True

        for adj in self.moralGraphNodes[node].adjacent:
            if not self.visited[adj]:
                self.dfs_moral(node=adj)

    def check_dseparation(
        self, set_nodes_1: list[str], set_nodes_2: list[str], conditioned_nodes: list[str]
    ) -> bool:
        """
        Given two sets of nodes (nodes1 and nodes2), the function returns true if every node in nodes1
        is independent of every node in nodes2, given that the nodes in conditionedNodes are conditioned.
        """
        nodes1 = [self.labelToIndex[node] for node in set_nodes_1]
        nodes2 = [self.labelToIndex[node] for node in set_nodes_2]
        conditionedNodes = [self.labelToIndex[node] for node in conditioned_nodes]

        self.build_moral(
            consideredNodes=list(range(0, self.numberOfNodes)),
            conditionedNodes=conditionedNodes,
        )

        self.visited = [False] * self.numberOfNodes
        for node in nodes1:
            if not self.visited[node]:
                self.dfs_moral(node)

        areDseparated = True
        for node in nodes2:
            if self.visited[node]:
                areDseparated = False
                break
        return areDseparated
