from SinglePacketRoutingEnv import SinglePacketRoutingEnv
from LinkFailureEnv import LinkFailureEnv
from DijkstraAgent import DijkstraAgent


def create_node_param(node_num):
    nodes = []
    for i in range(node_num):
        nodes.append("n{}".format(i))
    return nodes


nodes1 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"]
links1 = [["n1", "n2", 1], ["n1", "n0", 1], ["n0", "n2", 1], ["n3", "n0", 1], ["n2", "n5", 2], ["n3", "n5", 1], ["n4", "n3", 1], ["n4", "n6", 10], ["n6", "n3", 1], ["n4", "n7", 1],  ["n6", "n7", 1],  ["n5", "n6", 1]]

nodes2 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16",
          "n17", "n18", "n19", "n20", "n21", "n22", "n23", "n24", "n25"]
links2 = [["n15", "n16", 1], ["n15", "n17", 1], ["n15", "n13", 10], ["n14", "n13", 1], ["n15", "n18", 10],
          ["n13", "n7", 10], ["n10", "n7", 1], ["n9", "n7", 1], ["n11", "n7", 1], ["n8", "n7", 1], ["n7", "n12", 1],
          ["n7", "n18", 1], ["n18", "n19", 1], ["n7", "n4", 1], ["n4", "n5", 1], ["n4", "n6", 1], ["n4", "n3", 10],
          ["n1", "n3", 1], ["n4", "n20", 1], ["n20", "n21", 1], ["n20", "n22", 1], ["n20", "n23", 1],
          ["n20", "n24", 1], ["n20", "n25", 1], ["n2", "n3", 10], ["n20", "n2", 10], ["n0", "n2", 1], ["n18", "n20", 1]]

myEnv = LinkFailureEnv(nodes=nodes2, edges=links2, seed=None, packet=None, broken_link_num=1)
state = [myEnv.state[0].id, myEnv.state[1].id, myEnv.state[2].id]
print(state)
# myEnv.step(0)
"""for i in range(20):
    myEnv.step(0)
state = [myEnv.state[0].id, myEnv.state[1].id, myEnv.state[2].id]
print(state)"""

"""myEnv.reset()
for _ in range(100):
    myEnv.render()
    state, reward, done, info = myEnv.step(myEnv.action_space.sample())
    print(reward)

print(myEnv.observation_space.sample())"""

myAgent = DijkstraAgent(env=myEnv, nodes=nodes2, edges=links2)

