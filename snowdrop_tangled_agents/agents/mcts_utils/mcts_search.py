# called by mcts_agent
class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node
        self.Q = {}

    def best_action(self, simulations_number=10):
        """
        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------
        """

        for _ in range(0, simulations_number):    # for each simulation
            v = self._tree_policy()               # v is the node to run rollout/playout for
            reward = v.rollout()                  # returns reward from random rollout from v
            # updates _number_of_visits and _results[result =+1, -1, 0] for each node in chain back up to root
            v.backpropagate(reward)

        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def q_table(self):
        return self.root.compute_q_table()

    def _tree_policy(self):
        # selects node to run rollout/playout for

        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
