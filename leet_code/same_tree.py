"""
https://leetcode.com/problems/same-tree/
Given the roots of two binary trees p and q, write a function to check if they are the same or not.
Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isSameTree(self, p, q):
        """
        :type p: Optional[TreeNode]
        :type q: Optional[TreeNode]
        :rtype: bool
        """
        if p is None and q is None:
            # None equal to none
            return True
        elif p is None or q is None:
            # only one is none
            return False
        elif p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False


# Test cases
def test_solution():
    # Create tree1: [1,2,3]
    tree1 = TreeNode(1)
    tree1.left = TreeNode(2)
    tree1.right = TreeNode(3)

    # Create tree2: [1,2,3]
    tree2 = TreeNode(1)
    tree2.left = TreeNode(2)
    tree2.right = TreeNode(3)

    # Create different tree: [1,2]
    tree3 = TreeNode(1)
    tree3.left = TreeNode(2)

    solution = Solution()
    print("Test 1 (Same trees):", solution.isSameTree(tree1, tree2))  # Should print True
    print("Test 2 (Different trees):", solution.isSameTree(tree1, tree3))  # Should print False


if __name__ == "__main__":
    test_solution()
