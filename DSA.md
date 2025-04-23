# DSA

Here is a list of all the question headings from the context:

1. Diameter of Binary Tree
2. Lowest Common Ancestor of a Binary Search Tree
3. Balanced Binary Tree
4. Linked List Cycle
5. Longest Substring Without Repeating Characters
6. Implement Queue using Stacks
7. Binary Tree Level Order Traversal
8. 3Sum
9. Course Schedule
10. [Course Schedule II](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
11. Implement Trie (Prefix Tree)
12. Coin Change
13. Product of Array Except Self
14. Climbing Stairs
15. Min Stack
16. Reverse Linked List
17. Validate Binary Search Tree
18. Number of Islands
19. Rotting Oranges
20. [Search in Rotated Sorted Array](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
21. Combination Sum
22. Permutations
23. Merge Intervals
24. Insert Interval
25. Lowest Common Ancestor of a Binary Tree
26. Minimum Window Substring
27. Serialize and Deserialize Binary Tree
28. Find Median from Data Stream
29. Trapping Rain Water
30. [Diameter of Binary Tree](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
31. Add Binary
32. Majority Element
33. Middle of the Linked List
34. Accounts Merge
35. Sort Colors
36. Longest Palindromic Substring
37. Partition Equal Subset Sum
38. Binary Tree Right Side View
39. Subsets
40. [Word Ladder](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
41. Maximum Depth of Binary Tree
42. Contains Duplicate
43. Unique Paths
44. Construct Binary Tree from Preorder and Inorder Traversal
45. Kth Smallest Element in a BST
46. LRU Cache
47. Longest Substring Without Repeating Characters
48. Container With Most Water
49. Two Sum II - Input Array Is Sorted
50. [Missing Ranges](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
51. Longest Substring with At Most Two Distinct Characters
52. Plus One
53. Kth Largest Element in an Array
54. Meeting Rooms II
55. Valid Parentheses
56. Add Two Numbers
57. Remove Nth Node From End of List
58. Merge Two Sorted Lists
59. Copy List with Random Pointer
60. [Binary Tree Maximum Path Sum](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
61. Count Complete Tree Nodes
62. Longest Increasing Path in a Matrix
63. Decode String
64. Flip Equivalent Binary Trees
65. Peak Index in a Mountain Array
66. Maximum Subarray
67. Maximum Product Subarray
68. Split Array Largest Sum
69. Longest Consecutive Sequence
70. [Palindrome Number](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
71. Remove Element
72. Add Two Integers
73. Roman to Integer
74. Reverse Integer
75. Number of Provinces
76. Graph Valid Tree
77. Number of Connected Components in an Undirected Graph
78. Find if Path Exists in Graph
79. All Paths From Source to Target
80. [Populating Next Right Pointers in Each Node](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
81. N-ary Tree Level Order Traversal
82. [Min Cost to Connect All Points](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
83. Network Delay Time
84. Parallel Courses
85. Reverse Words in a String
86. Increasing Triplet Subsequence
87. String Compression
88. Max Number of K-Sum Pairs
89. Maximum Number of Vowels in a Substring of Given Length
90. [Determine if Two Strings Are Close](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
91. Equal Row and Column Pairs
92. Asteroid Collision
93. Odd Even Linked List
94. Maximum Twin Sum of a Linked List
95. Count Good Nodes in Binary Tree
96. [Path Sum III](https://www.notion.so/DSA-1d7ee6f103f1472a8b79a280f3bef1a4?pvs=21)
97. 

# [**543. Diameter of Binary Tree**](https://leetcode.com/problems/diameter-of-binary-tree/)

Given the `root` of a binary tree, return *the length of the **diameter** of the tree*.

The **diameter** of a binary tree is the **length** of the longest path between any two nodes in a tree. This path may or may not pass through the `root`.

The **length** of a path between two nodes is represented by the number of edges between them.

```jsx
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        
        def dfs(node):
            nonlocal max_path
            if not node:
                return 0
            
            left_path = dfs(node.left)
            right_path = dfs(node.right)
            max_path = max(max_path, left_path + right_path)
            
            return max(left_path, right_path) + 1
            
        max_path = 0
        dfs(root)
        return max_path
```

# [**235. Lowest Common Ancestor of a Binary Search Tree**](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the [definition of LCA on Wikipedia](https://en.wikipedia.org/wiki/Lowest_common_ancestor): “The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow **a node to be a descendant of itself**).”

**Example 1:**

![https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png](https://assets.leetcode.com/uploads/2018/12/14/binarysearchtree_improved.png)

```python
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        p_val = p.val
        q_val = q.val

        node = root

        while node:
            parent_val = node.val

            if p_val> parent_val and q_val> parent_val:
                node = node.right
            elif p_val<parent_val and q_val<parent_val:
                node = node.left
            else:
                return node
               
```

# [**110. Balanced Binary Tree**](https://leetcode.com/problems/balanced-binary-tree/)

Given a binary tree, determine if it is

**height-balanced**

.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg](https://assets.leetcode.com/uploads/2020/10/06/balance_1.jpg)

```

Input: root = [3,9,20,null,null,15,7]
Output: true
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def height(self, root: TreeNode):
        if not root:
            return -1
        return 1+max(self.height(root.left), self.height(root.right))

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        return (
            abs(self.height(root.left) - self.height(root.right)) < 2
            and self.isBalanced(root.left)
            and self.isBalanced(root.right)
        )
```

# [**141. Linked List Cycle**](https://leetcode.com/problems/linked-list-cycle/)

Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter**.

Return `true` *if there is a cycle in the linked list*. Otherwise, return `false`.

**Example 1:**

![https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None:
            return False

        slow, fast = head, head.next

        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True
        
```

# [**3. Longest Substring Without Repeating Characters**](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

Given a string `s`, find the length of the **longest**

**substring**

without repeating characters.

**Example 1:**

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

```python
from collections import Counter
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        chars = Counter()

        l, r = 0, 0

        max_len = 0

        while r < len(s):
            chars[s[r]] += 1

            while chars[s[r]] > 1:
                chars[s[l]] -= 1
                l += 1
            
            max_len = max(max_len, r-l+1)

            r += 1

        return max_len

```

# [**232. Implement Queue using Stacks**](https://leetcode.com/problems/implement-queue-using-stacks/)

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (`push`, `peek`, `pop`, and `empty`).

```python
class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        while self.stack1:
            val = self.stack1.pop()
            self.stack2.append(val)
        self.stack1.append(x)
        while self.stack2:
            val = self.stack2.pop()
            self.stack1.append(val)

    def pop(self) -> int:
        if not self.empty():
            return self.stack1.pop()
        
    def peek(self) -> int:
        if not self.empty():
            return self.stack1[-1]
        
    def empty(self) -> bool:
        return len(self.stack1) == 0

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

# [**102. Binary Tree Level Order Traversal**](https://leetcode.com/problems/binary-tree-level-order-traversal/)

Given the `root` of a binary tree, return *the level order traversal of its nodes' values*. (i.e., from left to right, level by level).

**Example 1:**

![https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg](https://assets.leetcode.com/uploads/2021/02/19/tree1.jpg)

```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return None
        queue = deque([root])
        res = []

        while queue:
            size = len(queue)
            level_res = []

            for i in range(size):
                node = queue.popleft()
                level_res.append(node.val)
                
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)

            res.append(level_res)

        return res
        
```

# [**278. First Bad Version**](https://leetcode.com/problems/first-bad-version/)

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

```python
class Solution:
    def firstBadVersion(self, n: int) -> int:
        l, r = 0, n

        while l<r:
            mid = (r+l)//2
            if isBadVersion(mid):
                r = mid
            else:
                l = mid+1
        return l
```

# [**15. 3Sum**](https://leetcode.com/problems/3sum/)

Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

**Example 1:**

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation:
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.
```

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i ==0 or nums[i-1] != nums[i]:
                self.twoSumII(i, nums, res)
        return res

    def twoSumII(self, i, nums, res, target=0):
        left, right = i+1, len(nums) - 1

        while left < right:
            vsum = nums[i] + nums[left] + nums[right]
            if vsum < 0:
                left += 1
            elif vsum > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left-1]:
                    left += 1

```

Given a reference of a node in a [**connected**](https://en.wikipedia.org/wiki/Connectivity_(graph_theory)#Connected_graph) undirected graph.

Return a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) (clone) of the graph.

Each node in the graph contains a value (`int`) and a list (`List[Node]`) of its neighbors.

```
class Node {
    public int val;
    public List<Node> neighbors;
}
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return node

        stack = [node]
        
        clone_map: Dict[Node, Node] = {}
        clone_map[node] = Node(node.val)

        while stack:
            vertics = stack.pop()
            clone_node = clone_map[vertics]

            for child in vertics.neighbors:
                if child not in clone_map:
                    clone_map[child] = Node(child.val)
                    stack.append(child)

                clone_node.neighbors.append(clone_map[child])

        return clone_map[node]

```

**150. Evaluate Reverse Polish Notation**

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        operations = {
            '+': lambda a, b: a+b,
            '-': lambda a, b: a-b,
            '*': lambda a, b: a*b,
            '/': lambda a, b: int(a/b),
        }
        stack = []
        for token in tokens:
            if token in operations:
                val2 = stack.pop()
                val1 = stack.pop()   
                operation = operations[token]
                stack.append(operation(val1, val2))  
            else:
                stack.append(int(token))
        return stack.pop()

Example 1:

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
Example 2:

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
```

# [**207. Course Schedule**](https://leetcode.com/problems/course-schedule/)

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you **must** take course `bi` first if you want to take course `ai`.

- For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return `true` if you can finish all courses. Otherwise, return `false`.

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take.
To take course 1 you should have finished course 0. So it is possible.
```

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # kahn's algorithm topological sorting
        indegree = [0]*numCourses
        adj_list = {i: [] for i in range(numCourses)}

        for prerequisite in prerequisites:
            adj_list[prerequisite[1]].append(prerequisite[0])
            indegree[prerequisite[0]] += 1

        queue = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)
        
        nodeVisited = 0

        while queue:
            node = queue.popleft()
            nodeVisited += 1

            for neighbor in adj_list[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        
        return nodeVisited == numCourses
        
```

# [**210. Course Schedule II**](https://leetcode.com/problems/course-schedule-ii/)

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you **must** take course `bi` first if you want to take course `ai`.

- For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return *the ordering of courses you should take to finish all courses*. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return **an empty array**.

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].

```

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

        adj_list = defaultdict(list)
        indegree = {}

        for dest, src in prerequisites:
            adj_list[src].append(dest)

            indegree[dest] = indegree.get(dest, 0) + 1

        zero_indgree_queue = deque([k for k in range(numCourses) if k not in indegree])

        topological_sorted_order = []

        while zero_indgree_queue:
            vertex = zero_indgree_queue.popleft()
            topological_sorted_order.append(vertex)

            if vertex in adj_list:
                for neighbor in adj_list[vertex]:
                    indegree[neighbor] -= 1

                    if indegree[neighbor] == 0:
                        zero_indgree_queue.append(neighbor)

        return (topological_sorted_order if len(topological_sorted_order) == numCourses else [])

```

# [**208. Implement Trie (Prefix Tree)**](https://leetcode.com/problems/implement-trie-prefix-tree/)

A [**trie**](https://en.wikipedia.org/wiki/Trie) (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string `word` into the trie.
- `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.
- `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
        

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

# [**322. Coin Chang](https://leetcode.com/problems/coin-change/)e**

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return *the fewest number of coins that you need to make up that amount*. If that amount of money cannot be made up by any combination of the coins, return `-1`.

You may assume that you have an infinite number of each kind of coin.

**Example 1:**

```
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
```

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')]*(amount+1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount+1):
                dp[x] = min(dp[x], dp[x-coin]+1)
        return dp[amount] if dp[amount] != float('inf') else -1
        
```

# [**238. Product of Array Except Self**](https://leetcode.com/problems/product-of-array-except-self/)

Given an integer array `nums`, return *an array* `answer` *such that* `answer[i]` *is equal to the product of all the elements of* `nums` *except* `nums[i]`.

The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in `O(n)` time and without using the division operation.

**Example 1:**

```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

```

**Example 2:**

```
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        total_product = 1
        for num in nums:
            if num: total_product *= num

        ans = [0]*len(nums)
        zero_count = 0
        for i, num in enumerate(nums):
            if num == 0: 
                zero_count += 1
                ans[i] = total_product
        if zero_count > 1: return [0]*len(nums)
        if zero_count: return ans

        for i, num in enumerate(nums):
            ans[i] = total_product//num
        return ans
        
```

# [**70. Climbing Stair](https://leetcode.com/problems/climbing-stairs/)s**

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

**Example 1:**

```
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```

```python
class Solution:
    def climbStairs(self, n: int) -> int:

        # recurrsion (top-dow)

        def dp(i):
            if i<3:
                return i

            if i not in memo:
                memo[i] = dp(i-1) + dp(i-2)
            
            return memo[i]

        memo = {}
        return dp(n)

        # iterative (botton-up)
        if n < 2:
            return n
        dp = [0]*n
        dp[0] = 1
        dp[1] = 2

        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[-1]
        
```

# [**155. Min Stac](https://leetcode.com/problems/min-stack/)k**

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:

- `MinStack()` initializes the stack object.
- `void push(int val)` pushes the element `val` onto the stack.
- `void pop()` removes the element on the top of the stack.
- `int top()` gets the top element of the stack.
- `int getMin()` retrieves the minimum element in the stack.

You must implement a solution with `O(1)` time complexity for each function.

```python
class MinStack:

    def __init__(self):
        self.stack = []
        

    def push(self, val: int) -> None:
        if self.stack:
            min_val = min(self.stack[-1][1], val)
        else:
            min_val = val
        self.stack.append((val, min_val))
        

    def pop(self) -> None:
        return self.stack.pop()
        

    def top(self) -> int:
        if self.stack:
            return self.stack[-1][0]
        

    def getMin(self) -> int:
        if self.stack:
            return self.stack[-1][1]
        

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```

# [**206. Reverse Linked List**](https://leetcode.com/problems/reverse-linked-list/)

Given the `head` of a singly linked list, reverse the list, and return *the reversed list*.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = forw = None
        curr = head 
        while curr:
            forw = curr.next
            curr.next = prev
            prev = curr
            curr = forw        
        return prev
        
```

# [**98. Validate Binary Search Tree**](https://leetcode.com/problems/validate-binary-search-tree/)

Given the `root` of a binary tree, *determine if it is a valid binary search tree (BST)*.

A **valid BST** is defined as follows:

- The left  of a node contains only nodes with keys **less than** the node's key.
    
    subtree
    
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if not root: return None

        stack = [(root, -inf, inf)]

        while stack:
            root, lower, upper = stack.pop()
            if not root:
                continue

            val = root.val

            if val <= lower or val>=upper:
                return False
            stack.append((root.right, val, upper))
            stack.append((root.left, lower, val))

        return True
        
```

# [**200. Number of Islands**](https://leetcode.com/problems/number-of-islands/)

Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return *the number of islands*.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**Example 1:**

```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

```python
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [1]*size

    def find(self, x):
        # path comparision
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # union by rank
        parentx, parenty = self.find(x), self.find(y)
        if parentx == parenty:
            return
        elif self.rank[parentx] > self.rank[parenty]:
            self.rank[parenty] = parentx
        elif self.rank[parentx] < self.rank[parenty]:
            self.rank[parentx] = parenty
        else:
            self.parent[parenty] = parentx
            self.rank[parentx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

class Solution:

    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dirs = {(0, 1), (1, 0), (-1, 0), (0, -1)}
        count = 0
        
        def dfs(grid, r, c):
            for dx, dy in dirs:
                nr, nc = r+dx, c+dy
                if 0<=nr < m and 0<=nc<n and grid[nr][nc] == '1':
                    grid[nr][nc] = '0'
                    dfs(grid, nr, nc)
        
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1':
                    count += 1
                    dfs(grid, r, c)

        return count

        
```

# [**994. Rotting Oranges**](https://leetcode.com/problems/rotting-oranges/)

You are given an `m x n` `grid` where each cell can have one of three values:

- `0` representing an empty cell,
- `1` representing a fresh orange, or
- `2` representing a rotten orange.

Every minute, any fresh orange that is **4-directionally adjacent** to a rotten orange becomes rotten.

Return *the minimum number of minutes that must elapse until no cell has a fresh orange*. If *this is impossible, return* `-1`.

**Example 1:**

![https://assets.leetcode.com/uploads/2019/02/16/oranges.png](https://assets.leetcode.com/uploads/2019/02/16/oranges.png)

```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

```python
from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        dir = [(0,1), (1, 0), (0, -1), (-1, 0)]
        fresh_count = 0

        queue = deque()
        rows, cols = len(grid), len(grid[0])
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c, 0))
                elif grid[r][c] == 1:
                    fresh_count += 1

        minutes_count = 0

        while queue:
            r, c, minutes_count = queue.popleft()
            for dr, dc in dir:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh_count -= 1
                    queue.append((nr, nc, minutes_count+1))
        return minutes_count if fresh_count == 0 else -1

```

# [**33. Search in Rotated Sorted Array**](https://leetcode.com/problems/search-in-rotated-sorted-array/)

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is **possibly rotated** at an unknown pivot index `k` (`1 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` **after** the possible rotation and an integer `target`, return *the index of* `target` *if it is in* `nums`*, or* `-1` *if it is not in* `nums`.

You must write an algorithm with `O(log n)` runtime complexity.

**Example 1:**

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left)//2
            if nums[mid] == target:
                return mid
            # subarray on mid's left is sorted
            elif nums[mid] >= nums[left]:
                if nums[mid] > target and target >= nums[left]:
                    right = mid-1
                else:
                    left = mid+1
            # subarray on mid's right is sorted
            else:
                if nums[mid] < target and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid -1
            
        return -1
        
```

# [**39. Combination Sum**](https://leetcode.com/problems/combination-sum/)

Given an array of **distinct** integers `candidates` and a target integer `target`, return *a list of all **unique combinations** of* `candidates` *where the chosen numbers sum to* `target`*.* You may return the combinations in **any order**.

The **same** number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the

frequency

of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.

**Example 1:**

```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
```

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        def backtrack(remain, comb, start):

            if remain == 0:
                result.append(list(comb))
                return
            
            if remain < 0:
                return

            for i in range(start, len(candidates)):
                
                comb.append(candidates[i])

                backtrack(remain-candidates[i], comb, i)

                comb.pop()

        backtrack(target, [], 0)

        return result
        
```

# [**46. Permutation](https://leetcode.com/problems/permutations/)s**

Given an array `nums` of distinct integers, return *all the possible permutations*. You can return the answer in **any order**.

**Example 1:**

```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        result = []

        def backtrack(curr):
            if len(curr) == len(nums):
                result.append(list(curr))
                return

            for num in nums:
                if num not in curr:
                    curr.append(num)
                    backtrack(curr)
                    curr.pop()

        backtrack([])
        return result
```

# [**56. Merge Intervals**](https://leetcode.com/problems/merge-intervals/)

Given an array of `intervals` where `intervals[i] = [starti, endi]`, merge all overlapping intervals, and return *an array of the non-overlapping intervals that cover all the intervals in the input*.

**Example 1:**

```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
```

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []

        intervals.sort(key = lambda x: x[0])
        merged = [intervals[0]]

        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        return merged
        
```

# [**57. Insert Interval**](https://leetcode.com/problems/insert-interval/)

You are given an array of non-overlapping intervals `intervals` where `intervals[i] = [starti, endi]` represent the start and the end of the `ith` interval and `intervals` is sorted in ascending order by `starti`. You are also given an interval `newInterval = [start, end]` that represents the start and end of another interval.

Insert `newInterval` into `intervals` such that `intervals` is still sorted in ascending order by `starti` and `intervals` still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return `intervals` *after the insertion*.

**Note** that you don't need to modify `intervals` in-place. You can make a new array and return it.

**Example 1:**

```
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
```

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # if not intervals: return [intervals]

        n = len(intervals)

        target = newInterval[0]
        left, right = 0, n-1
        
        while left <= right:
            mid = left + (right-left)//2

            if intervals[mid][0] < target:
                left = mid+1
            else:
                right = mid-1

        intervals.insert(left, newInterval)
        res = []

        # merge overlapping intervals
        for interval in intervals:
            if not res or res[-1][1] < interval[0]:
                res.append(interval)
            else:
                res[-1][1] = max(res[-1][1], interval[1])

        return res
        
```

# [**236. Lowest Common Ancestor of a Binary Tree**](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the [definition of LCA on Wikipedia](https://en.wikipedia.org/wiki/Lowest_common_ancestor): “The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow **a node to be a descendant of itself**).”

**Example 1:**

![https://assets.leetcode.com/uploads/2018/12/14/binarytree.png](https://assets.leetcode.com/uploads/2018/12/14/binarytree.png)

```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        stack = [root]
        parent = {root: None}

        while p not in parent or q not in parent:
            node = stack.pop()

            if node.left:
                stack.append(node.left)
                parent[node.left] = node
            if node.right:
                stack.append(node.right)
                parent[node.right] = node

        ancestors = set()

        while p:
            ancestors.add(p)
            p = parent[p]
        
        while q not in ancestors:
            q = parent[q]
        return q
        
```

# [**76. Minimum Window Substring**](https://leetcode.com/problems/minimum-window-substring/)

Given two strings `s` and `t` of lengths `m` and `n` respectively, return *the **minimum window***

The testcases will be generated such that the answer is **unique**.

**Example 1:**

```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

```

```python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""

        tchars = Counter(t)
        required_chars = len(tchars)
        l, r = 0, 0
        min_len = float('inf')
        min_window = ""

        window_counts = Counter()
        formed = 0

        while r < len(s):
            char = s[r]
            window_counts[char] += 1

            if char in tchars and window_counts[char] == tchars[char]:
                formed += 1

            while l <= r and formed == required_chars:
                char = s[l]
                
                # Update the result if the current window is smaller than the previously found one
                if r - l + 1 < min_len:
                    min_len = r - l + 1
                    min_window = s[l:r+1]

                # Remove characters from the left of the window
                window_counts[char] -= 1
                if char in tchars and window_counts[char] < tchars[char]:
                    formed -= 1

                l += 1

            r += 1

        return min_window if min_len != float('inf') else ""

```

# [**297. Serialize and Deserialize Binary Tree**](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Clarification:** The input/output format is the same as [how LeetCode serializes a binary tree](https://support.leetcode.com/hc/en-us/articles/360011883654-What-does-1-null-2-3-mean-in-binary-tree-representation-). You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg](https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg)

```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
```

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """

        def serialize_helper(node):
            if not node:
                return 'None,'
            return str(node.val) + ',' + serialize_helper(node.left) + serialize_helper(node.right)
        
        return serialize_helper(root)
        # stack = [root]
        # string = ''

        # while stack:
        #     node = stack.pop()
        #     string += str(node.val)

        #     if node.left: stack.append(node.left)
        #     if node.right: stack.append(node.right)
        # print(string)
        # return string

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def rdeserialize(values):
            if not values:
                return None
            value = values.pop(0)
            if value == 'None':
                return None
            node = TreeNode(int(value))
            node.left = rdeserialize(values)
            node.right = rdeserialize(values)
            return node
        
        values = data.split(',')
        return rdeserialize(values)

        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
```

# [**295. Find Median from Data Stream**](https://leetcode.com/problems/find-median-from-data-stream/)

The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

- For example, for `arr = [2,3,4]`, the median is `3`.
- For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the MedianFinder class:

- `MedianFinder()` initializes the `MedianFinder` object.
- `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
- `double findMedian()` returns the median of all elements so far. Answers within `105` of the actual answer will be accepted.

**Example 1:**

```
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]
```

```python
import heapq
class MedianFinder:

    def __init__(self):
        self.max_heap = []
        self.min_heap = []
        

    def addNum(self, num: int) -> None:
        heapq.heappush(self.max_heap, -num)

        # balance the heap
        if self.min_heap and (-self.max_heap[0] > self.min_heap[0]):
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)

        # balance both the heaps
        if len(self.max_heap) > len(self.min_heap)+1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.max_heap) < len(self.min_heap):
            val = -heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, val)

    def findMedian(self) -> float:
        if not self.max_heap and not self.min_heap:
            return 0
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0])/2

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

# [**42. Trapping Rain Water**](https://leetcode.com/problems/trapping-rain-water/)

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

**Example 1:**

![https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.

```

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        max_water = 0
        l, r = 0, len(height) -1
        max_left, max_right = height[l], height[r]

        while l<r:
            if max_left < max_right:
                l += 1
                max_left = max(max_left, height[l])
                max_water += max_left-height[l]
            else:
                r -= 1
                max_right = max(max_right, height[r])
                max_water += max_right-height[r]

        return max_water
        
```

# [**543. Diameter of Binary Tree**](https://leetcode.com/problems/diameter-of-binary-tree/)

Given the `root` of a binary tree, return *the length of the **diameter** of the tree*.

The **diameter** of a binary tree is the **length** of the longest path between any two nodes in a tree. This path may or may not pass through the `root`.

The **length** of a path between two nodes is represented by the number of edges between them.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg](https://assets.leetcode.com/uploads/2021/03/06/diamtree.jpg)

```
Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        
        def dfs(node):
            nonlocal max_path
            if not node:
                return 0
            
            left_path = dfs(node.left)
            right_path = dfs(node.right)
            max_path = max(max_path, left_path + right_path)
            
            return max(left_path, right_path) + 1
            
        max_path = 0
        dfs(root)
        return max_path
        
        
```

# [**67. Add Binary**](https://leetcode.com/problems/add-binary/)

Given two binary strings `a` and `b`, return *their sum as a binary string*.

**Example 1:**

```
Input: a = "11", b = "1"
Output: "100"
```

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        x, y = int(a, 2), int(b, 2)
        while y:
            answer = x^y
            carry = (x & y) << 1
            x, y = answer, carry
        print(x)
        return bin(x)[2:]
        
```

# [**169. Majority Element**](https://leetcode.com/problems/majority-element/)

Given an array `nums` of size `n`, return *the majority element*.

The majority element is the element that appears more than `⌊n / 2⌋` times. You may assume that the majority element always exists in the array.

**Example 1:**

```
Input: nums = [3,2,3]
Output: 3
```

```python
# Boyer-Moore Voting Algorithm
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1

        return candidate
        
```

# [**876. Middle of the Linked List**](https://leetcode.com/problems/middle-of-the-linked-list/)

Given the `head` of a singly linked list, return *the middle node of the linked list*.

If there are two middle nodes, return **the second middle** node.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/07/23/lc-midlist1.jpg](https://assets.leetcode.com/uploads/2021/07/23/lc-midlist1.jpg)

```
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

```

**Example 2:**

![https://assets.leetcode.com/uploads/2021/07/23/lc-midlist2.jpg](https://assets.leetcode.com/uploads/2021/07/23/lc-midlist2.jpg)

```
Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.

```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow
        
```

# [**721. Accounts Merge**](https://leetcode.com/problems/accounts-merge/)

Given a list of `accounts` where each element `accounts[i]` is a list of strings, where the first element `accounts[i][0]` is a name, and the rest of the elements are **emails** representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails **in sorted order**. The accounts themselves can be returned in **any order**.

**Example 1:**

```
Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Explanation:
The first and second John's are the same person as they have the common email "johnsmith@mail.com".
The third John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'],
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
```

```python
from collections import defaultdict

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
    
    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
class Solution:
    def accountsMerge(self, accounts):
        uf = UnionFind()
        email_to_name = {}

        # Step 1: Initialize the Union-Find structure
        for account in accounts:
            name = account[0]
            emails = account[1:]
            for email in emails:
                email_to_name[email] = name
                uf.add(email)
            
            # Union all emails in the same account
            for i in range(1, len(emails)):
                uf.union(emails[0], emails[i])
        
        # Step 2: Collect emails for each root
        root_to_emails = defaultdict(set)
        for email in email_to_name:
            root = uf.find(email)
            root_to_emails[root].add(email)
        
        # Step 3: Format the result
        result = []
        for root, emails in root_to_emails.items():
            name = email_to_name[root]
            result.append([name] + sorted(emails))
        
        return result

```

# [**75. Sort Colors**](https://leetcode.com/problems/sort-colors/)

Given an array `nums` with `n` objects colored red, white, or blue, sort them [**in-place](https://en.wikipedia.org/wiki/In-place_algorithm)** so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

**Example 1:**

```
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

```

**Example 2:**

```
Input: nums = [2,0,1]
Output: [0,1,2]
```

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Dutch National Flag problem solution.
        """
        # For all idx < p0 : nums[idx < p0] = 0
        # curr is an index of elements under consideration
        p0 = curr = 0

        # For all idx > p2 : nums[idx > p2] = 2
        p2 = len(nums) - 1

        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                p0 += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[p2] = nums[p2], nums[curr]
                p2 -= 1
            else:
                curr += 1
```

# [**5. Longest Palindromic Substring**](https://leetcode.com/problems/longest-palindromic-substring/)

Given a string `s`, return *the longest palindromic substring*

**Example 1:**

```
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

```

**Example 2:**

```
Input: s = "cbbd"
Output: "bb"
```

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        sr = s[::-1]
        
        n = len(s)
        if n == 0:
            return ""
        
        dp = [[0]*(n+1) for _ in range(n+1)]
        
        for i in range(1, n+1):
            for j in range(1, n+1):
                if s[i-1] == sr[j-1]:
                    dp[i][j] = 1+dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        # reconstruct
        i, j = n, n
        sp = []

        while i > 0 and j > 0:
            if s[i-1] == sr[j-1]:
                sp.append(s[i-1])
                j -= 1
                i -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        return ''.join(sp)
  
```

# [**416. Partition Equal Subset Sum**](https://leetcode.com/problems/partition-equal-subset-sum/)

Given an integer array `nums`, return `true` *if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or* `false` *otherwise*.

**Example 1:**

```
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

```

**Example 2:**

```
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.

```

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum %2 != 0:
            return False

        subset_sum = total_sum//2
        n = len(nums)
        
        dp = [[False] * (subset_sum + 1) for _ in range(n+1)]
        dp[0][0] = True
        

        for i in range(1, n+1):
            curr = nums[i-1]
            for j in range(subset_sum + 1):
                if j < curr:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-curr]

        return dp[n][subset_sum]
        
        
```

# [**199. Binary Tree Right Side View**](https://leetcode.com/problems/binary-tree-right-side-view/)

Given the `root` of a binary tree, imagine yourself standing on the **right side** of it, return *the values of the nodes you can see ordered from top to bottom*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/02/14/tree.jpg](https://assets.leetcode.com/uploads/2021/02/14/tree.jpg)

```
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

```

**Example 2:**

```
Input: root = [1,null,3]
Output: [1,3]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return None

        queue = []
        result = []
        queue.append(root)

        while queue:
            qlen = len(queue)
            level_res = []
            for i in range(qlen):
                node = queue.pop(0)
                level_res.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level_res[-1])

        return result

```

# [**78. Subsets**](https://leetcode.com/problems/subsets/)

Given an integer array `nums` of **unique** elements, return *all possible*

*subsets*

The solution set **must not** contain duplicate subsets. Return the solution in **any order**.

**Example 1:**

```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:

        self.output = []

        self.n, self.k = len(nums), None
        for self.k in range(self.n+1):
            self.backtrack(0, [], nums)
        return self.output

    def backtrack(self, first, curr, nums):
        if len(curr) == self.k:
            self.output.append(curr[:])

        for i in range(first, self.n):
            curr.append(nums[i])
            self.backtrack(i+1, curr, nums)
            curr.pop()        
```

# [**127. Word Ladder**](https://leetcode.com/problems/word-ladder/)

A **transformation sequence** from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:

- Every adjacent pair of words differs by a single letter.
- Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
- `sk == endWord`

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return *the **number of words** in the **shortest transformation sequence** from* `beginWord` *to* `endWord`*, or* `0` *if no such sequence exists.*

**Example 1:**

```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
```

```python
from collections import defaultdict, deque 

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        adj_map = defaultdict(list)

        for word in wordList:
            for j in range(len(word)):
                pattern = word[:j] + '*' + word[j+1:]
                adj_map[pattern].append(word)
        print(adj_map)

        visited = set([beginWord])
        q = [beginWord]
        res = 1

        while q:
            size = len(q)
            for j in range(size):
                word = q.pop(0)
                if word == endWord:
                    return res
                for j in range(len(word)):
                    pattern = word[:j] + '*' + word[j+1:]
                    for nword in adj_map[pattern]:
                        if nword not in visited:
                            q.append(nword)
                            visited.add(nword)
            res += 1
        return 0
```

# [**104. Maximum Depth of Binary Tre](https://leetcode.com/problems/maximum-depth-of-binary-tree/)e**

Given the `root` of a binary tree, return *its maximum depth*.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg](https://assets.leetcode.com/uploads/2020/11/26/tmp-tree.jpg)

```
Input: root = [3,9,20,null,null,15,7]
Output: 3
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        
```

# [**217. Contains Duplicate**](https://leetcode.com/problems/contains-duplicate/)

Given an integer array `nums`, return `true` if any value appears **at least twice** in the array, and return `false` if every element is distinct.

**Example 1:**

**Input:** nums = [1,2,3,1]

**Output:** true

**Explanation:**

The element 1 occurs at the indices 0 and 3.

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        map = set()
        for num in nums:
            if num in map:
                return True
            map.add(num)
        return False
```

# [**62. Unique Paths**](https://leetcode.com/problems/unique-paths/)

There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.

Given the two integers `m` and `n`, return *the number of possible unique paths that the robot can take to reach the bottom-right corner*.

The test cases are generated so that the answer will be less than or equal to `2 * 109`.

**Example 1:**

![https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)

```
Input: m = 3, n = 7
Output: 28
```

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # if m == 1 or n == 1:
        #     return 1

        # return self.uniquePaths(m, n-1) + self.uniquePaths(m-1, n)

        dp = [[1]*n for _ in range(m)]

        for col in range(1, m):
            for row in range(1, n):
                dp[col][row] = dp[col-1][row] + dp[col][row-1]
        
        return dp[m-1][n-1]
        
```

# [**105. Construct Binary Tree from Preorder and Inorder Traversal**](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return *the binary tree*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/02/19/tree.jpg](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        val = preorder.pop()
        mid = inorder.index(val)

        root = TreeNode(val)

        root.left = self.buildTree(preorder, inorder[:mid])
        root.right = self.buildTree(preorder, inorder[mid+1:])

        return root

        
```

# [**230. Kth Smallest Element in a BST**](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

Given the `root` of a binary search tree, and an integer `k`, return *the* `kth` *smallest value (**1-indexed**) of all the values of the nodes in the tree*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg](https://assets.leetcode.com/uploads/2021/01/28/kthtree1.jpg)

```
Input: root = [3,1,4,null,2], k = 1
Output: 1

```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        if not root:
            return None
        
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right

```

# [**146. LRU Cache**](https://leetcode.com/problems/lru-cache/)

Design a data structure that follows the constraints of a [**Least Recently Used (LRU) cache**](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU).

Implement the `LRUCache` class:

- `LRUCache(int capacity)` Initialize the LRU cache with **positive** size `capacity`.
- `int get(int key)` Return the value of the `key` if the key exists, otherwise return `1`.
- `void put(int key, int value)` Update the value of the `key` if the `key` exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, **evict** the least recently used key.

The functions `get` and `put` must each run in `O(1)` average time complexity.

**Example 1:**

```
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]
```

```python
class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = {}
        self.head = ListNode(-1, -1)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key not in self.dic:
            return -1

        node = self.dic[key]
        self.remove(node)
        self.add(node)
        return node.val

    def put(self, key, val):
        if key in self.dic:
            old_node = self.dic[key]
            self.remove(old_node)

        node = ListNode(key, val)
        self.dic[key] = node
        self.add(node)

        if len(self.dic) > self.capacity:
            node_to_delete = self.head.next
            self.remove(node_to_delete)
            del self.dic[node_to_delete.key]

    def add(self, node):
        previous_node = self.tail.prev
        previous_node.next = node
        node.prev = previous_node
        node.next = self.tail
        self.tail.prev = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

Google:

# [**3. Longest Substring Without Repeating Characters**](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

Given a string `s`, find the length of the **longest**

**substring**

without repeating characters.

**Example 1:**

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

```

**Example 2:**

```
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

```python
from collections import Counter
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        chars = Counter()

        l, r = 0, 0

        max_len = 0

        while r < len(s):
            chars[s[r]] += 1

            while chars[s[r]] > 1:
                chars[s[l]] -= 1
                l += 1
            
            max_len = max(max_len, r-l+1)

            r += 1

        return max_len

```

# [**11. Container With Most Water**](https://leetcode.com/problems/container-with-most-water/)

You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `ith` line are `(i, 0)` and `(i, height[i])`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return *the maximum amount of water a container can store*.

**Notice** that you may not slant the container.

**Example 1:**

![https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:

        left, right = 0, len(height)-1
        max_area = 0

        while left<right:
            width = right-left
            max_area = max(max_area, min(height[left], height[right])*width)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area

```

# [**167. Two Sum II - Input Array Is Sorted**](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

Given a **1-indexed** array of integers `numbers` that is already ***sorted in non-decreasing order***, find two numbers such that they add up to a specific `target` number. Let these two numbers be `numbers[index1]` and `numbers[index2]` where `1 <= index1 < index2 <= numbers.length`.

Return *the indices of the two numbers,* `index1` *and* `index2`*, **added by one** as an integer array* `[index1, index2]` *of length 2.*

The tests are generated such that there is **exactly one solution**. You **may not** use the same element twice.

Your solution must use only constant extra space.

**Example 1:**

```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
```

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        left = 0
        right = n-1
        while left < right:
            sumvalues = numbers[left] + numbers[right]
            if target == sumvalues:
                return [left+1, right+1]
            elif target > sumvalues:
                left += 1
            else:
                right -= 1
            print(left, right)
        return [-1, -1]
        
```

# [**163. Missing Ranges**](https://leetcode.com/problems/missing-ranges/)

You are given an inclusive range `[lower, upper]` and a **sorted unique** integer array `nums`, where all elements are within the inclusive range.

A number `x` is considered **missing** if `x` is in the range `[lower, upper]` and `x` is not in `nums`.

Return *the **shortest sorted** list of ranges that **exactly covers all the missing numbers***. That is, no element of `nums` is included in any of the ranges, and each missing number is covered by one of the ranges.

**Example 1:**

```
Input: nums = [0,1,3,50,75], lower = 0, upper = 99
Output: [[2,2],[4,49],[51,74],[76,99]]
Explanation: The ranges are:
[2,2]
[4,49]
[51,74]
[76,99]
```

```python
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        ranges = []
        if len(nums) == 0:
            ranges.append([lower, upper])
            return ranges
        if nums[0] > lower:
            ranges.append([lower, nums[0]-1])
        
        for i in range(len(nums)-1):
            if nums[i] + 1 <nums[i+1]:
                ranges.append([nums[i]+1, nums[i+1]-1]) 
        if nums[-1] < upper:
            ranges.append([nums[-1]+1, upper])
        print(ranges)
        return ranges
        
```

# [**159. Longest Substring with At Most Two Distinct Characters**](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/)

Given a string `s`, return *the length of the longest*

*substring*

*that contains at most **two distinct characters***

.

**Example 1:**

```
Input: s = "eceba"
Output: 3
Explanation: The substring is "ece" which its length is 3.

```

**Example 2:**

```
Input: s = "ccaabbb"
Output: 5
Explanation: The substring is "aabbb" which its length is 5.

```

```python
from collections import defaultdict
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        l, r = 0, 0
        map = {}
        max_len = 0
        
        while r < len(s):
            
            map[s[r]] = map.get(s[r], 0) + 1
            # print(map, max_len, l, r)
            while len(map) > 2:
                map[s[l]] -= 1
                if map[s[l]] == 0:
                    del map[s[l]]
                l += 1
                # print(map, l, r)
            max_len = max(max_len, r-l+1)
            r += 1
        # print(max_len)
        return max_len
        
```

# [**66. Plus One**](https://leetcode.com/problems/plus-one/)

You are given a **large integer** represented as an integer array `digits`, where each `digits[i]` is the `ith` digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading `0`'s.

Increment the large integer by one and return *the resulting array of digits*.

**Example 1:**

```
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].

```

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        rd = digits[::-1]
        c = 1
        for i, d in enumerate(rd):
            print(d)
            nd = d+c
            if nd > 9:
                rd[i] = nd%10
                c = nd//10
            else:
                rd[i] = nd
                c = nd//10
                break
        if c:
            rd.append(c)
        
        return rd[::-1]
        
```

# [**215. Kth Largest Element in an Array**](https://leetcode.com/problems/kth-largest-element-in-an-array/)

Given an integer array `nums` and an integer `k`, return *the* `kth` *largest element in the array*.

Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Can you solve it without sorting?

**Example 1:**

```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

```

**Example 2:**

```
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

```python
import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
                
        return min_heap[0]
```

# [**253. Meeting Rooms II**](https://leetcode.com/problems/meeting-rooms-ii/)

Given an array of meeting time intervals `intervals` where `intervals[i] = [starti, endi]`, return *the minimum number of conference rooms required*.

**Example 1:**

```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
```

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        free_rooms = []

        intervals.sort(key=lambda x: x[0])

        heapq.heappush(free_rooms, intervals[0][1])

        for i in intervals[1:]:
            if free_rooms[0] <= i[0]:
                heapq.heappop(free_rooms)

            heapq.heappush(free_rooms, i[1])

        return len(free_rooms)
        
```

# [**20. Valid Parentheses**](https://leetcode.com/problems/valid-parentheses/)

Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        map = {
            '[': ']',
            '{': '}',
            '(': ')'
        }
        inp = '[{('
        oup = ']})'

        for char in s:
            if char in inp:
                stack.append(char)
            if char in oup:
                if len(stack) == 0: return False
                if map.get(stack[-1]) == char:
                    stack.pop()
                else:
                    return False
        if len(stack) == 0:
            return True
        return False

        
```

# [**2. Add Two Numbers**](https://leetcode.com/problems/add-two-numbers/)

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/10/02/addtwonumber1.jpg](https://assets.leetcode.com/uploads/2020/10/02/addtwonumber1.jpg)

```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummyNode = ListNode(0)
        curr = dummyNode
        carry = 0
        while l2 != None or l1 != None or carry != 0:
            l1val = l1.val if l1 else 0
            l2val = l2.val if l2 else 0

            asum = l1val + l2val + carry
            carry = asum//10
            newNode = ListNode(asum%10)
            curr.next = newNode
            curr = newNode
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummyNode.next

        
```

# [**19. Remove Nth Node From End of List**](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        slow, fast = dummy, dummy

        for i in range(n+1):
            fast = fast.next
        
        while fast:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next
        return dummy.next
            
        
        
```

# [**21. Merge Two Sorted Lists**](https://leetcode.com/problems/merge-two-sorted-lists/)

You are given the heads of two sorted linked lists `list1` and `list2`.

Merge the two lists into one **sorted** list. The list should be made by splicing together the nodes of the first two lists.

Return *the head of the merged linked list*.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg](https://assets.leetcode.com/uploads/2020/10/03/merge_ex1.jpg)

```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        prehead = ListNode(-1)
        prev = prehead
        while list1 and list2:
            if list1.val <= list2.val:
                prev.next = list1
                list1 = list1.next
            else:
                prev.next = list2
                list2 = list2.next
            prev = prev.next
        prev.next = list1 if list1 is not None else list2

        return prehead.next

```

# [**138. Copy List with Random Pointer**](https://leetcode.com/problems/copy-list-with-random-pointer/)

A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.

Construct a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) of the list. The deep copy should consist of exactly `n` **brand new** nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. **None of the pointers in the new list should point to nodes in the original list**.

For example, if there are two nodes `X` and `Y` in the original list, where `X.random --> Y`, then for the corresponding two nodes `x` and `y` in the copied list, `x.random --> y`.

Return *the head of the copied linked list*.

The linked list is represented in the input/output as a list of `n` nodes. Each node is represented as a pair of `[val, random_index]` where:

- `val`: an integer representing `Node.val`
- `random_index`: the index of the node (range from `0` to `n-1`) that the `random` pointer points to, or `null` if it does not point to any node.

Your code will **only** be given the `head` of the original linked list.

**Example 1:**

![https://assets.leetcode.com/uploads/2019/12/18/e1.png](https://assets.leetcode.com/uploads/2019/12/18/e1.png)

```
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        curr = head
        random_map = {None: None}
        while curr:
            copy = Node(curr.val)
            random_map[curr] = copy
            curr = curr.next
        
        curr = head
        while curr:
            copy = random_map[curr]
            copy.next = random_map[curr.next]
            copy.random = random_map[curr.random]
            curr = curr.next
        
        return random_map[head]
```

# [**124. Binary Tree Maximum Path Sum**](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, return *the maximum **path sum** of any **non-empty** path*.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg](https://assets.leetcode.com/uploads/2020/10/13/exx1.jpg)

```
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

```

**Example 2:**

![https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg](https://assets.leetcode.com/uploads/2020/10/13/exx2.jpg)

```
Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        def dfs(node):
            if not node:
                return 0

            left_max = max(dfs(node.left), 0)
            right_max = max(dfs(node.right), 0)

            curr_max = node.val + left_max + right_max

            self.max_path = max(self.max_path, curr_max)

            return node.val + max(left_max, right_max)

        self.max_path = float('-inf')
        dfs(root)
        return self.max_path

```

# [**222. Count Complete Tree Nodes**](https://leetcode.com/problems/count-complete-tree-nodes/)

Given the `root` of a **complete** binary tree, return the number of the nodes in the tree.

According to [**Wikipedia**](http://en.wikipedia.org/wiki/Binary_tree#Types_of_binary_trees), every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between `1` and `2h` nodes inclusive at the last level `h`.

Design an algorithm that runs in less than `O(n)` time complexity.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/01/14/complete.jpg](https://assets.leetcode.com/uploads/2021/01/14/complete.jpg)

```
Input: root = [1,2,3,4,5,6]
Output: 6
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def compute_depth(self, node: TreeNode):
        d = 0
        while node.left:
            node = node.left
            d += 1
        return d

    def exist(self, idx, d, node):
        left, right = 0, 2**d - 1
        for _ in range(d):
            pivot = (left+right)//2
            if idx <= pivot:
                node = node.left
                right = pivot
            else:
                node = node.right
                left = pivot + 1
        return node is not None

    def countNodes(self, root: Optional[TreeNode]):
        if not root:
            return 0
        
        d = self.compute_depth(root)
        
        if d == 0:
            return 1
        
        left, right = 0, 2**d - 1
        while left <= right:
            pivot = (left+right)//2
            if self.exist(pivot, d, root):
                left = pivot + 1
            else:
                right = pivot -1

        return (2**d -1) + left

    # def countNodes(self, root: Optional[TreeNode]) -> int:
    #     count = 0
    #     if not root: return count

    #     stack = [root]

    #     while stack:
    #         node = stack.pop()
    #         count += 1
    #         if node.left: stack.append(node.left)
    #         if node.right: stack.append(node.right)

    #     return count

```

# [**329. Longest Increasing Path in a Matrix**](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)

Given an `m x n` integers `matrix`, return *the length of the longest increasing path in* `matrix`.

From each cell, you can either move in four directions: left, right, up, or down. You **may not** move **diagonally** or move **outside the boundary** (i.e., wrap-around is not allowed).

**Example 1:**

![https://assets.leetcode.com/uploads/2021/01/05/grid1.jpg](https://assets.leetcode.com/uploads/2021/01/05/grid1.jpg)

```
Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
Output: 4
Explanation: The longest increasing path is[1, 2, 6, 9].

```

```python
from typing import List

class Solution:
    def __init__(self):
        self.dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
        self.n = 0
        self.m = 0
        self.cache = []

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        self.n, self.m = len(matrix), len(matrix[0])
        self.cache = [[0] * self.m for _ in range(self.n)]

        ans = 0

        for i in range(self.n):
            for j in range(self.m):
                ans = max(ans, self.dfs(matrix, i, j))
        return ans

    def dfs(self, matrix: List[List[int]], i: int, j: int) -> int:
        if self.cache[i][j] != 0:
            return self.cache[i][j]
        
        max_length = 1
        for dir in self.dirs:
            x, y = dir[0] + i, dir[1] + j
            if 0 <= x < self.n and 0 <= y < self.m and matrix[x][y] > matrix[i][j]:
                max_length = max(max_length, 1 + self.dfs(matrix, x, y))
        
        self.cache[i][j] = max_length
        return self.cache[i][j]
```

# [**394. Decode String**](https://leetcode.com/problems/decode-string/)

Given an encoded string, return its decoded string.

The encoding rule is: `k[encoded_string]`, where the `encoded_string` inside the square brackets is being repeated exactly `k` times. Note that `k` is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, `k`. For example, there will not be input like `3a` or `2[4]`.

The test cases are generated so that the length of the output will never exceed `105`.

**Example 1:**

```
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

```

**Example 2:**

```
Input: s = "3[a2[c]]"
Output: "accaccacc"
```

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []

        for c in s:
            print(stack, c)
            if c != ']':
                stack.append(c)
            else:
                a = ''
                while True:
                    i = stack[-1]
                    if i == '[':
                        stack.pop()
                        break
                    a = i + a
                    stack.pop()
                n = ''
                while True:
                    if not stack:
                        break
                    if not stack[-1].isdigit():
                        break
                    if stack:
                        n = stack.pop() + n
                # print(a, n)
                stack.append(a*int(n))
        final_str = ''
        while stack:
            final_str = stack.pop() + final_str
        return final_str

        
```

# [**951. Flip Equivalent Binary Trees**](https://leetcode.com/problems/flip-equivalent-binary-trees/)

For a binary tree **T**, we can define a **flip operation** as follows: choose any node, and swap the left and right child subtrees.

A binary tree **X** is *flip equivalent* to a binary tree **Y** if and only if we can make **X** equal to **Y** after some number of flip operations.

Given the roots of two binary trees `root1` and `root2`, return `true` if the two trees are flip equivalent or `false` otherwise.

**Example 1:**

![https://assets.leetcode.com/uploads/2018/11/29/tree_ex.png](https://assets.leetcode.com/uploads/2018/11/29/tree_ex.png)

```
Input: root1 = [1,2,3,4,5,6,null,null,null,7,8], root2 = [1,3,2,null,6,4,5,null,null,null,null,8,7]
Output: true
Explanation:We flipped at nodes with values 1, 3, and 5.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if root1 is root2:
            return True
        
        if not root1 or not root2 or root1.val != root2.val:
            return False

        return (
            self.flipEquiv(root1.left, root2.left) and 
            self.flipEquiv(root1.right, root2.right) or
            self.flipEquiv(root1.left, root2.right) and
            self.flipEquiv(root1.right, root2.left)
        )
```

# [**852. Peak Index in a Mountain Array**](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

You are given an integer **mountain** array `arr` of length `n` where the values increase to a **peak element** and then decrease.

Return the index of the peak element.

Your task is to solve it in `O(log(n))` time complexity.

**Example 1:**

**Input:** arr = [0,1,0]

**Output:** 1

**Example 2:**

**Input:** arr = [0,2,1,0]

**Output:** 1

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left, right = 0, len(arr)

        while left<right:
            mid = (left+right)//2

            if arr[mid] < arr[mid-1]:
                right = mid
            else:
                left = mid+1
        return left-1

            
        
```

# [**53. Maximum Subarray**](https://leetcode.com/problems/maximum-subarray/)

Given an integer array `nums`, find the subarray with the largest sum, and return *its sum*.

**Example 1:**

```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_till_now = 0
        max_so_far = -inf
        
        for num in nums:
            max_till_now = max(num, max_till_now + num)
            max_so_far = max(max_so_far, max_till_now)
            
        return max_so_far
```

# [**152. Maximum Product Subarray**](https://leetcode.com/problems/maximum-product-subarray/)

Given an integer array `nums`, find a subarray that has the largest product, and return *the product*.

The test cases are generated so that the answer will fit in a **32-bit** integer.

**Example 1:**

```
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
```

```python
class Solution:
    def maxProduct(self, nums):
        if len(nums) == 0:
            return 0

        max_so_far = nums[0]
        min_so_far = nums[0]
        result = max_so_far

        for i in range(1, len(nums)):
            curr = nums[i]
            temp_max = max(curr, max(max_so_far * curr, min_so_far * curr))
            min_so_far = min(curr, min(max_so_far * curr, min_so_far * curr))

            # Update max_so_far after updates to min_so_far to avoid overwriting it
            max_so_far = temp_max
            # Update the result with the maximum product found so far
            result = max(max_so_far, result)

        return result
        
```

# [**410. Split Array Largest Sum**](https://leetcode.com/problems/split-array-largest-sum/)

Given an integer array `nums` and an integer `k`, split `nums` into `k` non-empty subarrays such that the largest sum of any subarray is **minimized**.

Return *the minimized largest sum of the split*.

A **subarray** is a contiguous part of the array.

**Example 1:**

```
Input: nums = [7,2,5,10,8], k = 2
Output: 18
Explanation: There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8], where the largest sum among the two subarrays is only 18.
```

```python
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        def feasible(largestSum):
            currentSum, subarray = 0, 1
            for num in nums:
                currentSum += num
                if currentSum > largestSum:
                    currentSum = num
                    subarray += 1
                    if subarray > k:
                        return False
            return True

        left, right = max(nums), sum(nums)
        while left < right:
            mid = left + (right-left)//2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left
        
```

# [**128. Longest Consecutive Sequence**](https://leetcode.com/problems/longest-consecutive-sequence/)

Given an unsorted array of integers `nums`, return *the length of the longest consecutive elements sequence.*

You must write an algorithm that runs in `O(n)` time.

**Example 1:**

```
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is[1, 2, 3, 4]. Therefore its length is 4.

```

**Example 2:**

```
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num-1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                
                longest_streak = max(longest_streak, current_streak)

        return longest_streak

```

# [**9. Palindrome Number**](https://leetcode.com/problems/palindrome-number/)

Given an integer `x`, return `true` *if* `x` *is a **palindrome** and false otherwise.*.

**Example 1:**

```
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

```

**Example 2:**

```
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
```

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x<0 or (x%10 == 0 and x!=0):
            return False

        revertedNumber = 0

        while x>revertedNumber:
            revertedNumber = revertedNumber *10 + x%10
            x //= 10

        return x == revertedNumber or x==revertedNumber//10
```

# [**27. Remove Element**](https://leetcode.com/problems/remove-element/)

Given an integer array `nums` and an integer `val`, remove all occurrences of `val` in `nums` [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm). The order of the elements may be changed. Then return *the number of elements in* `nums` *which are not equal to* `val`.

Consider the number of elements in `nums` which are not equal to `val` be `k`, to get accepted, you need to do the following things:

- Change the array `nums` such that the first `k` elements of `nums` contain the elements which are not equal to `val`. The remaining elements of `nums` are not important as well as the size of `nums`.

If all assertions pass, then your solution will be **accepted**.

**Example 1:**

```
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).

```

**Example 2:**

```
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).

```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        pointer = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[pointer] = nums[i]
                pointer += 1
        return pointer
```

# [**2235. Add Two Integers**](https://leetcode.com/problems/add-two-integers/)

*the **sum** of the two integers*

.

**Example 1:**

```
Input: num1 = 12, num2 = 5
Output: 17
Explanation: num1 is 12, num2 is 5, and their sum is 12 + 5 = 17, so 17 is returned.

```

**Example 2:**

```
Input: num1 = -10, num2 = 4
Output: -6
Explanation: num1 + num2 = -6, so -6 is returned.

```

```python
class Solution:
    def sum(self, num1: int, num2: int) -> int:
        # 32-bit integer range
        MAX = 0xFFFFFFFF
        
        while num2 != 0:
            carry = num1 & num2
            num1 = (num1 ^ num2) & MAX
            num2 = carry << 1
            
        return num1 if num1 <= 0x7FFFFFFF else ~(num1 ^ MAX)
        
```

# [**13. Roman to Integer**](https://leetcode.com/problems/roman-to-integer/)

Roman numerals are represented by seven different symbols: `I`, `V`, `X`, `L`, `C`, `D` and `M`.

```
SymbolValue
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, `2` is written as `II` in Roman numeral, just two ones added together. `12` is written as `XII`, which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as `IX`. There are six instances where subtraction is used:

- `I` can be placed before `V` (5) and `X` (10) to make 4 and 9.
- `X` can be placed before `L` (50) and `C` (100) to make 40 and 90.
- `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.

**Example 2:**

```
Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.

```

**Example 3:**

```
Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

```

```python
values = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

class Solution:
    def romanToInt(self, s: str) -> int:
        total = values.get(s[-1])
        for i in reversed(range(len(s)-1)):
            if values[s[i]] < values[s[i+1]]:
                total -= values[s[i]]
            else:
                total += values[s[i]]
        return total
```

# [**7. Reverse Integer**](https://leetcode.com/problems/reverse-integer/)

Given a signed 32-bit integer `x`, return `x` *with its digits reversed*. If reversing `x` causes the value to go outside the signed 32-bit integer range `[-231, 231 - 1]`, then return `0`.

**Assume the environment does not allow you to store 64-bit integers (signed or unsigned).**

**Example 1:**

```
Input: x = 123
Output: 321

```

**Example 2:**

```
Input: x = -123
Output: -321
```

```python
class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x<0 else 1
        rev, x = 0, abs(x)

        while x:
            x, mod = divmod(x, 10)
            rev = rev *10 + mod
            if rev > 2**31-1:
                return 0
        return sign * rev
        
```

# [**547. Number of Provinces**](https://leetcode.com/problems/number-of-provinces/)

There are `n` cities. Some of them are connected, while some are not. If city `a` is connected directly with city `b`, and city `b` is connected directly with city `c`, then city `a` is connected indirectly with city `c`.

A **province** is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an `n x n` matrix `isConnected` where `isConnected[i][j] = 1` if the `ith` city and the `jth` city are directly connected, and `isConnected[i][j] = 0` otherwise.

Return *the total number of **provinces***.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/12/24/graph1.jpg](https://assets.leetcode.com/uploads/2020/12/24/graph1.jpg)

```
Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2
```

```python
class UnionFind:

    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [1]*size

    def find(self, x):
        # using path comparision
        if x != self.parent[x]:
		        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # using union by rank
        parentX, parentY = self.find(x), self.find(y)
        if parentX == parentY:
            return
        elif self.rank[parentX] > self.rank[parentY]:
            self.parent[parentY] = parentX
        elif self.rank[parentX] < self.rank[parentY]:
            self.parent[parentX] = parentY
        else:
            self.parent[parentY] = parentX
            self.rank[parentX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        unionFind = UnionFind(n)
        numOfComponents = n
        for i in range(n):
            for j in range(i+1, n):
                if isConnected[i][j] == 1 and not unionFind.connected(i, j):
                    numOfComponents -= 1
                    unionFind.union(i, j)
        return numOfComponents

        
```

# [**261. Graph Valid Tree**](https://leetcode.com/problems/graph-valid-tree/)

You have a graph of `n` nodes labeled from `0` to `n - 1`. You are given an integer n and a list of `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi` in the graph.

Return `true` *if the edges of the given graph make up a valid tree, and* `false` *otherwise*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/03/12/tree1-graph.jpg](https://assets.leetcode.com/uploads/2021/03/12/tree1-graph.jpg)

```
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true
```

```python
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [1]*size

    def find(self, x):
        # path comparision 
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # union by rank
        parentX, parentY = self.find(x), self.find(y)
        if parentX == parentY:
            return False
        elif self.rank[parentX] > self.rank[parentY]:
            self.parent[parentY] = parentX
        elif self.rank[parentY] > self.rank[parentX]:
            self.parent[parentX] = parentY
        else:
            self.parent[parentY] = parentX
            self.rank[parentX] += 1
        return True

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if n-1 != len(edges): return False
        uf = UnionFind(n)
        for i, j in edges:
            if not uf.union(i, j):
                return False
        return True
            

        
```

# [**323. Number of Connected Components in an Undirected Graph**](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

You have a graph of `n` nodes. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an edge between `ai` and `bi` in the graph.

Return *the number of connected components in the graph*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/03/14/conn1-graph.jpg](https://assets.leetcode.com/uploads/2021/03/14/conn1-graph.jpg)

```
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2
```

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1]*size

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        parentx, parenty = self.find(x), self.find(y)
        if parentx==parenty:
            return
        elif self.rank[parentx] > self.rank[parenty]:
            self.parent[parenty] = parentx
        elif self.rank[parentx] < self.rank[parenty]:
            self.parent[parentx] = parenty
        else:
            self.parent[parenty] = parentx
            self.rank[parentx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)
        

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        connectedComponent = n
        uf = UnionFind(n)
        for a, b in edges:
            if not uf.connected(a,b):
                connectedComponent -= 1
                uf.union(a, b)
        return connectedComponent
        
```

# [**1971. Find if Path Exists in Graph**](https://leetcode.com/problems/find-if-path-exists-in-graph/)

There is a **bi-directional** graph with `n` vertices, where each vertex is labeled from `0` to `n - 1` (**inclusive**). The edges in the graph are represented as a 2D integer array `edges`, where each `edges[i] = [ui, vi]` denotes a bi-directional edge between vertex `ui` and vertex `vi`. Every vertex pair is connected by **at most one** edge, and no vertex has an edge to itself.

You want to determine if there is a **valid path** that exists from vertex `source` to vertex `destination`.

Given `edges` and the integers `n`, `source`, and `destination`, return `true` *if there is a **valid path** from* `source` *to* `destination`*, or* `false` *otherwise.*

**Example 1:**

![https://assets.leetcode.com/uploads/2021/08/14/validpath-ex1.png](https://assets.leetcode.com/uploads/2021/08/14/validpath-ex1.png)

```
Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
Output: true
Explanation: There are two paths from vertex 0 to vertex 2:
- 0 → 1 → 2
- 0 → 2
```

```python
import collections
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        graph = collections.defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)

        seen = [False]*n

        seen[source] = True
        queue = collections.deque([source])

        while queue:
            curr_node = queue.popleft()
            if curr_node == destination:
                return True
            for next_node in graph[curr_node]:
                if not seen[next_node]:
                    seen[next_node] = True
                    queue.append(next_node)
        return False
        
```

# [**797. All Paths From Source to Target**](https://leetcode.com/problems/all-paths-from-source-to-target/)

Given a directed acyclic graph (**DAG**) of `n` nodes labeled from `0` to `n - 1`, find all possible paths from node `0` to node `n - 1` and return them in **any order**.

The graph is given as follows: `graph[i]` is a list of all nodes you can visit from node `i` (i.e., there is a directed edge from node `i` to node `graph[i][j]`).

**Example 1:**

![https://assets.leetcode.com/uploads/2020/09/28/all_1.jpg](https://assets.leetcode.com/uploads/2020/09/28/all_1.jpg)

```
Input: graph = [[1,2],[3],[3],[]]
Output: [[0,1,3],[0,2,3]]
Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.
```

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        n = len(graph)
        all_paths = []

        stack = [(0, [0])]

        while stack:
            node, path = stack.pop()

            if node == n-1:
                all_paths.append(path)
                continue

            for child in graph[node]:
                stack.append((child, path + [child]))

        return all_paths
        
```

# [**116. Populating Next Right Pointers in Each Node**](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

You are given a **perfect binary tree** where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}

```

Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to `NULL`.

Initially, all next pointers are set to `NULL`.

**Example 1:**

![https://assets.leetcode.com/uploads/2019/02/14/116_sample.png](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

```
Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation:Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root: return None

        queue = collections.deque([root])

        while queue:
            size = len(queue)

            for i in range(size):
                node = queue.popleft()

                if i < size -1:
                    node.next = queue[0]

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return root

        
```

# [**429. N-ary Tree Level Order Traversal**](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)

Given an n-ary tree, return the *level order* traversal of its nodes' values.

*Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).*

**Example 1:**

![https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png](https://assets.leetcode.com/uploads/2018/10/12/narytreeexample.png)

```
Input: root = [1,null,3,2,4,null,5,6]
Output: [[1],[3,2,4],[5,6]]
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
from collections import deque

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root: return None

        res = []
        queue = deque([root])

        while queue:
            size = len(queue)
            level = []

            for _ in range(size):
                node = queue.popleft()
                level.append(node.val)
                queue.extend(node.children)
            res.append(level)

        return res
        
```

# [**1584. Min Cost to Connect All Points**](https://leetcode.com/problems/min-cost-to-connect-all-points/)

You are given an array `points` representing integer coordinates of some points on a 2D-plane, where `points[i] = [xi, yi]`.

The cost of connecting two points `[xi, yi]` and `[xj, yj]` is the **manhattan distance** between them: `|xi - xj| + |yi - yj|`, where `|val|` denotes the absolute value of `val`.

Return *the minimum cost to make all points connected.* All points are connected if there is **exactly one** simple path between any two points.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/08/26/d.png](https://assets.leetcode.com/uploads/2020/08/26/d.png)

```
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20
Explanation:
We can connect the points as shown above to get the minimum cost of 20.
Notice that there is a unique path between every pair of points.
```

![https://assets.leetcode.com/uploads/2020/08/26/c.png](https://assets.leetcode.com/uploads/2020/08/26/c.png)

```python
#krushkal algo for minimum spanning tree
class UnionFind:

    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0]*size

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        parentx, parenty = self.find(x), self.find(y)
        if parentx != parenty:
            if self.rank[parentx] > self.rank[parenty]:
                self.parent[parenty] = parentx
            elif self.rank[parentx] < self.rank[parenty]:
                self.parent[parentx] = parenty
            else:
                self.parent[parenty] = parentx
                self.rank[parentx] += 1
            return True
        else:
            return False
    
    def connected(self, x, y):
        return self.parent[x] == self.parent[y]

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        all_edges = []

        for curr_node in range(n):
            for next_node in range(curr_node+1, n):
                weight = abs(points[curr_node][0] - points[next_node][0]) + abs(points[curr_node][1] - points[next_node][1])

                all_edges.append((weight, curr_node, next_node))

        # sort all the edges in increasing order
        all_edges.sort()

        uf = UnionFind(n)
        mst_cost = 0
        edges_used = 0

        for weight, node1, node2 in all_edges:
            if uf.union(node1, node2):
                mst_cost += weight
                edges_used += 1
                if edges_used == n-1:
                    break

        return mst_cost

```

# [**743. Network Delay Time**](https://leetcode.com/problems/network-delay-time/)

You are given a network of `n` nodes, labeled from `1` to `n`. You are also given `times`, a list of travel times as directed edges `times[i] = (ui, vi, wi)`, where `ui` is the source node, `vi` is the target node, and `wi` is the time it takes for a signal to travel from source to target.

We will send a signal from a given node `k`. Return *the **minimum** time it takes for all the* `n` *nodes to receive the signal*. If it is impossible for all the `n` nodes to receive the signal, return `-1`.

**Example 1:**

![https://assets.leetcode.com/uploads/2019/05/23/931_example_1.png](https://assets.leetcode.com/uploads/2019/05/23/931_example_1.png)

```
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

```

**Example 2:**

```
Input: times = [[1,2,1]], n = 2, k = 1
Output: 1

```

```python
import heapq
import sys
from typing import List

class Solution:
    def dijkstra(self, signal_received_at: List[int], source: int, n: int) -> None:
        # Priority queue to store (time, node)
        pq = [(0, source)]
        signal_received_at[source] = 0
        
        while pq:
            curr_node_time, curr_node = heapq.heappop(pq)
            
            # If the current node time is greater than the recorded time, skip it
            if curr_node_time > signal_received_at[curr_node]:
                continue
            
            # Update the times for adjacent nodes
            for time, neighbor_node in self.adj[curr_node]:
                if signal_received_at[neighbor_node] > curr_node_time + time:
                    signal_received_at[neighbor_node] = curr_node_time + time
                    heapq.heappush(pq, (signal_received_at[neighbor_node], neighbor_node))
    
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Initialize adjacency list
        self.adj = [[] for _ in range(n + 1)]
        
        # Build the adjacency list
        for u, v, w in times:
            self.adj[u].append((w, v))
        
        # Initialize the signal received times with infinity
        signal_received_at = [float('inf')] * (n + 1)
        
        # Perform Dijkstra's algorithm
        self.dijkstra(signal_received_at, k, n)
        
        # Find the maximum delay time
        max_time = max(signal_received_at[1:])  # Ignore the 0-th index
        
        # If any node is unreachable, return -1
        return max_time if max_time < float('inf') else -1

```

# [**1136. Parallel Course](https://leetcode.com/problems/parallel-courses/)s**

You are given an integer `n`, which indicates that there are `n` courses labeled from `1` to `n`. You are also given an array `relations` where `relations[i] = [prevCoursei, nextCoursei]`, representing a prerequisite relationship between course `prevCoursei` and course `nextCoursei`: course `prevCoursei` has to be taken before course `nextCoursei`.

In one semester, you can take **any number** of courses as long as you have taken all the prerequisites in the **previous** semester for the courses you are taking.

Return *the **minimum** number of semesters needed to take all courses*. If there is no way to take all the courses, return `-1`.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/02/24/course1graph.jpg](https://assets.leetcode.com/uploads/2021/02/24/course1graph.jpg)

```
Input: n = 3, relations = [[1,3],[2,3]]
Output: 2
Explanation: The figure above represents the given graph.
In the first semester, you can take courses 1 and 2.
In the second semester, you can take course 3.
```

```python
# kahn's algorithm for topological sor
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        graph = {i: [] for i in range(1, n+1)}
        in_count = {i: 0 for i in range(1, n+1)}

        for start_node, end_node in relations:
            graph[start_node].append(end_node)
            in_count[end_node] += 1

        queue = []

        for node in graph:
            if in_count[node] == 0:
                queue.append(node)

        step = 0
        studied_count = 0

        while queue:

            step += 1
            next_queue = []

            for node in queue:
                studied_count += 1
                end_nodes = graph[node]

                for end_node in end_nodes:
                    in_count[end_node] -= 1

                    if in_count[end_node] == 0:
                        next_queue.append(end_node)

            queue = next_queue

        return step if studied_count == n else -1
        
```

# [**151. Reverse Words in a String**](https://leetcode.com/problems/reverse-words-in-a-string/)

Given an input string `s`, reverse the order of the **words**.

A **word** is defined as a sequence of non-space characters. The **words** in `s` will be separated by at least one space.

Return *a string of the words in reverse order concatenated by a single space.*

**Note** that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

**Example 1:**

```
Input: s = "the sky is blue"
Output: "blue is sky the"

```

**Example 2:**

```
Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s_list = [i.strip() for i in s.split()]
        print(s_list)

        left, right = 0, len(s_list)-1
        while left<right:
            s_list[left], s_list[right] = s_list[right], s_list[left]
            left += 1
            right -= 1
        return ' '.join(s_list)
        
```

# [**334. Increasing Triplet Subsequence**](https://leetcode.com/problems/increasing-triplet-subsequence/)

Given an integer array `nums`, return `true` *if there exists a triple of indices* `(i, j, k)` *such that* `i < j < k` *and* `nums[i] < nums[j] < nums[k]`. If no such indices exists, return `false`.

**Example 1:**

```
Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.

```

**Example 2:**

```
Input: nums = [5,4,3,2,1]
Output: false
Explanation: No triplet exists.
```

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first_number = inf
        second_number = inf
        for n in nums:
            if n<=first_number:
                first_number = n
            elif n<=second_number:
                second_number = n
            else:
                return True
        return False
```

# [**443. String Compression**](https://leetcode.com/problems/string-compression/)

Given an array of characters `chars`, compress it using the following algorithm:

Begin with an empty string `s`. For each group of **consecutive repeating characters** in `chars`:

- If the group's length is `1`, append the character to `s`.
- Otherwise, append the character followed by the group's length.

The compressed string `s` **should not be returned separately**, but instead, be stored **in the input character array `chars`**. Note that group lengths that are `10` or longer will be split into multiple characters in `chars`.

After you are done **modifying the input array,** return *the new length of the array*.

You must write an algorithm that uses only constant extra space.

**Example 1:**

```
Input: chars = ["a","a","b","b","c","c","c"]
Output: Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
Explanation: The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".
```

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        pointer = 0
        counter = 1
        i = 0
        while i < len(chars):
            if i == len(chars)-1 or chars[i] != chars[i+1]:
                chars[pointer] = chars[i]
                pointer += 1
                if 1 < counter <10:
                        chars[pointer] = str(counter)
                        pointer += 1
                        counter = 1
                elif counter >= 10:
                    counter_str = str(counter)
                    for num in counter_str:
                        chars[pointer] = num
                        pointer += 1
                    counter = 1
            else:
                counter += 1                    
            i += 1
        return pointer

        
```

# [**1679. Max Number of K-Sum Pairs**](https://leetcode.com/problems/max-number-of-k-sum-pairs/)

You are given an integer array `nums` and an integer `k`.

In one operation, you can pick two numbers from the array whose sum equals `k` and remove them from the array.

Return *the maximum number of operations you can perform on the array*.

**Example 1:**

```
Input: nums = [1,2,3,4], k = 5
Output: 2
Explanation: Starting with nums = [1,2,3,4]:
- Remove numbers 1 and 4, then nums = [2,3]
- Remove numbers 2 and 3, then nums = []
There are no more pairs that sum up to 5, hence a total of 2 operations.
```

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        print(nums)
        left, right = 0, len(nums)-1
        sum_count = 0
        while left < right:
            if nums[left]+nums[right] < k:
                left += 1
            elif nums[left]+nums[right] > k:
                right -= 1
            else:
                left += 1
                right -= 1
                sum_count += 1
        return sum_count
        
```

# [**1456. Maximum Number of Vowels in a Substring of Given Length**](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

Given a string `s` and an integer `k`, return *the maximum number of vowel letters in any substring of* `s` *with length* `k`.

**Vowel letters** in English are `'a'`, `'e'`, `'i'`, `'o'`, and `'u'`.

**Example 1:**

```
Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.

```

**Example 2:**

```
Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.

```

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        max_so_far = 0
        for i in range(k):
            if s[i] in vowels:
                max_so_far += 1
        current_max = max_so_far
        for i in range(k, len(s)):
            if s[i-k] in vowels:
                current_max -= 1
            if s[i] in vowels:
                current_max += 1
            max_so_far = max(max_so_far, current_max)
        return max_so_far
        
```

# [**1657. Determine if Two Strings Are Close**](https://leetcode.com/problems/determine-if-two-strings-are-close/)

Two strings are considered **close** if you can attain one from the other using the following operations:

- Operation 1: Swap any two **existing** characters.
    - For example, `abcde -> aecdb`
- Operation 2: Transform **every** occurrence of one **existing** character into another **existing** character, and do the same with the other character.
    - For example, `aacabb -> bbcbaa` (all `a`'s turn into `b`'s, and all `b`'s turn into `a`'s)

You can use the operations on either string as many times as necessary.

Given two strings, `word1` and `word2`, return `true` *if* `word1` *and* `word2` *are **close**, and* `false` *otherwise.*

**Example 1:**

```
Input: word1 = "abc", word2 = "bca"
Output: true
Explanation: You can attain word2 from word1 in 2 operations.
Apply Operation 1: "abc" -> "acb"
Apply Operation 1: "acb" -> "bca"
```

```python
from collections import Counter
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        word1c = Counter(word1)
        word2c = Counter(word2)
        return sorted(word1c.keys()) == sorted(word2c.keys()) and sorted(word1c.values()) == sorted(word2c.values())
        
        
```

# [**2352. Equal Row and Column Pairs**](https://leetcode.com/problems/equal-row-and-column-pairs/)

Given a **0-indexed** `n x n` integer matrix `grid`, *return the number of pairs* `(ri, cj)` *such that row* `ri` *and column* `cj` *are equal*.

A row and column pair is considered equal if they contain the same elements in the same order (i.e., an equal array).

**Example 1:**

![https://assets.leetcode.com/uploads/2022/06/01/ex1.jpg](https://assets.leetcode.com/uploads/2022/06/01/ex1.jpg)

```
Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
Output: 1
Explanation: There is 1 equal row and column pair:
- (Row 2, Column 1): [2,7,7]
```

```python
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        row_map = {}
        for i in range(len(grid[0])):
            if tuple(grid[i]) in row_map:
                row_map[tuple(grid[i])] += 1
            else:
                row_map[tuple(grid[i])] = 1
        count = 0
        print(row_map)
        for j in range(len(grid[0])):
            col = [grid[i][j] for i in range(len(grid[0]))]
            print(col)
            if tuple(col) in row_map:
                count += row_map[tuple(col)]

        return count
        
```

# [**735. Asteroid Collision**](https://leetcode.com/problems/asteroid-collision/)

We are given an array `asteroids` of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

**Example 1:**

```
Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

```

```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for a in asteroids:
            while stack and a<0<stack[-1]:
                if stack[-1] < -a:
                    stack.pop()
                    continue
                if stack[-1] == -a:
                    stack.pop()
                break
            else:
                stack.append(a)
        return stack
```

# [**328. Odd Even Linked Lis](https://leetcode.com/problems/odd-even-linked-list/)t**

Given the `head` of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return *the reordered list*.

The **first** node is considered **odd**, and the **second** node is **even**, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.

You must solve the problem in `O(1)` extra space complexity and `O(n)` time complexity.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/03/10/oddeven-linked-list.jpg](https://assets.leetcode.com/uploads/2021/03/10/oddeven-linked-list.jpg)

```
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: return None
        odd = head
        even = odd.next
        evenHead = even

        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = evenHead
        return head

            

        
```

# [**2130. Maximum Twin Sum of a Linked List**](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/)

In a linked list of size `n`, where `n` is **even**, the `ith` node (**0-indexed**) of the linked list is known as the **twin** of the `(n-1-i)th` node, if `0 <= i <= (n / 2) - 1`.

- For example, if `n = 4`, then node `0` is the twin of node `3`, and node `1` is the twin of node `2`. These are the only nodes with twins for `n = 4`.

The **twin sum** is defined as the sum of a node and its twin.

Given the `head` of a linked list with even length, return *the **maximum twin sum** of the linked list*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/12/03/eg1drawio.png](https://assets.leetcode.com/uploads/2021/12/03/eg1drawio.png)

```
Input: head = [5,4,2,1]
Output: 6
Explanation:
Nodes 0 and 1 are the twins of nodes 3 and 2, respectively. All have twin sum = 6.
There are no other nodes with twins in the linked list.
Thus, the maximum twin sum of the linked list is 6.
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        # find mid
        slow, fast = head, head
        maximumSum = 0
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # reverse half of the linked list
        curr, prev = slow, None

        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        start = head
        while prev:
            maximumSum = max(maximumSum, start.val+prev.val)
            prev, start =  prev.next, start.next

        return maximumSum
            
        
        
```

# [**1448. Count Good Nodes in Binary Tree**](https://leetcode.com/problems/count-good-nodes-in-binary-tree/)

Given a binary tree `root`, a node *X* in the tree is named **good** if in the path from root to *X* there are no nodes with a value *greater than* X.

Return the number of **good** nodes in the binary tree.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/04/02/test_sample_1.png](https://assets.leetcode.com/uploads/2020/04/02/test_sample_1.png)

```
Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation: Nodes in blue aregood.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        stack = []
        good_nodes = 0

        # save (val, max_val) in stack
        stack.append((root, root.val))

        while stack:
            node, node_max_val = stack.pop()
            if node_max_val <= node.val:
                good_nodes += 1
            if node.left:
                stack.append((node.left, max(node_max_val, node.val)))
            if node.right:
                stack.append((node.right, max(node_max_val, node.val)))

        return good_nodes

```

# [**437. Path Sum III**](https://leetcode.com/problems/path-sum-iii/)

Given the `root` of a binary tree and an integer `targetSum`, return *the number of paths where the sum of the values along the path equals* `targetSum`.

The path does not need to start or end at the root or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).

**Example 1:**

![https://assets.leetcode.com/uploads/2021/04/09/pathsum3-1-tree.jpg](https://assets.leetcode.com/uploads/2021/04/09/pathsum3-1-tree.jpg)

```
Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3
Explanation: The paths that sum to 8 are shown.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        def preorder(node: TreeNode, curr_sum) -> None:
            nonlocal count
            if not node:
                return 
            
            # The current prefix sum
            curr_sum += node.val
            
            # Here is the sum we're looking for
            if curr_sum == k:
                count += 1
            
            # The number of times the curr_sum − k has occurred already, 
            # determines the number of times a path with sum k 
            # has occurred up to the current node
            count += h[curr_sum - k]
            
            # Add the current sum into a hashmap
            # to use it during the child nodes' processing
            h[curr_sum] += 1
            
            # Process the left subtree
            preorder(node.left, curr_sum)
            # Process the right subtree
            preorder(node.right, curr_sum)
            
            # Remove the current sum from the hashmap
            # in order not to use it during 
            # the parallel subtree processing
            h[curr_sum] -= 1
            
        count, k = 0, targetSum
        h = defaultdict(int)
        preorder(root, 0)
        return count
        # if not root: return 0
        # stack = []
        # stack.append((root, [root.val]))

        # valid_paths = []

        # while stack:
        #     node, paths = stack.pop()
        #     print(node, paths, valid_paths)
        #     if sum(paths) == targetSum:
        #         valid_paths.append(paths)
        #     elif sum(paths) > targetSum:
        #         while sum(paths) > targetSum:
        #             num = paths.pop(0)
        #             if sum(paths) == targetSum:
        #                 valid_paths.append(paths)
            
        #     if node.left:
        #         stack.append((node.left, paths + [node.left.val]))
        #     if node.right:
        #         stack.append((node.right, paths + [node.right.val]))

        # print(valid_paths)
        # return len(valid_paths)            
        
```

# [**1372. Longest ZigZag Path in a Binary Tree**](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/)

You are given the `root` of a binary tree.

A ZigZag path for a binary tree is defined as follow:

- Choose **any** node in the binary tree and a direction (right or left).
- If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
- Change the direction from right to left or from left to right.
- Repeat the second and third steps until you can't move in the tree.

Zigzag length is defined as the number of nodes visited - 1. (A single node has a length of 0).

Return *the longest **ZigZag** path contained in that tree*.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/01/22/sample_1_1702.png](https://assets.leetcode.com/uploads/2020/01/22/sample_1_1702.png)

```
Input: root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1]
Output: 3
Explanation: Longest ZigZag path in blue nodes (right -> left -> right).
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        if not root:
            return None

        max_so_far = 0
        stack = [(root, 0, 0)]

        while stack:
            node, side, path = stack.pop()
            print(node.val, side, path)
            
            if node.left:
                if side == -1:
                    stack.append((node.left, -1, 1))
                else:
                    stack.append((node.left, -1, path+1))
                    max_so_far = max(max_so_far, path+1)
            if node.right:
                if side == 1:
                    stack.append((node.right, 1, 1))
                else:
                    stack.append((node.right, 1, path+1))
                    max_so_far = max(max_so_far, path+1)

        return max_so_far
        
```

# [**199. Binary Tree Right Side View**](https://leetcode.com/problems/binary-tree-right-side-view/)

Given the `root` of a binary tree, imagine yourself standing on the **right side** of it, return *the values of the nodes you can see ordered from top to bottom*.

**Example 1:**

![https://assets.leetcode.com/uploads/2021/02/14/tree.jpg](https://assets.leetcode.com/uploads/2021/02/14/tree.jpg)

```
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return None

        queue = []
        result = []
        queue.append(root)

        while queue:
            qlen = len(queue)
            level_res = []
            for i in range(qlen):
                node = queue.pop(0)
                level_res.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level_res[-1])

        return result

```

# [**1161. Maximum Level Sum of a Binary Tree**](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/)

Given the `root` of a binary tree, the level of its root is `1`, the level of its children is `2`, and so on.

Return the **smallest** level `x` such that the sum of all the values of nodes at level `x` is **maximal**.

**Example 1:**

![https://assets.leetcode.com/uploads/2019/05/03/capture.JPG](https://assets.leetcode.com/uploads/2019/05/03/capture.JPG)

```
Input: root = [1,7,0,7,-8,null,null]
Output: 2
Explanation:
Level 1 sum = 1.
Level 2 sum = 7 + 0 = 7.
Level 3 sum = 7 + -8 = -1.
So we return the level with the maximum sum which is level 2.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        if not root: return None
        queue = [root]
        max_sum_min_level_so_far = (0, -inf)
        level = 0

        while queue:
            qlen = len(queue)
            curr_level_sum = 0
            level += 1

            for i in range(qlen): 
                node = queue.pop(0)
                curr_level_sum += node.val
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            if curr_level_sum > max_sum_min_level_so_far[1]:
                max_sum_min_level_so_far = (level, curr_level_sum)
            print(max_sum_min_level_so_far)

        return max_sum_min_level_so_far[0]

```

# [**450. Delete Node in a BST**](https://leetcode.com/problems/delete-node-in-a-bst/)

Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return *the **root node reference** (possibly updated) of the BST*.

Basically, the deletion can be divided into two stages:

1. Search for a node to remove.
2. If the node is found, delete the node.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/09/04/del_node_1.jpg](https://assets.leetcode.com/uploads/2020/09/04/del_node_1.jpg)

```
Input: root = [5,3,6,2,4,null,7], key = 3
Output: [5,4,6,2,null,null,7]
Explanation: Given key to delete is 3. So we find the node with value 3 and delete it.
One valid answer is [5,4,6,2,null,null,7], shown in the above BST.
Please notice that another valid answer is [5,2,6,null,4,null,7] and it's also accepted.

```

![https://assets.leetcode.com/uploads/2020/09/04/del_node_supp.jpg](https://assets.leetcode.com/uploads/2020/09/04/del_node_supp.jpg)

**Example 2:**

```
Input: root = [5,3,6,2,4,null,7], key = 0
Output: [5,3,6,2,4,null,7]
Explanation: The tree does not contain a node with value = 0.
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def successor(self, root):
        root = root.right
        while root.left:
            root = root.left
        return root.val
    
    def predecessor(self, root):
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root: return None

        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left and not root.right:
                root = None
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
        return root
        
        
```

# [**841. Keys and Rooms**](https://leetcode.com/problems/keys-and-rooms/)

There are `n` rooms labeled from `0` to `n - 1` and all the rooms are locked except for room `0`. Your goal is to visit all the rooms. However, you cannot enter a locked room without having its key.

When you visit a room, you may find a set of **distinct keys** in it. Each key has a number on it, denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.

Given an array `rooms` where `rooms[i]` is the set of keys that you can obtain if you visited room `i`, return `true` *if you can visit **all** the rooms, or* `false` *otherwise*.

**Example 1:**

```
Input: rooms = [[1],[2],[3],[]]
Output: true
Explanation:
We visit room 0 and pick up key 1.
We then visit room 1 and pick up key 2.
We then visit room 2 and pick up key 3.
We then visit room 3.
Since we were able to visit every room, we return true.
```

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        stack = [0]
        seen = [False]*len(rooms)
        seen[0] = True
        while stack:
            node = stack.pop()
            for nei in rooms[node]:
                if not seen[nei]:
                    seen[nei] = True
                    stack.append(nei)
            
        return all(seen)
```

# [**1466. Reorder Routes to Make All Paths Lead to the City Zero**](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/)

There are `n` cities numbered from `0` to `n - 1` and `n - 1` roads such that there is only one way to travel between two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in one direction because they are too narrow.

Roads are represented by `connections` where `connections[i] = [ai, bi]` represents a road from city `ai` to city `bi`.

This year, there will be a big event in the capital (city `0`), and many people want to travel to this city.

Your task consists of reorienting some roads such that each city can visit the city `0`. Return the **minimum** number of edges changed.

It's **guaranteed** that each city can reach city `0` after reorder.

**Example 1:**

![https://assets.leetcode.com/uploads/2020/05/13/sample_1_1819.png](https://assets.leetcode.com/uploads/2020/05/13/sample_1_1819.png)

```
Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
Output: 3
Explanation:Change the direction of edges show in red such that each node can reach the node 0 (capital).
```

```python
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]

    def find(self, x):
        # path comparision 
        if x == self.parent[x]:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # union by rank
        parentX, parentY = self.find(x), self.find(y)
        if parentX == parentY:
            return 
        else:
            self.parent[x] = y

class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        reorders = 0
        uf = UnionFind(n)
        for a, b in connections:
            uf.union(a, b)
        print(uf.parent)
        for a, b in connections:
            if uf.find(b) != uf.find(0):
                print(a, b)
                reorders += 1
                uf.union(b, a)
        return reorders
        
```

# [**875. Koko Eating Bananas**](https://leetcode.com/problems/koko-eating-bananas/)

Koko loves to eat bananas. There are `n` piles of bananas, the `ith` pile has `piles[i]` bananas. The guards have gone and will come back in `h` hours.

Koko can decide her bananas-per-hour eating speed of `k`. Each hour, she chooses some pile of bananas and eats `k` bananas from that pile. If the pile has less than `k` bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return *the minimum integer* `k` *such that she can eat all the bananas within* `h` *hours*.

**Example 1:**

```
Input: piles = [3,6,7,11], h = 8
Output: 4

```

**Example 2:**

```
Input: piles = [30,11,23,4,20], h = 5
Output: 30
```

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:

        def feasible(K):
            # return sum((math.ceil(pile/K) for pile in piles) <= h
            return sum((pile-1)// K +1 for pile in piles) <= h

        
        left, right = 1, max(piles)
        while left < right:
            mid = left + (right - left)//2
            if feasible(mid):
                right = mid
            else:
                left = mid+1
        return left
        
```

# [**162. Find Peak Elemen](https://leetcode.com/problems/find-peak-element/)t**

A peak element is an element that is strictly greater than its neighbors.

Given a **0-indexed** integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any of the peaks**.

You may imagine that `nums[-1] = nums[n] = -∞`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in `O(log n)` time.

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
```

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        l, r = 0, len(nums)-1
        while l < r:
            mid = l + (r-l)//2

            if nums[mid] > nums[mid+1]:
                r = mid
            else:
                l = mid + 1
        return l
        
```

# [**216. Combination Sum III**](https://leetcode.com/problems/combination-sum-iii/)

Find all valid combinations of `k` numbers that sum up to `n` such that the following conditions are true:

- Only numbers `1` through `9` are used.
- Each number is used **at most once**.

Return *a list of all possible valid combinations*. The list must not contain the same combination twice, and the combinations may be returned in any order.

**Example 1:**

```
Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
```

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:

        result = []

        def backtrack(remain, comb, next_start):
            if remain==0 and len(comb) == k:
                result.append(list(comb))
                return
            
            elif remain < 0 or len(comb) == k:
                return

            for i in range(next_start, 9):
                comb.append(i+1)
                backtrack(remain-i-1, comb, i+1)
                comb.pop()

        backtrack(n, [], 0)

        return result
        
```

# [**746. Min Cost Climbing Stairs**](https://leetcode.com/problems/min-cost-climbing-stairs/)

You are given an integer array `cost` where `cost[i]` is the cost of `ith` step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index `0`, or the step with index `1`.

Return *the minimum cost to reach the top of the floor*.

**Example 1:**

```
Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.
```

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        minimum_cost = [0]*(len(cost)+1)

        for i in range(2, len(cost)+1):
            take_one_step = minimum_cost[i-1] + cost[i-1]
            take_two_step = minimum_cost[i-2] + cost[i-2]
            minimum_cost[i] = min(take_one_step, take_two_step)

        return minimum_cost[-1]
        
```

# [**198. House Robber**](https://leetcode.com/problems/house-robber/)

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
```

```python
class Solution:

    # def __init__(self):
    #     self.memo = {}

    # def rob(self, nums: List[int]) -> int:
    #     return self.robFrom(0, nums)

    # def robFrom(self, i, nums):
    #     if i>=len(nums): return 0

    #     if i in self.memo:
    #         return self.memo[i]

    #     self.memo[i] = max(self.robFrom(i+1, nums), self.robFrom(i+2, nums)+nums[i])
    #     return self.memo[i]
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0

        n = len(nums)
        dp = [None for _ in range(n+1)]

        dp[n], dp[n-1] = 0, nums[n-1]

        for i in range(n-2, -1, -1):
            dp[i] = max(dp[i+1], dp[i+2] + nums[i])

        return dp[0]

        
        
```

# [**1143. Longest Common Subsequence**](https://leetcode.com/problems/longest-common-subsequence/)

Given two strings `text1` and `text2`, return *the length of their longest **common subsequence**.* If there is no **common subsequence**, return `0`.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

- For example, `"ace"` is a subsequence of `"abcde"`.

A **common subsequence** of two strings is a subsequence that is common to both strings.

**Example 1:**

```
Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.
```

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp_grid = [[0]*(len(text2)+1) for _ in range(len(text1)+1)]

        for col in reversed(range(len(text2))):
            for row in reversed(range(len(text1))):
                if text2[col] == text1[row]:
                    dp_grid[row][col] = 1 + dp_grid[row+1][col+1]
                else:
                    dp_grid[row][col] = max(dp_grid[row+1][col], dp_grid[row][col+1])

        return dp_grid[0][0]
        
```

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = [0, 0]

        for i in range(n):
            dp[i][i] = True

        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                ans = [i, i + 1]

        for diff in range(2, n):
            for i in range(n - diff):
                j = i + diff
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    ans = [i, j]

        i, j = ans
        return s[i : j + 1]
```

# [**72. Edit Distance**](https://leetcode.com/problems/edit-distance/)

Given two strings `word1` and `word2`, return *the minimum number of operations required to convert `word1` to `word2`*.

You have the following three operations permitted on a word:

- Insert a character
- Delete a character
- Replace a character

**Example 1:**

```
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
```

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        w1l = len(word1)
        w2l = len(word2)

        if w1l == 0: return w2l
        if w2l == 0: return w1l

        dp = [[0 for _ in range(w2l+1)] for _ in range(w1l+1)]

        for w1i in range(1, w1l+1):
            dp[w1i][0] = w1i
        for w2i in range(1, w2l+1):
            dp[0][w2i] = w2i

        for w1i in range(1, w1l+1):
            for w2i in range(1, w2l+1):
                if word2[w2i-1] == word1[w1i-1]:
                    dp[w1i][w2i] = dp[w1i-1][w2i-1]
                else:
                    dp[w1i][w2i] = 1 + min(
                        dp[w1i-1][w2i], dp[w1i][w2i-1],
                        dp[w1i-1][w2i-1]
                    )
        return dp[w1l][w2l]

        
```

# [**208. Implement Trie (Prefix Tree)**](https://leetcode.com/problems/implement-trie-prefix-tree/)

A [**trie**](https://en.wikipedia.org/wiki/Trie) (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string `word` into the trie.
- `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.
- `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.

**Example 1:**

```
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
```

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
        

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

# [**1268. Search Suggestions System**](https://leetcode.com/problems/search-suggestions-system/)

You are given an array of strings `products` and a string `searchWord`.

Design a system that suggests at most three product names from `products` after each character of `searchWord` is typed. Suggested products should have common prefix with `searchWord`. If there are more than three products with a common prefix return the three lexicographically minimums products.

Return *a list of lists of the suggested products after each character of* `searchWord` *is typed*.

**Example 1:**

```
Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"
Output: [["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]
Explanation: products sorted lexicographically = ["mobile","moneypot","monitor","mouse","mousepad"].
After typing m and mo all products match and we show user ["mobile","moneypot","monitor"].
After typing mou, mous and mouse the system suggests ["mouse","mousepad"].

```

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = ''

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word

    def search(self, word):
        node = self.root
        result = []
        for char in word:
            if char not in node.children:
                return []
            node = node.children[char]
        return self.dfs(node)

    def dfs(self, node):
        result = []
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr.is_end_of_word:
                result.append(curr.word)
            for child in sorted(curr.children.keys(), reverse=True):
                stack.append(curr.children[child])
            if len(result)==3:
                break
        return result     

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        trie = Trie()
        for product in products:
            trie.insert(product)
        
        ans = []
        for i in range(len(searchWord)):
            prefix = searchWord[:i+1]
            print(prefix)
            res = trie.search(prefix)
            ans.append(res)

        return ans

        
```

# [**435. Non-overlapping Intervals**](https://leetcode.com/problems/non-overlapping-intervals/)

Given an array of intervals `intervals` where `intervals[i] = [starti, endi]`, return *the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping*.

**Example 1:**

```
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

```

**Example 2:**

```
Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

```

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        ans = 0
        k = -inf

        for x, y in intervals:
            if x>=k:
                k=y
            else:
                ans += 1
        return ans
        
```

# [**739. Daily Temperatures**](https://leetcode.com/problems/daily-temperatures/)

Given an array of integers `temperatures` represents the daily temperatures, return *an array* `answer` *such that* `answer[i]` *is the number of days you have to wait after the* `ith` *day to get a warmer temperature*. If there is no future day for which this is possible, keep `answer[i] == 0` instead.

**Example 1:**

```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        res = [0]*len(temperatures)

        for i in range(len(temperatures)-1, -1, -1):
            print(i, stack, res)
            while stack and temperatures[stack[-1]]<=temperatures[i]:
                stack.pop()
            if stack:
                res[i] = stack[-1] -i

            stack.append(i)
        
        return res
        
```