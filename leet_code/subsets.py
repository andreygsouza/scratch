"""
https://leetcode.com/problems/subsets/description/
Given an integer array nums of unique elements, return all possible
subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Example 2:
Input: nums = [0]
Output: [[],[0]]
"""


def generate_combinations(input_list):
    n = len(input_list)
    result = [[]]

    for i in range(n):
        current_len = len(result)
        print("current len", current_len)
        for j in range(current_len):
            print(f"i: {i}; j: {j}")
            new_combination = result[j] + [input_list[i]]
            print(new_combination)
            result.append(new_combination)

    return result


# Example usage
input_list = [1, 2, 3]
combinations = generate_combinations(input_list)
print(combinations)
