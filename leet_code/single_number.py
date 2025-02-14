"""
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
You must implement a solution with a linear runtime complexity and use only constant extra space.

Example 1:
Input: nums = [2,2,1]
Output: 1

Example 2:
Input: nums = [4,1,2,1,2]
Output: 4

Example 3:
Input: nums = [1]
Output: 1
"""

# class Solution(object):
#     def singleNumber(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         unique_nums = set(nums)
#         for num in unique_nums:
#             if len([n for n in nums if num==n])==1:
#                 return num


# class Solution:
#     def singleNumber(self, nums):
#         result = 0
#         for i, num in enumerate(nums):
#             result ^= num
#             print(f"Step {i+1}: {result} (binary: {bin(result)[2:]:>03})")
#         return result


class Solution:
    def singleNumber(self, nums) -> int:
        """
        Find number that appears only once in array.

        Time: O(n) where n is length of nums
        Space: O(n) for dictionary storage
        """
        num_counts = {}

        # Count occurrences
        for num in nums:
            num_counts[num] = num_counts.get(num, 0) + 1

        # Find single occurrence
        for num in nums:
            if num_counts[num] == 1:
                return num


# print(Solution().singleNumber([2,2,1]))
# print(Solution().singleNumber([4,1,2,1,2]))
# print(Solution().singleNumber([1]))

print(Solution().singleNumber([0, 3, 2, 3, 2]))
