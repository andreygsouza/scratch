def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    for i, num in enumerate(nums):
        for j in range(i + 1, len(nums)):
            if num + nums[j] == target:
                return [i, j]


print(twoSum([2, 7, 11, 15], 9))
print(twoSum([3, 2, 4], 6))
