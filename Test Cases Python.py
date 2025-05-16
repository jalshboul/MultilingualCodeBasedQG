# 1. Binary Search
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1  # Target not found

# 2. Linear Search
def linear_search(arr, target):
    for element in arr:
        if element == target:
            return True
    return False

# 3. Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # swap
    return arr

# 4. Insertion Sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 5. Selection Sort
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 6. Merge Sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Split the array into two halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Recursively sort each half
    left = merge_sort(left)
    right = merge_sort(right)
    
    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 7. Quick Sort
def quick_sort(arr, low, high):
    if low < high:
        # Partition the array and get the pivot index
        pivot_index = partition(arr, low, high)
        
        # Recursively sort the elements before and after the pivot
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
    
# def quick_sort(arr):
    # if len(arr) <= 1:
        # return arr
    
    # pivot = arr[len(arr) // 2]
    # left = [x for x in arr if x < pivot]
    # middle = [x for x in arr if x == pivot]
    # right = [x for x in arr if x > pivot]
    
    # return quick_sort(left) + middle + quick_sort(right)

# def partition(arr, low, high):
    # pivot = arr[high]
    # i = low - 1
    
    # for j in range(low, high):
        # if arr[j] <= pivot:
            # i += 1
            # arr[i], arr[j] = arr[j], arr[i]
    
    # arr[i + 1], arr[high] = arr[high], arr[i + 1]
    # return i + 1

# 8. DFS (Depth-First Search)
def dfs(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    
    return visited

# 9. BFS (Breadth-First Search)
def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    
    while queue:
        vertex = queue.pop(0)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

# 10. Dijkstra's Algorithm
def dijkstra(graph, start):
    import heapq
    
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# 11. Dynamic Programming
def fibonacci_dp(n):
    # Initialize memoization table
    memo = [0] * (n + 1)
    memo[1] = 1
    
    # Fill the table
    for i in range(2, n + 1):
        memo[i] = memo[i - 1] + memo[i - 2]
    
    return memo[n]

def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# 12. Greedy Algorithm
def activity_selection(start, finish):
    n = len(start)
    # Sort activities by finish time
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    
    # Select the first activity
    selected = [activities[0]]
    last_finish = activities[0][1]
    
    # Greedy selection of activities
    for i in range(1, n):
        if activities[i][0] >= last_finish:
            selected.append(activities[i])
            last_finish = activities[i][1]
    
    return selected

# 13. KMP Algorithm (Pattern Matching)
def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    
    lps = compute_lps(pattern)
    
    i = 0  # index for text
    j = 0  # index for pattern
    
    results = []
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            results.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return results

# 14. Kruskal's Algorithm
def kruskal_mst(graph):
    # Sort all edges in non-decreasing order of their weight
    edges = []
    for u in range(len(graph)):
        for v, weight in graph[u].items():
            edges.append((u, v, weight))
    
    edges.sort(key=lambda x: x[2])  # Sort edges by weight
    
    parent = list(range(len(graph)))
    
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    minimum_spanning_tree = []
    
    for u, v, weight in edges:
        if find(u) != find(v):  # Check if including this edge creates a cycle
            union(u, v)
            minimum_spanning_tree.append((u, v, weight))
    
    return minimum_spanning_tree

# 15. Prim's Algorithm
def prim_mst(graph):
    import heapq
    
    start_vertex = 0
    mst = []
    visited = {start_vertex}
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex].items()]
    heapq.heapify(edges)
    
    while edges:
        weight, frm, to = heapq.heappop(edges)
        
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, weight))
            
            for next_vertex, next_weight in graph[to].items():
                if next_vertex not in visited:
                    heapq.heappush(edges, (next_weight, to, next_vertex))
    
    return mst

# 16. Floyd-Warshall Algorithm
def floyd_warshall(graph):
    n = len(graph)
    dist = [row[:] for row in graph]
    
    # Initialize distances
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] == 0:
                dist[i][j] = float('inf')
    
    # Find all pairs shortest paths
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

# 17. Topological Sort
def topological_sort(graph):
    visited = set()
    temp = set()
    order = []
    
    def dfs(node):
        if node in temp:
            return False  # Cycle detected
        if node in visited:
            return True
        
        temp.add(node)
        
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        
        temp.remove(node)
        visited.add(node)
        order.append(node)
        return True
    
    # Process all vertices of the directed acyclic graph
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []  # Graph is not a DAG
    
    return order[::-1]  # Reverse to get topological order

# 18. A* Search Algorithm
def a_star_search(graph, start, goal, heuristic):
    import heapq
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {node: float('infinity') for node in graph}
    g_score[start] = 0
    
    f_score = {node: float('infinity') for node in graph}
    f_score[start] = heuristic(start, goal)
    
    open_set_hash = {start}
    
    while open_set:
        current_f_score, current = heapq.heappop(open_set)
        open_set_hash.remove(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor, weight in graph[current].items():
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
    
    return None  # No path found

# 19. Huffman Coding
def huffman_coding(data):
    from collections import Counter
    import heapq
    
    # Calculate frequency of each character
    frequency = Counter(data)
    
    # Create a priority queue with items (frequency, character)
    priority_queue = [[weight, [char, ""]] for char, weight in frequency.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        lo = heapq.heappop(priority_queue)
        hi = heapq.heappop(priority_queue)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(priority_queue, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    huffman_codes = sorted(priority_queue[0][1:], key=lambda p: (len(p[-1]), p))
    
    # Create a mapping of character to its prefix code
    huffman_mapping = {char: code for char, code in huffman_codes}
    
    # Compress the data
    compressed = ''.join(huffman_mapping[char] for char in data)
    
    return compressed, huffman_mapping