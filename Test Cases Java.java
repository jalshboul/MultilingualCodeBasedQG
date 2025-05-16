// 1. Binary Search
public class BinarySearch {
    public static int binarySearch(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        
        while (low <= high) {
            int mid = low + (high - low) / 2;
            
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        return -1; // Element not found
    }
    
    public static void main(String[] args) {
        int[] sortedArray = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};
        int target = 23;
        int result = binarySearch(sortedArray, target);
        
        if (result != -1) {
            System.out.println("Element found at index: " + result);
        } else {
            System.out.println("Element not found in the array");
        }
    }
}

// 2. Linear Search
public class LinearSearch {
    public static int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1; // Element not found
    }
    
    public static void main(String[] args) {
        int[] array = {64, 34, 25, 12, 22, 11, 90};
        int target = 12;
        int result = linearSearch(array, target);
        
        if (result != -1) {
            System.out.println("Element found at index: " + result);
        } else {
            System.out.println("Element not found in the array");
        }
    }
}

// 3. Bubble Sort
public class BubbleSort {
    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    // Swap arr[j] and arr[j+1]
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
    
    public static void main(String[] args) {
        int[] array = {64, 34, 25, 12, 22, 11, 90};
        bubbleSort(array);
        
        System.out.println("Sorted array:");
        for (int element : array) {
            System.out.print(element + " ");
        }
    }
}

// 4. Insertion Sort
public class InsertionSort {
    public static void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
    
    public static void main(String[] args) {
        int[] array = {64, 34, 25, 12, 22, 11, 90};
        insertionSort(array);
        
        System.out.println("Sorted array:");
        for (int element : array) {
            System.out.print(element + " ");
        }
    }
}

// 5. Selection Sort
public class SelectionSort {
    public static void selectionSort(int[] arr) {
        int n = arr.length;
        
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            
            // Swap the minimum element with the current element
            int temp = arr[minIndex];
            arr[minIndex] = arr[i];
            arr[i] = temp;
        }
    }
    
    public static void main(String[] args) {
        int[] array = {64, 34, 25, 12, 22, 11, 90};
        selectionSort(array);
        
        System.out.println("Sorted array:");
        for (int element : array) {
            System.out.print(element + " ");
        }
    }
}

// 6. Merge Sort
public class MergeSort {
    public static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            // Find the middle point to divide the array into two halves
            int mid = left + (right - left) / 2;
            
            // Sort first and second halves
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            
            // Merge the sorted halves
            merge(arr, left, mid, right);
        }
    }
    
    public static void merge(int[] arr, int left, int mid, int right) {
        // Calculate sizes of two subarrays to be merged
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        // Create temp arrays
        int[] leftArray = new int[n1];
        int[] rightArray = new int[n2];
        
        // Copy data to temp arrays
        for (int i = 0; i < n1; i++) {
            leftArray[i] = arr[left + i];
        }
        for (int j = 0; j < n2; j++) {
            rightArray[j] = arr[mid + 1 + j];
        }
        
        // Merge the temp arrays
        int i = 0, j = 0;
        int k = left;
        
        while (i < n1 && j < n2) {
            if (leftArray[i] <= rightArray[j]) {
                arr[k] = leftArray[i];
                i++;
            } else {
                arr[k] = rightArray[j];
                j++;
            }
            k++;
        }
        
        // Copy remaining elements of leftArray if any
        while (i < n1) {
            arr[k] = leftArray[i];
            i++;
            k++;
        }
        
        // Copy remaining elements of rightArray if any
        while (j < n2) {
            arr[k] = rightArray[j];
            j++;
            k++;
        }
    }
    
    public static void main(String[] args) {
        int[] array = {64, 34, 25, 12, 22, 11, 90};
        mergeSort(array, 0, array.length - 1);
        
        System.out.println("Sorted array:");
        for (int element : array) {
            System.out.print(element + " ");
        }
    }
}

// 7. Quick Sort
public class QuickSort {
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            // Partition the array and get the pivot point
            int pivotIndex = partition(arr, low, high);
            
            // Sort elements before and after pivot
            quickSort(arr, low, pivotIndex - 1);
            quickSort(arr, pivotIndex + 1, high);
        }
    }
    
    public static int partition(int[] arr, int low, int high) {
        // Choose the rightmost element as pivot
        int pivot = arr[high];
        
        // Index of smaller element
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            // If current element is smaller than the pivot
            if (arr[j] < pivot) {
                i++;
                
                // Swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        
        // Swap arr[i+1] and arr[high] (pivot)
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        return i + 1;
    }
    
    public static void main(String[] args) {
        int[] array = {64, 34, 25, 12, 22, 11, 90};
        quickSort(array, 0, array.length - 1);
        
        System.out.println("Sorted array:");
        for (int element : array) {
            System.out.print(element + " ");
        }
    }
}

// 8. Depth-First Search (DFS)
import java.util.*;

public class DFS {
    private int V; // Number of vertices
    private LinkedList<Integer>[] adj; // Adjacency list
    
    @SuppressWarnings("unchecked")
    public DFS(int v) {
        V = v;
        adj = new LinkedList[v];
        for (int i = 0; i < v; i++) {
            adj[i] = new LinkedList<>();
        }
    }
    
    // Add an edge to the graph
    public void addEdge(int v, int w) {
        adj[v].add(w);
    }
    
    // DFS traversal from a given source vertex
    public void dfsTraversal(int startVertex) {
        // Mark all vertices as not visited
        boolean[] visited = new boolean[V];
        
        // Create a stack for DFS
        Stack<Integer> stack = new Stack<>();
        
        // Push the starting vertex
        stack.push(startVertex);
        
        while (!stack.empty()) {
            // Pop a vertex from stack and print it
            startVertex = stack.pop();
            
            // If the popped vertex is not visited, mark it as visited and process it
            if (!visited[startVertex]) {
                System.out.print(startVertex + " ");
                visited[startVertex] = true;
            }
            
            // Get all adjacent vertices of the popped vertex
            // If an adjacent vertex is not visited, push it to the stack
            Iterator<Integer> iterator = adj[startVertex].listIterator();
            while (iterator.hasNext()) {
                int adjacent = iterator.next();
                if (!visited[adjacent]) {
                    stack.push(adjacent);
                }
            }
        }
    }
    
    public static void main(String[] args) {
        DFS graph = new DFS(7);
        
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        
        System.out.println("DFS traversal starting from vertex 0:");
        graph.dfsTraversal(0);
    }
}

// 9. Breadth-First Search (BFS)
import java.util.*;

public class BFS {
    private int V; // Number of vertices
    private LinkedList<Integer>[] adj; // Adjacency list
    
    @SuppressWarnings("unchecked")
    public BFS(int v) {
        V = v;
        adj = new LinkedList[v];
        for (int i = 0; i < v; i++) {
            adj[i] = new LinkedList<>();
        }
    }
    
    // Add an edge to the graph
    public void addEdge(int v, int w) {
        adj[v].add(w);
    }
    
    // BFS traversal from a given source vertex
    public void bfsTraversal(int startVertex) {
        // Mark all vertices as not visited
        boolean[] visited = new boolean[V];
        
        // Create a queue for BFS
        LinkedList<Integer> queue = new LinkedList<>();
        
        // Mark the current node as visited and enqueue it
        visited[startVertex] = true;
        queue.add(startVertex);
        
        while (!queue.isEmpty()) {
            // Dequeue a vertex from queue and print it
            startVertex = queue.poll();
            System.out.print(startVertex + " ");
            
            // Get all adjacent vertices of the dequeued vertex
            // If an adjacent vertex is not visited, mark it as visited and enqueue it
            Iterator<Integer> iterator = adj[startVertex].listIterator();
            while (iterator.hasNext()) {
                int adjacent = iterator.next();
                if (!visited[adjacent]) {
                    visited[adjacent] = true;
                    queue.offer(adjacent);
                }
            }
        }
    }
    
    public static void main(String[] args) {
        BFS graph = new BFS(7);
        
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);
        
        System.out.println("BFS traversal starting from vertex 0:");
        graph.bfsTraversal(0);
    }
}

// 10. Dijkstra's Algorithm
import java.util.*;

public class Dijkstra {
    private static final int INF = Integer.MAX_VALUE;
    
    public static void dijkstra(int[][] graph, int source) {
        int vertices = graph.length;
        
        // The output array distance[i] holds the shortest distance from source to i
        int[] distance = new int[vertices];
        
        // sptSet[i] will be true if vertex i is included in shortest path tree
        boolean[] sptSet = new boolean[vertices];
        
        // Initialize all distances as INFINITE and sptSet[] as false
        for (int i = 0; i < vertices; i++) {
            distance[i] = INF;
            sptSet[i] = false;
        }
        
        // Distance from source vertex to itself is always 0
        distance[source] = 0;
        
        // Use priority queue for getting the minimum distance vertex
        PriorityQueue<Node> pq = new PriorityQueue<>(vertices, Comparator.comparingInt(o -> o.distance));
        pq.add(new Node(source, 0));
        
        while (!pq.isEmpty()) {
            // Extract the vertex with minimum distance value
            int u = pq.poll().vertex;
            
            // Mark the picked vertex as processed
            sptSet[u] = true;
            
            // Update distance value of all adjacent vertices
            for (int v = 0; v < vertices; v++) {
                // Update distance[v] only if is not in sptSet, there is an
                // edge from u to v, and total weight of path from source to
                // v through u is smaller than current value of distance[v]
                if (!sptSet[v] && graph[u][v] != 0 && distance[u] != INF
                        && distance[u] + graph[u][v] < distance[v]) {
                    distance[v] = distance[u] + graph[u][v];
                    pq.add(new Node(v, distance[v]));
                }
            }
        }
        
        // Print the calculated distances
        System.out.println("Vertex \t Distance from Source");
        for (int i = 0; i < vertices; i++) {
            System.out.println(i + " \t\t " + distance[i]);
        }
    }
    
    static class Node {
        int vertex;
        int distance;
        
        public Node(int vertex, int distance) {
            this.vertex = vertex;
            this.distance = distance;
        }
    }
    
    public static void main(String[] args) {
        // Example graph represented as adjacency matrix
        int[][] graph = {
            {0, 4, 0, 0, 0, 0},
            {4, 0, 8, 0, 0, 0},
            {0, 8, 0, 7, 0, 4},
            {0, 0, 7, 0, 9, 14},
            {0, 0, 0, 9, 0, 10},
            {0, 0, 4, 14, 10, 0}
        };
        
        dijkstra(graph, 0);
    }
}

// 11. Dynamic Programming (Fibonacci)
/* public class DynamicProgramming {
    public static int fibonacciDP(int n) {
        int[][] dp = new int[2][n + 1];
        dp[0][0] = 0;
        dp[0][1] = 1;
        
        for (int i = 2; i <= n; i++) {
            dp[0][i] = dp[0][i - 1] + dp[0][i - 2];
        }
        
        return dp[0][n];
    }
    
    public static void main(String[] args) {
        int n = 10;
        System.out.println("Fibonacci number " + n + " is: " + fibonacciDP(n));
        
        // Print the Fibonacci sequence up to n
        System.out.print("Fibonacci sequence up to " + n + ": ");
        for (int i = 0; i <= n; i++) {
            System.out.print(fibonacciDP(i) + " ");
        }
    }
} */

public class DynamicProgramming {
    public static int fibonacciDP(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;

        int prev1 = 1, prev2 = 0, current = 0;
        for (int i = 2; i <= n; i++) {
            current = prev1 + prev2;
            prev2 = prev1;
            prev1 = current;
        }
        return current;
    }

    public static void main(String[] args) {
        int n = 10;
        System.out.println("Fibonacci number " + n + " is: " + fibonacciDP(n));

        // Print the Fibonacci sequence up to n
        System.out.print("Fibonacci sequence up to " + n + ": ");
        int prev2 = 0, prev1 = 1;
        System.out.print(prev2 + " " + prev1 + " ");
        for (int i = 2; i <= n; i++) {
            int current = prev1 + prev2;
            System.out.print(current + " ");
            prev2 = prev1;
            prev1 = current;
        }
    }
}

// 12. Greedy Algorithm (Fractional Knapsack)
import java.util.*;

public class GreedyAlgorithm {
    // Item value class
    static class Item {
        int value, weight;
        
        public Item(int value, int weight) {
            this.value = value;
            this.weight = weight;
        }
    }
    
    // Function to get maximum value that can be put in knapsack of capacity W
    public static double fractionalKnapsack(int W, Item[] items, int n) {
        // Sort items based on value/weight ratio in descending order
        Arrays.sort(items, (a, b) -> {
            double r1 = (double) a.value / a.weight;
            double r2 = (double) b.value / b.weight;
            return Double.compare(r2, r1); // For descending order
        });
        
        // Current weight in knapsack
        int currentWeight = 0;
        
        // Result (value in Knapsack)
        double finalValue = 0.0;
        
        // Greedy approach: Take items with highest value/weight ratio first
        for (int i = 0; i < n; i++) {
            // If adding the complete item doesn't exceed the weight constraint
            if (currentWeight + items[i].weight <= W) {
                currentWeight += items[i].weight;
                finalValue += items[i].value;
            } else {
                // If we can't add the complete item, add a fraction of it for optimal local choice
                int remainingWeight = W - currentWeight;
                finalValue += items[i].value * ((double) remainingWeight / items[i].weight);
                break;
            }
        }
        
        return finalValue;
    }
    
    public static void main(String[] args) {
        int W = 50; // Knapsack capacity
        Item[] items = {
            new Item(60, 10),
            new Item(100, 20),
            new Item(120, 30)
        };
        
        int n = items.length;
        
        System.out.println("Maximum value we can obtain = " + 
                        fractionalKnapsack(W, items, n));
    }
}

// 13. KMP Algorithm
public class KMPAlgorithm {
    public static void KMPSearch(String pattern, String text) {
        int M = pattern.length();
        int N = text.length();
        
        // Create lps[] that will hold the longest prefix suffix values for pattern
        int[] lps = new int[M];
        
        // Preprocess the pattern
        computeLPSArray(pattern, M, lps);
        
        int i = 0; // index for text
        int j = 0; // index for pattern
        
        while (i < N) {
            if (pattern.charAt(j) == text.charAt(i)) {
                j++;
                i++;
            }
            
            if (j == M) {
                System.out.println("Pattern found at index " + (i - j));
                j = lps[j - 1];
            } 
            // Mismatch after j matches
            else if (i < N && pattern.charAt(j) != text.charAt(i)) {
                // Do not match lps[0..lps[j-1]] characters,
                // they will match anyway
                if (j != 0) {
                    j = lps[j - 1];
                } else {
                    i++;
                }
            }
        }
    }
    
    public static void computeLPSArray(String pattern, int M, int[] lps) {
        // Length of the previous longest prefix suffix
        int len = 0;
        int i = 1;
        lps[0] = 0; // lps[0] is always 0
        
        // Calculate lps[i] for i = 1 to M-1
        while (i < M) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                len++;
                lps[i] = len;
                i++;
            } else {
                // This is tricky. Consider the example: AAACAAAA and i = 7
                if (len != 0) {
                    len = lps[len - 1];
                    // Note that we do not increment i here
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
    }
    
    public static void main(String[] args) {
        String text = "ABABDABACDABABCABAB";
        String pattern = "ABABCABAB";
        KMPSearch(pattern, text);
    }
}

// 14. Kruskal's Algorithm
import java.util.*;

public class KruskalAlgorithm {
    // Edge class to represent the weighted edges in the graph
    static class Edge implements Comparable<Edge> {
        int src, dest, weight;
        
        public Edge(int src, int dest, int weight) {
            this.src = src;
            this.dest = dest;
            this.weight = weight;
        }
        
        // Comparator for sorting edges based on their weights
        @Override
        public int compareTo(Edge edge) {
            return this.weight - edge.weight;
        }
    }
    
    // Subset class for union-find
    static class Subset {
        int parent, rank;
        
        public Subset(int parent, int rank) {
            this.parent = parent;
            this.rank = rank;
        }
    }
    
    // Number of vertices and edges
    private int V, E;
    private Edge[] edges;
    
    // Constructor
    public KruskalAlgorithm(int v, int e) {
        V = v;
        E = e;
        edges = new Edge[E];
    }
    
    // Find function for union-find
    private int find(Subset[] subsets, int i) {
        if (subsets[i].parent != i) {
            subsets[i].parent = find(subsets, subsets[i].parent);
        }
        return subsets[i].parent;
    }
    
    // Union function for union-find
    private void union(Subset[] subsets, int x, int y) {
        int rootX = find(subsets, x);
        int rootY = find(subsets, y);
        
        if (subsets[rootX].rank < subsets[rootY].rank) {
            subsets[rootX].parent = rootY;
        } else if (subsets[rootX].rank > subsets[rootY].rank) {
            subsets[rootY].parent = rootX;
        } else {
            subsets[rootY].parent = rootX;
            subsets[rootX].rank++;
        }
    }
    
    // Kruskal's algorithm to find minimum spanning tree
    public void kruskalMST() {
        // Result array to store the resultant minimum spanning tree
        Edge[] result = new Edge[V - 1];
        
        // Sort all edges in non-decreasing order of their weight
        Arrays.sort(edges);
        
        // Create V subsets with single elements
        Subset[] subsets = new Subset[V];
        for (int i = 0; i < V; i++) {
            subsets[i] = new Subset(i, 0);
        }
        
        int e = 0; // Index used for result[]
        int i = 0; // Index used for sorted edges[]
        
        // Number of edges to be taken is V-1
        while (e < V - 1 && i < E) {
            // Pick the smallest edge
            Edge nextEdge = edges[i++];
            
            int x = find(subsets, nextEdge.src);
            int y = find(subsets, nextEdge.dest);
            
            // If including this edge doesn't cause cycle, include it in result
            if (x != y) {
                result[e++] = nextEdge;
                union(subsets, x, y);
            }
            // Else discard the edge
        }
        
        // Print the constructed MST
        System.out.println("Edges in the constructed minimum spanning tree:");
        int minimumCost = 0;
        for (i = 0; i < e; i++) {
            System.out.println(result[i].src + " -- " + result[i].dest + " == " + result[i].weight);
            minimumCost += result[i].weight;
        }
        System.out.println("Minimum Cost Spanning Tree: " + minimumCost);
    }
    
    public static void main(String[] args) {
        int V = 4; // Number of vertices
        int E = 5; // Number of edges
        KruskalAlgorithm graph = new KruskalAlgorithm(V, E);
        
        // Add edges with their weights
        graph.edges[0] = new Edge(0, 1, 10);
        graph.edges[1] = new Edge(0, 2, 6);
        graph.edges[2] = new Edge(0, 3, 5);
        graph.edges[3] = new Edge(1, 3, 15);
        graph.edges[4] = new Edge(2, 3, 4);
        
        // Find minimum spanning tree
        graph.kruskalMST();
    }
}

// 15. Prim's Algorithm
import java.util.*;

public class PrimAlgorithm {
    private static final int V = 5; // Number of vertices
    
    // Find the vertex with minimum key value, from the set of vertices
    // not yet included in MST
    private static int minKey(int[] key, boolean[] mstSet) {
        int min = Integer.MAX_VALUE, minIndex = -1;
        
        for (int v = 0; v < V; v++) {
            if (!mstSet[v] && key[v] < min) {
                min = key[v];
                minIndex = v;
            }
        }
        
        return minIndex;
    }
    
    // Print the constructed MST stored in parent[]
    private static void printMST(int[] parent, int[][] graph) {
        System.out.println("Edge \tWeight");
        for (int i = 1; i < V; i++) {
            System.out.println(parent[i] + " - " + i + "\t" + graph[i][parent[i]]);
        }
    }
    
    // Function to construct and print MST for a graph represented as adjacency matrix
    private static void primMST(int[][] graph) {
        // Array to store constructed MST
        int[] parent = new int[V];
        
        // Key values used to pick minimum weight edge
        int[] key = new int[V];
        
        // To represent set of vertices included in MST
        boolean[] mstSet = new boolean[V];
        
        // Initialize all keys as INFINITE
        for (int i = 0; i < V; i++) {
            key[i] = Integer.MAX_VALUE;
            mstSet[i] = false;
        }
        
        // Priority queue to pick minimum weight edges
        PriorityQueue<Integer> pq = new PriorityQueue<>(V, Comparator.comparingInt(v -> key[v]));
        
        // Always include first vertex in MST
        // Make key 0 so that this vertex is picked as first vertex
        key[0] = 0;
        parent[0] = -1; // First node is always root of MST
        pq.add(0);
        
        // The MST will have V vertices
        while (!pq.isEmpty()) {
            // Extract minimum key vertex from priority queue
            int u = pq.poll();
            
            // Add the picked vertex to the MST Set
            mstSet[u] = true;
            
            // Update key value and parent index of adjacent vertices
            // Consider only those vertices which are not yet included in MST
            for (int v = 0; v < V; v++) {
                // graph[u][v] is non zero only for adjacent vertices of u
                // mstSet[v] is false for vertices not yet included in MST
                // Update the key only if graph[u][v] is smaller than key[v]
                if (graph[u][v] != 0 && !mstSet[v] && graph[u][v] < key[v]) {
                    parent[v] = u;
                    key[v] = graph[u][v];
                    
                    // If key[v] was already in the priority queue, this will just add a duplicate
                    // which isn't perfect but simpler than checking if it's already there
                    pq.add(v);
                }
            }
        }
        
        // Print the constructed minimum spanning tree
        printMST(parent, graph);
    }
    
    public static void main(String[] args) {
        /* Let us create the following graph
             2    3
        (0)--(1)--(2)
        |    / \   |
        6|  8/   \5 |7
        | /       \ |
        (3)-------(4)
              9         */
        int[][] graph = new int[][] {
            {0, 2, 0, 6, 0},
            {2, 0, 3, 8, 5},
            {0, 3, 0, 0, 7},
            {6, 8, 0, 0, 9},
            {0, 5, 7, 9, 0}
        };
        
        // Print the minimum spanning tree
        primMST(graph);
    }
}


// 16. Floyd Warshall Algorithm (continued)
public class FloydWarshall {
    private static final int INF = 9999;
    
    public static void floydWarshall(int[][] graph, int V) {
        int[][] dist = new int[V][V];
        
        // Initialize the distance matrix as the same as the input graph
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                dist[i][j] = graph[i][j];
            }
        }
        
        // Find all pairs shortest path for all vertices
        for (int k = 0; k < V; k++) {
            // Pick all vertices as source one by one
            for (int i = 0; i < V; i++) {
                // Pick all vertices as destination for the source
                for (int j = 0; j < V; j++) {
                    // If vertex k is on the shortest path from i to j,
                    // then update the value of dist[i][j]
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        
        // Print the shortest distance matrix
        printSolution(dist, V);
    }
    
    public static void printSolution(int[][] dist, int V) {
        System.out.println("All pairs shortest path matrix:");
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == INF) {
                    System.out.print("INF ");
                } else {
                    System.out.print(dist[i][j] + "   ");
                }
            }
            System.out.println();
        }
    }
    
    public static void main(String[] args) {
        int V = 4;
        int[][] graph = {
            {0,   5,  INF, 10},
            {INF, 0,   3, INF},
            {INF, INF, 0,   1},
            {INF, INF, INF, 0}
        };
        
        floydWarshall(graph, V);
    }
}

// 17. Topological Sort
import java.util.*;

public class TopologicalSort {
    private int V; // Number of vertices
    private LinkedList<Integer>[] adj; // Adjacency list
    
    @SuppressWarnings("unchecked")
    public TopologicalSort(int v) {
        V = v;
        adj = new LinkedList[v];
        for (int i = 0; i < v; i++) {
            adj[i] = new LinkedList<>();
        }
    }
    
    // Add an edge to the directed acyclic graph
    public void addEdge(int v, int w) {
        adj[v].add(w);
    }
    
    // Recursive function used by topologicalSort
    private void topologicalSortUtil(int v, boolean[] visited, Stack<Integer> stack) {
        // Mark the current node as visited
        visited[v] = true;
        
        // Recur for all the vertices adjacent to this vertex
        Iterator<Integer> it = adj[v].iterator();
        while (it.hasNext()) {
            int n = it.next();
            if (!visited[n]) {
                topologicalSortUtil(n, visited, stack);
            }
        }
        
        // Push current vertex to stack which stores result
        stack.push(v);
    }
    
    // The function to do Topological Sort
    public void topologicalSort() {
        Stack<Integer> stack = new Stack<>();
        
        // Mark all the vertices as not visited
        boolean[] visited = new boolean[V];
        
        // Call the recursive helper function to store Topological Sort
        // starting from all vertices one by one
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                topologicalSortUtil(i, visited, stack);
            }
        }
        
        // Print contents of stack
        System.out.println("Topological ordering of the directed acyclic graph:");
        while (!stack.empty()) {
            System.out.print(stack.pop() + " ");
        }
    }
    
    public static void main(String[] args) {
        // Create a graph given in the example
        TopologicalSort g = new TopologicalSort(6);
        g.addEdge(5, 2);
        g.addEdge(5, 0);
        g.addEdge(4, 0);
        g.addEdge(4, 1);
        g.addEdge(2, 3);
        g.addEdge(3, 1);
        
        System.out.println("Topological Sort of the directed acyclic graph:");
        g.topologicalSort();
    }
}

// 18. A* Search Algorithm
import java.util.*;

public class AStarSearch {
    // Cell class to store node details
    static class Cell {
        int x, y;
        int f, g, h; // f = g + h
        Cell parent;
        
        public Cell(int x, int y) {
            this.x = x;
            this.y = y;
            this.f = 0;
            this.g = 0;
            this.h = 0;
            this.parent = null;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (obj == this) return true;
            if (!(obj instanceof Cell)) return false;
            Cell other = (Cell) obj;
            return this.x == other.x && this.y == other.y;
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }
    }
    
    // A* Search Algorithm
    public static void aStarSearch(int[][] grid, Cell start, Cell goal) {
        // If the start or goal is out of bounds or blocked
        if (!isValid(grid, start.x, start.y) || !isValid(grid, goal.x, goal.y)) {
            System.out.println("Start or goal is invalid");
            return;
        }
        
        // If the start is the goal
        if (start.equals(goal)) {
            System.out.println("Start is the goal");
            return;
        }
        
        // Create closed list and initialize it to false
        boolean[][] closedList = new boolean[grid.length][grid[0].length];
        
        // Create an open list containing only the starting node
        PriorityQueue<Cell> openList = new PriorityQueue<>(
            Comparator.comparingInt(c -> c.f)
        );
        
        // Add the start node to the open list
        start.f = start.g + calculateHValue(start, goal);
        openList.add(start);
        
        // Loop until the open list is empty
        while (!openList.isEmpty()) {
            // Get the node with the lowest f value
            Cell current = openList.poll();
            
            // Add the current node to the closed list
            closedList[current.x][current.y] = true;
            
            // If we have reached the goal
            if (current.equals(goal)) {
                System.out.println("Path found!");
                printPath(current);
                return;
            }
            
            // Check all adjacent cells
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    // Skip the current cell
                    if (dx == 0 && dy == 0) continue;
                    
                    // Skip diagonal movements
                    if (dx != 0 && dy != 0) continue;
                    
                    int nx = current.x + dx;
                    int ny = current.y + dy;
                    
                    // Check if the adjacent cell is valid
                    if (isValid(grid, nx, ny) && grid[nx][ny] == 0 && !closedList[nx][ny]) {
                        Cell neighbor = new Cell(nx, ny);
                        neighbor.parent = current;
                        
                        // Calculate g, h, and f values
                        neighbor.g = current.g + 1;
                        neighbor.h = calculateHValue(neighbor, goal);
                        neighbor.f = neighbor.g + neighbor.h;
                        
                        // If cell is already in open list with a lower f value, skip it
                        boolean skip = false;
                        for (Cell cell : openList) {
                            if (cell.equals(neighbor) && cell.f <= neighbor.f) {
                                skip = true;
                                break;
                            }
                        }
                        
                        // If cell should not be skipped, add it to open list
                        if (!skip) {
                            openList.add(neighbor);
                        }
                    }
                }
            }
        }
        
        System.out.println("No path found");
    }
    
    // Calculate the heuristic value (Manhattan distance)
    private static int calculateHValue(Cell cell, Cell goal) {
        return Math.abs(cell.x - goal.x) + Math.abs(cell.y - goal.y);
    }
    
    // Check if a given cell is valid
    private static boolean isValid(int[][] grid, int x, int y) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] == 0;
    }
    
    // Print the path from start to goal
    private static void printPath(Cell cell) {
        if (cell.parent != null) {
            printPath(cell.parent);
        }
        System.out.println("(" + cell.x + ", " + cell.y + ")");
    }
    
    public static void main(String[] args) {
        // 0 = passable, 1 = obstacle
        int[][] grid = {
            {0, 0, 0, 0, 0},
            {0, 1, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 1, 0},
            {0, 0, 0, 0, 0}
        };
        
        Cell start = new Cell(0, 0);
        Cell goal = new Cell(4, 4);
        
        System.out.println("A* Search from (" + start.x + ", " + start.y + ") to (" + goal.x + ", " + goal.y + "):");
        aStarSearch(grid, start, goal);
    }
}

// 19. Huffman Coding
import java.util.*;

public class HuffmanCoding {
    // Node class for Huffman Tree
    static class Node implements Comparable<Node> {
        char character;
        int frequency;
        Node left, right;
        
        public Node(char character, int frequency) {
            this.character = character;
            this.frequency = frequency;
            this.left = null;
            this.right = null;
        }
        
        public Node(char character, int frequency, Node left, Node right) {
            this.character = character;
            this.frequency = frequency;
            this.left = left;
            this.right = right;
        }
        
        @Override
        public int compareTo(Node node) {
            return this.frequency - node.frequency;
        }
    }
    
    // Build Huffman Tree and get the codes
    public static void huffmanCoding(String text) {
        // Calculate frequency of each character
        Map<Character, Integer> frequencyMap = new HashMap<>();
        for (char c : text.toCharArray()) {
            frequencyMap.put(c, frequencyMap.getOrDefault(c, 0) + 1);
        }
        
        // Create a priority queue
        PriorityQueue<Node> pq = new PriorityQueue<>();
        
        // Create a leaf node for each character and add it to the priority queue
        for (Map.Entry<Character, Integer> entry : frequencyMap.entrySet()) {
            pq.add(new Node(entry.getKey(), entry.getValue()));
        }
        
        // Build Huffman Tree: combine the two nodes with lowest frequency
        while (pq.size() > 1) {
            Node left = pq.poll();
            Node right = pq.poll();
            
            // Create a new internal node with these two nodes as children
            // and with frequency equal to the sum of frequencies
            pq.add(new Node('\0', left.frequency + right.frequency, left, right));
        }
        
        // Get the root of the Huffman Tree
        Node root = pq.poll();
        
        // Generate Huffman codes for each character
        Map<Character, String> huffmanCodes = new HashMap<>();
        generateCodes(root, "", huffmanCodes);
        
        // Print the Huffman codes
        System.out.println("Huffman Codes:");
        for (Map.Entry<Character, String> entry : huffmanCodes.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // Encode the input text
        StringBuilder encodedText = new StringBuilder();
        for (char c : text.toCharArray()) {
            encodedText.append(huffmanCodes.get(c));
        }
        
        System.out.println("\nEncoded Text: " + encodedText);
        
        // Decode the encoded text
        System.out.println("\nDecoded Text: " + decode(root, encodedText.toString()));
    }
    
    // Generate Huffman codes recursively
    private static void generateCodes(Node root, String code, Map<Character, String> huffmanCodes) {
        if (root == null) return;
        
        // If this is a leaf node, then it contains a character
        if (root.left == null && root.right == null) {
            huffmanCodes.put(root.character, code);
            return;
        }
        
        // Assign 0 to left edge and recur
        generateCodes(root.left, code + "0", huffmanCodes);
        
        // Assign 1 to right edge and recur
        generateCodes(root.right, code + "1", huffmanCodes);
    }
    
    // Decode function to get the original text from Huffman encoded text
    private static String decode(Node root, String encodedText) {
        StringBuilder result = new StringBuilder();
        Node current = root;
        
        for (int i = 0; i < encodedText.length(); i++) {
            if (encodedText.charAt(i) == '0') {
                current = current.left;
            } else {
                current = current.right;
            }
            
            // If this is a leaf node, add the character to result and reset to root
            if (current.left == null && current.right == null) {
                result.append(current.character);
                current = root;
            }
        }
        
        return result.toString();
    }
    
    public static void main(String[] args) {
        String text = "This is a test for Huffman coding with frequency based prefix compression";
        huffmanCoding(text);
    }
}		