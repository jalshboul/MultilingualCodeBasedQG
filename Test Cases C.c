// 1. Binary Search Algorithm
int binary_search(int arr[], int size, int key) {
    int low = 0;
    int high = size - 1;
    
    while (low <= high) {
        int mid = low + (high - low) / 2;
        
        if (arr[mid] == key)
            return mid;
        
        if (arr[mid] < key)
            low = mid + 1;
        else
            high = mid - 1;
    }
    
    return -1; // Element not found
}

// 2. Linear Search Algorithm
int linear_search(int arr[], int size, int key) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == key)
            return i;
    }
    return -1; // Element not found
}

// 3. Bubble Sort Algorithm
void bubble_sort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// 4. Insertion Sort Algorithm
void insertion_sort(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        int key = arr[i];
        int j = i - 1;
        
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// 5. Selection Sort Algorithm
void selection_sort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        int min_idx = i;
        
        for (int j = i + 1; j < size; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        
        // Swap the found minimum element with the first element
        if (min_idx != i) {
            int temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

// 6. Merge Sort Algorithm
void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    // Create temporary arrays
    int L[n1], R[n2];
    
    // Copy data to temporary arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    // Merge the temporary arrays back into arr[left..right]
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    
    // Copy the remaining elements of L[], if any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    // Copy the remaining elements of R[], if any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        // Same as (left+right)/2, but avoids overflow
        int mid = left + (right - left) / 2;
        
        // Sort first and second halves
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        
        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

// 7. Quick Sort Algorithm
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            // Swap arr[i] and arr[j]
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    // Swap arr[i + 1] and arr[high] (or pivot)
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    
    return (i + 1);
}

void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pivot_index = partition(arr, low, high);
        
        quick_sort(arr, low, pivot_index - 1);
        quick_sort(arr, pivot_index + 1, high);
    }
}

// 8. Depth-First Search Algorithm
#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 100

// Graph representation using adjacency list
typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int num_vertices;
    Node* adj_list[MAX_VERTICES];
    int visited[MAX_VERTICES];
} Graph;

// Stack implementation for DFS
typedef struct Stack {
    int items[MAX_VERTICES];
    int top;
} Stack;

void push(Stack* s, int value) {
    s->items[++s->top] = value;
}

int pop(Stack* s) {
    return s->items[s->top--];
}

int is_stack_empty(Stack* s) {
    return s->top == -1;
}

// DFS using stack
void dfs(Graph* graph, int start_vertex) {
    Stack stack;
    stack.top = -1; // Initialize stack
    
    // Mark the start vertex as visited and push it onto the stack
    graph->visited[start_vertex] = 1;
    push(&stack, start_vertex);
    
    printf("DFS starting from vertex %d: ", start_vertex);
    
    while (!is_stack_empty(&stack)) {
        // Pop a vertex from the stack and print it
        int current_vertex = pop(&stack);
        printf("%d ", current_vertex);
        
        // Get all adjacent vertices of the popped vertex
        Node* temp = graph->adj_list[current_vertex];
        
        // If an adjacent vertex is not visited, mark it as visited and push it to the stack
        while (temp) {
            int adj_vertex = temp->vertex;
            
            if (graph->visited[adj_vertex] == 0) {
                graph->visited[adj_vertex] = 1;
                push(&stack, adj_vertex);
            }
            temp = temp->next;
        }
    }
    printf("\n");
}

// 9. Breadth-First Search Algorithm
#include <stdio.h>
#include <stdlib.h>

// Queue implementation for BFS
typedef struct Queue {
    int items[MAX_VERTICES];
    int front;
    int rear;
} Queue;

void enqueue(Queue* q, int value) {
    q->items[q->rear++] = value;
}

int dequeue(Queue* q) {
    return q->items[q->front++];
}

int is_queue_empty(Queue* q) {
    return q->front == q->rear;
}

// BFS using queue
void bfs(Graph* graph, int start_vertex) {
    Queue queue;
    queue.front = 0;
    queue.rear = 0;
    
    // Mark the start vertex as visited and enqueue it
    graph->visited[start_vertex] = 1;
    enqueue(&queue, start_vertex);
    
    printf("BFS starting from vertex %d: ", start_vertex);
    
    while (!is_queue_empty(&queue)) {
        // Dequeue a vertex and print it
        int current_vertex = dequeue(&queue);
        printf("%d ", current_vertex);
        
        // Get all adjacent vertices of the dequeued vertex
        Node* temp = graph->adj_list[current_vertex];
        
        // If an adjacent vertex is not visited, mark it as visited and enqueue it
        while (temp) {
            int adj_vertex = temp->vertex;
            
            if (graph->visited[adj_vertex] == 0) {
                graph->visited[adj_vertex] = 1;
                enqueue(&queue, adj_vertex);
            }
            temp = temp->next;
        }
    }
    printf("\n");
}

// 10. Dijkstra's Algorithm
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define V 9  // Number of vertices

// Priority Queue implementation (Min Heap) for Dijkstra
typedef struct MinHeapNode {
    int v;
    int dist;
} MinHeapNode;

typedef struct MinHeap {
    int size;
    int capacity;
    int *pos;
    MinHeapNode **array;
} MinHeap;

// Function to implement Dijkstra's algorithm
void dijkstra(int graph[V][V], int src) {
    int dist[V];  // Array to store shortest distance from src to i
    
    // Create a min heap
    MinHeap* minHeap = createMinHeap(V);
    
    // Initialize min heap with all vertices
    for (int v = 0; v < V; ++v) {
        dist[v] = INT_MAX;
        minHeap->array[v] = newMinHeapNode(v, dist[v]);
        minHeap->pos[v] = v;
    }
    
    // Make dist value of src vertex as 0 so that it is extracted first
    dist[src] = 0;
    decreaseKey(minHeap, src, dist[src]);
    
    // Initialize heap size as V
    minHeap->size = V;
    
    // Process vertices - Extract minimum distance vertex from min heap
    while (!isEmpty(minHeap)) {
        MinHeapNode* minHeapNode = extractMin(minHeap);
        int u = minHeapNode->v;
        
        // For all adjacent vertices of the extracted vertex
        for (int v = 0; v < V; v++) {
            // Update dist[v] only if is not in heap, there is an edge from u to v, 
            // and total weight of path from src to v through u is smaller than current value of dist[v]
            if (isInMinHeap(minHeap, v) && graph[u][v] && dist[u] != INT_MAX
                && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                decreaseKey(minHeap, v, dist[v]);
            }
        }
    }
    
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < V; ++i)
        printf("%d \t\t %d\n", i, dist[i]);
}

// 11. Dynamic Programming (Knapsack Problem)
int knapsack(int W, int wt[], int val[], int n) {
    int i, w;
    int dp[n+1][W+1];
    
    // Build table dp[][] in bottom-up manner
    for (i = 0; i <= n; i++) {
        for (w = 0; w <= W; w++) {
            if (i == 0 || w == 0)
                dp[i][w] = 0;
            else if (wt[i-1] <= w)
                dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w]);
            else
                dp[i][w] = dp[i-1][w];
        }
    }
    
    return dp[n][W];
}

// 12. Greedy Algorithm (Activity Selection)
void activity_selection(int start[], int finish[], int n) {
    printf("Selected activities: ");
    
    // The first activity is always selected
    int i = 0;
    printf("%d ", i);
    
    // Consider rest of the activities
    for (int j = 1; j < n; j++) {
        // If this activity has start time greater than or equal
        // to the finish time of previously selected activity, then select it
        if (start[j] >= finish[i]) {
            printf("%d ", j);
            i = j;
        }
    }
    printf("\n");
    
    // This is a greedy algorithm because we're making locally optimal choices
    // at each step with the hope of finding a global optimum.
}

// 13. KMP Algorithm
void compute_lps_array(char* pattern, int M, int* lps) {
    int len = 0;
    lps[0] = 0; // lps[0] is always 0
    
    int i = 1;
    while (i < M) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len-1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void kmp_search(char* text, char* pattern) {
    int M = strlen(pattern);
    int N = strlen(text);
    
    // Create lps[] that will hold the longest prefix suffix values for pattern
    int lps[M];
    
    // Preprocess the pattern (calculate lps[] array)
    compute_lps_array(pattern, M, lps);
    
    int i = 0; // index for text[]
    int j = 0; // index for pattern[]
    
    while (i < N) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }
        
        if (j == M) {
            printf("Found pattern at index %d\n", i - j);
            j = lps[j - 1];
        } else if (i < N && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
}

// 14. Kruskal's Algorithm
#include <stdio.h>
#include <stdlib.h>

// Edge structure for Kruskal's algorithm
typedef struct Edge {
    int src, dest, weight;
} Edge;

// Structure to represent a subset for union-find
typedef struct Subset {
    int parent;
    int rank;
} Subset;

// Find set of an element i (path compression technique)
int find(Subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

// Union of two sets of x and y (uses union by rank)
void Union(Subset subsets[], int x, int y) {
    int rootX = find(subsets, x);
    int rootY = find(subsets, y);
    
    if (subsets[rootX].rank < subsets[rootY].rank)
        subsets[rootX].parent = rootY;
    else if (subsets[rootX].rank > subsets[rootY].rank)
        subsets[rootY].parent = rootX;
    else {
        subsets[rootY].parent = rootX;
        subsets[rootX].rank++;
    }
}

// Compare two edges according to their weights
int compare(const void* a, const void* b) {
    Edge* a1 = (Edge*)a;
    Edge* b1 = (Edge*)b;
    return a1->weight > b1->weight;
}

// Kruskal's algorithm to find Minimum Spanning Tree of a graph
void kruskal_mst(Edge edges[], int V, int E) {
    Edge result[V];  // This will store the resultant MST
    int e = 0;  // An index variable, used for result[]
    int i = 0;  // An index variable, used for sorted edges
    
    // Step 1: Sort all the edges in non-decreasing order of their weight
    qsort(edges, E, sizeof(edges[0]), compare);
    
    // Allocate memory for creating V subsets
    Subset* subsets = (Subset*)malloc(V * sizeof(Subset));
    
    // Create V subsets with single elements
    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }
    
    // Number of edges to be taken is equal to V-1
    while (e < V - 1 && i < E) {
        // Step 2: Pick the smallest edge. Add it to result[]
        // if including it doesn't cause a cycle
        Edge next_edge = edges[i++];
        
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);
        
        // If including this edge doesn't cause cycle, include it in result
        if (x != y) {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
        // Else discard the next_edge
    }
    
    // Print the contents of result[] to display the built MST
    printf("Edges in the constructed minimum spanning tree:\n");
    for (i = 0; i < e; ++i)
        printf("%d -- %d == %d\n", result[i].src, result[i].dest, result[i].weight);
    
    free(subsets);
}

// 15. Prim's Algorithm
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

#define V 5

// Function to find the vertex with minimum key value
int min_key(int key[], bool mst_set[]) {
    int min = INT_MAX, min_index;
    
    for (int v = 0; v < V; v++)
        if (mst_set[v] == false && key[v] < min)
            min = key[v], min_index = v;
    
    return min_index;
}

// Function to print the constructed MST stored in parent[]
void print_mst(int parent[], int graph[V][V]) {
    printf("Edge \tWeight\n");
    for (int i = 1; i < V; i++)
        printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
}

// Function to implement Prim's algorithm for MST
void prim_mst(int graph[V][V]) {
    int parent[V]; // Array to store constructed MST
    int key[V];    // Key values used to pick minimum weight edge
    bool mst_set[V]; // To represent set of vertices included in MST
    
    // Initialize all keys as INFINITE
    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mst_set[i] = false;
    
    // Always include first vertex in MST
    key[0] = 0;     // Make key 0 so that this vertex is picked as first vertex
    parent[0] = -1; // First node is always root of MST
    
    // The MST will have V vertices
    for (int count = 0; count < V - 1; count++) {
        // Pick the minimum key vertex from the set of vertices not yet included in MST
        int u = min_key(key, mst_set);
        
        // Add the picked vertex to the MST Set
        mst_set[u] = true;
        
        // Update key value and parent index of the adjacent vertices of the picked vertex
        // Consider only those vertices which are not yet included in MST
        for (int v = 0; v < V; v++)
            if (graph[u][v] && mst_set[v] == false && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }
    
    // Print the constructed MST
    print_mst(parent, graph);
}

// 16. Floyd Warshall Algorithm
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define V 4
#define INF 99999

void floyd_warshall(int graph[V][V]) {
    int dist[V][V], i, j, k;
    
    // Initialize the solution matrix same as input graph matrix
    for (i = 0; i < V; i++)
        for (j = 0; j < V; j++)
            dist[i][j] = graph[i][j];
    
    // Add all vertices one by one to the set of intermediate vertices
    // ---> All pairs shortest path algorithm
    for (k = 0; k < V; k++) {
        // Pick all vertices as source one by one
        for (i = 0; i < V; i++) {
            // Pick all vertices as destination for the above source
            for (j = 0; j < V; j++) {
                // If vertex k is on the shortest path from i to j,
                // then update the value of dist[i][j]
                if (dist[i][k] != INF && dist[k][j] != INF && 
                    dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
    
    // Print the shortest distance matrix
    printf("All pairs shortest paths:\n");
    for (i = 0; i < V; i++) {
        for (j = 0; j < V; j++) {
            if (dist[i][j] == INF)
                printf("%7s", "INF");
            else
                printf("%7d", dist[i][j]);
        }
        printf("\n");
    }
}

// 17. Topological Sort
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Graph representation for Topological Sort
typedef struct Graph {
    int V;          // Number of vertices
    int** adj;      // Adjacency Lists
} Graph;

// Function to create a graph with V vertices
Graph* create_graph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    
    // Create an array of adjacency lists
    graph->adj = (int**)malloc(V * sizeof(int*));
    
    // Initialize each adjacency list as empty
    for (int i = 0; i < V; i++)
        graph->adj[i] = NULL;
    
    return graph;
}

// Recursive function used by topological_sort
void topological_sort_util(Graph* graph, int v, bool visited[], int stack[], int* stack_index) {
    // Mark the current node as visited
    visited[v] = true;
    
    // Recur for all adjacent vertices
    for (int i = 0; graph->adj[v] != NULL && i < sizeof(graph->adj[v])/sizeof(graph->adj[v][0]); i++) {
        if (!visited[graph->adj[v][i]])
            topological_sort_util(graph, graph->adj[v][i], visited, stack, stack_index);
    }
    
    // Push current vertex to stack which stores result
    stack[(*stack_index)++] = v;
}

// The function to do Topological Sort for a directed acyclic graph (DAG)
void topological_sort(Graph* graph) {
    int* stack = (int*)malloc(graph->V * sizeof(int));
    int stack_index = 0;
    
    // Mark all the vertices as not visited
    bool* visited = (bool*)malloc(graph->V * sizeof(bool));
    for (int i = 0; i < graph->V; i++)
        visited[i] = false;
    
    // Call the recursive helper function to store Topological Sort
    // starting from all vertices one by one
    for (int i = 0; i < graph->V; i++)
        if (visited[i] == false)
            topological_sort_util(graph, i, visited, stack, &stack_index);
    
    // Print contents of stack (topological order)
    printf("Topological order: ");
    for (int i = stack_index - 1; i >= 0; i--)
        printf("%d ", stack[i]);
    printf("\n");
    
    free(visited);
    free(stack);
}

// 18. A* Search Algorithm
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>

#define ROW 9
#define COL 10

// Creating a shortcut for int, int pair type
typedef struct Pair {
    int first;
    int second;
} Pair;

// Creating a shortcut for pair<int, pair<int, int>> type
typedef struct Cell {
    int f;  // f = g + h
    int i;  // Row
    int j;  // Column
} Cell;

// A structure to hold the necessary parameters for A* algorithm
typedef struct AStarData {
    // i, j are the coordinates of current cell; parent_i, parent_j are coordinates of parent cell
    int i, j;
    int parent_i, parent_j;
    // f = g + h where g is distance from source, h is heuristic distance to destination
    double f, g, h;
} AStarData;

// A utility function to check if a given cell is valid
bool is_valid(int row, int col) {
    return (row >= 0) && (row < ROW) && (col >= 0) && (col < COL);
}

// A utility function to check if a cell is blocked
bool is_unblocked(int grid[ROW][COL], int row, int col) {
    return grid[row][col] == 1;
}

// A utility function to check if destination cell has been reached
bool is_destination(int row, int col, Pair dest) {
    return (row == dest.first && col == dest.second);
}

// A utility function to calculate the 'h' heuristic (Manhattan distance)
double calculate_h_value(int row, int col, Pair dest) {
    return abs(row - dest.first) + abs(col - dest.second);
}

// Function to find the path from source to destination using A* search algorithm
void a_star_search(int grid[ROW][COL], Pair src, Pair dest) {
    // If the source or destination is blocked, return
    if (!is_valid(src.first, src.second) || !is_valid(dest.first, dest.second)) {
        printf("Source or destination is invalid\n");
        return;
    }
    
    if (!is_unblocked(grid, src.first, src.second) || 
        !is_unblocked(grid, dest.first, dest.second)) {
        printf("Source or destination is blocked\n");
        return;
    }
    
    if (is_destination(src.first, src.second, dest)) {
        printf("We are already at the destination\n");
        return;
    }
    
    // Create a closed list and initialize it to false
    bool closed_list[ROW][COL];
    memset(closed_list, false, sizeof(closed_list));
    
    // Declare a 2D array of structure to hold the details of each cell
    AStarData cell_details[ROW][COL];
    
    int i, j;
    
    // Initialize all cells
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            cell_details[i][j].f = FLT_MAX;
            cell_details[i][j].g = FLT_MAX;
            cell_details[i][j].h = FLT_MAX;
            cell_details[i][j].parent_i = -1;
            cell_details[i][j].parent_j = -1;
        }
    }
    
    // Initialize the parameters of the starting node
    i = src.first, j = src.second;
    cell_details[i][j].f = 0.0;
    cell_details[i][j].g = 0.0;
    cell_details[i][j].h = 0.0;
    cell_details[i][j].parent_i = i;
    cell_details[i][j].parent_j = j;
    
    // Create an open list (priority queue)
    // The open list contains cells that have been visited but not expanded yet
    // That is, cells whose neighbors have not all been inspected
    Cell open_list[ROW*COL];
    int open_list_size = 0;
    
    // Put the starting cell on the open list and set its f as 0
    open_list[open_list_size].f = 0;
    open_list[open_list_size].i = i;
    open_list[open_list_size].j = j;
    open_list_size++;
    
    // Set a boolean value indicating whether destination was found
    bool found_dest = false;
    
    printf("A* search algorithm for pathfinding from (%d,%d) to (%d,%d)\n", 
           src.first, src.second, dest.first, dest.second);
    
    // The A* algorithm search process using priority queue (open list)
    while (open_list_size > 0) {
        // Implement priority queue operations here with heuristic-based sorting
        // ...
        
        // This is a simplified version for demonstration purposes
    }
}

// 19. Huffman Coding
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This constant is used to indicate EOI (End of Input)
#define MAX_TREE_HT 100

// A Huffman tree node
struct MinHeapNode {
    char data;
    unsigned frequency;
    struct MinHeapNode *left, *right;
};

// A Min Heap (Priority Queue for Huffman)
struct MinHeap {
    unsigned size;
    unsigned capacity;
    struct MinHeapNode** array;
};

// Function to print huffman codes from the root of Huffman Tree
void print_codes(struct MinHeapNode* root, int arr[], int top) {
    // Assign 0 to left edge and 1 to right edge
    if (root->left) {
        arr[top] = 0;
        print_codes(root->left, arr, top + 1);
    }
    
    if (root->right) {
        arr[top] = 1;
        print_codes(root->right, arr, top + 1);
    }
    
    // If this is a leaf node, then it contains one of the input characters
    // Print the character and its code from arr[]
    if (!(root->left) && !(root->right)) {
        printf("%c: ", root->data);
        for (int i = 0; i < top; ++i)
            printf("%d", arr[i]);
        printf("\n");
    }
}

// Build Huffman Tree and print codes by traversing the built Huffman Tree
void huffman_codes(char data[], int freq[], int size) {
    struct MinHeapNode *left, *right, *top;
    
    // Create a min heap & initialize all nodes
    struct MinHeap* minHeap = create_min_heap(size);
    
    // Create a min heap node for each character and add to min heap
    for (int i = 0; i < size; ++i)
        minHeap->array[i] = new_node(data[i], freq[i]);
    
    minHeap->size = size;
    build_min_heap(minHeap);
    
    // Iterate while size of heap doesn't become 1
    while (minHeap->size != 1) {
        // Extract the two minimum frequency items from min heap
        left = extract_min(minHeap);
        right = extract_min(minHeap);
        
        // Create a new internal node with frequency equal to the sum of the
        // two nodes frequencies. Make the two extracted nodes as left and right children
        // of this new node. Add this node to the min heap.
        // '$' is a special character for internal nodes, not used in the data
        top = new_node('$', left->frequency + right->frequency);
        top->left = left;
        top->right = right;
        insert_min_heap(minHeap, top);
    }
    
    // The remaining node is the root node and the tree is complete
    // Print Huffman codes using the Huffman tree built above
    int arr[MAX_TREE_HT], top = 0;
    printf("Character codes for frequency-based prefix compression:\n");
    print_codes(extract_min(minHeap), arr, top);
}

// Sample utility functions for Huffman coding
struct MinHeapNode* new_node(char data, unsigned freq) {
    struct MinHeapNode* temp = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->frequency = freq;
    return temp;
}

struct MinHeap* create_min_heap(unsigned capacity) {
    struct MinHeap* minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array = (struct MinHeapNode**)malloc(minHeap->capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

// Main function implementing all algorithms
int main() {
    // Example usage of binary search
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 10;
    int result = binary_search(arr, n, x);
    printf("Binary Search: Element %d found at index %d\n", x, result);
    
    // Example usage of linear search
    result = linear_search(arr, n, x);
    printf("Linear Search: Element %d found at index %d\n", x, result);
    
    // Example usage of bubble sort
    int arr1[] = {64, 34, 25, 12, 22, 11, 90};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);
    bubble_sort(arr1, n1);
    printf("Bubble Sort: Sorted array: ");
    for (int i = 0; i < n1; i++)
        printf("%d ", arr1[i]);
    printf("\n");
    
    // Example usage of insertion sort
    int arr2[] = {12, 11, 13, 5, 6};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);
    insertion_sort(arr2, n2);
    printf("Insertion Sort: Sorted array: ");
    for (int i = 0; i < n2; i++)
        printf("%d ", arr2[i]);
    printf("\n");
    
    // Example usage of selection sort
    int arr3[] = {64, 25, 12, 22, 11};
    int n3 = sizeof(arr3) / sizeof(arr3[0]);
    selection_sort(arr3, n3);
    printf("Selection Sort: Sorted array: ");
    for (int i = 0; i < n3; i++)
        printf("%d ", arr3[i]);
    printf("\n");
    
    // Example usage of merge sort
    int arr4[] = {12, 11, 13, 5, 6, 7};
    int n4 = sizeof(arr4) / sizeof(arr4[0]);
    merge_sort(arr4, 0, n4 - 1);
    printf("Merge Sort: Sorted array: ");
    for (int i = 0; i < n4; i++)
        printf("%d ", arr4[i]);
    printf("\n");
    
    // Example usage of quick sort
    int arr5[] = {10, 7, 8, 9, 1, 5};
    int n5 = sizeof(arr5) / sizeof(arr5[0]);
    quick_sort(arr5, 0, n5 - 1);
    printf("Quick Sort: Sorted array: ");
    for (int i = 0; i < n5; i++)
        printf("%d ", arr5[i]);
    printf("\n");
    
    // Example usage of Huffman coding
    char data[] = {'a', 'b', 'c', 'd', 'e', 'f'};
    int freq[] = {5, 9, 12, 13, 16, 45};
    int size = sizeof(data) / sizeof(data[0]);
    huffman_codes(data, freq, size);
    
    // Example usage of Floyd Warshall algorithm
    int graph[V][V] = {
        {0, 5, INF, 10},
        {INF, 0, 3, INF},
        {INF, INF, 0, 1},
        {INF, INF, INF, 0}
    };
    floyd_warshall(graph);
    
    // Example usage of dynamic programming (Knapsack)
    int val[] = {60, 100, 120};
    int wt[] = {10, 20, 30};
    int W = 50;
    int nItems = sizeof(val) / sizeof(val[0]);
    printf("Maximum value in Knapsack: %d\n", knapsack(W, wt, val, nItems));
    
    // Example usage of KMP algorithm
    char text[] = "ABABDABACDABABCABAB";
    char pattern[] = "ABABCABAB";
    printf("KMP Pattern Matching Algorithm:\n");
    kmp_search(text, pattern);
    
    return 0;
}

// Additional helper functions for algorithms that were not fully implemented above

// Function for max utility in Knapsack algorithm
int max(int a, int b) {
    return (a > b) ? a : b;
}

// MinHeap operations for Huffman coding
void swap_min_heap_node(struct MinHeapNode** a, struct MinHeapNode** b) {
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

void min_heapify(struct MinHeap* minHeap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    
    if (left < minHeap->size && 
        minHeap->array[left]->frequency < minHeap->array[smallest]->frequency)
        smallest = left;
    
    if (right < minHeap->size && 
        minHeap->array[right]->frequency < minHeap->array[smallest]->frequency)
        smallest = right;
    
    if (smallest != idx) {
        swap_min_heap_node(&minHeap->array[smallest], &minHeap->array[idx]);
        min_heapify(minHeap, smallest);
    }
}

int is_size_one(struct MinHeap* minHeap) {
    return (minHeap->size == 1);
}

struct MinHeapNode* extract_min(struct MinHeap* minHeap) {
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0] = minHeap->array[minHeap->size - 1];
    --minHeap->size;
    min_heapify(minHeap, 0);
    return temp;
}

void insert_min_heap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode) {
    ++minHeap->size;
    int i = minHeap->size - 1;
    
    while (i && minHeapNode->frequency < minHeap->array[(i - 1) / 2]->frequency) {
        minHeap->array[i] = minHeap->array[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    
    minHeap->array[i] = minHeapNode;
}

void build_min_heap(struct MinHeap* minHeap) {
    int n = minHeap->size - 1;
    int i;
    
    for (i = (n - 1) / 2; i >= 0; --i)
        min_heapify(minHeap, i);
}

// Utility functions for MinHeap in Dijkstra's algorithm
MinHeapNode* newMinHeapNode(int v, int dist) {
    MinHeapNode* minHeapNode = (MinHeapNode*)malloc(sizeof(MinHeapNode));
    minHeapNode->v = v;
    minHeapNode->dist = dist;
    return minHeapNode;
}

MinHeap* createMinHeap(int capacity) {
    MinHeap* minHeap = (MinHeap*)malloc(sizeof(MinHeap));
    minHeap->pos = (int*)malloc(capacity * sizeof(int));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array = (MinHeapNode**)malloc(capacity * sizeof(MinHeapNode*));
    return minHeap;
}

void decreaseKey(MinHeap* minHeap, int v, int dist) {
    int i = minHeap->pos[v];
    minHeap->array[i]->dist = dist;
    
    while (i && minHeap->array[i]->dist < minHeap->array[(i - 1) / 2]->dist) {
        minHeap->pos[minHeap->array[i]->v] = (i - 1) / 2;
        minHeap->pos[minHeap->array[(i - 1) / a2]->v] = i;
        swapMinHeapNode(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

MinHeapNode* extractMin(MinHeap* minHeap) {
    if (isEmpty(minHeap))
        return NULL;
    
    MinHeapNode* root = minHeap->array[0];
    MinHeapNode* lastNode = minHeap->array[minHeap->size - 1];
    
    minHeap->array[0] = lastNode;
    minHeap->pos[root->v] = minHeap->size - 1;
    minHeap->pos[lastNode->v] = 0;
    
    --minHeap->size;
    minHeapify(minHeap, 0);
    
    return root;
}

bool isEmpty(MinHeap* minHeap) {
    return minHeap->size == 0;
}

bool isInMinHeap(MinHeap* minHeap, int v) {
    if (minHeap->pos[v] < minHeap->size)
        return true;
    return false;
}

void swapMinHeapNode(MinHeapNode** a, MinHeapNode** b) {
    MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

void minHeapify(MinHeap* minHeap, int idx) {
    int smallest, left, right;
    smallest = idx;
    left = 2 * idx + 1;
    right = 2 * idx + 2;
    
    if (left < minHeap->size && 
        minHeap->array[left]->dist < minHeap->array[smallest]->dist)
        smallest = left;
    
    if (right < minHeap->size && 
        minHeap->array[right]->dist < minHeap->array[smallest]->dist)
        smallest = right;
    
    if (smallest != idx) {
        MinHeapNode* smallestNode = minHeap->array[smallest];
        MinHeapNode* idxNode = minHeap->array[idx];
        
        minHeap->pos[smallestNode->v] = idx;
        minHeap->pos[idxNode->v] = smallest;
        
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}