// 1. Binary Search
#include <iostream>
#include <vector>
using namespace std;

int binary_search(vector<int>& arr, int target) {
    int low = 0;
    int high = arr.size() - 1;
    
    while (low <= high) {
        int mid = low + (high - low) / 2;
        
        if (arr[mid] == target)
            return mid;
            
        if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }
    
    return -1;
}

// 2. Linear Search
#include <iostream>
#include <vector>
using namespace std;

int linear_search(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target)
            return i;
    }
    return -1;
}

// 3. Bubble Sort
#include <iostream>
#include <vector>
using namespace std;

void bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// 4. Insertion Sort
#include <iostream>
#include <vector>
using namespace std;

void insertion_sort(vector<int>& arr) {
    int n = arr.size();
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

// 5. Selection Sort
#include <iostream>
#include <vector>
using namespace std;

void selection_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(arr[min_idx], arr[i]);
    }
}

// 6. Merge Sort
#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    vector<int> L(n1), R(n2);
    
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
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
    
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void merge_sort(vector<int>& arr, int left, int right) {
    // Using divide and conquer approach
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        
        merge(arr, left, mid, right);
    }
}

// 7. Quick Sort
#include <iostream>
#include <vector>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quick_sort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// 8. DFS (Depth-First Search)
#include <iostream>
#include <vector>
#include <stack>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;
    
public:
    Graph(int v) : V(v) {
        adj.resize(v);
    }
    
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }
    
    void DFS(int s) {
        vector<bool> visited(V, false);
        stack<int> stack;
        
        stack.push(s);
        
        while (!stack.empty()) {
            s = stack.top();
            stack.pop();
            
            if (!visited[s]) {
                cout << s << " ";
                visited[s] = true;
            }
            
            for (auto i = adj[s].rbegin(); i != adj[s].rend(); ++i) {
                if (!visited[*i])
                    stack.push(*i);
            }
        }
    }
};

// 9. BFS (Breadth-First Search)
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;
    
public:
    Graph(int v) : V(v) {
        adj.resize(v);
    }
    
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }
    
    void BFS(int s) {
        vector<bool> visited(V, false);
        queue<int> queue;
        
        visited[s] = true;
        queue.push(s);
        
        while (!queue.empty()) {
            s = queue.front();
            cout << s << " ";
            queue.pop();
            
            for (auto adjacent : adj[s]) {
                if (!visited[adjacent]) {
                    visited[adjacent] = true;
                    queue.push(adjacent);
                }
            }
        }
    }
};

// 10. Dijkstra's Algorithm
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
using namespace std;

typedef pair<int, int> pii;

class Graph {
    int V;
    vector<vector<pii>> adj;
    
public:
    Graph(int v) : V(v) {
        adj.resize(v);
    }
    
    void addEdge(int u, int v, int w) {
        adj[u].push_back(make_pair(v, w));
        adj[v].push_back(make_pair(u, w));
    }
    
    void dijkstra(int src) {
        priority_queue<pii, vector<pii>, greater<pii>> pq;
        vector<int> distance(V, numeric_limits<int>::max());
        
        pq.push(make_pair(0, src));
        distance[src] = 0;
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            for (auto x : adj[u]) {
                int v = x.first;
                int weight = x.second;
                
                if (distance[v] > distance[u] + weight) {
                    distance[v] = distance[u] + weight;
                    pq.push(make_pair(distance[v], v));
                }
            }
        }
        
        cout << "Vertex Distance from Source:\n";
        for (int i = 0; i < V; i++)
            cout << i << "\t\t" << distance[i] << endl;
    }
};

// 11. Dynamic Programming (0-1 Knapsack)
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int knapsack(vector<int>& weights, vector<int>& values, int W, int n) {
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i - 1] <= w)
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
            else
                dp[i][w] = dp[i - 1][w];
        }
    }
    
    return dp[n][W];
}

// 12. Greedy Algorithm (Activity Selection)
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Activity {
    int start, finish;
};

bool activityCompare(Activity s1, Activity s2) {
    return (s1.finish < s2.finish);
}

void printMaxActivities(vector<Activity>& activities, int n) {
    // Sort activities according to finish time
    sort(activities.begin(), activities.end(), activityCompare);
    
    cout << "Selected activities: ";
    
    // The first activity is always selected
    int i = 0;
    cout << i << " ";
    
    // Greedy choice: Consider activities in sorted order
    for (int j = 1; j < n; j++) {
        // If this activity has start time greater than or equal to the finish
        // time of previously selected activity, then select it
        if (activities[j].start >= activities[i].finish) {
            cout << j << " ";
            i = j;
        }
    }
}

// 13. KMP (Knuth-Morris-Pratt) Algorithm
#include <iostream>
#include <vector>
#include <string>
using namespace std;

void computeLPSArray(string pattern, vector<int>& lps) {
    int m = pattern.length();
    int len = 0;
    
    lps[0] = 0;
    
    int i = 1;
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(string text, string pattern) {
    int n = text.length();
    int m = pattern.length();
    
    vector<int> lps(m);
    
    // Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pattern, lps);
    
    int i = 0;  // index for text[]
    int j = 0;  // index for pattern[]
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }
        
        if (j == m) {
            cout << "Pattern found at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
}

// 14. Kruskal's Algorithm
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Edge {
public:
    int src, dest, weight;
};

class Graph {
public:
    int V, E;
    vector<Edge> edges;
    
    Graph(int v, int e) : V(v), E(e) {
        edges.resize(e);
    }
};

class DisjointSets {
public:
    vector<int> parent, rank;
    
    DisjointSets(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }
    
    int find(int u) {
        if (u != parent[u])
            parent[u] = find(parent[u]);
        return parent[u];
    }
    
    void merge(int x, int y) {
        x = find(x);
        y = find(y);
        
        if (rank[x] > rank[y])
            parent[y] = x;
        else
            parent[x] = y;
        
        if (rank[x] == rank[y])
            rank[y]++;
    }
};

bool compareEdges(Edge a, Edge b) {
    return a.weight < b.weight;
}

void kruskalMST(Graph& graph) {
    int V = graph.V;
    vector<Edge> result(V - 1);
    
    // Sort all edges in non-decreasing order of their weight
    sort(graph.edges.begin(), graph.edges.end(), compareEdges);
    
    // Allocate memory for creating V subsets
    DisjointSets ds(V);
    
    // Create a minimum spanning tree with V-1 edges
    int e = 0;  // An index variable for result[]
    int i = 0;  // An index variable for sorted edges
    
    while (e < V - 1 && i < graph.E) {
        Edge next_edge = graph.edges[i++];
        
        int x = ds.find(next_edge.src);
        int y = ds.find(next_edge.dest);
        
        if (x != y) {
            result[e++] = next_edge;
            ds.merge(x, y);
        }
    }
    
    cout << "Edges in the constructed minimum spanning tree:\n";
    for (i = 0; i < e; i++) {
        cout << result[i].src << " -- " << result[i].dest << " == " << result[i].weight << endl;
    }
}

// 15. Prim's Algorithm
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
using namespace std;

typedef pair<int, int> pii;

class Graph {
    int V;
    vector<vector<pii>> adj;
    
public:
    Graph(int v) : V(v) {
        adj.resize(v);
    }
    
    void addEdge(int u, int v, int w) {
        adj[u].push_back(make_pair(v, w));
        adj[v].push_back(make_pair(u, w));
    }
    
    void primMST() {
        // Create a priority queue to store vertices that are being processed
        priority_queue<pii, vector<pii>, greater<pii>> pq;
        
        int src = 0;  // Starting vertex
        
        // Create a vector for keys and initialize all keys as infinite
        vector<int> key(V, numeric_limits<int>::max());
        
        // Create an array to store constructed MST
        vector<int> parent(V, -1);
        
        // To keep track of vertices included in MST
        vector<bool> inMST(V, false);
        
        // Insert source into priority queue and initialize its key as 0
        pq.push(make_pair(0, src));
        key[src] = 0;
        
        while (!pq.empty()) {
            // Extract minimum key vertex from priority queue
            int u = pq.top().second;
            pq.pop();
            
            if (inMST[u])
                continue;
                
            inMST[u] = true;  // Include vertex in MST
            
            // Traverse all adjacent vertices of u
            for (auto x : adj[u]) {
                // Get vertex label and weight of current adjacent
                int v = x.first;
                int weight = x.second;
                
                // If v is not in MST and weight of (u,v) is smaller than current key of v
                if (!inMST[v] && key[v] > weight) {
                    // Update key of v
                    key[v] = weight;
                    pq.push(make_pair(key[v], v));
                    parent[v] = u;
                }
            }
        }
        
        // Print edges of MST
        cout << "Edges in the constructed minimum spanning tree:\n";
        for (int i = 1; i < V; i++)
            cout << parent[i] << " - " << i << endl;
    }
};

// 16. Floyd-Warshall Algorithm
#include <iostream>
#include <vector>
#include <limits>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> dist;
    
public:
    Graph(int v) : V(v) {
        dist.resize(v, vector<int>(v, numeric_limits<int>::max()));
        for (int i = 0; i < v; i++)
            dist[i][i] = 0;
    }
    
    void addEdge(int u, int v, int w) {
        dist[u][v] = w;
    }
    
    void floydWarshall() {
        // Calculate all pairs shortest path
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != numeric_limits<int>::max() && 
                        dist[k][j] != numeric_limits<int>::max() && 
                        dist[i][j] > dist[i][k] + dist[k][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        
        // Print the shortest distance matrix
        cout << "All pairs shortest path distances:\n";
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == numeric_limits<int>::max())
                    cout << "INF ";
                else
                    cout << dist[i][j] << " ";
            }
            cout << endl;
        }
    }
};

// 17. Topological Sort
#include <iostream>
#include <vector>
#include <stack>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;
    
public:
    Graph(int v) : V(v) {
        adj.resize(v);
    }
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }
    
    void topologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack) {
        visited[v] = true;
        
        for (auto i : adj[v]) {
            if (!visited[i])
                topologicalSortUtil(i, visited, Stack);
        }
        
        Stack.push(v);
    }
    
    void topologicalSort() {
        stack<int> Stack;
        vector<bool> visited(V, false);
        
        // Create a directed acyclic graph for topological order
        for (int i = 0; i < V; i++) {
            if (!visited[i])
                topologicalSortUtil(i, visited, Stack);
        }
        
        cout << "Topological order: ";
        while (!Stack.empty()) {
            cout << Stack.top() << " ";
            Stack.pop();
        }
    }
};

// 18. A* Search Algorithm
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <unordered_set>
using namespace std;

typedef pair<int, int> Pair;
typedef pair<double, pair<int, int>> pPair;

class Cell {
public:
    int parent_i, parent_j;
    double f, g, h;
};

class AStar {
    int ROW, COL;
    vector<vector<int>> grid;
    
public:
    AStar(int row, int col, vector<vector<int>>& map) : ROW(row), COL(col), grid(map) {}
    
    bool isValid(int row, int col) {
        return (row >= 0) && (row < ROW) && (col >= 0) && (col < COL);
    }
    
    bool isUnBlocked(int row, int col) {
        return grid[row][col] == 1;
    }
    
    bool isDestination(int row, int col, Pair dest) {
        return row == dest.first && col == dest.second;
    }
    
    double calculateHValue(int row, int col, Pair dest) {
        // Use Euclidean distance as heuristic
        return sqrt(pow(row - dest.first, 2) + pow(col - dest.second, 2));
    }
    
    void tracePath(vector<vector<Cell>>& cellDetails, Pair dest) {
        int row = dest.first;
        int col = dest.second;
        
        stack<Pair> Path;
        
        while (!(cellDetails[row][col].parent_i == row && cellDetails[row][col].parent_j == col)) {
            Path.push(make_pair(row, col));
            int temp_row = cellDetails[row][col].parent_i;
            int temp_col = cellDetails[row][col].parent_j;
            row = temp_row;
            col = temp_col;
        }
        
        Path.push(make_pair(row, col));
        
        cout << "Path: ";
        while (!Path.empty()) {
            Pair p = Path.top();
            Path.pop();
            cout << "-> (" << p.first << "," << p.second << ") ";
        }
    }
    
    void aStarSearch(Pair src, Pair dest) {
        if (!isValid(src.first, src.second) || !isValid(dest.first, dest.second)) {
            cout << "Source or destination is invalid\n";
            return;
        }
        
        if (!isUnBlocked(src.first, src.second) || !isUnBlocked(dest.first, dest.second)) {
            cout << "Source or destination is blocked\n";
            return;
        }
        
        if (isDestination(src.first, src.second, dest)) {
            cout << "Already at destination\n";
            return;
        }
        
        vector<vector<bool>> closedList(ROW, vector<bool>(COL, false));
        vector<vector<Cell>> cellDetails(ROW, vector<Cell>(COL));
        
        int i, j;
        
        for (i = 0; i < ROW; i++) {
            for (j = 0; j < COL; j++) {
                cellDetails[i][j].f = numeric_limits<double>::max();
                cellDetails[i][j].g = numeric_limits<double>::max();
                cellDetails[i][j].h = numeric_limits<double>::max();
                cellDetails[i][j].parent_i = -1;
                cellDetails[i][j].parent_j = -1;
            }
        }
        
        i = src.first, j = src.second;
        cellDetails[i][j].f = 0.0;
        cellDetails[i][j].g = 0.0;
        cellDetails[i][j].h = 0.0;
        cellDetails[i][j].parent_i = i;
        cellDetails[i][j].parent_j = j;
        
        // Open list with heuristic values
        priority_queue<pPair, vector<pPair>, greater<pPair>> openList;
        openList.push(make_pair(0.0, make_pair(i, j)));
        
        bool foundDest = false;
        
        while (!openList.empty()) {
            pPair p = openList.top();
            openList.pop();
            
            i = p.second.first;
            j = p.second.second;
            closedList[i][j] = true;
            
            // Generate all 4 neighbors
            for (int add_i = -1; add_i <= 1; add_i++) {
                for (int add_j = -1; add_j <= 1; add_j++) {
                    if ((add_i == 0 && add_j == 0) || (add_i != 0 && add_j != 0))
                        continue;
                        
                    int new_i = i + add_i;
                    int new_j = j + add_j;
                    
                    if (isValid(new_i, new_j)) {
                        if (isDestination(new_i, new_j, dest)) {
                            cellDetails[new_i][new_j].parent_i = i;
                            cellDetails[new_i][new_j].parent_j = j;
                            cout << "Destination found\n";
                            tracePath(cellDetails, dest);
                            foundDest = true;
                            return;
                        } else if (!closedList[new_i][new_j] && isUnBlocked(new_i, new_j)) {
                            double gNew = cellDetails[i][j].g + 1.0;
                            double hNew = calculateHValue(new_i, new_j, dest);
                            double fNew = gNew + hNew;
                            
                            if (cellDetails[new_i][new_j].f == numeric_limits<double>::max() || 
                                cellDetails[new_i][new_j].f > fNew) {
                                
                                openList.push(make_pair(fNew, make_pair(new_i, new_j)));
                                
                                cellDetails[new_i][new_j].f = fNew;
                                cellDetails[new_i][new_j].g = gNew;
                                cellDetails[new_i][new_j].h = hNew;
                                cellDetails[new_i][new_j].parent_i = i;
                                cellDetails[new_i][new_j].parent_j = j;
                            }
                        }
                    }
                }
            }
        }
        
        if (!foundDest)
            cout << "Failed to find the destination\n";
    }
};

// 19. Huffman Coding
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
using namespace std;

struct Node {
    char data;
    unsigned frequency;
    Node *left, *right;
    
    Node(char data, unsigned frequency) : data(data), frequency(frequency), left(nullptr), right(nullptr) {}
};

struct compare {
    bool operator()(Node* left, Node* right) {
        return left->frequency > right->frequency;
    }
};

void printCodes(struct Node* root, string str, unordered_map<char, string>& huffmanCode) {
    if (!root)
        return;
        
    if (root->data != '$')
        huffmanCode[root->data] = str;
        
    printCodes(root->left, str + "0", huffmanCode);
    printCodes(root->right, str + "1", huffmanCode);
}

void huffmanCoding(string text) {
    unordered_map<char, unsigned> freq;
    for (char ch : text)
        freq[ch]++;
        
    priority_queue<Node*, vector<Node*>, compare> minHeap;
    
    for (auto pair : freq)
        minHeap.push(new Node(pair.first, pair.second));
        
    while (minHeap.size() > 1) {
        Node* left = minHeap.top();
        minHeap.pop();
        
        Node* right = minHeap.top();
        minHeap.pop();
        
        Node* top = new Node('$', left->frequency + right->frequency);
        top->left = left;
        top->right = right;
        
        minHeap.push(top);
    }
    
    Node* root = minHeap.top();
    
    unordered_map<char, string> huffmanCode;
    printCodes(root, "", huffmanCode);
    
    cout << "Huffman Codes for frequency-based prefix compression:\n";
    for (auto pair : huffmanCode)
        cout << pair.first << " " << pair.second << endl;
        
    string encodedString = "";
    for (char ch : text)
        encodedString += huffmanCode[ch];
        
    cout << "Encoded string: " << encodedString << endl;
}