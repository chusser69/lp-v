#include<iostream>
#include<stack>
#include<queue>
using namespace std;

class Treenode{
    public:
    Treenode *left;
    Treenode *right;
    int data;
    
    Treenode(int n)
    {
        data=n;
        left=right=NULL;
    }
};

class Tree
{
    public:
    
    Treenode *root;
    Tree()
    {
        root=NULL;
    };
    
    void insert(int data)
    {
        Treenode *n=new Treenode(data);
        if(root==NULL)
        {
            root=n;
        }
        else
        {
            queue<Treenode *>q;
            q.push(root);
            
            int flag=0;
            
            while(!q.empty() && !flag)
            {
                for(int i=0;i<q.size();i++)
                {
                    Treenode *temp=q.front();
                    q.pop();
                    
                    if(temp->left==NULL)
                    {
                        temp->left=n;
                        flag=1;
                    }
                    else
                    {
                        q.push(temp->left);
                        if(temp->right==NULL)
                        {
                            temp->right=n;
                            flag=1;
                        }
                        else
                        {
                            q.push(temp->right);
                        }
                        
                    }
                }
            }
        }
    }
    
    void bfs()
    {
        queue<Treenode *>q;
        q.push(root);
        
        while(!q.empty())
        {
            int n=q.size();
            
            #pragma omp parallel for
            for(int i=0;i<n;i++)
            {
                Treenode *temp;
                #pragma omp critical
                {
                    temp=q.front();
                    q.pop();
                    
                    cout<<temp->data<<" | ";
                }
               
                if(temp->left!=NULL)
                {
                    q.push(temp->left);
                }
                if(temp->right!=NULL)
                {
                    q.push(temp->right);
                }
            }
        }
        cout<<endl;
        
        
    }
    
    void dfs()
    {
        Treenode *temp=root;
        stack<Treenode *>s;
        s.push(root);
        
        
        while(!s.empty())
        {
            int n=s.size();
            #pragma omp parallel for
            for(int i=0;i<n;i++)
            {
                #pragma omp critical
                {
                    while(temp!=NULL)
                    {
                        temp=temp->left;
                        if(temp!=NULL)
                        {
                            s.push(temp);
                        }
                    }
                        temp=s.top();
                        s.pop();
                        
                        cout<<temp->data<<" | ";
                        
                        if(temp->right!=NULL)
                        {
                             s.push(temp->right);
                        }
                        temp=temp->right;
                    
                } 
                
            }
        }
        cout<<endl;
    }
};

int main()
{
    Tree *t=new Tree();
    
    while(true)
    {
        int ch,n;
        cout<<"Enter 1- Insert || 2- BFS || 3- DFS || 4- Exit"<<endl;
        cin>>ch;
        
        if(ch==1)
        {
            cout<<"Enter NO"<<endl;
            cin>>n;
            t->insert(n);
        }
        else if(ch==2)
        {
            t->bfs();
        }
        else if(ch==3)
        {
            t->dfs();
        }
        else if(ch==4)
        {
            break;
        }
        else
        {
            return 0;
        }
    
    }
    
}














##################explainantion#####
Of course!  
Here’s a **simple, clear line-by-line explanation** of your C++ code (written in **British English**, as you asked):

---

### Header files
```cpp
#include<iostream>
#include<stack>
#include<queue>
using namespace std;
```
- Import **input/output**, **stack**, and **queue** libraries.
- Use the **standard namespace** so you don't have to write `std::` each time.

---

### Class: Treenode
```cpp
class Treenode {
public:
    Treenode *left;
    Treenode *right;
    int data;

    Treenode(int n) {
        data = n;
        left = right = NULL;
    }
};
```
- This defines a **node** of the binary tree.
- Each node holds:
  - An **integer value** (`data`).
  - A **left child** and a **right child**.
- Constructor: sets the node's data and initialises both children to `NULL`.

---

### Class: Tree
```cpp
class Tree {
public:
    Treenode *root;
    Tree() {
        root = NULL;
    };
```
- This defines the **Tree** class.
- It contains a **root node** (starting point).
- Constructor: sets the `root` to `NULL` (empty tree).

---

### Insert function
```cpp
void insert(int data)
```
- Insert a new node into the tree **in level order** (left to right).

Inside `insert`:
- Create a new `Treenode`.
- If tree is empty, set it as the root.
- Otherwise:
  - Use a **queue** to traverse level-by-level.
  - Find the first spot where a left or right child is empty, and insert there.

---

### BFS (Breadth-First Search)
```cpp
void bfs()
```
- Traverses the tree **level-by-level** (left to right).

Inside `bfs`:
- Use a queue to store nodes level-wise.
- For each node:
  - Print its data.
  - Push its left and right children (if they exist) into the queue.
- **OpenMP (`#pragma omp`)** is used to try to run parts of the loop in **parallel**.

---

### DFS (Depth-First Search)
```cpp
void dfs()
```
- Traverses the tree **depth-first** (explores as far down the left before the right).

Inside `dfs`:
- Use a **stack** to explore the tree.
- Push the root.
- For each node:
  - Go as left as possible, pushing nodes onto the stack.
  - Pop from stack, print data, and push right child if any.
- Again uses **OpenMP** for some parallelism, but the critical section makes it serial for safety.

---

### Main Function
```cpp
int main()
```
- Starts the program.
- Create a `Tree` object dynamically.

Inside the infinite loop:
- Ask the user to **choose an option**:
  - `1` → Insert a number into the tree.
  - `2` → Perform BFS and display the tree.
  - `3` → Perform DFS and display the tree.
  - `4` → Exit the program.
  - Anything else → End the program.

---

### Notes:
- `#pragma omp` commands are used to try to **speed up** BFS and DFS by parallelising loops. (OpenMP is a parallel programming API.)
- The **critical sections** prevent race conditions, but they slow down the benefits of parallelism.
- The `insert` method creates a **complete binary tree** as much as possible (fills left to right at each level).

---

### In super simple words:

| Part | Meaning |
|:----|:-----|
| `Treenode` | A small block that holds a number and links to two other blocks. |
| `Tree` | A collection of blocks starting from the first one called "root". |
| `insert` | Adds a new number into the tree, level by level. |
| `bfs` | Visits all numbers level-by-level from top to bottom. |
| `dfs` | Visits numbers deeply: go left as much as possible, then back up and go right. |
| `main` | A menu for the user to insert numbers or view the tree.

