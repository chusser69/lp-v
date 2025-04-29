
#include<iostream>
#include<omp.h>
#define _BSD_SOURCE_
#include <sys/time.h>
using namespace std;

void bubble(int arr[],int n)
{
    cout<<"Sequential Bubble Sort"<<endl;
    for(int i=0;i<n-1;i++)
    {
        for(int j=0;j<n-i-1;j++)
        {
            if(arr[j]>arr[j+1])
            {
                int temp=arr[j];
                arr[j]=arr[j+1];
                arr[j+1]=temp;
            }
        }
    }
}

void bubbleParallel(int arr[],int n)
{   
    cout<<"Parallel Bubble Sort"<<endl;;
    for(int i=0;i<n-1;i++)
    {
        #pragma omp parallel 
        for(int j=0;j<n-i-1;j++)
        {
            #pragma omp critical
            {
                if(arr[j]>arr[j+1])
                {
                    int temp=arr[j];
                    arr[j]=arr[j+1];
                    arr[j+1]=temp;
                }
            }
            
        }
    }
}

void merge(int arr[],int left,int mid,int right)
{
    int p1=left;
    int p2=mid+1;
    
    int narr[right-left+1];
    int i=0;
    
    
    while(p1<=mid && p2<=right)
    {
        if(arr[p1]<=arr[p2])
        {
            narr[i]=arr[p1];
            i++;
            p1++;
        }
        
        else
        {
            narr[i]=arr[p2];
            i++;
            p2++;
        }
    }
    while(p1<=mid )
    {
            narr[i]=arr[p1];
            i++;
            p1++;
    }
    
    while(p2<=right )
    {
            narr[i]=arr[p2];
            i++;
            p2++;
    }
    
    
    for(int k=0,x=left;k<sizeof(narr)/sizeof(narr[0]);k++,x++)
    {
        arr[x]=narr[k];
    }
}
void mergeSeq(int arr[],int left,int right)
{
   
    if(left<right)
    {
        int mid=left+(right-left)/2;
        mergeSeq(arr,left,mid);
        mergeSeq(arr,mid+1,right);
        merge(arr,left,mid,right);
        
    }
}

void mergePar(int arr[],int left,int right)
{
    
    if(left<right)
    {
        int mid=left+(right-left)/2;
        #pragma omp parallel
        { mergeSeq(arr,left,mid);}
        #pragma omp parallel
        { mergeSeq(arr,mid+1,right);}
        #pragma omp critical
        { merge(arr,left,mid,right);}
        
        
    }
}


int main()
{
    int n,ch;
    cout<<"Enter the size of Array"<<endl;
    cin>>n;
    int arr[n];
    
    cout<<"Enter the elements"<<endl;
    for(int i=0;i<n;i++)
    {
        cin>>arr[i];
    }
    
    cout<<"Enter: 1- Sequential Bubble sort || 2- Parallel Bubble sort || 3- Sequential Merge sort || 4-Parallel Merge sort "<<endl;
    cin>>ch;
    
    
    struct timeval start,end;
    if(ch==1)
    {
        gettimeofday(&start,NULL);
        bubble(arr,n);
    }
    else if(ch==2)
    {
        gettimeofday(&start,NULL);
        bubbleParallel(arr,n);
    }
    else if(ch==3)
    {
        cout<<"Sequential Merge Sort"<<endl;
        gettimeofday(&start,NULL);
        mergeSeq(arr,0,n-1);
    }
    else if(ch==4)
    {
        cout<<"Parallel Merge Sort"<<endl;
        gettimeofday(&start,NULL);
        mergePar(arr,0,n-1);
    }
    else
    {
        cout<<"Invalid Choice :"<<endl;
        return 0;
    }
    cout<<"Sorted array :"<<endl;
    for(int i=0;i<n;i++)
    {
        cout<<arr[i]<<" ";
    }
    cout<<endl;
    
    gettimeofday(&end,NULL);
    cout<<"TIME "<<(end.tv_sec-start.tv_sec)*1e9+(end.tv_usec-start.tv_usec)*1000<<" sec";
    
}

























Of course!  
Let’s go through your **C++ code line by line** in **clear English (UK)**.

---

# 📜 Line-by-Line Explanation

---

```cpp
#include<iostream>
```
- Includes the **input-output stream** library.
- Needed for using `cout` and `cin`.

---

```cpp
#include<omp.h>
```
- Includes the **OpenMP** header file.
- Needed for using **OpenMP directives** (parallelism).

---

```cpp
#define _BSD_SOURCE_
#include <sys/time.h>
```
- `_BSD_SOURCE_` enables some system features (mainly for older Linux versions).
- `sys/time.h` provides **gettimeofday()** function to **measure time** precisely.

---

```cpp
using namespace std;
```
- So you don’t need to write `std::` before `cout`, `cin`, etc.

---

## 🧹 Function: Sequential Bubble Sort
```cpp
void bubble(int arr[],int n)
{
    cout<<"Sequential Bubble Sort"<<endl;
```
- Defines the `bubble()` function.
- Prints that **sequential** bubble sort is starting.

---
```cpp
for(int i=0;i<n-1;i++)
{
    for(int j=0;j<n-i-1;j++)
    {
        if(arr[j]>arr[j+1])
        {
            int temp=arr[j];
            arr[j]=arr[j+1];
            arr[j+1]=temp;
        }
    }
}
```
- Outer loop (`i`) controls **passes**.
- Inner loop (`j`) compares adjacent elements.
- If elements are **out of order**, swap them.
- Largest element **bubbles up** to the end after each pass.

---

## 🧹 Function: Parallel Bubble Sort
```cpp
void bubbleParallel(int arr[],int n)
{   
    cout<<"Parallel Bubble Sort"<<endl;
```
- Defines `bubbleParallel()` function.
- Prints that **parallel** bubble sort is starting.

---
```cpp
for(int i=0;i<n-1;i++)
{
    #pragma omp parallel 
    for(int j=0;j<n-i-1;j++)
    {
        #pragma omp critical
        {
            if(arr[j]>arr[j+1])
            {
                int temp=arr[j];
                arr[j]=arr[j+1];
                arr[j+1]=temp;
            }
        }
        
    }
}
```
- Outer loop (`i`) remains sequential.
- Inner loop (`j`) is **parallelised** using OpenMP.
- Inside the critical section:
  - Compare and swap elements if necessary.
  - **`critical`** ensures only one thread swaps at a time.

---

## 🧹 Function: Merge Two Sorted Parts
```cpp
void merge(int arr[],int left,int mid,int right)
{
```
- Defines `merge()` function to **merge two sorted parts** of an array.

---
```cpp
int p1=left;
int p2=mid+1;
int narr[right-left+1];
int i=0;
```
- Pointers `p1` and `p2` track left and right parts.
- `narr` is a **temporary array** for merged elements.

---
```cpp
while(p1<=mid && p2<=right)
{
    if(arr[p1]<=arr[p2])
    {
        narr[i]=arr[p1];
        i++;
        p1++;
    }
    else
    {
        narr[i]=arr[p2];
        i++;
        p2++;
    }
}
```
- Compare elements from both halves.
- Insert the smaller element into `narr`.

---
```cpp
while(p1<=mid)
{
    narr[i]=arr[p1];
    i++;
    p1++;
}
while(p2<=right)
{
    narr[i]=arr[p2];
    i++;
    p2++;
}
```
- If one half is left, copy all remaining elements.

---
```cpp
for(int k=0,x=left;k<sizeof(narr)/sizeof(narr[0]);k++,x++)
{
    arr[x]=narr[k];
}
```
- Copy merged elements back into the **original array**.

---

## 🧹 Function: Sequential Merge Sort
```cpp
void mergeSeq(int arr[],int left,int right)
{
```
- Defines `mergeSeq()` for **sequential** merge sort.

---
```cpp
if(left<right)
{
    int mid=left+(right-left)/2;
    mergeSeq(arr,left,mid);
    mergeSeq(arr,mid+1,right);
    merge(arr,left,mid,right);
}
```
- If more than one element:
  - Find `mid`.
  - Recursively sort left half and right half.
  - Merge sorted halves.

---

## 🧹 Function: Parallel Merge Sort
```cpp
void mergePar(int arr[],int left,int right)
{
```
- Defines `mergePar()` for **parallel** merge sort.

---
```cpp
if(left<right)
{
    int mid=left+(right-left)/2;
    #pragma omp parallel
    { mergeSeq(arr,left,mid);}
    #pragma omp parallel
    { mergeSeq(arr,mid+1,right);}
    #pragma omp critical
    { merge(arr,left,mid,right);}
}
```
- Recursively call `mergeSeq` for both halves **in parallel**.
- Merge them inside a **critical section**.

---
⚡ (Note: Ideal way would be to use **`task`** instead of separate parallel blocks.)

---

# 🖥️ Main Program

```cpp
int main()
{
```
- **Main program starts**.

---
```cpp
int n,ch;
cout<<"Enter the size of Array"<<endl;
cin>>n;
int arr[n];
```
- Ask user for array size `n`.
- Declare array `arr` of size `n`.

---
```cpp
cout<<"Enter the elements"<<endl;
for(int i=0;i<n;i++)
{
    cin>>arr[i];
}
```
- Input array elements from the user.

---
```cpp
cout<<"Enter: 1- Sequential Bubble sort || 2- Parallel Bubble sort || 3- Sequential Merge sort || 4-Parallel Merge sort "<<endl;
cin>>ch;
```
- Ask the user to **choose sorting method**.

---
```cpp
struct timeval start,end;
```
- Declare structures to store **start and end time**.

---

### ⏳ Choice Handling and Sorting
```cpp
if(ch==1)
{
    gettimeofday(&start,NULL);
    bubble(arr,n);
}
```
- If choice = 1:
  - Record start time.
  - Call sequential bubble sort.

---
```cpp
else if(ch==2)
{
    gettimeofday(&start,NULL);
    bubbleParallel(arr,n);
}
```
- If choice = 2:
  - Record start time.
  - Call parallel bubble sort.

---
```cpp
else if(ch==3)
{
    cout<<"Sequential Merge Sort"<<endl;
    gettimeofday(&start,NULL);
    mergeSeq(arr,0,n-1);
}
```
- If choice = 3:
  - Print heading.
  - Call sequential merge sort.

---
```cpp
else if(ch==4)
{
    cout<<"Parallel Merge Sort"<<endl;
    gettimeofday(&start,NULL);
    mergePar(arr,0,n-1);
}
```
- If choice = 4:
  - Print heading.
  - Call parallel merge sort.

---
```cpp
else
{
    cout<<"Invalid Choice :"<<endl;
    return 0;
}
```
- If invalid choice, exit program.

---

### 📋 Print Sorted Array
```cpp
cout<<"Sorted array :"<<endl;
for(int i=0;i<n;i++)
{
    cout<<arr[i]<<" ";
}
cout<<endl;
```
- Print the **sorted array**.

---

### 🕒 Print Time Taken
```cpp
gettimeofday(&end,NULL);
cout<<"TIME "<<(end.tv_sec-start.tv_sec)*1e9+(end.tv_usec-start.tv_usec)*1000<<" sec";
```
- Record end time.
- Calculate time difference.
- Print **execution time**.

---
# 🔥 In Short
- Code sorts an array using different methods (sequential/parallel) and shows the time taken.

---

Would you also like me to show you a **small diagram** (flowchart style) of how your code flows during execution?  
It'll take just a minute to draw! 🚀  
(Just say "Yes, show diagram!")