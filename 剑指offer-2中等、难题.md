剑指 Offer 35. 复杂链表的复制
实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

class Solution {
public:
    Node* copyRandomList(Node* head) {
        //方法一：哈希表
        // if(head==nullptr) return nullptr;
        // unordered_map<Node*,Node*> map;
        // Node *cur=head;
        // //1.构建哈希表,建立新旧节点的键值对关系
        // while(cur!=nullptr){
        //     map[cur]=new Node(cur->val);
        //     cur=cur->next;
        // }
        // cur=head;
        // //2.建立random引用指向
        // while(cur!=nullptr){
        //     map[cur]->next=map[cur->next];
        //     map[cur]->random=map[cur->random];
        //     cur=cur->next;
        // }
        // return map[head];

        //方法二：拼接+拆分.1->2->3_null------->>  1->1->2->2->3->3->null
        if(head==nullptr) return nullptr;
        //1.复制
        Node* cur=head;
        while(cur!=nullptr){
            Node* tmp=new Node(cur->val);
            tmp->next=cur->next;
            cur->next=tmp;
            cur=tmp->next;
        }
        //2.构建random关系
        cur=head;
        while(cur!=nullptr){
            if(cur->random!=nullptr){
                cur->next->random=cur->random->next;
            }
            cur=cur->next->next;
        }
        //拆分
        Node* pre=head;
        Node* res=head->next;
        cur=res;
        while(cur->next!=nullptr){
            pre->next=pre->next->next;
            cur->next=cur->next->next;
            pre=pre->next;
            cur=cur->next;
        }
        pre->next=nullptr;
        return res;
    }
};

剑指 Offer 36. 二叉搜索树与双向链表
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。




class Solution {
private:
    Node* head,*tail;
public:
    Node* treeToDoublyList(Node* root) {
        if(!root) return nullptr;
        inorder(root);  //构造链表的所有结构，除了头和尾
        head->left=tail;  //补上头尾的连接   头连尾
        tail->right=head;  //尾连头
        return head;

    }
    void inorder(Node *root){
        if(!root) return;
        inorder(root->left);
        if(!tail){
            head=root;   //tail还不存在时，root此时在二叉树最左边的节点，将此节点设为头
        }
        else{
            tail->right=root;   //前一个节点的right是当前节点
            root->left=tail;    //当前节点的left是前一个节点
        }
        tail=root;   //前一个节点更新为当前节点，当更新tail=root时，root再回溯到上一级节点
        inorder(root->right);
    }
};

剑指 Offer 14- I. 剪绳子
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？
class Solution {
public:
    int cuttingRope(int n) {
        //贪心法，分解出更多的3,3的个数为a，得到的余数b可能为0,1,2
        //1.任何大于1的数都可以由2或3组成
        //2.因为2*2*2<3*3,3越多积越大
        if(n<4) return n-1;
        int a=n/3,b=n%3;
        if(b==0) return a=pow(3,a);
        if(b==1) return a=pow(3,a-1)*4;
        if(b==2) return a=pow(3,a)*2;
        return a;
    }
};

剑指 Offer 45. 把数组排成最小的数
输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
class Solution {
private:
    static bool compare(const string &a,const string &b){
        return a+b<b+a;
    }
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        string res;
        for(auto num:nums){
            strs.push_back(to_string(num));
        }
        sort(strs.begin(),strs.end(),compare);
        for(auto str:strs){
            res+=str;
        }
        return res;

    }
};

剑指 Offer 38. 字符串的排列
输入一个字符串，打印出该字符串中字符的所有排列
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
class Solution {
public:
    vector<string> permutation(string s) {
        //回溯函数
        if(s.empty()) return {};

        sort(s.begin(),s.end());//使用sort对字符排序，使重复的字符相邻
        vector<string> res;  //返回的排列结果
        string path;  //当前的路径
        vector<bool> visited(s.size(),false);//使用visited标记节点在当前决策树中是否遍历过
        dfs(res,s,path,visited);
        return res;
    }
    void dfs(vector<string>& res,string& s,string &path,vector<bool>&visited){
        //若当前决策树已经遍历到叶子节点
        if(path.size()==s.size()){
            res.push_back(path);
            return;
        }
        for(int i=0;i<s.size();i++){
            //排除被访问过的，和重复的
            if(visited[i]) continue;
            if(i>0&&s[i-1]==s[i]&&(!visited[i-1])){
                continue;
            }
            //做选择
            visited[i]=true;  //标记为已访问
            path.push_back(s[i]);  //加入排列字符中
            dfs(res,s,path,visited);  //继续往后做决策，依次递归遍历
            path.pop_back();  //回溯，撤销，要换一条路径了
            visited[i]=false;  
        }
    }
};

141. 环形链表
判断链表里是否有环
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        // //1.快慢指针，一个跑快点，一个跑慢点，如果有环，快的肯定能追上慢的，如果快的跑到后面为空了，就说明没有环
        // if(head==nullptr||head->next==nullptr) return false;
        // ListNode *fast=head,*slow=head;
        // while(fast!=nullptr){
        //     fast=fast->next;
        //     slow=slow->next;
        //     if(!fast) return false;
        //     fast=fast->next;
        //     if(fast==slow) return true;
        // }
        // return false;

        //2.hash，每次把遍历到的节点放入set中，如果节点在set已经存在了，说明有环
        if(head==nullptr) return false;
        unordered_set<ListNode*>set;
        while(head){
            if(set.count(head)){
                return true;
            }
            set.insert(head);
            head=head->next;
        }
        return false;
    }
};

剑指 Offer 46. 把数字翻译成字符串
0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
public:
    int translateNum(int num) {
        /*
        解题思路：动态规划，转移方程:
        10<=s[i-2,i]<=25时，f(i)=f(i-2)+f(i-1);
        0<=s[i-2,i]<10,或者在（25，99）时,f(i)=f(i-1)
        1.将整型转换为字符串
        2.状态定义，dp[0]=dp[1]=1
        3.遍历字符串
        4.返回dp[n]
        */
        string s=to_string(num);
        vector<int> dp(s.size()+1,1);
        for(int i=2;i<=s.size();i++){
            if(10*(s[i-2]-'0')+(s[i-1]-'0')>=10&&10*(s[i-2]-'0')+(s[i-1]-'0')<=25){
                dp[i]=dp[i-2]+dp[i-1];
            }
            else{
                dp[i]=dp[i-1];
            }
            // auto tmp=s.substr(i-2,i);
            // int c=tmp>="10"&&tmp<="25"?a+b:a;
        }
        return dp[s.size()];
    }
};

剑指 Offer 13. 机器人的运动范围
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
输入：m = 2, n = 3, k = 1
输出：3
方法一：深度优先遍历，dfs，先朝一个方向搜到底，再回溯到上一个节点，沿另一个方向搜索，搜索过程中，遇到数位和超出目标值，此元素已访问或者越界，则立即返回
递归参数：元素在矩阵中的行列索引，以及两者的数位和si，sj
终止条件：行列越界；数位和超出目标值k；当前元素已访问，返回0
递推工作：1.将索引i,j放入visited中，代表访问过；   2.搜索下一个单元格，计算当前元素的下、右两个方向元素的数位和，并开启下层递归
回溯返回值：1+右方搜索的可达解数+下方搜索的可达解数
方法二：广度优先遍历，bfs，按照平推的方式向前搜索，利用队列实现
初始化：初始点（0，0）加入队列queue
迭代终止条件：队列已空，已遍历完所有可达解
迭代工作：1单元格出队，将索引，数位和都弹出，作为当前搜索单元和；2.判断是否跳过（越界，超过数位和，访问过）；  3.标记当前单元格访问过；  4.下、右方单元格入队
返回值：res的值
class Solution {
public:
    int movingCount(int m, int n, int k) {
        /*回溯，dfs*/
//         vector<vector<bool>> visited(m,vector<bool>(n,0));
//         return dfs(0,0,0,0,visited,m,n,k);
//     }
// private:
//     int dfs(int i,int j,int si,int sj,vector<vector<bool>>& visited,int m,int n,int k){
//         if(i>=m||j>=n||k<si+sj||visited[i][j]) return 0;
//         visited[i][j]=true;
//         return 1+dfs(i+1,j,(i+1)%10!=0?si+1:si-8,sj,visited,m,n,k)+dfs(i,j+1,si,(j+1)%10!=0?sj+1:sj-8,visited,m,n,k);

        //BFS 广度优先遍历
        vector<vector<bool>> visited(m, vector<bool>(n, 0));//访问数组
        queue<vector<int>>que; //队列，存放的数组值是方格位置的下标i,j以及下标的数位和si,sj
        int res=0;  //返回值，初始化为0
        que.push({0,0,0,0});  //将第一个方格入队
        while(que.size()>0){   //队列不空时
            vector<int>x=que.front();  //队首元素
            que.pop();  //出队
            int i=x[0],j=x[1],si=x[2],sj=x[3];
            //1.边界越过2.k小于下标数位和3.放个已经被访问过，跳过
            if(i>=m||j>=n||k<(si+sj)||visited[i][j]) continue;
            res++;
            visited[i][j]=true;
            que.push({i+1,j,(i+1)%10!=0?si+1:si-8,sj}); //将右边的方格入队
            que.push({i,j+1,si,(j+1)%10!=0?sj+1:sj-8});  //将下边的方格入队
        }
        return res;
    }
};

剑指 Offer 12. 矩阵中的路径
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]
class Solution {
public:
    //使用深度优先遍历dfs，回溯，从四个方向进行递归
    //方格下标过界，或者方格的值和word[k]不相等时，进行剪枝，返回false
    //如果k=word.size()-1，k遍历到字符串尾部，说明是一条正确路径
    bool exist(vector<vector<char>>& board, string word) {
        rows=board.size();
        cols=board[0].size();
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(dfs(board,word,i,j,0)) return true;
            }
        }
        return false;
    }
private:
    int cols,rows;
    bool dfs(vector<vector<char>>& board,string word,int i,int j,int k){
        if(i>=rows||i<0||j>=cols||j<0||board[i][j]!=word[k]) return false;
        if(k==word.size()-1) return true;
        board[i][j]=' ';
        // k++;
        bool res=dfs(board,word,i+1,j,k+1)||dfs(board,word,i-1,j,k+1)||dfs(board,word,i,j+1,k+1)||dfs(board,word,i,j-1,k+1);
        board[i][j]=word[k];
        return res;
     }
};

剑指 Offer 26. 树的子结构


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        //中序遍历，只要A中的任意节点包含B子树，就返回true
        if(A==NULL||B==NULL) return false;
        return isContain(A,B)||isSubStructure(A->left,B)||isSubStructure(A->right,B);
    }
    bool isContain(TreeNode* A,TreeNode* B){
        if(B==NULL) return true;
        if(A==NULL||A->val!=B->val) return false;
        return isContain(A->left,B->left)&&isContain(A->right,B->right);
    }
};

