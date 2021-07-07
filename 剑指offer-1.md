剑指 Offer 05. 替换空格
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
class Solution {
public:
    string replaceSpace(string s) {
        //先找出空格个数,然后用双指针法替换空格，用resize函数重置字符串长度
        int blank_count=0;
        int old_size=s.size();
        for(int i=0;i<s.size();i++){
            if(s[i]==' ') blank_count++;
        }
        s.resize(old_size+2*blank_count);
        int i=old_size-1,j=s.size()-1;
        while(i>=0){
            if(s[i]!=' '){
                s[j--]=s[i--];
            }
            else{
                s[j]='0';
                s[j-1]='2';
                s[j-2]='%';
                j-=3;
                i--;
            }
        }
        return s;
    }
};

剑指 Offer 03. 数组中重复的数字
在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        //一个萝卜一个坑，如果序号和对应萝卜对不上，就和另一个萝卜号对应的数字交换，
        int size=nums.size();
        if(size==0) return 0;
        for(int i=0;i<size;i++){
            while(nums[i]!=i){
                if(nums[i]==nums[nums[i]]) return nums[i];
                swap(nums[i],nums[nums[i]]);
            }
        }
        return 0;
    }
};

剑指 Offer 06. 从尾到头打印链表
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        vector<int> res;
        while(head){
            res.push_back(head->val);
            head=head->next;
        }
        reverse(res.begin(),res.end());
        return res;
        
        //用递归
        if(head==NULL) return {};
        vector<int> res=reversePrint(head->next);
        res.push_back(head->val);
        return res;
    }
};

方法三：使用辅助栈
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        // 使用辅助栈
        if(!head) return{};
        stack<int> stk;
        while(head){
            stk.push(head->val);
            head=head->next;
        }
        vector<int>res;
        while(!stk.empty()){
            res.push_back(stk.top());
            stk.pop();
        }
        return res;
    }
};


剑指 Offer 09. 用两个栈实现队列
class CQueue {
    stack<int> stack1,stack2;   //stack1用来插入，stack2用来删除，实现队列的先进先出
public:
    CQueue() {
        //先把两个栈清空
        while(!stack1.empty()){
            stack1.pop();
        }
        while(!stack2.empty()){
            stack2.pop();
        }
    }
    void appendTail(int value) {
        stack1.push(value);   //插入元素，压入stack1
    }
    
    int deleteHead() {
        //如果第二个栈为空，将第一个栈元素压入stack2
        if(stack2.empty()){
            while(!stack1.empty()){
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        if(stack2.empty()){ return -1;}
        else{
            //如果stack2不为空，将第一个元素出栈并返回删除元素
            int deleteItem=stack2.top();
            stack2.pop();
            return deleteItem;
        }
    }
};

剑指 Offer 10- I. 斐波那契数列
class Solution {
public:
    int fib(int n) {
        if(n==0||n==1) return n;
        int a=0,b=1,temp;
        for(int i=1;i<n;i++){
            temp=(a+b)%1000000007;
            a=b;
            b=temp;
        }
        return b;
    }
};

剑指 Offer 10- II. 青蛙跳台阶问题
class Solution {
public:
    int numWays(int n) {
        if(n<=1) return 1;
        int dp[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<n+1;i++){
            dp[i]=(dp[i-1]+dp[i-2])%1000000007;//取模是防止溢出
        }
        return dp[n];
    }
};

剑指 Offer 11. 旋转数组的最小数字
排序数组的查找问题首先客户以考虑二分法，可以把复杂度从线性级别降到对数级别
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。
https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/solution/mian-shi-ti-11-xuan-zhuan-shu-zu-de-zui-xiao-shu-3/
class Solution {
public:
    int minArray(vector<int>& numbers) {
        //二分法
        int i=0,j=numbers.size()-1;
        while(i<j){
            int m=i+(j-i)/2;
            if(numbers[m]>numbers[j]) i=m+1;
            else if(numbers[m]<numbers[j]) j=m;
            else j--;
        }
        return numbers[i];
    }
};

剑指 Offer 15. 二进制中1的个数
实现一个函数，输入一个整数（二进制串形式），输出该数二进制表示中1的个数，9的二进制1001，有两位1

class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count=0;
        while(n!=0){
            count++;
            //把一个整数减去1，再和原来的整数进行与运算，会让整个二进制位最后那个1变为0
            n=n&(n-1);
        }
        return count;
    }
};

剑指 Offer 17. 打印从1到最大的n位数（考虑大数问题）
剑指 Offer 18. 删除链表的节点
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        ListNode *p=head;
        if(!head) return NULL;
        if(head->val==val) return head->next;
        while(p->next!=NULL){
            if(p->next->val==val){
                p->next=p->next->next;
                return head;
            }
            p=p->next;
        }
        return head;
    }
};

剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        //双指针法，左边遇到的偶数和右边遇到的奇数交换
        int i=0,j=nums.size()-1;
        while(i<j){
            if(nums[i]%2!=0) {
                i++;
                continue;
            }
            if(nums[j]%2==0) {
                j--;
                continue;
            }
            swap(nums[i++],nums[j--]);
        }
        return nums;
    }
};

剑指 Offer 22. 链表中倒数第k个节点
输入一个链表，输出该链表中倒数第k个节点。例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode *res_node;
        ListNode *fast=head,*slow=head;
        int count=0;
        while(fast!=NULL){
            count++;
            if(count<=k) {
                fast=fast->next;
            }
            else{
                fast=fast->next;
                slow=slow->next;
            }
        }
        if(count<k) return NULL;
        return slow;
    }
};

剑指 Offer 24. 反转链表
输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
//递归
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head==NULL||head->next==NULL) return head;
        ListNode *node=reverseList(head->next);
        head->next->next=head;
        head->next=NULL;
        return node;
    }
};
//双指针法
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *cur=NULL,*pre=head;
        while(pre!=NULL){
            ListNode *temp=pre->next;
            pre->next=cur;
            cur=pre;
            pre=temp;
        }
        return cur;
    }
};

剑指 Offer 25. 合并两个排序的链表
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        //定义一个虚拟头节点，双指针遍历l1和l2，两者节点值比较大小，小的结点依次放到虚拟头节点的后面
        ListNode *res=new ListNode(0);
        ListNode *p=res;
        // if(l1==NULL&&l1==NULL) return NULL;
        if(l1==NULL) return l2;
        if(l2==NULL) return l1;
        while(l1!=NULL&&l2!=NULL){
            if(l1->val<l2->val){
                p->next=l1;
                l1=l1->next;
            }
            else{
                p->next=l2;
                l2=l2->next;
            }
            p=p->next;
        }
        if(l1!=NULL) p->next=l1;
        if(l2!=NULL) p->next=l2;
        return res->next;
    }
};

剑指 Offer 27. 二叉树的镜像
输入一个二叉树，该函数输出它的镜像
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(root==nullptr) return nullptr;
        //递归。结点为空时终止条件，先递归root右子节点作为root的左子节点，再递归左子节点左右root的右子节点
        // TreeNode *left=mirrorTree(root->right);
        // TreeNode *right=mirrorTree(root->left);
        // root->left=left;
        // root->right=right;
        // return root;

        //辅助栈，将根节点入栈，在栈不空时，将左右结点入栈，再将左右结点交换
        stack<TreeNode*> stk;
        stk.push(root);
        while(!stk.empty()){
            TreeNode *p=stk.top();
            stk.pop();
            if(p->left!=nullptr) stk.push(p->left);
            if(p->right!=nullptr) stk.push(p->right);
            TreeNode *tmp=p->left;
            p->left=p->right;
            p->right=tmp;
        }
        return root;
    }
};

剑指 Offer 28. 对称的二叉树
判断二叉树是否是对称的】
递归，比较左右子树，比较的是内侧节点和外侧节点是否相等
class Solution {
public:
    bool compare(TreeNode* left,TreeNode* right){
        //排除空节点的情况
        if(left==NULL&&right!=NULL) return false;
        else if(left!=NULL&&right==NULL) return false;
        else if(left==NULL&&right==NULL) return true;
        //排除数值不相同的情况
        else if(left->val!=right->val) return false;
        //左右节点不为空，且数值相同的情况，比较内测节点和外侧节点是否相等
        bool outside=compare(left->left,right->right);
        bool inside=compare(left->right,right->left);
        bool isSame=outside&&inside;
        return isSame;
    }
    bool isSymmetric(TreeNode* root) {
        if(root==NULL) return true;
        return compare(root->left,root->right);
    }
};

剑指 Offer 30. 包含min函数的栈
数据栈和辅助栈
请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int> stack1,stack2;//一个数据栈，一个辅助栈，辅助栈的数据个数始终和数据栈相同，同时辅助栈的栈顶始终时数据栈中的最小元素
    MinStack() {

    }
    void push(int x) {
        stack1.push(x);
        //如果辅助栈为空，或者x比辅助栈的栈顶元素小，入栈
        if(stack2.empty()||x<=stack2.top()){
            stack2.push(x);
        }
        //如果x比辅助栈的栈顶元素大，则把辅助栈的栈顶元素再入栈一次
        if(x>stack2.top()){
            int tmp=stack2.top();
            stack2.push(tmp);
        }
    }
    void pop() {
        //保持两个栈的数据个数相同，同时出栈
        if(!stack1.empty()&&!stack2.empty()){
            stack1.pop();
            stack2.pop();
        }
    }
    int top() {
        return stack1.top();
    }
    int min() {
        return stack2.top();
    }
};

剑指 Offer 29. 顺时针打印矩阵
按层模拟

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> order;
        if(matrix.size()==0||matrix[0].size()==0){
            return {};
        }
        int rows=matrix.size(),cols=matrix[0].size();
        int left=0,right=cols-1,top=0,button=rows-1;
        while(left<=right&&top<=button){
            //(top,left) to (top,right)
            for(int i=left;i<=right;i++){
                order.push_back(matrix[top][i]);
            }
            //(top+1,right) to (button,right)
            for(int j=top+1;j<=button;j++){
                order.push_back(matrix[j][right]);
            }
            if(left<right&&top<button){
                //(right-1,button)to(left+1,button)
                for(int m=right-1;m>left;m--){
                    order.push_back(matrix[button][m]);
                }
                //(left,button)to(left,top+1)
                for(int n=button;n>top;n--){
                    order.push_back(matrix[n][left]);
                }
            }
            left++;
            right--;
            top++;
            button--;
        }
        return order;
    }
};

剑指 Offer 32 - I. 从上到下打印二叉树
从上到下打印出二叉树的每个节点，同一层节点按照从左到右的顺序打印
//层序遍历，用队列
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        if(root==NULL) return {};
        vector<int> res;
        que.push(root);
        while(!que.empty()){
            TreeNode* node=que.front();
            que.pop();
            res.push_back(node->val);
            if(node->left) que.push(node->left);
            if(node->right) que.push(node->right);
        } 
        return res;
    }
};

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root==NULL) return {};
        vector<vector<int>> res; //结果数组
        queue<TreeNode*> que;   //辅助队列
        que.push(root);
        while(!que.empty()){
            int count=que.size();  //保存当前层的节点数
            vector<int> level; //保存当前层的元素
            //处理当前层的元素，每个元素出队时将子节点入队，作为下一层元素
            for(int i=0;i<count;i++){
                TreeNode* node=que.front();
                if(node->left) que.push(node->left);
                if(node->right) que.push(node->right);
                level.push_back(node->val);
                que.pop();
            }
            res.push_back(level);  //得到一层元素
        }
        return res;
    }
};

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root==NULL) return {};
        vector<vector<int>> res;
        queue<TreeNode*>que;
        bool isLeft=false;  //bool值表示是否需要将这一层的节点反转
        que.push(root);
        while(!que.empty()){
            vector<int> lev_vec;
            int count=que.size();
            for(int i=0;i<count;i++){
                TreeNode *node=que.front();
                if(node->left) que.push(node->left);
                if(node->right) que.push(node->right);
                lev_vec.push_back(node->val);
                que.pop();
            }
            isLeft=!isLeft;
            if(!isLeft){
                reverse(lev_vec.begin(),lev_vec.end());
                res.push_back(lev_vec);
            }
            else{
                res.push_back(lev_vec);
            }
        }
        return res;
    }
};

剑指 Offer 39. 数组中出现次数超过一半的数字
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        //哈希
        unordered_map<int,int> hash;
        int result;
        int len=nums.size();
        for(int i=0;i<len;i++){
            hash[nums[i]]++;
            if(hash[nums[i]]>len/2){
                result=nums[i];
                return result;
            }
        }
        return result;

        //摩尔投票法，不同的两者遇到就同归于尽，最后活下来的就是相同的
        int res=0,count=0;
        for(int i=0;i<nums.size();i++){
            if(count==0){
                res=nums[i];
                count++;
            }
            else{
                res==nums[i]?count++:count--;//投我就++，不投就--
            }
        }
        return res;
    }
};

剑指 Offer 40. 最小的k个数
//递归，哨兵划分，然后递归
class Solution{
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k){
        vector<int> res;
        quickSort(arr,0,arr.size()-1);
        //排序后取前k个元素
        res.assign(arr.begin(),arr.begin()+k);
        return res;
    }
private:
    void quickSort(vector<int>& arr,int l,int r){
        if(l>=r) return;//子数组长度为1时终止递归
        int i=l,j=r;
        while(i<j){
            while(i<j&&arr[j]>=arr[l]) j--;
            while(i<j&&arr[i]<=arr[l]) i++;
            swap(arr[i],arr[j]);
        }
        swap(arr[i],arr[l]);
        //递归左右子树进行哨兵划分
        quickSort(arr,l,i-1);
        quickSort(arr,i+1,r);
    }    
}

剑指 Offer 33. 二叉搜索树的后序遍历序列
输入整数数组，判断该数组是不是某二叉搜索树的后序遍历结果
class Solution{
public:
//二叉搜索树，根节点左侧的所有节点值一定小于根节点的值，根节点右侧所有节点的值一定大于根节点
//后序遍历的末尾存放的树的根节点
//递归
    bool verifyPostorder(vector<int>& postorder){
        if(postorder.empty()) return true;
        return isBST(postorder,0,postorder.size()-1);
     }  
     bool isBST(vector<int>& postorder,int l,int r){
         if(l>=r) return true;//超过叶节点，终止递归
         int flag=postorder[r];
         int i=1;
         //找到第一个大于根节点值的位置，将该节点到尾节点之前的区间视为右子树对应的序列
         for(;i<r;i++){
             if(postorder[i]>flag) break;
         }
         //检查右子树对应的序列有无小于根节点值的节点，有则返回false
         for(int j=i+1;j<r;j++){
             if(postorder[j]<flag) return false;
         }
         //递归判断左子树及右子树 [l,i-1]左子树,[i,r-1]右子树
         return isBST(postorder,l,i-1)&&isBST(postorder,i,r-1);
     }  
}

剑指 Offer 42. 连续子数组的最大和
输入整型数组，数组中的一个或连续多个整数组成一个子数组，求所有子数组的和的最大值
动态规划：dp[i]代表以nums[i]为结尾的连续子数组最大和
dp[i-1]>0时，dp[i]=dp[i-1]+nums[i]
dp[i-1]<=0,dp[i]=nums[i]
class Solution{
public:
    int maxSubArray(vector<int>& nums){
        int res=-500,sum=0;
        for(int i=0;i<nums.size();i++){
            if(sum<0) sum=0;
            sum+=nums[i];
            res=max(res,max(nums[i],sum));
        }
        return res;
        
        //法二
        int res=nums[0];
        for(int i=1;i<nums.size();i++){
            nums[i]+=max(nums[i-1],0);
            res=max(res,nums[i]);
        }
    }  
}

剑指 Offer 50. 第一个只出现一次的字符
在字符串s中找出第一个只出现一次的字符，如果没有，返回一个单空格，s只包含小写字母
使用哈希，第一次遍历字符串，使用哈希表统计字符出现的次数，再遍历字符串，在哈希表中找到首个数量为1的字符并返回
class Solution{
public:
    char firstUniqChar(string s){
        unordered_map<char,int>hash;
        for(char c:s) hash[c]++;
        for(char c:s){
            if(hash[c]==1) return c;
        }
        return ' ';
    }    
}

剑指 Offer 52. 两个链表的第一个公共节点
双指针法，node1和node2分别指向headA，headB的头节点，然后同时分组逐个遍历，当node1到达链表headA的末尾时，重新定位到headB的头节点，当node2到达链表B的末尾时，重新定位到headA的头节点，当他们相遇时，所指向的节点就是第一个公共节点
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA==NULL||headB==NULL) return NULL;
        ListNode *node1=headA;
        ListNode *node2=headB;
        while(node1!=node2){
            node1=node1!=NULL?node1->next:headB;
            node2=node2!=NULL?node2->next:headA;
        }
        return node1;
    }
};

剑指 Offer 53 - I. 在排序数组中查找数字 I
统计一个数字在排序数组中出现的次数
采用二分查找，找到一排相同数字的左边界和右边界
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int len=nums.size();
        if(len==0) return 0;
        int left=0,right=len-1;
        int boundaryRight,boundaryLeft;
        //找右边界
        while(left<=right){   //正是left=right的这次循环，left又右移了一次成为右边界
            int mid=left+(right-left)/2;
            if(nums[mid]<=target){
                left=mid+1;
            }
            else{
                right=mid-1;
            }//退出循环的时候left是最右target右边的元素，right为最右边的target
        }
        if(right>=0&&nums[right]!=target) return 0;
        boundaryRight=left;

        //找左边界
        right=left;   //这里重新赋值一定不要忘
        left=0;
        while(left<=right){     //正是left=right时，right又左移了一次成为左边界
            int mid=left+(right-left)/2;
            if(nums[mid]>=target){
                right=mid-1;
            }
            else{
                left=mid+1;
            }//退出循环时right为最左target的左边元素
        }
        boundaryLeft=right;
        return boundaryRight-boundaryLeft-1;
    }
};

剑指 Offer 53 - II. 0～n-1中缺失的数字
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
//二分查找
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        //二分法
        int left=0,right=nums.size()-1;
        while(left<=right){
            int mid=(left+right)/2;
            if(nums[mid]==mid) left=mid+1;
            else right=mid-1;
        }
        return left;
    }
};

剑指 Offer 54. 二叉搜索树的第k大节点
给定一颗二叉搜索树，找出第k大的节点
class Solution {
public:
//中序遍历，但是先遍历右子树再遍历左子树就是从小到大的顺序,第k大所以右根左
    int count=0;
    int res=0;
    int kthLargest(TreeNode* root, int k) {
        inorder(root,k);
        return res;
    }
    void inorder(TreeNode *root,int k){
        if(!root) return;
        inorder(root->right,k);
        ++count;
        if(count==k) {
            res=root->val;
            return;
        }
        inorder(root->left,k);
    }
};

剑指 Offer 55 - I. 二叉树的深度
输入一颗二叉树的根节点，求该树的深度（根节点到叶节点依次经过的节点形成的树的路径，最长路径长度为树的深度）
class Solution {
public:
    int maxDepth(TreeNode* root) {
        queue<TreeNode*> que;
        if(!root) return 0;
        int level=0;
        que.push(root);
        while(!que.empty()){
            level++;
            int count=que.size();
            for(int i=0;i<count;i++){
                TreeNode *node=que.front();
                que.pop();
                if(node->left) que.push(node->left);
                if(node->right) que.push(node->right);
            }
        }
        return level;
        //方法2 递归，不过效率没有队列的高
        //if(!root) return 0;
        //return 1+max(maxDepth(root->left),maxDepth(root->right));
    }
};

剑指 Offer 55 - II. 平衡二叉树
输入一棵树的根节点，判断该树是不是平衡二叉树（某二叉树中任意节点的左右子树深度相差不超过1，则是一颗平衡二叉树）
class Solution {
public:
//终止条件，root=null
//判断条件：如果左子树或右子树不平衡则不平衡，如果左右子树相差大于1则不平衡
    bool isBalanced(TreeNode* root) {
        if(root==NULL) return true;
        return isBalanced(root->left)&&isBalanced(root->right)&&(abs(MaxHeight(root->left)-MaxHeight(root->right))<=1);
    }
    int MaxHeight(TreeNode* root){
        if(root==NULL) return 0;
        return 1+max(MaxHeight(root->left),MaxHeight(root->right));
    }
};

剑指 Offer 56 - I. 数组中数字出现的次数
一个整型数组nums除了两个数字以外，其他数字都出现两次，找出这两个只出现一次的数字，时间复杂度o(n)，空间复杂度o(1)
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        //法一：先将数组排序，
        // vector<int> res;
        // sort(nums.begin(),nums.end());   
        // if(nums[0]!=nums[1]) res.push_back(nums[0]);  //处理第一个数
        // if(nums.back()!=nums[nums.size()-2]) res.push_back(nums.back());  //处理最后一个数
        // for(int i=1;i<nums.size()-1;i++){
        //     if(nums[i]!=nums[i-1]&&nums[i]!=nums[i+1]) res.push_back(nums[i]);
        // }
        // return res;
        //使用异或，相同的异或为0，不同的异或为1，0和任意数异或等于它本身
        int a=0,b=0;
        int res=0;    //记录a,b的异或结果
        for(auto n:nums){
            res^=n;
        }
        int fab=res&(-res);//使用a&(-a)的作用是可以得到最低位的1
        //利用上面的1，可以来对原数组进行分组，相同的数字会被分到同一组，
        //不同的两个数字会被分到不同的组
        //每组的数一路异或下去，最终会得到结果a,b
        for(auto n:nums){
            if((n&fab)==0){
                a^=n;
            }
            else{
                b^=n;
            }
        }
        vector<int> result;
        result.push_back(a);
        result.push_back(b);
        return result;
    }
};

剑指 Offer 56 - II. 数组中数字出现的次数 II
一个数组nums中除一个数字只出现一次外，其他数字都出现了三次，找出这个落单的数
法一：用上面一题排序的那个方法，通用
法二：每三个数检查一次
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size()-2;i+=3){
            if(nums[i]!=nums[i+2]) return nums[i];
        }
        return nums.back();
    }
};

剑指 Offer 57. 和为s的两个数字
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        //两种方法：哈希和双指针
        // unordered_set<int> hash;
        // for(auto n:nums){
        //     if(hash.count(target-n)) return {n,target-n};
        //     hash.insert(n);
        // }
        // return {};
        int i=0,j=nums.size()-1;
        while(i<j){
         int sum=nums[i]+nums[j];
         if(sum==target) return{nums[i],nums[j]};
         else if(sum>target) j--;
         else i++;  
        }
        return {-1,-1};
    }
};

剑指 Offer 57 - II. 和为s的连续正数序列
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
思路：使用滑动窗口（窗口左闭右开），双指针指向窗口的头和尾i,j，窗口内的所有数的和小于target，j++，窗口内元素和大于Target，i++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        int i=1,j=1,sum=0;
        while(i<=(target/2)){
            if(sum<target){
                sum+=j;
                ++j;
            }
            else if(sum>target){
                sum-=i;
                ++i;
            }
            else{
                vector<int> temp;
                for(int k=i;k<j;k++){
                    temp.push_back(k);
                }
                res.push_back(temp);
                sum-=i;
                ++i;
            }
        }
        return res;
    }
};

剑指 Offer 58 - I. 翻转单词顺序
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。标点符号和普通字符一样处理
class Solution {
public:
    string reverseWords(string s) {
        //遍历字符串，用tmp存放每一个新单词，遇到空格时将tmp置空，每次将单词加到res前面
        //以the sky is blue为例：
        //初始时res为空
        //res=""
        //res=the+" "
        //res=sky the+" "
       //res=is sky the+" "
       //res=blue is sky the+" "
       //最后如果res不为空，将the 后面的那个空格删掉
        int i=0,n=s.size();
        string res,tmp;
        while(i<n){
            while(i<n&&s[i]==' ') ++i;
            if(i==n) break;
            tmp="";
            while(i<n&&s[i]!=' ')   tmp+=s[i++];
            res=tmp+' '+res;
        }
        if(res!="") res.pop_back(); //res加第一个单词时后面有个空格加上了，最后将它删掉
        return res;
    }
};

这题更高效的方法：用堆栈，先将单词都入栈，然后弹出
class Solution {
public:
    string reverseWords(string s) {
        //堆栈来实现
        stack<string> sta;
        int len=s.size();
        for(int i=0;i<s.size();i++){
            string s1;
            while(i<len&&s[i]!=' '){
                s1+=s[i++];
            }
            if(!s1.empty()){
                sta.push(s1);
            }
        }
        string res="";
        while(!sta.empty()){
            res+=sta.top();
            sta.pop();
            if(!sta.empty()) res+=' ';
        }
        return res;
    }
};

剑指 Offer 58 - II. 左旋转字符串
把字符串前面的若干个字符串转移到字符串转移到字符串的尾部。输入abcdefg，输出
cdefgab
三次翻转，s=s1+s2,首先将s2反转，再将s翻转，最后将s1翻转
没有直接用c++的reverse函数，感觉面试官不喜欢这种直接调库函数的
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        if(s.size()<2||n==0) return s;
        reverse(s,n,s.size()-1);
        reverse(s,0,s.size()-1);
        reverse(s,s.size()-n,s.size()-1);
        return s;
    }
    void reverse(string& s,int start,int end){
        if(end>=s.size()||(end-start)<1) return ;
        while(start<end){
            char temp;
            temp=s[start];
            s[start]=s[end];
            s[end]=temp;
            start++;
            end--;
        }
    }
};

剑指 Offer 59 - I. 滑动窗口的最大值
给定一个数组nums和滑动窗口的大小k，找出所有滑动窗口里的最大值
class Solution {
public:
    //始终保证双端队列是递减的，队首是最大值
    //1.判断特殊情况，数组为空或k为0
    //2.如果出窗口的元素刚好等于队首元素，出队
    //3.队列不空时，如果队尾元素小于nums[j]，队尾出队
    //4.i大于等于0，将队首元素存入返回数组中
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(nums.size()==0||k==0) return {};
        vector<int> res;
        deque<int> dq;
        for(int i=1-k,j=0;j<nums.size();i++,j++){
            if(i>0&&nums[i-1]==dq[0]) dq.pop_front();
            while(!dq.empty()&&dq[dq.size()-1]<nums[j])
            {
                //如果队尾元素小于窗口里的新元素，出队，后面让大的进来
                dq.pop_back();
            }
            //如果队尾大于新元素，后面的直接站过来
            dq.push_back(nums[j]);
            if(i>=0) res.push_back(dq[0]);
        }
        return res;
    }
};


下面用双指针，不过效率不高，好理解而已，考试实在写不出来就用这个
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int len=nums.size();
        vector<int>res;
        if(k<=1) return nums;
        for(int i=0;i<len-k+1;i++){
            int left=i,right=i+k-1;
            int tmp=nums[left];
            for(int j=left;j<=right;j++){
                if(nums[j]>tmp){
                    tmp=nums[j];
                }
            }
            res.push_back(tmp);
        }
        return res;
    }
};

剑指 Offer 59 - II. 队列的最大值
定义一个队列实现函数max_value得到队列里的最大值，要求函数max_value、push_back和pop_front的均摊时间复杂度都是O(1)，若队列为空，pop_front和max_value需要返回-1
class MaxQueue {
    //维护一个辅助双端队列（递减队列，队首是最大值）
    queue<int> que;
    deque<int> dq;
public:
    MaxQueue() {}
    
    int max_value() {
        return dq.empty()?-1:dq.front();
    }

    //入队，同时递减队列中的元素如果小于入队元素，将队尾小的元素弹出
    void push_back(int value) {
        que.push(value);
        while(!dq.empty()&&dq.back()<value) {dq.pop_back();}
        dq.push_back(value);
    }
    //出队，如果出队的元素和递减队列的首元素相等，为了保持一致，递减队列的首元素也要弹出
    int pop_front() {
        if(que.empty()) return -1;
        int val=que.front();     //保存要弹出的首元素
        if(val==dq.front()) dq.pop_front();
        que.pop();
        return val;
    }
};

剑指 Offer 60. n个骰子的点数

class Solution {
public:
    vector<double> dicesProbability(int n) {
        //1.以投2个骰子为例。点数出现的范围就是2-12，有11种点数的可能，
        //再对所有点数分别求出出现这个点数的概率
        //2.求n个骰子和为s的数量时，可以把n-1个骰子和为s-1,s-2,s-3,..,s-6的可能累加起来
        //原因：对于最后一个阶段，第n个骰子，出现的点数就是1，2，，，6，那么投完n个之后的
        //点数s出现的次数就是s-1,s-2,s-3,s-4,s-5,s-6出现的次数的和
        //3.最底层，1个骰子和为1 2 3 4 5 6时的可能数都是1
        unordered_map<int,unordered_map<int,int>>dp;
        for(int s=1;s<=6;s++){
            dp[1][s]=1;
        }
        for(int n_touzi=2;n_touzi<=n;n_touzi++){
            for(int s=n_touzi;s<=n_touzi*6;s++){
                for(int i=1;i<=6;i++){
                    if(dp[n_touzi-1].find(s-i)!=dp[n_touzi-1].end()){
                        dp[n_touzi][s]+=dp[n_touzi-1][s-i];
                    }
                }
            }
        }
        double p=pow(1.0/6,n);
        vector<double> res;
        for(int s=n;s<=n*6;s++){
            res.push_back(dp[n][s]*p);
        }
        return res;
    }
};

剑指 Offer 61. 扑克牌中的顺子
从扑克牌随机抽5张牌，判断是不是一个顺子，大小王为0，可以看成任意数字
输入[0,0,1,2,5],输出true
class Solution {
public:
    //因为0可以代替任意数字
    //如果0的个数大于等于元素之间相差的扑克牌数总和，返回true，否则返回false
    //如果连续的两个数相同，返回false
    bool isStraight(vector<int>& nums) {
        int n_zero=0,n_diff=0;  
        sort(nums.begin(),nums.end());//先将数组排序
        for(int i=0;i<nums.size()-1;i++){
            if(nums[i]==0) n_zero++;
            else {
                if(nums[i]==nums[i+1]) return false;
                if(nums[i]+1!=nums[i+1]) n_diff+=nums[i+1]-nums[i]-1;
            }
        }
        return n_zero>=n_diff;
    }
};

剑指 Offer 63. 股票的最大利润
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        //遍历数组时，维护一个最小值，最小值初值为prices[0]
        //如果prices[i]大于min，则更新一下利润，否则说明比min还小，更新min
        if(prices.empty()) return 0;
        int res=0,min=prices[0];
        for(int i=1;i<prices.size();i++){
            if(prices[i]<=min){
                min=prices[i];
            }
            else{
                res=max(res,prices[i]-min);
            }
        }
        return res;
    }
};

剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
给定一个二叉搜索树，找到该树中两个指定节点的最近公共祖先
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //方法一：递归
        if(root==NULL)  return NULL;
        // if(root==p||root==q) return root;
        // TreeNode *left=lowestCommonAncestor(root->left,p,q);
        // TreeNode *right=lowestCommonAncestor(root->right,p,q);
        // if(left==NULL) return right;  //p,q都在右子树
        // else if(right==NULL) return left;  //p,q都在左子树
        // else return root;  //p,q各在一边，当前根就是最近公共祖先

        //方法二：利用二叉搜索树的性质：左子树所有节点值小于根节点，右子树所有节点值大于根节点
        //如果p,q都小于root，则根节点指向左子树，如果p,q都大于root,指向右子树，如果p,q在根的两边，则最近公共祖先就是根
        TreeNode *res=root;
        while(true){
            if(res->val>p->val&&res->val>q->val){
                res=res->left;
            }
            else if(res->val<p->val&&res->val<q->val){
                res=res->right;
            }
            else{break;}
        }
        return res;
    }
};

剑指 Offer 68 - II. 二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==NULL) return NULL;
        if(root==p||root==q) return root;
        TreeNode* left=lowestCommonAncestor(root->left,p,q);
        TreeNode* right=lowestCommonAncestor(root->right,p,q);
        if(left&&right) return root;  //一个在左，一个在右
        if(left) return left;//都在左子树
        if(right) return right;//都在右子树
        return NULL;
    }
};

105. 从前序与中序遍历序列构造二叉树
根据一棵树的前序遍历与中序遍历构造二叉树。


class Solution {
private:
    unordered_map<int,int> index;
public:
    TreeNode* construct(vector<int>& preorder,vector<int>& inorder,int pre_left,int pre_right,int in_left,int in_right){
        /*
        1.找到根节点在中序遍历中的位置
        2.构造根节点
        3.计算中序遍历中根节点左子树的长度
        4.递归根节点左边，递归根节点右边
        */
        if(in_left>in_right) return nullptr;
        int in_root=index[preorder[pre_left]];
        TreeNode* root=new TreeNode(preorder[pre_left]);
        int len=in_root-in_left;
        root->left=construct(preorder,inorder,pre_left+1,pre_left+len,in_left,in_root-1);
        root->right=construct(preorder,inorder,pre_left+len+1,pre_right,in_root+1,in_right);
        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n=preorder.size();
        for(int i=0;i<inorder.size();i++){
            index[inorder[i]]=i;
        }
        return construct(preorder,inorder,0,n-1,0,n-1);
    }
};

