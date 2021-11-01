---
layout:     post
title:      2021-11-01-链表SOTA操作
subtitle:   leetcode-链表SOTA操作
date:       2021-11-01
author:     Fans
header-img: img/post-bg-swift.jpg
catalog: 	  true
tags:
    - leetcode

---

# 链表操作
为什么单独把链表拎出来？因为最近刷题经常碰到链表，然后以前的知识以及写法老是忘记，就很烦，特此总结一下链表的一些SOTA操作，以及比较综合的一些题

## 链表特性
首先因为链表没法使用下标索引访问数据，因此搜索时间复杂度是On
但是链表的插入以及修改操作的复杂度达到O1，因为不需要对其余的数据发生变动
但也正因为这个特性，导致我们在很多题中必须要模拟解法，但是并不能保证解法的复杂度最低，因此我们总结出一些SOTA的写法来记忆。

### 1. 找到链表的中间节点
1. 首先最简单的解法肯定是，先访问一遍所有节点，确认链表的长度n，然后再访问n/2的位置节点
这样的时间复杂度是O3n/2,因为要访问一遍半的链表，需要的时间复杂度较高
2. 要么就将链表元素转换到数组中，然后采用下标直接访问，但是需要On的空间复杂度，并且时间复杂度也有On

**SOTA解法**： **最好的解法使用快慢指针解法**
使用一个slow指针每次走一步，一个fast指针每次走两步，快指针走到尽头的时候，slow恰在中点
注意边界条件，**当链表长度为偶数的时候，快指针最后一步不走，slow到达链表中间节点（偶数长度链表两个中间节点）的左边。**

```python
def middleNode(self, head: ListNode) -> ListNode:
	slow = fast = head
	while fast.next and fast.next.next:
	    slow = slow.next
	    fast = fast.next.next
	return slow
```
时间复杂度达到O n/2，比第一种解法快三倍。
空间复杂度O1


### 2. 反转链表
链表的反转比数组更难，但是也可以找到比较好的解法
1. 迭代法：使用一个额外none节点，从链表头部反复改变next指针的指向即可，时间复杂度On，空间复杂度O1
2. 递归：解决好边界。剩下的交给递归

迭代解法：
```python
def reverseList(self, head: ListNode) -> ListNode:
	prev = None
	 curr = head
	 while curr:
	     nextTemp = curr.next
	     curr.next = prev
	     prev = curr
	     curr = nextTemp
	 return prev
```
递归解法：
```python
def reverseList(self, head: ListNode) -> ListNode:
    # 1. 递归终止条件
    if head is None or head.next is None:
        return head
    
    p = self.reverseList(head.next)
    head.next.next = head
    head.next = None

    return p
```

### 3.删除倒数第N个节点
思路很明确，快慢指针，一个fast指针先走n步，然后slow从起点出发，等到fast指针到末尾，slow到达删除的位置，注意处理好边界条件。

```python
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
	dummy=ListNode(0,head)
	first=head
	second=dummy
	for i in range(n):
	    first=first.next
	while first:
	    first=first.next
	    second=second.next
	second.next=second.next.next
	return dummy.next
```

### 4.重排链表
![在这里插入图片描述](https://img-blog.csdnimg.cn/855b3eecafcb400ab259c4b10fb120e6.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)

分析题意，我们发现是要将后半部分逆序然后和前半部分做一个重排因此我们可以用到之前的SOTA操作
- 首先找到链表的中点
- 然后将后半部分逆序
- 然后将前半部分的中点部分斩断，然后和后半部分拼在一起

```python
def reorderList(self, head: ListNode) -> None:
	if not head:
	    return
	mid = self.middleNode(head)
	l1 = head
	l2 = mid.next
	mid.next = None
	l2 = self.reverseList(l2)
	self.mergeList(l1, l2)
    
def middleNode(self, head: ListNode) -> ListNode:
	slow = fast = head
	while fast.next and fast.next.next:
	    slow = slow.next
	    fast = fast.next.next
	return slow

def reverseList(self, head: ListNode) -> ListNode:
	prev = None
	curr = head
	while curr:
	    nextTemp = curr.next
	    curr.next = prev
	    prev = curr
	    curr = nextTemp
	return prev

def mergeList(self, l1: ListNode, l2: ListNode):
	while l1 and l2:
	    l1_tmp = l1.next
	    l2_tmp = l2.next
	
	    l1.next = l2
	    l1 = l1_tmp
	
	    l2.next = l1
	    l2 = l2_tmp
```
### 5.判断回文链表
![在这里插入图片描述](https://img-blog.csdnimg.cn/f8ad9cd52f3a41ea916bd215dcc1c623.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)

SOTA解法：和上面一题的思路差不多，也是找到链表的中点，然后将后半部分逆序，再前后比对一下

```python
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        def findmid(head):
            slow=fast=head
            while fast.next and fast.next.next:
                slow=slow.next
                fast=fast.next.next
            return slow
        def reverselist(head):
            pre=None
            cur=head
            while cur:
                tmp=cur.next
                cur.next=pre
                pre=cur
                cur=tmp
            return pre
        def issame(f1,f2):
            p1=f1
            p2=f2
            while p1 and p2:
                if p1.val!=p2.val:
                    return False
                p1=p1.next
                p2=p2.next
            return True

        if not head:
            return True

        mid=findmid(head)
        right=mid.next

        mid.next=None
        right=reverselist(right)
        left=head
        return issame(left,right)
```
### 6.链表的两数相加
![在这里插入图片描述](https://img-blog.csdnimg.cn/43e83e7f7634412da653752089dbb71c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)

SOTA解法：思路和之前做的两数之和差不多，但是需要将链表逆序，因为模拟法需要的是低位在前，后面高位补0，然后利用进位相加。

```python
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        def reverseList(head):
            dummy=None
            first=head
            while first:
                tmp=first
                first=first.next
                tmp.next=dummy
                dummy=tmp
        
            return dummy

        l1=reverseList(l1)
        l2=reverseList(l2)

        pre=ListNode(0,None)
        cur=pre
        cint=0

        while l1 or l2:
            x=l1.val if l1 else 0
            y=l2.val if l2 else 0

            sum1=x+y+cint
            cint=sum1//10
            sum1=sum1%10

            cur.next=ListNode(sum1,None)
            cur=cur.next

            if l1: l1=l1.next
            if l2: l2=l2.next

        if cint==1:
            cur.next=ListNode(1,None)
        res=reverseList(pre.next)
        return res
```
### 7.循环链表插入
![在这里插入图片描述](https://img-blog.csdnimg.cn/9269a373a13540f39abb9b8edf9a4d09.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)

需要分三种情况解决：
- 如果没有元素，则以要插入的元素建立循环链表
- 如果有一个元素，则直接两个元素相连建立循环链表
- 如果大于1个元素，则沿着链表寻找当前节点小于要插入的节点，并且下一个节点大于节点的位置
- 并且保存链表中最大节点，如果要插入的值找不到位置，则插入到最大值和最小值中间即可，即最大值的后面

```python
class Solution(object):
    def insertCore(self,head,insertnode):
        cur=head
        curnext=head.next
        maxval=head
        while not (cur.val<=insertnode.val and curnext.val>=insertnode.val)and curnext!=head:
            cur=curnext
            curnext=curnext.next
            if cur.val>=maxval.val:
                maxval=cur

        if cur.val<=insertnode.val and curnext.val>=insertnode.val:
            
            cur.next=insertnode
            insertnode.next=curnext
        else:
            
            insertnode.next=maxval.next
            maxval.next=insertnode
        
    def insert(self, head, insertVal):
        """
        :type head: Node
        :type insertVal: int
        :rtype: Node
        """
        insertnode=ListNode(insertVal,None)
        if not head:
            head=insertnode
            head.next=head
        elif head.next==head:
            head.next=insertnode
            insertnode.next=head
        else:
            self.insertCore(head,insertnode)
        return head
```
### 8.链表的第一个交点
![在这里插入图片描述](https://img-blog.csdnimg.cn/663d8b0801164b229bff24b080b39319.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)

SOTA：这题有个浪漫的解法，即我走过你来时的路和你走过我走过的路，然后我们相遇
即AB同时从起点出发，然后到达末尾时换到对方的路，这样的话，AB走的路程为A左半段+公共路段+B左半段，所以他们会在交点处相遇：

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        f1=headA
        f2=headB
        if not headA or not headB:
            return None
        while f1!=f2:
            if f1:
                f1=f1.next
            else:
                f1=headB
            if f2:
                f2=f2.next
            else:
                f2=headA

        return f1
```
### 9.链表环的入口
![在这里插入图片描述](https://img-blog.csdnimg.cn/f88103b47c5d4288b81707f07468ab90.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)

SOTA:这题采用快慢指针的解法
当快指针在环中绕圈时，与慢指针相遇，相遇点距离环的入口等于起点到环入口的距离。
![在这里插入图片描述](https://img-blog.csdnimg.cn/5d11a9b133b74ff0a64cd7a484a20e49.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5peg5p6S,size_20,color_FFFFFF,t_70,g_se,x_16)
解法：

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow=head
        fast=head
        while fast:
            slow=slow.next
            if not fast.next:
                return 
            fast=fast.next.next

            if fast==slow:
                ptr=head
                while ptr!=slow:
                    slow=slow.next
                    ptr=ptr.next
                return ptr
        return 
```
## 总结：
链表知识点就到这，还有更好的题再行补充
