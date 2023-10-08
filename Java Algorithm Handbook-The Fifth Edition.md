# Algorithm Handbook - The Fifth Edition
版本号1.0.14 20231007更新
[TOC]
## Preface to the Fifth Edition

> 第五版介绍

Java算法刷题宝典，是作者在刷Leetcode中总结出的精华，对知识点进行了系统全面的梳理，并在此之上总结出解法思路、代码模版、变式题分析、专题总结等内容，本书面向攻克Java算法面试、笔试的同学，题目难度合理，循序渐进，是一部优秀的算法学习资料。

> 第五版更新内容如下

- 重新组织了部分章节的顺序和内容。
- 标注了每个章节知识点笔试、面试出现的概率。
- 减少了部分重复例题，相似题目采用练习题单的方式给出。
- 增加了大量题解，以及思考题、相关知识点总结、时间复杂度分析。
- 章节、题目之间关联性、铺垫性显著增强。
- 增加了随机算法章节。
- 补充了差分数组、树上倍增、SPFA算法、数位DP、最小费用最大流等内容。
- 图论章节做了大量删减。
- 根据作者个人理解不断加深，90%以上代码都进行了重写，代码风格统一且尽可能精简。

> 第五版中作者比较满意的部分

- 基础篇数学/数组/字符串部分，补充了大量实用的结论和模版。
- 图论部分，删减了很多内容，进一步突出了重点。
- 排序，较为全面地介绍了排序算法，并梳理了面试的知识点。
- 二分查找，从一个全新的角度进行书写，统一了二分代码模版规范。
- 动态规划部分，重新安排了章节，把一些不常规的动态规划例题全线删减。

> 第五版后续还会完善的部分

- 树章节，还有很多内容没有补充到位。
- 贪心章节，例题覆盖度较少，证明不到位。
- 递归回溯与分治部分，有一些题解还不够清晰。

> 本书适合的学习方法

本书的大纲非常丰富，读者可以按照章节，学习例题并完成练习题，进而系统性地学习知识点。题解的代码风格非常统一，可读性较高，适合反复阅读理解。

> 本书不适合的读者群体

由于本书中部分题解较为简略，对于一些较为基础的知识点并没有展开讲解。本书不适合编程零基础的读者。本书知识点较为密集，适合静下心来阅读，不适合想要速成的读者。但读者仍然可以挑感兴趣或者薄弱的章节针对性地学习。

> 本书参考内容

本书参考了大量课程、网上题解、博客等内容，主要来源有：

- 慕课网 liuyobobobo 《玩转算法系列-图论精讲》《算法与数据结构体体系课》
- LeetCode 灵茶山艾府 大量题解
- 机械工业出版社 《算法笔记》
- 左程云《程序员代码面试指南》
- 《算法导论》
- 《算法 第四版》

因作者水平有限，错误疏漏在所难免，如有错误，还请多多指教。

## Coding Style

本题解中，所有代码基本遵循如下规范。

1. 矩阵的第一个维度变量命名为$m$，第二个维度命名为$n$。
2. 二分左端点命名为$l$，右端点命名为$r$。
3. 循环从外到内的变量命名为$i,j,k$。
4. 集合的使用统一采用多态的方式，父类接口指向子类实现类。
5. 一行的语句必须加括号。
6. 变量声明时，相同类型变量尽量放在一行。
7. 逻辑等价时，采用三目运算符简化if判断。
8. 返回值命名通常为ret，res或者ans。
9. 数组长度、字符串长度、容器长度通常提取为变量。
10. 二分查找尽量使用二段式，并按照二分查找章节的规范书写。
11. 可以用Stream流或者lambda表达式简化的尽量简化书写。
12. 不使用var语法。
13. Java推荐用双端队列模拟Stack的功能，但本书为了简便，依然采用Stack类进行书写。
## Part 1 Basic Knowledge

### 1.1 Java Common Library Functions

注意：JDK要求大于等于8

> 数组与List的转化

```java
// int[]转化为List
int[] arr = new int[]{1, 2, 3};
List<Integer> list = Arrays.stream(arr).boxed().toList();
// Integer[]转化为List
Integer[] arr2 = new Integer[]{1, 2, 3};
List<Integer> list2 = Arrays.asList(arr2);
// List转化为int[]
int[] arr3 = list.stream().mapToInt(e -> e).toArray();
// List转化为Integer[]
Integer[] arr4 = list.toArray(list.toArray(new Integer[0]));
```

> 位运算

```java
int i = 8;   // 1000
// 统计二进制有多少个1
int bit = Integer.bitcount(i);  // 1
// 统计二进制从后往前到第一个1时有多少个0
int trailingZeros = Integer.numberOfTrailingZeros(i);  // 3
// 统计二进制从前往后到第一个1时有多少个0
int leadingZeros = Integer.numberOfLeadingZeros(i);  // 28
```

> Arrays操作

```java
int[] arr = new int[]{1, 2, 3};
int[][] arr2 = new int[0][0];
int n = 5;
int sum = Arrays.stream(arr).sum();  // 求和
int max = Arrays.stream(arr).max().getAsInt();  // 求最大值
Arrays.sort(arr);  // 排序
Arrays.sort(arr2, Comparator.comparingInt(a -> a[0]));  // 自定义排序规则
Arrays.sort(arr2, (a, b) -> Integer.compare(b[0], a[0]))  // 逆序
// 注意点1. 基础数据类型的数组，如int[]，无法自定义排序器，只能升序。
// 注意点2. 降序排序时，尽量避免写成 (a, b) -> b - a，防止数据溢出时导致排序出错，使用Integer.compare(b, a)的方式。
Arrays.fill(arr, 1);  // 赋值
List<Integer[]>[] graph = new List[n];
// Arrays.setAll和Arrays.fill的区别
Arrays.setAll(graph, k -> new ArrayList<>()); // 数组中每个元素的List不相同
Arrays.fill(graph, new ArrayList<>());  //  数组中每个元素共享同一个List
```

> Map相关操作

```java
// 简化写法
Map<Integer, Integer> map = new HashMap<>();
int key = 114514, value = 666;
if (!map.containsKey(key)) {
	map.put(key, value);
}
// 简化为 map.putIfAbsent(key, value);
if (!map.containsKey(key)) {
	map.put(key, 0);
}
map.put(key, map.get(key) + 1);
// 简化为 map.put(key, map.getOrDefault(key, 0) + 1);
// 第二种写法：map.merget(key, 1, Integer::sum);
Map<Integer, List<Integer>> multiMap = new HashMap<>();
if (!multiMap.containsKey(key)) {
	multiMap.put(key, new ArrayList<>());
}
multiMap.get(key).add(666);
// 简化为 map.computeIfAbsent(key, k -> new ArrayList<>()).add(666);

// 遍历Entry
for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
	int k = entry.getKey(), v = entry.getValue();
}
```

> 生成随机数

```java
Random random = new Random();
int l = 3, r = 6;
random.nextInt(r - l + 1) + l;  // 生成[l, r)的随机数
```

> 流

```java
// 生成从0到m不含m的数组
int[] range = IntStream.range(0, n).toArray(); 
// 统计数组nums元素出现频率
Map<Integer, Long> freqMap = Arrays.stream(range).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
String[] str = new String[]{"aaa", "bb", "c"};
// 根据key的长度分组
Map<Integer, List<String>> collect = Arrays.stream(str).collect(Collectors.groupingBy(String::length));
```
Java语言特有的易产生难以排查的bug总结（血泪史555）

| 类型                                                         |
| ------------------------------------------------------------ |
| 参数是List\<Integer>，进行值比较一定用equals，大于127后用==判断会出错。 |
| 递归与回溯时，向List<List\<Integer>>中添加答案时一定要添加List的拷贝。（参见递归、回溯与分治78题) |
| 优先队列、TreeMap，传入比较器时，采用如下写法可能出错 Queue\<Integer\> queue = new PriorityQueue<>((a, b) -> a - b); 如果a-b产生数据溢出，优先级比较可能出错。建议使用Comparator.comparingInt的方式 |
|                                                              |

### 1.2 Math

#### 1.2.1 Bit Manipulation

位运算与集合

| 术语   | 集合                      | 位运算      | 集合例子                             | 位运算例子             |
| ------ | ------------------------- | ----------- | ------------------------------------ | ---------------------- |
| 交集   | $A\cap B$                 | $a \& b$    | $\{0,2,3\}\cap\{0,1,2\}=\{0,2\}$     | $1101\&0111=0101$      |
| 并集   | $A\cup B$                 | $a|b$       | $\{0,2,3\}\cup\{0,1,2\}=\{0,1,2,3\}$ | $1101|0111=1111$       |
| 差集   | $A\verb|\|B$              | $a\&\sim b$ | $\{0,2,3\}\verb|\|\{1,2\}=\{0,3\}$   | $1101\&1001=1001$      |
| 差集   | $A\verb|\|B,B\subseteq A$ | $a\oplus b$ | $\{0,2,3\}\verb|\|\{0,2\}=\{3\}$     | $1101\oplus 0101=1000$ |
| 包含于 | $A\subseteq B$            | $a\&b=a$    | $\{0,2\}\subseteq\{0,2,3\}$          | $0101\&1101=0101$      |

位运算与元素

| 术语    | 集合                 | 位运算               |
| ------- | -------------------- | -------------------- |
| 空集    | $\varnothing$        | 0                    |
| 元素$i$ | $\{i\}$              | $1 << i$             |
| 全集    | $U=\{0,1,2...,n-1\}$ | $(1<<n)-1$           |
| 补集    | $U \verb|\| S$       | $((1<<n)-1)\oplus S$ |
| 属于    | $i\in S$             | $(s >> i) \& 1 = 1$  |
| 不属于  | $i\notin S$                     |           $(s>>i)\&1=0$           |
| 添加元素 | $S\cup\{i\}$ | $s|(1<<i)$ |
| 删除元素 | $S\verb|\|\{i\}$ | $s\&\sim(1<<i)$ |
| 删除集合元素 | $S\verb|\|\{i\}$ | $s\oplus(1<<i)$ |
| 删除最小元素 |  | $s\&(s-1)$ |

枚举集合：

```java
for(int mask = 0; mask < (1 << n); mask ++) {
}
```

遍历集合：

```java
for(int i = 0; i < n; i++) {
	if((mask >> i) & 1 == 1) {
		// i在集合中
	}
}
```

例题：[78. 子集](https://leetcode.cn/problems/subsets/)

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        List<List<Integer>> ret = new ArrayList<>();
        for(int mask = 0; mask < (1 << n); mask ++) {
            List<Integer> list = new ArrayList<>();
            for(int i = 0; i < n; i ++) {
                if((mask >> i & 1) != 0) {
                    list.add(nums[i]);
                }
            }
            ret.add(list);
        }
        return ret;
    }
}
```

枚举子集：常用于**状态压缩DP**

```java
for(int sub = mask; sub > 0; sub = (sub - 1) & mask) {
}
```

其他位运算操作：

得到最低位的1：x & -x

该操作用于获得x二进制表示下最低位的1及其后面的0构成的数值，常用于树状数组里。

| 例子         | 二进制    |
| ------------ | --------- |
| x            | 1 0 1 0 0 |
| x的反码      | 0 1 0 1 1 |
| x的补码（-x) | 0 1 1 0 0 |
| x & -x       | 0 0 1 0 0 |

清除最低位的1 ：x & (x - 1)

该操作每执行一次，会将x最右边的一个1变为0。

例如，判断一个数是否是2的$n$次方：

例题：[231. 2 的幂](https://leetcode.cn/problems/power-of-two/)

```java
class Solution {
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}
```

又如，计算一个数中二进制的1的个数：

例题：[191. 位1的个数](https://leetcode.cn/problems/number-of-1-bits/)

```java
public class Solution {
    public int hammingWeight(int n) {  // 等价于Integer.bitCount(n)
        int res = 0;
        while(n != 0) {
            n = n & (n - 1);
            res ++;
        }
        return res;
    }
}
```

用一道经典难题，串联位运算的知识：

例题：[52. N 皇后 II](https://leetcode.cn/problems/n-queens-ii/)

```java
class Solution {
    public int totalNQueens(int n) {
        if(n < 1) {
            return 0;
        }
        return dfs((1 << n) - 1, 0, 0, 0);
    }

    private int dfs(int limit, int col, int left, int right) {
        // col表示列的限制，left表示左下右上对角线的限制，right表示左上右下对角线的限制
        if(col == limit) {
            return 1;
        }
        // 总限制
        int ban = col | left | right;
        // 差集
        int candidate = limit & (~ban);
        int place = 0;
        int ans = 0;
        while(candidate != 0) {
            // 提取最右侧的1
            place = candidate & (-candidate);
            // 取差集，去掉该位的1
            candidate ^= place;
            ans += dfs(limit, col | place, (left | place) >> 1, (right | place) << 1);
        }
        return ans;
    }
}
```
时间复杂度：$O(n!)$

异或的性质 

- 异或可以理解为不进位加法，如$1 \oplus 1=0, 0\oplus 0 =0,1\oplus 0 = 1$
- $x \oplus x = 0,x \oplus 0 = x$
- 自反性：$A \oplus B \oplus B = A$
- $a \oplus b = c \to a \oplus c = b$

应用一：利用异或交换a, b：
```java
a = a ^ b
b = a ^ b
a = a ^ b
```

应用二：利用异或寻找缺失数字

例题：[268. 丢失的数字](https://leetcode.cn/problems/missing-number/)

```java
class Solution {
    public int missingNumber(int[] nums) {
        int all = 0, missing = 0;
        for(int i = 0; i < nums.length; i ++) {
            all ^= i;
            missing ^= nums[i];
        }
        all ^= nums.length;
        return all ^ missing;
    }
}
```

应用三：找出出现特殊次数的数字

类型1：仅一个元素出现奇数次，其余元素出现偶数次，找出出现奇数次的元素。

例题：[136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

思路：0与所有元素异或，异或完的结果即为答案，代码略。

类型2：有两个元素出现奇数次，其余元素出现偶数次，找出出现奇数次的两个元素。

例题：[260. 只出现一次的数字 III](https://leetcode.cn/problems/single-number-iii/)

分析：将所有元素异或，得到a^b，其中a,b为出现次数为奇数的元素。a与b不相同，异或的结果一定有一位存在1，根据该位是否为1，可以将a,b分到不同的组中。

```java
class Solution {
    public int[] singleNumber(int[] nums) {
        int eor1 = 0, eor2 = 0;
        for(int num : nums) {
            eor1 ^= num;      // a ^ b
        }
        int rightOne = eor1 & (-eor1);
        for(int num : nums) {
            if((num & rightOne) == 0) {
                eor2 ^= num;  // 要么为a，要么为b
            }
        }
        return new int[]{eor2, eor1 ^ eor2};
    }
}
```
类型3：除一个元素出现次数小于m次外，其他元素都出现了恰好m次，找出该数。

例题：[137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)

分析：本题和异或无关，因为知识连续性放在一起。可以统计每一位出现1的次数，对m取模，如果模不为0，则该位一定是特殊元素所在的位。

```java
class Solution {
    public int singleNumber(int[] nums) {
        int[] cnt = new int[32];
        for(int num : nums) {
            for(int i = 0; i < 32; i ++) {
                cnt[i] += (num >> i) & 1;
            }
        }
        int ans = 0;
        for(int i = 0; i < 32; i ++) {
            if(cnt[i] % 3 != 0) {
                ans |= 1 << i;
            }
        }
        return ans;
    }
}
```

应用四：利用异或自反性

下一例题结合了异或的性质和前缀和数组，读者可以在学习完**前缀和数组**知识之后再学习这道例题。

例题：[1542. 找出最长的超赞子字符串](https://leetcode.cn/problems/find-longest-awesome-substring/)

```java
class Solution {
    public int longestAwesome(String s) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int ans = 0, mask = 0;
        for(int i = 0; i < s.length(); i ++) {
            mask ^= (1 << (s.charAt(i) - '0'));
            if(map.containsKey(mask)) {
                ans = Math.max(ans, i - map.get(mask));
            }
            for(int d = 0; d < 10; d ++) {
                int oddMask = mask ^ (1 << d);
                if(map.containsKey(oddMask)) {
                    ans = Math.max(ans, i - map.get(oddMask));
                }
            }
            map.putIfAbsent(mask, i);
        }
        return ans;
    }
}
```
练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1371. 每个元音包含偶数次的最长子字符串](https://leetcode.cn/problems/find-the-longest-substring-containing-vowels-in-even-counts/) | 中等 |
| [318. 最大单词长度乘积](https://leetcode.cn/problems/maximum-product-of-word-lengths/) | 中等 |
| [1177. 构建回文串检测](https://leetcode.cn/problems/can-make-palindrome-from-substring/) | 中等 |
| [1457. 二叉树中的伪回文路径](https://leetcode.cn/problems/pseudo-palindromic-paths-in-a-binary-tree/) | 中等 |


#### 1.2.2 Round Up

求x/y向上取整的代码

```
x / y + (x % y == 0 ? 0 : 1);
(int)Math.ceil(x*1.0/y);
(x + y - 1) / y;
```

#### 1.2.3 Base

该部分待补充（todo）

#### 1.2.4 Greatest Common Divisor

```java
public int gcd(int x, int y) {  // 最大公因数
	return y == 0 ? x : gcd(y, x % y);
}
public int lcm(int a, int b) {  // 最小公倍数
	return a * b / gcd(a, b);
}
```

例题：[878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/)

```java
class Solution {
    public int nthMagicalNumber(int n, int a, int b) {
        long l = Math.min(a, b), r = (long) n * Math.min(a, b);
        int c = lcm(a, b);
        while(l < r) {
            long mid = l + r >> 1;
            long cnt = mid / a + mid / b - mid / c;
            if(cnt < n) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return (int)(l % 10000_00007);
    }

    private int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
}
```

#### 1.2.5 High Precision Computation

例题：[43. 字符串相乘](https://leetcode.cn/problems/multiply-strings/)

```java
class Solution {
    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] result = new int[m+n];
        for(int i = m - 1; i >= 0; i --) {
            for(int j = n - 1; j >= 0; j --) {
                int a = num1.charAt(i) - '0';
                int b = num2.charAt(j) - '0';
                int c = a * b + result[i + j + 1];
                result[i + j + 1] = c % 10;
                result[i + j] += c / 10;
            }
        }
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while(i + 1 < result.length && result[i] == 0) {
            i ++;
        }
        while(i < result.length) {
            sb.append(result[i]);
            i ++;
        }
        return sb.toString();
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/) | 中等 |
| [66. 加一](https://leetcode.cn/problems/plus-one/)           | 简单 |
| [369. 给单链表加一](https://leetcode.cn/problems/plus-one-linked-list/) | 中等 |

#### 1.2.6 Modulo

$(a+b) \bmod m=((a\bmod m)+(b\bmod m))\bmod m$

$(a*b)\bmod m = ((a\bmod m)*(b\bmod m))\bmod m$

#### 1.2.7 Fast Power

例题：[50. Pow(x, n)](https://leetcode.cn/problems/powx-n/)

```java
class Solution {
    public double myPow(double x, int n) {
        long ln = n;
        return n > 0 ? myPow(x, ln) : 1.0 / myPow(x, -ln);
    }

    private double myPow(double x, long n) {
        if(n == 0) {
            return 1.0;
        }
        double res = 1.0;
        while(n > 0) {
            if((n & 1) == 1) {
                res *= x;
            }
            x *= x;
            n >>= 1;
        }
        return res;
    }
}
```

时间复杂度$O(\log n)$

> 矩阵快速幂

例题：[509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/)

斐波那契数列的递推公式：$F(n)=F(n-1)+F(n-2)$，线性递推的时间复杂度为$O(n)$

使用矩阵快速幂，可以将时间复杂度优化到$O(\log n)$

$\begin{pmatrix}1&1\\1&0\end{pmatrix}\begin{pmatrix}F(n)\\F(n-1)\end{pmatrix}=\begin{pmatrix}F(n)+F(n-1)\\F(n)\end{pmatrix}=\begin{pmatrix}F(n+1)\\F(n)\end{pmatrix}$

故$\begin{pmatrix}F(n+1)\\F(n)\end{pmatrix}=\begin{pmatrix}1&1\\1&0\end{pmatrix}^n\begin{pmatrix}F(1)\\F(0)\end{pmatrix}$

令$M=\begin{pmatrix}1&1\\1&0\end{pmatrix}$，用矩阵快速幂求解$M^n$

```java
class Solution {
    public int fib(int n) {
        if(n < 2) {
            return n;
        }
        int[][] m = {{1,1},{1,0}};
        int[][] res = pow(m, n - 1);
        return res[0][0];
    }

    public int[][] pow(int[][] m, int n) {
        int[][] ret = {{1, 0}, {0, 1}};
        while(n > 0) {
            if((n & 1) == 1) {
                ret = multiply(ret, m);
            }
            n >>= 1;
            m = multiply(m, m);
        }
        return ret;
    }

    public int[][] multiply(int[][] a, int[][] b) {
        int m = a.length, n = a[0].length, t = b[0].length;
        int[][] c = new int[m][t];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for(int k = 0; k < t; k ++) {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }
        return c;
    }
}
```

#### 1.2.8 Permutation and Combination

> 下一个排列

例题：[31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

分析：

```java
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while(i >= 0 && nums[i] >= nums[i+1]) {
            i --;
        }
        if(i >= 0) {
            int j = nums.length - 1;
            while(j >= 0 && nums[j] <= nums[i]) {
                j --;
            }
            swap(nums, i, j);
        }
        invert(nums, i+1, nums.length - 1);
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void invert(int[] nums, int l, int r) {
        while(l < r) {
            swap(nums, l, r);
            l ++;
            r --;
        }
    }
}
```

将其封装为模版

```java
public class Permutation<T extends Comparable<T>> {

    public boolean nextPermutation(int[] arr) {   // 直接修改arr获得下一个排列
        if(arr.length <= 1) {
            return false;
        }
        int i = arr.length - 2;
        for(; i >= 0 && arr[i] >= arr[i+1]; i--);
        if(i == -1) {
            return false;
        }
        int j = arr.length - 1;
        while(j >= 0 && arr[j] <= arr[i]) {
            j --;
        }
        swap(arr, i, j);
        invert(arr, i+1, arr.length - 1);
        return true;
    }

    public boolean nextPermutation(List<T> list) {  // 直接修改list获得下一个排列
        if (list == null || list.size() < 2) {
            return false;
        }
        int i = list.size() - 2;
        for(; i >= 0 && list.get(i).compareTo(list.get(i+1)) >= 0; i--);
        if (i == -1) {
            return false;
        }
        int j = list.size() - 1;
        while (j >= 0 && list.get(j).compareTo(list.get(i)) <= 0) {
            j --;
        }
        Collections.swap(list, i, j);
        Collections.reverse(list.subList(i+1, list.size()));
        return true;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void invert(int[] nums, int l, int r) {
        while(l < r) {
            swap(nums, l++, r--);
        }
    }
}
```

下面将该模版用于以下问题。

例题：[1947. 最大兼容性评分和](https://leetcode.cn/problems/maximum-compatibility-score-sum/)

分析：由于学生和导师的数量较少，可以枚举学生或者老师的全排列，计算所有排列中最大的兼容性评分和。

```java
class Solution {
    public int maxCompatibilitySum(int[][] students, int[][] mentors) {
        int m = students.length, ans = 0;
        int[] range = IntStream.range(0, m).toArray();
        do {
            int score = 0;
            for(int i = 0; i < m; i ++) {
                int j = range[i];
                score += getScore(students[i], mentors[j]);
            }
            ans = Math.max(ans, score);
        } while(nextPermutation(range));
        return ans;
    }

    public boolean nextPermutation(int[] arr) {
        if(arr.length <= 1) {
            return false;
        }
        int i = arr.length - 2;
        for(; i >= 0 && arr[i] >= arr[i+1]; i--);
        if(i == -1) {
            return false;
        }
        int j = arr.length - 1;
        while(j >= 0 && arr[j] <= arr[i]) {
            j --;
        }
        swap(arr, i, j);
        invert(arr, i+1, arr.length - 1);
        return true;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void invert(int[] nums, int l, int r) {
        while(l < r) {
            swap(nums, l++, r--);
        }
    }

    private int getScore(int[] student, int[] mentor) {
        int score = 0;
        for(int i = 0; i < student.length; i ++) {
            score += student[i] == mentor[i] ? 1 : 0;
        }
        return score;
    }
}
```

时间复杂度：$O(m·m!)$

$m >12$时，基于枚举全排列的方法可能会超时，可以阅读**状态压缩DP**相关章节，学习时间复杂度更优的解法。

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [2850. 将石头分散到网格图的最少移动次数](https://leetcode.cn/problems/minimum-moves-to-spread-stones-over-grid/) | 中等 |

> 第$k$个排列

例题：[60. 排列序列](https://leetcode.cn/problems/permutation-sequence/)

```java
class Solution {
    public String getPermutation(int n, int k) {
        int[] fac = new int[n];
        fac[0] = 1;
        for(int i = 1; i < n; i ++) {
            fac[i] = fac[i-1] * i;
        }
        List<Integer> range = IntStream.range(1, n + 1).boxed().collect(Collectors.toList());
        k --;
        StringBuilder sb = new StringBuilder();
        for(int i = n - 1; i >= 0; i --) {
            int index = k / fac[i];
            int candidate = range.remove(index);
            sb.append(candidate);
            k -= index * fac[i];
        }
        return sb.toString();
    }
}
```

时间复杂度：$O(n^2)$

反向思考：给定排列$a_1,a_2,...,a_n$，如何求$k$？

$k=(\sum_{i=1}^norder(a_i)·(n-i)!)+1$

> 组合数

直接使用BigInteger计算

```java
import java.math.BigInteger;

public class Main {
    public static BigInteger factorial(int n) {
        BigInteger result = BigInteger.valueOf(1);
        for (int i = 2; i <= n; i++) {
            result = result.multiply(BigInteger.valueOf(i));
        }
        return result;
    }

    public static BigInteger permutation(int n, int m) {
        return factorial(n).divide(factorial(n - m));
    }

    public static BigInteger combination(int n, int m) {
        return factorial(n).divide(factorial(m).multiply(factorial(n - m)));
    }
   
}
```

基于公式计算

$C_n^m=\frac{n!}{m!(n-m)!}(n<21)$

由于long存储的限制，最多只支持$n\le 20$的计算，因为$21!$会超过Long.MAX_VALUE

```java
public static long comb(long n, long m) {
    long ans = 1;
    for (long i = 1; i <= n ; i++) {
        ans *= i;
    }
    for (long i = 1; i <= m; i++) {
        ans /= i;
    }
    for (long i = 1; i <= n - m; i++) {
        ans /= i;
    }
    return ans;
}
```
$C_n^m=\frac{(n-m+1)\times(n-m+2)+...\times(n-m+m)}{1\times2\times...\times m}(n < 62, m < 31)$
```java
public static long comb(long n, long m) {
    long ans = n;
    for(long i = 2; i <= m; i ++) {
        ans = ans * (--n) / i;
    }
    return ans;
}
```

$C_n^m=C_{n-1}^m+C_{n-1}^{m-1}(n<67,m<33)$

```java
private static long[][] res = new long[67][67];
public static long comb(int n, int m) {
    if(m == 0 || m == n) {
        return 1;
    }
    if(res[n][m] != 0) {
        return res[n][m];
    }
    return res[n][m] = comb(n-1,m) + comb(n-1,m-1);
}
```

预处理所有组合数：

```java
private static long[][] res = new long[67][67];
public static void calculate(int n) {
	for (int i = 0; i <= n; i++) {
        res[i][0] = res[i][i] = 1;
    }
    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= i / 2; j ++) {
            res[i][j] = res[i-1][j] + res[i-1][j-1];
            res[i][i-j] = res[i][j];
        }
    }
}
```

时间复杂度：$O(n^2),n\le 66$

例题：[2842. 统计一个字符串的 k 子序列美丽值最大的数目](https://leetcode.cn/problems/count-k-subsequences-of-a-string-with-maximum-beauty/)

```java
class Solution {
    public int countKSubsequencesWithMaxBeauty(String s, int k) {
        int[] cnt = new int[26];
        for(char c : s.toCharArray()) {
            cnt[c - 'a'] ++;
        }
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for(int freq : cnt) {
            if(freq > 0) {
                map.put(freq, map.getOrDefault(freq, 0) + 1);
            }
        }
        long ans = 1;
        for(Map.Entry<Integer, Integer> entry : map.descendingMap().entrySet()) {
            int freq = entry.getKey(), count = entry.getValue();
            if(count >= k) {
                return (int) (ans * pow(freq, k) % mod * comb(count, k) % mod);
            }
            ans = ans * pow(freq, count) % mod;
            k -= count;
        }
        return 0;
    }

    private long mod = 10000_00007;

    private long pow(long x, int n) {
        long res = 1;
        while(n > 0) {
            if((n & 1) == 1) {
                res = res * x % mod;
            }
            x = x * x % mod;
            n >>= 1;
        }
        return res;
    }

    private long comb(long n, int k) {
        long res = n;
        for(int i = 2; i <= k; i ++) {
            res = res * (-- n) / i;
        }
        return res % mod;
    }
}
```

#### 1.2.9 Prime
筛法求素数模版：预处理[0...max]之间的素数。
```java
private static int max = (int)1e5;
private static boolean[] np = new boolean[max + 1];
static {
    np[1] = true;
    for(int i = 2; i * i <= max; i ++) {
        if(!np[i]) {
            for(int j = i * i; j <= max; j += i) {
                np[j] = true;
            }
        }
    }
}
```

### 1.3 Array

#### 1.3.1 Loop Invariant

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

循环不变量，是一组在循环体内、每一次迭代均保持为真的性质，通常用于证明程序或伪代码的正确性。后续讲解快速排序的划分操作时，循环不变量起到了很大的作用。

从一道例题看循环不变量：

例题：[26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

循环不变量：$nums[0...i)$中不存在重复元素。

初始化：$i = 1$，一个元素显然不存在重复元素。

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int i = 1;
        for(int j = 1; j < nums.length; j ++) {
            if(nums[j] != nums[i - 1]) {
                nums[i] = nums[j];
                i ++;
            }
        }
        return i;
    }
}
```
该代码还可以实现数据的离散化：

```java
class Solution {
    public Map<Integer, Integer> discretization(int[] nums) {
        int i = 1;
        Arrays.sort(nums);
        for(int j = 1; j < nums.length; j ++) {
            if(nums[j] != nums[i - 1]) {
                nums[i++] = nums[j];
            }
        }
        Map<Integer, Integer> map = new HashMap<>();
        for(int j = 0; j < i; j ++) {
            map.put(nums[j], j);
        }
        return map;
    }
}
```

例题：[80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/)

循环不变量：$nums[0...i)$中的元素最多重复出现2次。

初始化：$i = 2$

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        if(nums.length < 2) {
            return nums.length;
        }
        int i = 2;
        for(int j = 2; j < nums.length; j ++) {
            if(nums[j] != nums[i - 2]) {
                nums[i] = nums[j];
                i ++;
            }
        }
        return i;
    }
}
```

思考：原地删除重复出现多次的元素，使得每个元素只出现$k$次，应该如何书写代码？

例题：[41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)

分析：循环不变量为nums[0...l]范围内的i满足nums[i] = i + 1，nums[0...r-1]范围内的数为最好情况下能满足nums[i] = i + 1的区间。

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int l = 0, r = nums.length;
        while(l < r) {
            if(nums[l] == l + 1) {
                l ++;
            }else if(nums[l] <= l || nums[l] > r ||  nums[l] == nums[nums[l] - 1]) {
                // 进入else if条件，说明 nums[l] - 1 != l，但相等，说明存在重复数字。
                swap(nums, l, --r);
            }else {
                // nums[l] 在 [l+1...r]之间，先放置到该放置的位置。
                swap(nums, l, nums[l] - 1);
            }
        }
        return l + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

进阶：学习完快速排序的三路划分操作后，再结合41题进行深入理解二者共同点。

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [283. 移动零](https://leetcode.cn/problems/move-zeroes/)     | 简单 |
| [27. 移除元素](https://leetcode.cn/problems/remove-element/) | 简单 |

#### 1.3.2 Prefix Sum Array

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 中       | 高       | 必须掌握 |

从一个简单问题谈起：

例题：[303. 区域和检索 - 数组不可变](https://leetcode.cn/problems/range-sum-query-immutable/)

分析：定义presum[i]表示原数组nums[0...i-1]的累加和。区间nums[left...right]的和为presum[right+1] - presum[left]。

```java
class NumArray {
    private int[] presum;
    public NumArray(int[] nums) {
        int n = nums.length;
        presum = new int[n+1];  // 前缀数组比原始数组长度大一
        for(int i = 0; i < n; i ++) {
            presum[i+1] = presum[i] + nums[i];  // presum[0] = 0
        }
    }
    public int sumRange(int left, int right) {   
        return presum[right + 1] - presum[left];
    }
}
```

前缀和数组的优势在于，只需要预处理一次，后续求解子数组[left, right]的和，只需要O(1)时间就能计算出。

前缀和数组能够非常方便地解决子数组问题，一类经典的题目如下：

例题：[560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

分析：子数组和为$k$，则转化为前缀和数组中存在两个下标，使得$presum[i]-presum[j]=k$。可以枚举$i$和$j$的值，时间复杂度为$O(n^2)$。当数据规模较大时，需要采用哈希表进行优化，时间复杂度优化到$O(n)$。

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] presum = new int[n+1];
        for(int i = 0; i < n; i ++) {
            presum[i+1] = presum[i] + nums[i];
        }
        Map<Integer, Integer> map = new HashMap<>();
        int count = 0;
        for(int i = 0; i <= n; i ++) {
            if(map.containsKey(presum[i] - k)) {
                count += map.get(presum[i] - k);
            }
            map.put(presum[i], map.getOrDefault(presum[i], 0) + 1);
        }
        return count;
    }
}
```

以上代码还有优化空间，前缀和数组的计算可以在循环过程中计算，转而用变量sum代替。需要注意，前缀和数组的第0个元素0并没有加入计算，所以哈希表需要在一开始时添加put(0, 1)。同时，循环次数为$n$而非$n+1$。

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int n = nums.length, sum = 0, count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for(int i = 0; i < n; i ++) {
            sum += nums[i];
            if(map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return count;
    }
}
```

560题的等式关系是前缀和最简单的一种形式，有时候需要对等式进行一些变形，将其转化为能利用前缀和+哈希表处理的形式。

例题：[974. 和可被 K 整除的子数组](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)

分析：首先得到前缀和数组presum，假设存在区间$[i...j]$满足$(presum[i+1]-presum[j])\%k=0$

整理变形，得到$presum[i+1]\%k=presum[j]\%k$。在哈希表中存储以$presum[j]\%k$为键，出现次数为值的记录。

```java
class Solution {
    public int subarraysDivByK(int[] nums, int k) {
        int n = nums.length, sum = 0, count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for(int i = 0; i < n; i ++) {
            sum += nums[i];
            int key = (sum % k + k) % k;  // 取模后可能为负数，需要再加上k
            count += map.getOrDefault(key, 0);
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return count;
    }
}
```
真题链接：美团20230826笔试 https://codefun2000.com/p/P1497

提示：
$(presum[j] - presum[i]) / (j - i) = k$

$pre[j] - kj = pre[i] - ki$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [523. 连续的子数组和](https://leetcode.cn/problems/continuous-subarray-sum/) | 中等 |
| [525. 连续数组](https://leetcode.cn/problems/contiguous-array/) | 中等 |
| [2845. 统计趣味子数组的数目](https://leetcode.cn/problems/count-of-interesting-subarrays/) | 中等 |
| [325. 和等于 k 的最长子数组长度](https://leetcode.cn/problems/maximum-size-subarray-sum-equals-k/) | 中等 |
| [1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/) | 中等 |
| [1590. 使数组和能被 P 整除](https://leetcode.cn/problems/make-sum-divisible-by-p/) | 中等 |
| [1371. 每个元音包含偶数次的最长子字符串](https://leetcode.cn/problems/find-the-longest-substring-containing-vowels-in-even-counts/) | 中等 |

> 二维前缀和

定义$sum[i+1][j+1]$表示左上角为$a[0][0]$，右下角为$a[i][j]$的子矩阵元素和。

$sum[i+1][j+1]=sum[i+1][j] + sum[i][j+1]-sum[i][j]+a[i][j]$

例题：[304. 二维区域和检索 - 矩阵不可变](https://leetcode.cn/problems/range-sum-query-2d-immutable/)

```java
class NumMatrix {
    private int[][] sum;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        sum = new int[m+1][n+1];
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                sum[i+1][j+1] = sum[i][j+1] + sum[i+1][j] - sum[i][j] + matrix[i][j];
            }
        }
    }   
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        return sum[row2+1][col2+1] + sum[row1][col1] - sum[row2+1][col1] - sum[row1][col2+1];
    }
}
```

例题：[1139. 最大的以 1 为边界的正方形](https://leetcode.cn/problems/largest-1-bordered-square/)

```java
class Solution {
    public int largest1BorderedSquare(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] sum = new int[m+1][n+1];
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                sum[i+1][j+1] = sum[i][j+1] + sum[i+1][j] - sum[i][j] + grid[i][j];
            }
        }
        if(sum[m][n] == 0) {
            return 0;
        }
        int ans = 1;
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                for(int a = i + ans, b = j + ans; a < m && b < n; a ++, b ++) {
                    if(sumRegion(sum, i, j, a, b) - sumRegion(sum, i+1, j+1, a-1, b-1) == (a - i) * 4) {
                        ans = a - i + 1;
                    }
                }
            }
        }
        return ans * ans;
    }

    public int sumRegion(int[][]sum, int row1, int col1, int row2, int col2) {
        return sum[row2+1][col2+1] + sum[row1][col1] - sum[row2+1][col1] - sum[row1][col2+1];
    }
}
```
时间复杂度：$O(mn\min(m,n))$

真题链接：美团20230812笔试 https://codefun2000.com/p/P1443

> 计数前缀和

定义前缀和数组$pre[i][c]$表示数组$a[0...i-1]$中包含元素$c$的个数，则区间$[l...r]$中$c$的个数为$pre[r+1][c]-pre[l][c]$。

注意：通常来说，若$c$的取值范围需要很小，否则容易超时。

例题：[1906. 查询差绝对值的最小值](https://leetcode.cn/problems/minimum-absolute-difference-queries)

```java
class Solution {
    public int[] minDifference(int[] nums, int[][] queries) {
        int n = nums.length, m = queries.length;
        int[][] presum = new int[n+1][101];
        for(int i = 0; i < n; i ++) {
            for(int c = 1; c <= 100; c ++) {
                presum[i+1][c] += presum[i][c] + (nums[i] == c ? 1 : 0);
            }
        }
        int[] ret = new int[m];
        for(int i = 0; i < m; i ++) {
            int[] q = queries[i];
            int l = q[0], r = q[1];
            int pre = 0, ans = Integer.MAX_VALUE;
            for(int c = 1; c <= 100; c ++) {
                if(presum[r+1][c] - presum[l][c] > 0) {
                    if(pre != 0) {
                        ans = Math.min(ans, c - pre);
                    }
                    pre = c;
                }
            }
            ret[i] = ans == Integer.MAX_VALUE ? -1 : ans;
        }
        return ret;
    }
}
```
时间复杂度：$O(cn),c=100$

真题链接：字节跳动20230827笔试 https://codefun2000.com/p/P1505

#### 1.3.3 Binary Index Tree
| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低 | 中 |

如果数组可修改呢？每一次修改元素，前缀和数组就需要重新计算，重新计算的时间复杂度为O(n)。

例题：[307. 区域和检索 - 数组可修改](https://leetcode.cn/problems/range-sum-query-mutable/)

```java
class NumArray {
    private int[] c;
    private int[] a;
    public NumArray(int[] nums) {
        int n = nums.length;
        a = nums;
        c = new int[n + 1];
        for(int i = 0; i < n; i ++) {
            add(i + 1, nums[i]);
        }
    }

    private void add(int index, int val) { 
        for(; index < c.length; index += lowbit(index)) {
            c[index] += val;
        }
    }

    private int lowbit(int x) {
        return x & -x;
    }

    private int query(int index) {
        int sum = 0;
        for(; index > 0; index -= lowbit(index)) {
            sum += c[index];
        }
        return sum;
    }
    
    public void update(int index, int val) {
        add(index + 1, val - a[index]);
        a[index] = val;
    }
    
    public int sumRange(int left, int right) {
        return query(right + 1) - query(left);
    }
}
```

此时需要用到另一种更高效的数据结构，**树状数组**。

树状数组是一种可以动态维护序列前缀和的数据结构，支持单点修改和区间查询。

树状数组下标从1开始。

假设原数组为$a$，包含$a[1],a[2],...,a[8]$，另有一等长数组$c$，其中$c[i]$满足如下条件：

若$i$的二进制表示末尾有$k$个连续的0，则$c[i]$存储的区间长度为$2^k$，即：

$c[i]=a[i-2^k+1]+a[i-2^k+2]+...+a[i]$

区间长度即为i的二进制表示下最低位的1及后面的0构成的数值。

核心操作是 lowbit

```java
private int lowbit(int x) {
	return x & -x;
}
```

举例

| 树状数组 | 二进制 | 原数组                | x & -x |
| -------- | ------ | --------------------- | ------ |
| $c[1]$   | 0001   | $a[1]$                | 1      |
| $c[2]$   | 0010   | $a[1]+a[2]$           | 2      |
| $c[3]$   | 0011   | $a[3]$                | 1      |
| $c[4]$   | 0100   | $a[1]+a[2]+a[3]+a[4]$ | 4      |
| $c[5]$   | 0101   | $a[5]$                | 1      |
| $c[6]$   | 0110   | $a[5]+a[6]$           | 2      |
| $c[7]$   | 0111   | $a[7]$                | 1      |
| $c[8]$   | 1000   | $a[1]+...+a[8]$        | 8      |

> 前驱和后继

直接前驱：$c[i]$直接前驱为$c[i-lowbit(i)]$

直接后继：$c[i]$直接后继为$c[i+lowbit(i)]$

> 查询前缀和：前i个元素的前缀和$sum[i]$等于$c[i]$加上$c[i]$的所有前驱。

| 树状数组c[i] | 直接前驱 | 前缀和sum[i]            |
| ------------ | -------- | ----------------------- |
| $c[1]$       | 0        | $sum[1]=c[1]$           |
| $c[2]$       | 0        | $sum[2]=c[2]$           |
| $c[3]$       | 2        | $sum[3]=c[3]+c[2]$      |
| $c[4]$       | 0        | $sum[4]=c[4]$           |
| $c[5]$       | 4        | $sum[5]=c[5]+c[4]$      |
| $c[6]$       | 4        | $sum[6]=c[6]+c[4]$      |
| $c[7]$       | 6        | $sum[7]=c[7]+c[6]+c[4]$ |
| $c[8]$       | 0        | $sum[8]=c[8]$           |

```java
public int presum(int index) {
	int sum = 0;
	for(; index > 0; index -= lowbit(index)) {
		sum += c[index];
	}
	return sum;
}
```

> 点更新

当对$a[i]$进行修改时，如加上$z$，只需更新$c[i]$及其后继，都加上$z$

| 树状数组c[i] | 直接后继 | 更新                  |
| ------------ | -------- | --------------------- |
| $c[1]$       | 2        | $c[1],c[2],c[4],c[8]$ |
| $c[2]$       | 4        | $c[2],c[4],c[8]$      |
| $c[3]$       | 4        | $c[3],c[4],c[8]$      |
| $c[4]$       | 8        | $c[4],c[8]$           |
| $c[5]$       | 6        | $c[5],c[6],c[8]$      |
| $c[6]$       | 8        | $c[6],c[8]$           |
| $c[7]$       | 8        | $c[7],c[8]$           |
| $c[8]$       | -        | -                     |

```java
private void add(int index, int value) {
	for(; index < tree.length; index += lowbit(index)) {
        c[index] += value;
    }
}
```

树状数组除了维护前缀和外，还可以维护区间的最值。

例题：[1626. 无矛盾的最佳球队](https://leetcode.cn/problems/best-team-with-no-conflicts/)

分析：因为ages取值范围较小，先根据scores分数进行从小到大排序。

```java
class Solution {
    private int[] c;

    public int bestTeamScore(int[] scores, int[] ages) {
        int maxAge = Arrays.stream(ages).max().getAsInt();
        c = new int[maxAge + 1];
        int n = scores.length;
        int[][] pair = new int[n][2];
        for(int i = 0; i < n; i ++) {
            pair[i][0] = scores[i];
            pair[i][1] = ages[i];
        }
        Arrays.sort(pair, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        int ans = 0;
        for(int[] p : pair) {
            int score = p[0], age = p[1], cur = score + query(age);
            // cur 表示选择当前球员的分数 + 选择年龄小于等于该球员小的分数的最大值
            update(age, cur);
            ans = Math.max(ans, cur);
        }
        return ans;
    }

    private int query(int index) {   // 查询年龄小于等于index的最大分数。
        int max = 0;
        for(; index > 0; index -= lowbit(index)) {
            max = Math.max(max, c[index]);
        }
        return max;
    }

    private int lowbit(int x) {
        return x & (-x);
    }

    private void update(int index, int val) {  // 更新最大分数。
        for(; index < c.length; index += lowbit(index)) {
            c[index] = Math.max(c[index], val);
        }
    }
}
```

时间复杂度：$O(n \log n + n \log m)$

其中$n,m$分别为scores的长度，m为最大年龄。若m取值很大，可以用离散化的方式优化。

> 树状数组的离散化

例题：[315. 计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/)

一个可行的思路，是从右到左遍历数组，维护前缀和。假设当前遍历的数组为$nums[i]$，则通过查询前缀和$0...nums[i]-1$，即是$nums[i]$右侧小于$nums[i]$的元素的数量。

此时树状数组的长度取决于数组中的最大元素值，且树状数组数据较为稀疏，可以通过离散化的方式优化。

假设输入数组为$[5,2,6,1]$，通过离散化，将其转化为$[2,1,3,0]$，元素的大小关系并没有变化，数据更加紧凑。

```java
class Solution {
    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        int[] arr = Arrays.copyOf(nums, n);
        Arrays.sort(arr);
        Map<Integer, Integer> map = new HashMap<>();
        for(int num : arr) {
            if(!map.containsKey(num)) {
                map.put(num, map.size());
            }
        }
        tree = new int[map.size() + 1];
        int[] ret = new int[n];
        for(int i = n - 1; i >= 0; i --) {
            int hash = map.get(nums[i]);
            ret[i] = query(hash);
            update(hash + 1, 1);
        }
        return Arrays.stream(ret).boxed().toList();
    }

    private int[] tree;

    private int lowbit(int x) {
        return x & -x;
    }

    private int query(int index) {
        int sum = 0;
        for(; index > 0; index -= lowbit(index)) {
            sum += tree[index];
        }
        return sum;
    }

    private void update(int index, int val) {
        for(; index < tree.length; index += lowbit(index)) {
            tree[index] += val;
        }
    }

}
```

#### 1.3.4 Differential Array

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 低       | 低       | 了解 |

假设原数组为$a$，差分数组$d$定义如下：

$d[i]=\begin{cases} a[0]&i=0 \\ a[i]-a[i-1]& i\ge 1 \end{cases}$

通过计算差分数组的前缀和，可以复原原数组。

差分数组具有如下性质：对于$a$的子数组的区间操作，如$a[i],a[i+1],...,a[j]$都加上$x$，等价于将$d[i]+x$，$d[j+1]-x$。若$j + 1=n$，则只需$d[i]+x$。即原数组的区间操作可以等价为差分数组的单点操作。

使用差分数组的场景：有多次区间操作，询问最终区间操作的结果。

例题：[1109. 航班预订统计](https://leetcode.cn/problems/corporate-flight-bookings/)

```java
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] diff = new int[n];
        for(int[] book : bookings) {
            diff[book[0] - 1] += book[2];
            if(book[1] < n) {
                diff[book[1]] -= book[2];
            }
        }
        for(int i = 1; i < n; i ++) {
            diff[i] += diff[i-1];
        }
        return diff;
    }   
}
```

时间复杂度：$O(n+m)$，$m$为bookings数组长度。

1109题，由于已知区间长度，可以直接求解差分数组。若区间长度未知，则差分数组无法直接计算，此时需要利用TreeMap来记录。

例题：[2406. 将区间分为最少组数](https://leetcode.cn/problems/divide-intervals-into-minimum-number-of-groups/)

```java
class Solution {
    public int minGroups(int[][] intervals) {
        Map<Integer, Integer> map = new TreeMap<>();
        for(int[] inter : intervals) {
            int left = inter[0], right = inter[1];
            map.put(left, map.getOrDefault(left, 0) + 1);
            map.put(right+1, map.getOrDefault(right+1, 0) - 1);
        }
        int ans = 0, sum = 0;
        for(int value : map.values()) {
            sum += value;
            ans = Math.max(ans, sum);
        }
        return ans;
    }
}
```

时间复杂度：$O(n\log n)$

有了差分数组的知识，解答下一题就非常容易

例题：[1526. 形成目标数组的子数组最少增加次数](https://leetcode.cn/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/)

```java
class Solution {
    public int minNumberOperations(int[] target) {
        int n = target.length, ans = target[0];
        for(int i = 1; i < n; i ++) {
            ans += Math.max(target[i] - target[i-1], 0);
        }
        return ans;
    }
}
```

分析该算法的正确性：

若$target[i] \ge target[i+1]$，则可以在给$target[i]$加1的同时给$target[i+1]$增加1，此时$target[i+1]$不会占用额外次数。

若$target[i]<target[i+1]$，即使每次给$target[i]$加1的同时给$target[i+1]$增加1，还需要$target[i+1]-target[i]$次操作。

所以操作次数的下界等于差分数组中所有值为正的元素之和。

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [2381. 字母移位 II](https://leetcode.cn/problems/shifting-letters-ii/) | 中等 |
| [1094. 拼车](https://leetcode.cn/problems/car-pooling/)      | 中等 |
| [2772. 使数组中的所有元素都等于零](https://leetcode.cn/problems/apply-operations-to-make-all-array-elements-equal-to-zero/) | 中等 |
| [253. 会议室 II](https://leetcode.cn/problems/meeting-rooms-ii/) | 中等 |

> 二维差分

例题：[2132. 用邮票贴满网格图](https://leetcode.cn/problems/stamping-the-grid/)

分析：枚举每个grid[a][b] = 0的端点，用二维前缀和数组快速判断是否被占用。用二维差分数组对区域进行加一。将差分数组进行复原，复原后如果diff[a+1][b+1] = 0，则说明没有邮票覆盖。

```java
class Solution {
    public boolean possibleToStamp(int[][] grid, int h, int w) {
        int m = grid.length, n = grid[0].length;
        int[][] sum = new int[m+1][n+1], diff = new int[m+2][n+2];
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                sum[i+1][j+1] = sum[i+1][j] + sum[i][j+1] + grid[i][j] - sum[i][j];
            }
        }
        for(int a = 1, c = a + h - 1; c <= m; a ++, c ++) {
            for(int b = 1, d = b + w - 1; d <= n; b ++, d ++) {
                int sumRegion = sum[c][d] - sum[c][b-1] - sum[a-1][d] + sum[a-1][b-1];
                if(sumRegion == 0) {
                    diff[c+1][d+1] ++;
                    diff[a][b] ++;
                    diff[c+1][b] --;
                    diff[a][d+1] --;
                }
            }
        }
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                diff[i+1][j+1] += diff[i][j+1] + diff[i+1][j] - diff[i][j];
                if(grid[i][j] == 0 && diff[i+1][j+1] == 0) {
                    return false;
                }
            }
        }
        return true;
    }
}
```
注意：diff长度为m+2,n+2，是为了后续恢复diff数组时方便，不用针对下标为0进行特殊判断。

练习题单

| 题号                                                         | 难度 | 知识点 |
| ------------------------------------------------------------ | ---- | ---- |
| [LCP 74. 最强祝福力场](https://leetcode.cn/problems/xepqZ5/) | 中等 | 二维差分+离散化|

#### 1.3.5 Merge Array

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

数组归并是归并排序算法的核心。

> 二路归并

二路归并，旨在将两个有序的数组合并为一个更大的数组，是归并排序算法的子操作。

二路归并通常需要开辟额外空间，防止更新时的数据覆盖。

```java
public void merge(int[] first, int[] second) {
	int m = first.length, n = second.length;
    int[] temp = new int[m + n];
    int i = 0, j = 0, k = 0;
    while(i < m || j < n) {
        if(i == m) {
            temp[k++] = second[j++];
        }else if(j == n) {
            temp[k++] = first[i++];
        }else if(first[i] <= second[j]) {
            temp[k++] = first[i++];
        }else {
            temp[k++] = second[j++];
        }
    }
}
```

在一些特殊情况下，二路归并可以原地进行。

例题：[88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

采用逆序的方式，不会发生数据的覆盖。

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, k = nums1.length - 1;
        while(i >= 0 || j >= 0) {
            if(i < 0) {
                nums1[k--] = nums2[j--];
            }else if(j < 0) {
                nums1[k--] = nums1[i--];
            }else if(nums1[i] >= nums2[j]) {
                nums1[k--] = nums1[i--];
            }else {
                nums1[k--] = nums2[j--];
            }
        }
    }
}
```

利用数组归并，可以高效地计算逆序对个数。

例题：[剑指 Offer 51. 数组中的逆序对](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```java
class Solution {
    private int[] temp;

    public int reversePairs(int[] nums) {
        temp = new int[nums.length];
        return mergeAndCount(nums, 0, nums.length - 1);
    }

    private int mergeAndCount(int[] nums, int l, int r) {
        if(l >= r) {
            return 0;
        }
        int mid = l + r >> 1;
        int leftCount = mergeAndCount(nums, l, mid);
        int rightCount = mergeAndCount(nums, mid + 1, r);
        for(int i = l; i <= r; i ++) {
            temp[i] = nums[i];
        }
        int ans = leftCount + rightCount;
        int i = l, j = mid + 1, k = l;
        while(i <= mid || j <= r) {
            if(i > mid) {
                nums[k++] = temp[j++];
            }else if(j > r) {
                nums[k++] = temp[i++];
            }else if(temp[i] <= temp[j]) {
                nums[k++] = temp[i++];
            }else {
                ans += mid - i + 1;
                nums[k++] = temp[j++];
            }
        }
        return ans;
    }
}
```

变式：[493. 翻转对](https://leetcode.cn/problems/reverse-pairs/)

```java
class Solution {
    public int reversePairs(int[] nums) {
        temp = new int[nums.length];
        return mergeAndCount(nums, 0, nums.length - 1);
    }

    private int[] temp;

    private int mergeAndCount(int[] nums, int l, int r) {
        if(l >= r) {
            return 0;
        }
        int mid = l + r >> 1;
        int left = mergeAndCount(nums, l, mid);
        int right = mergeAndCount(nums, mid + 1, r);
        int ans = left + right;
        for(int i = l; i <= r; i ++) {
            temp[i] = nums[i];
        }
        int i = l, j = mid + 1, k = l;
        while(i <= mid && j <= r) {
            if((long)nums[i] > (long)2 * nums[j]) {
                ans += mid - i + 1;
                j ++;
            }else {
                i ++;
            }
        }
        i = l;
        j = mid + 1;
        while(i <= mid || j <= r) {
            if(i > mid) {
                nums[k++] = temp[j++];
            }else if(j > r) {
                nums[k++] = temp[i++];
            }else if(temp[i] <= temp[j]){
                nums[k++] = temp[i++];
            } else {
                nums[k++] = temp[j++];
            }
        }
        return ans;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [327. 区间和的个数](https://leetcode.cn/problems/count-of-range-sum/) | 困难 |
| [315. 计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) | 困难 |

> 多路归并

二路归并时，可以通过if else语句块判断元素的大小情况，从而选择移动指针。多路归并时，需要利用堆这一数据结构，高效地判断出需要移动哪个指针。

例题：[373. 查找和最小的 K 对数字](https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/)

以示例$[1,7,11]$和$[2,4,6]$ 讲解多路归并过程

第一路：$(1,2),(7,2),(11,2)$

第二路：$(1,4),(7,4),(11,4)$

第三路：$(1,6),(7,6),(11,6)$

初始时，放入每一路的第一个元素对，即$(1,2),(1,4),(1,6)$。

优先队列根据元素对元素和的大小进行出对。

```java
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> nums1[a[0]]+ nums2[a[1]]));
        int m = nums1.length, n = nums2.length;
        for(int i = 0; i < n; i ++) {
            queue.offer(new int[]{0, i});
        }
        List<List<Integer>> ret = new ArrayList<>();
        while(!queue.isEmpty() && k > 0) {
            int[] cur = queue.poll();
            int i = cur[0], j = cur[1];
            ret.add(Arrays.asList(nums1[i], nums2[j]));
            k --;
            if(i + 1 < m) {
                queue.offer(new int[]{i + 1, j});
            }
        }
        return ret;
    }
}
```

时间复杂度：$O(k\log k)$

变式：[313. 超级丑数](https://leetcode.cn/problems/super-ugly-number/)

分析：超级丑数本质上也可以看作是多路归并。

假设primes数组中的值为$[2,7,13,19]$，则存在四个指针，对四个无限序列进行归并，假设丑数序列为$s$。

第一路：$[2*e,e\in s]$

第二路：$[7*e,e\in s]$

第三路：$[13*e,e\in s]$

第四路：$[19*e,e\in s]$

```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        int m = primes.length;
        int[] pointer = new int[m];  // m路归并
        int[] dp = new int[n+1];   // 记录第i个丑数的值
        dp[1] = 1;
        Arrays.fill(pointer, 1);
        for(int i = 2; i <= n; i ++ ) {
            int min = Integer.MAX_VALUE;
            for(int j = 0; j < m; j ++) {
                if(dp[pointer[j]] * primes[j] < 0) {  // 避免溢出
                    continue;
                }
                min = Math.min(min, dp[pointer[j]] * primes[j]);
            }
            dp[i] = min;
            for(int j = 0; j < m; j ++) {
                if(min == dp[pointer[j]] * primes[j]) {  // 判断当前的最小值来自哪一个指针
                    pointer[j] ++; // 指针向后移动一位
                }
            }
        }
        return dp[n];
    }
}
```

时间复杂度：$O(nm)$

采用优先队列优化

```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        int m = primes.length;
        Queue<int[]> queue = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        for(int i = 0; i < m; i ++) {
            queue.add(new int[]{primes[i], i, 0});  
        }
        int[] ans = new int[n];
        ans[0] = 1;
        for(int i = 1; i < n;) {
            int[] cur = queue.poll();
            // val表示当前的丑数，j表示primes的下标，idx表示丑数的下标
            int val = cur[0], j = cur[1], idx = cur[2];
            if(val != ans[i - 1]) {  // 去重
                ans[i++] = val;
            }
            queue.offer(new int[]{ans[idx + 1] * primes[j], j, idx + 1});
        }
        return ans[n - 1];
    }
}
```

时间复杂度：$O(\max(m\log m,n\log m))$

练习题单

| 题号                                                         | 难度           |
| ------------------------------------------------------------ | -------------- |
| [786. 第 K 个最小的素数分数](https://leetcode.cn/problems/k-th-smallest-prime-fraction/) | 中等           |
| [1508. 子数组和排序后的区间和](https://leetcode.cn/problems/range-sum-of-sorted-subarray-sums/) | 中等           |
| [719. 找出第 K 小的数对距离](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/) | 困难(可能超时) |
| [264. 丑数 II](https://leetcode.cn/problems/ugly-number-ii/) | 中等           |

#### 1.3.6 Array Partition

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 低       |

数组划分，给定一个元素t，将数组划分成$<t,t,>t$三部分。t叫做pivot，通常是取数组中最左侧的元素。

划分操作是快速排序的核心，其中一个最简单的实现如下：

```java
public int partition(int[] nums, int l, int r) {
    int pivot = nums[l];
    int j = l;
    for(int i = l + 1; i <= r; i ++) {
        if(nums[i] <= pivot) {
            swap(nums, ++j, i);
        }
    }
    swap(nums, l, j);
    return j;
}
private void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}
```

划分操作还有一个重要应用，可以在$O(n)$时间内找到无序数组的第$k$个元素，该问题称为**SELECT K**问题。

例题：[215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        return findKthLargest(nums, nums.length - k, 0, nums.length - 1);
    }

    private int findKthLargest(int[] nums, int k, int l, int r) {
        int p = partition(nums, l, r);
        if(p == k) {
            return nums[p];
        }
        return p < k ? findKthLargest(nums, k, p + 1, r) : findKthLargest(nums, k, l, p - 1);
    }

    private Random random = new Random();

    private int partition(int[] nums, int l, int r) {
        int index = random.nextInt(r - l + 1) + l;
        swap(nums, l, index);
        int pivot = nums[l];
        int j = l;
        for(int i = l + 1; i <= r; i++) {
            if(nums[i] <= pivot) {
                swap(nums, ++j, i);
            }
        }
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

注：该算法的理论时间复杂度为$O(n)$，但可能因为特殊用例，如大量相同元素导致时间复杂度退化。最新增加的测试用例，该算法无法通过，需要对算法进行优化。

> 二路划分

二路划分的核心思想是将数组划分为三部分：$\ge t,t,\le t$。对于包含大量相同元素的情况，相同元素会较为平均地分布在两端。

将上面代码的partition替换为如下代码，则可以通过。

```java
private int partition(int[] nums, int l, int r) {
    int index = random.nextInt(r - l + 1) + l;
    swap(nums, l, index);
    int pivot = nums[l];
    // nums[l+1...i-1] <= r, nums[j+1...r] >= r
    int i = l + 1, j = r;
    while(true) {
        while(i <= j && nums[i] < pivot) {
            i ++;
        }
        while(j >= i && nums[j] > pivot) {
            j --;
        }
        if(i >= j) {
            break;
        }
        // 交换前：nums[i] >= pivot, nums[j] <= pivot
        swap(nums, i, j);
        // 交换后：nums[i] <= pivot, nums[j] >= pivot
        i ++;
        j --;
    }
    swap(nums, l, j);
    return j;
}
```

> 三路划分

三路划分的核心思想是将数组划分为三部分：$> t,t,< t$。

循环不变量：

$arr[l+1...lt] < v$

$arr[lt+1,i-1]=v$

$arr[gt,r]>v$

初始化：$lt = l,gt=r+1,i=l+1$

循环条件：$i<gt$

```java
public void partition(int[] arr, int l, int r) {
    if(l >= r) {
        return;
    }
    int index = random.nextInt(r - l + 1) + l;
    swap(nums, l, index);
    int lt = l, i = l + 1, gt = r + 1, pivot = arr[l];
    while(i < gt) {
        if(arr[i] < pivot) {
            swap(arr, ++lt, i);
            i ++;
        }else if(arr[i] > pivot) {
            gt --;
            swap(arr, i, gt);
        }else {
            i ++;
        }
    }
}
```

例题：[75. 颜色分类](https://leetcode.cn/problems/sort-colors/)

分析：由于只有三种颜色，所以不需要选择pivot。

循环不变量：$nums[0...zero]=0$

$nums[zero+1,i-1]=1$

$nums[two,n-1]=2$

```java
class Solution {
    public void sortColors(int[] nums) {
        int zero = -1, i = 0, two = nums.length;
        while (i < two) {
            if (nums[i] == 0) {
                zero ++;
                swap(nums, zero, i);
                i ++;
            } else if (nums[i] == 1) {
                i ++;
            } else {
                two --;
                swap(nums, two, i);
            }
        }
    }
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### 1.3.7 Blocks and Buckets

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

> 分块

在探讨307题，我们使用了树状数组的方式求解。该题还可以采用分块的思想进行解决。

SQRT分解：将含有$N$个元素的数组分成$\sqrt N$份

假设有原数组

| 32   | 26   | 17   | 55   | 72   | 19   | 8    | 46   | 22   | 68   | 28   | 33   | 62   | 92   | 53   | 16   | 91   | 16   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |

分成5个子数组

|      |      |      |      | 数组和     |
| ---- | ---- | ---- | ---- | ---------- |
| 32   | 26   | 17   | 55   | 130(0~3)   |
| 72   | 19   | 8    | 46   | 145(4~7)   |
| 22   | 68   | 28   | 33   | 151(8~12)  |
| 62   | 92   | 53   | 16   | 223(13~15) |
| 91   | 16   |      |      | 107(16-17) |

查询区间在同一个组：遍历数组，时间复杂度$O(\sqrt n)$

查询区间在相邻组：遍历数组，时间复杂度$O(2\sqrt n)$

查询区间在非相邻组：首尾两组遍历，中间跨过的组，累加预处理的数组和，时间复杂度$O(3\sqrt n)$

整体查询的时间复杂度为$O(\sqrt n)$

例题：[307. 区域和检索 - 数组可修改](https://leetcode.cn/problems/range-sum-query-mutable/)

```java
class NumArray {

    private int[] data, blocks;
    private int n;  // 元素总数
    private int b;  // 每组元素个数
    private int bn; // 组数

    public NumArray(int[] nums) {
        n = nums.length;
        b = (int)Math.sqrt(n);
        bn = n / b + (n % b == 0 ? 0 : 1);
        data = Arrays.copyOf(nums, n);
        blocks = new int[bn];
        for(int i = 0; i < n; i ++) {
            blocks[i / b] += nums[i];
        }
    }
    
    public int sumRange(int left, int right) {
        int bStart = left / b, bEnd = right / b;
        int sum = 0;
        if(bStart == bEnd) {
            for(int i = left; i <= right; i ++) {
                sum += data[i];
            }
            return sum;
        }
        for(int i = left; i < (bStart + 1) * b; i ++) {
            sum += data[i];
        }
        for(int i = bEnd * b; i <= right; i ++) {
            sum += data[i];
        }
        for(int i = bStart + 1; i < bEnd; i ++) {
            sum += blocks[i];
        }
        return sum;
    }
    
    public void update(int index, int val) {
        int bi = index / b;
        blocks[bi] -= data[index];
        blocks[bi] += val;
        data[index] = val;
    }
    
}
```

时间复杂度：update操作$O(1)$，sumRange操作$O(\sqrt n)$

> 分桶

直接从一个例题学习分桶思想。

例题：[164. 最大间距](https://leetcode.cn/problems/maximum-gap/)

分析：对于有n个元素的数组，分配n+1个桶，则必然有一个空桶。最小值一定放在第1个桶，最大值一定放在第n+1个桶。排序后的相邻数可能在同一个桶或相邻的两个桶，但产生最大差值的两个相邻数一定不会来自于同一个桶，因为在同一个桶的差值小于c。由于有空桶的存在，空桶左侧和右侧一定存在非空桶，此时这两个非空桶任意相邻元素的差值一定大于等于c。因此，最大差值一定来源于某个非空桶的最小值与前一个非空桶的最大值的差值。用一个变量记录遍历过程中的最大差值即可。

```java
class Solution {
    public int maximumGap(int[] nums) {
        int n = nums.length;
        int min = Arrays.stream(nums).min().getAsInt();
        int max = Arrays.stream(nums).max().getAsInt();
        if(min == max) {
            return 0;
        }
        boolean[] used = new boolean[n+1];  // 记录当前桶是否被使用
        int[] minBucket = new int[n+1];  // 当前桶的最小元素
        int[] maxBucket = new int[n+1];  // 当前桶的最大元素
        int interval = (max - min) / (n + 1) + 1;
        for(int i = 0; i < n; i++) {
            int index = (nums[i] - min) / interval;
            minBucket[index] = used[index] ? Math.min(nums[i], minBucket[index]) : nums[i];
            maxBucket[index] = used[index] ? Math.max(nums[i], maxBucket[index]) : nums[i];
            used[index] = true;
        }
        int ret = 0;
        int lastMax = maxBucket[0];
        for(int i = 1; i <= n; i++) {
            if(used[i]) {
                ret = Math.max(minBucket[i]-lastMax, ret);
                lastMax = maxBucket[i];
            }
        }
        return ret;
    }
}
```

总结：

假设数组元素个数为$n$，最大值为$maxValue$，最小值为$minValue$，即数组元素取值范围为$[minValue, maxValue]$。假设桶的元素范围间隔为$c$，桶的数量为$b$。

对于第$1$个桶，存放数据范围为$[minValue, minValue + c)$的元素；

对于第$2$个桶，存放数据范围为$[minValue + c, minValue + 2*c)$的元素；

对于第$b-1$个桶，存放数据为$[minValue + (b - 2)*c, minValue + (b-1)*c)$的元素；

对于第$b$个桶，单独存放$[minValue + (b - 1)*c, minValue + b*c)$元素，其中$ minValue + (b - 1) * c \le maxValue < minValue + b*c$

若$c$已知，求$b$，则$b = (maxValue - minValue) / c + 1$，因为$ b > (maxValue - minValue) / c$

若$b$已知，求$c$，则$c = (maxValue - minValue) / b + 1$，理由同上。

若给定元素$num$，获取桶索引$i$，则为$(num - minValue) / c$，其中索引从$0$开始。

变式：[1630. 等差子数组](https://leetcode.cn/problems/arithmetic-subarrays/)

```java
class Solution {
    public List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
        List<Boolean> ret = new ArrayList<>();
        for(int i = 0; i < l.length; i++) {
            ret.add(check(nums, l[i], r[i]));
        }
        return ret;
    }
    private boolean check(int[] nums, int l, int r) {
        int max = nums[l], min = nums[l];
        for(int i = l+1; i <= r; i++) {
            max = Math.max(max, nums[i]);
            min = Math.min(min, nums[i]);
        }
        if(max == min) {
            return true;
        }
        if((max - min) % (r - l) != 0) {
            return false;
        }
        int d = (max - min) / (r - l);
        boolean[] bucket = new boolean[r - l + 1];
        for(int i = l; i <= r; i++) {
            if((nums[i] - min) % d != 0) {
                return false;
            }
            int bid = (nums[i] - min) / d;
            if(bucket[bid]) {
                return false;
            }
            bucket[bid] = true;
        }
        return true;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [621. 任务调度器](https://leetcode.cn/problems/task-scheduler/) | 中等 |

#### 1.3.8 Rotating Array

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

例题：[189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

将数组进行旋转，假设旋转位置为$k$，可以通过翻转操作等效。

若$k >= n$，先对$k$取模。

例子：$k = 3$

| 操作               | 结果              |
| ------------------ | ----------------- |
| 原数组             | $[1,2,3,4,5,6,7]$ |
| 翻转数组           | $[7,6,5,4,3,2,1]$ |
| 翻转区间$[0, k-1]$ | $[5,6,7,4,3,2,1]$ |
| 翻转区间$[k,n-1]$  | $[5,6,7,1,2,3,4]$ |



```java
class Solution {
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k %= n;
        reverse(nums, 0, n - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, n - 1);
        
    }
    private void reverse(int[] nums, int l, int r) {
        while(l < r) {
            int temp = nums[l];
            nums[l++] = nums[r];
            nums[r--] = temp;
        }
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/) | 中等 |

假设数组在旋转前是升序，旋转后的数组有两种情况。

- 完全升序：$k \% n = 0$
- 分段升序，在$(k-1+n)\%n$处取得最大值，$k\%n$处取得最小值。

若数组经过了多次旋转，上述性质依然成立。

> 旋转数组求最小值

例题：[153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

求解该题之前，可以先学习后续**二分查找**章节的知识点。

```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = l + r >> 1;
            if(nums[mid] >= nums[r]) {  // 不存在重复元素，中间有断点，最小值一定在右侧
                l = mid + 1;
            }else {  
                r = mid;
            }
        }
        return nums[l];
    }
}
```

若数组中存在重复元素，又应该怎样处理？

例题：[154. 寻找旋转排序数组中的最小值 II](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/)

```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = l + r >> 1;
            if(nums[mid] > nums[r]) {
                l = mid + 1;
            }else if(nums[mid] < nums[r]) {
                r = mid;
            }else {
                r --;  
            }
        }
        return nums[l];
    }
}
```

时间复杂度：$O(n)$，数组元素完全相同时

> 旋转数组查找指定元素

例题：[33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

由于是精确查找，采用三段式的写法。

```java
class Solution {
    public int search(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l <= r) {
            int mid = l + r >> 1;
            if(nums[mid] == target) {
                return mid;
            }
            if(nums[mid] <= nums[r]) {  // nums[mid...r]有序
                if(nums[mid] < target && target <= nums[r]) {  // 有序区间
                    l = mid + 1;
                } else {  // 无序区间
                    r = mid - 1;
                }
            }else {  // nums[l...mid]区间有序
                if(nums[l] <= target && target < nums[mid]) {  // 有序区间
                    r = mid - 1;
                }else {  // 无序区间
                    l = mid + 1;
                }
            }
        }
        return -1;
    }
}
```

若数组中存在重复元素，又应该怎样处理？

例题：[81. 搜索旋转排序数组 II](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)

```java
class Solution {
    public boolean search(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l <= r) {
            int mid = l + r >> 1;
            if(nums[mid] == target) {
                return true;
            }
            if(nums[mid] == nums[r]) {  // 只需要补充这段逻辑即可
                r --;
            }else if(nums[mid] < nums[r]) {  
                if(nums[mid] < target && target <= nums[r]) {  
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }else { 
                if(nums[l] <= target && target < nums[mid]) { 
                    r = mid - 1;
                }else {
                    l = mid + 1;
                }
            }
        }
        return false;
    }
}
```

#### 1.3.9 Median of Array

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

从一道例题看数组中位数的性质。

例题：[462. 最小操作次数使数组元素相等 II](https://leetcode.cn/problems/minimum-moves-to-equal-array-elements-ii/)

分析：当所有元素变为数组的中位数，需要的操作次数最少。

证明：假设有$2n+1$个数，从大到小排序后如下：$...,a,m,b$，其中$m$为中位数。

假设把$m$左边的数变为$m$需要代价$x$，把$m$右边的数变为$m$需要代价$y$，总代价$t=x+y$。

若把所有元素变为$a$才是最优解，则$m$右边的数代价为$y+(m-a)*n$，$m$左边的数代价为$x-(m-a)*n$，$m$变为$a$的代价为$m-a$，总代价$y+(m-a)*n+x-(m-a)*n+m-a=x+y+m-a>t$。

偶数的场景，选择左侧或右侧任意一个中位数都可，证明采用同样方式推导即可。

对于无序数组，可以利用数组划分操作，在$O(n)$时间复杂度内快速寻找中位数。

```java
class Solution {
    private Random random = new Random();

    public int minMoves2(int[] nums) {
        int n = nums.length;
        int mid = quickSelect(nums, 0, n - 1, n / 2);
        int ret = 0;
        for(int i = 0; i < n; i ++) {
            ret += Math.abs(nums[i] - mid);
        }
        return ret;
    }

    public int quickSelect(int[] nums, int left, int right, int index) {
        int p = randomPartition(nums, left, right);
        if(p == index) {
            return nums[p];
        } else {
            return p < index ? quickSelect(nums, p + 1, right, index) : quickSelect(nums, left, p - 1, index);
        }
    }

    public int randomPartition(int[] nums, int left, int right) {
        int i = random.nextInt(right - left + 1) + left;
        swap(nums, i, left);
        int pivot = nums[left], p = left;
        for(int j = left + 1; j <= right; j ++) {
            if(nums[j] <= pivot) {
                swap(nums, ++p, j);
            }
        }
        swap(nums, left, p);
        return p;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

时间复杂度：$O(n)$

真题链接：字节跳动20230820笔试 https://codefun2000.com/p/P1474

提示：排序数组，分别去掉最大值或者最小值，然后让数组都变为中位数。

刚才的例题，顺带讨论了无序数组如何快速寻找中位数，如果是两个有序数组，如何快速找到其归并后的数组的中位数？

例题：[4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

分析：该解法用到了**二分查找**章节的知识。

假设两个数组的长度为$m,n$，假设有一条分割线，将两个数组分为左右两半部分。其中$nums1$有$i$个元素，$nums2$有$j$个元素。$i + j = \lceil (m+n)/2\rceil$。

对于$nums1$，左侧边界为$nums1[i-1]$，右侧边界为$nums1[i]$；

对于$nums2$，左侧边界为$nums2[j-1]$，右侧边界为$nums2[j]$；

如果满足$nums1[i-1]<nums2[j],nums2[j-1]<nums1[i]$，则此时左边界元素的最大值小于右边界元素的最小值。

元素个数为奇数的情况下，取$\max(nums1[i-1], nums2[j-1])$即可；

元素个数为偶数的情况下，取$(\max(nums1[i-1], nums2[j-1])+\min(nums1[i], nums2[j]))/2$即可。

使用二分查找寻找分界线的边界。

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if(m > n) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int half = m + n + 1 >> 1;
        int left = 0, right = m;
        while(left < right) {
            int i = left + right >> 1;
            int j = half - i;
            if(nums1[i] < nums2[j-1]) {  // 不用担心j-1<0越界。i在循环中，最大取到m-1，j最小取到1。
                left = i + 1;
            }else {
                right = i;
            }
        }
        int nums1Left = left == 0 ? Integer.MIN_VALUE: nums1[left - 1];
        int nums1right = left == m ? Integer.MAX_VALUE : nums1[left];
        int nums2left = half - left == 0 ? Integer.MIN_VALUE : nums2[half - left - 1];
        int nums2right = half - left == n ? Integer.MAX_VALUE : nums2[half - left];
        if((m + n) % 2 == 0) {
            return (Math.max(nums1Left, nums2left) + Math.min(nums1right, nums2right)) / 2.0;
        } else {
            return Math.max(nums1Left, nums2left);
        }
    }
}

```

时间复杂度：$O(\log \min(m,n))$，$m,n$为两个数组的长度。

再进一步思考一个问题：如果是数据流，如何快速寻找中位数？

例题：[295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

分析：维护一个最大堆和最小堆，其中，最大堆的最大元素小于最小堆中的最小元素。

```java
class MedianFinder {
    private int count;
    private Queue<Integer> maxHeap;
    private Queue<Integer> minHeap;

    public MedianFinder() {
        count = 0;
        maxHeap = new PriorityQueue<>((a, b) -> b - a);
        minHeap = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        count ++;
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());
        if ((count & 1) != 0) {
            maxHeap.add(minHeap.poll());
        }
    }
    
    public double findMedian() {
        return (count & 1) != 0 ? (double) maxHeap.peek() : (double) (maxHeap.peek() + minHeap.peek()) / 2;
    }
}
```

### 1.4 String

#### 1.4.1 Substring Matching

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

> 字符串哈希

例题：[1147. 段式回文](https://leetcode.cn/problems/longest-chunked-palindrome-decomposition/)

分析：字符串哈希的思想，是将字符串转换为哈希值，在判断字符串相等时，时间复杂度即从$O(n)$降低为$O(1)$。由于本题中只存在小写字母，可以将字符串看作是一个26进制的数，由于哈希值可能非常大，计算过程中需要进行取模。由于取模操作，当两个串哈希值相同时，还需要判断其字符串内容是否相同。

前向计算哈希：$ab \to abc$ 

$prehash = (prehash * 26 + c) \% mod$

后向计算哈希：$ba\to cba$

$posthash = (c * 26^{n-1}+posthash)\% mod$

```java
class Solution {
    private long mod = 10000_00007;
    public int longestDecomposition(String text) {
        long[] pow26 = new long[text.length()];
        pow26[0] = 1;
        for(int i = 1; i < pow26.length; i ++) {
            pow26[i] = pow26[i - 1] * 26 % mod;
        }
        return solve(text, 0, text.length() - 1, pow26);
    }

    private int solve(String s, int left, int right, long[] pow26) {
        if(left > right) {
            return 0;
        }
        long prehash = 0, posthash = 0;
        for(int i = left, j = right; i < j; i ++, j --) {
            prehash = (prehash * 26 + (s.charAt(i) - 'a')) % mod;
            posthash = ((s.charAt(j) - 'a') * pow26[right - j] + posthash) % mod;
            if(prehash == posthash && equals(s, left, i, j, right)) {
                return 2 + solve(s, i + 1, j - 1, pow26);
            }
        }
        return 1;
    }

    private boolean equals(String s, int l1, int r1, int l2, int r2) {
        for(int i = l1, j = l2; i <= r1 && j <= r2; i ++, j ++) {
            if(s.charAt(i) != s.charAt(j)) {
                return false;
            }
        }
        return true;
    }
}
```

时间复杂度：$O(n)$

例题：[1392. 最长快乐前缀](https://leetcode.cn/problems/longest-happy-prefix/)

```java
class Solution {
    public String longestPrefix(String s) {
        int n = s.length();
        long mod = 10000_00007;
        long[] pre = new long[n], post = new long[n], pow26 = new long[n];
        pow26[0] = 1;
        for(int i = 1; i < n; i ++) {
            pow26[i] = (pow26[i-1] * 26) % mod;
        }
        pre[0] = s.charAt(0) - 'a';
        for(int i = 1; i < n; i ++) {
            pre[i] = (pre[i-1] * 26 + s.charAt(i) - 'a') % mod;
        }
        post[n - 1] = s.charAt(n - 1) - 'a';
        for(int i = n - 2; i >= 0; i --) {
            post[i] = ((s.charAt(i) - 'a') * pow26[n - i - 1] + post[i + 1]) % mod;
        }
        for(int len = n - 1; len >= 1; len --) {
            if(pre[len - 1] == post[n - len] && equals(s, 0, len - 1, n - len, n - 1)) {
                return s.substring(0, len);
            }
        }
        return "";
    }

    private boolean equals(String s, int l1, int r1, int l2, int r2) {
        for(int i = l1, j = l2; i <= r1 && j <= r2; i ++, j ++) {
            if(s.charAt(i) != s.charAt(j)) {
                return false;
            }
        }
        return true;
    }
}
```

> Rabin Karp算法

Rabin Karp算法核心是基于滚动哈希。

首先计算出target串的哈希值thash，由于哈希值可能很大，需要进行取模操作。

假设source串为1234，target串为234

在source中计算长度为target串的子串哈希值，采用滚动的方式计算。

$123 = 1*10^2+2*10^1+3*10^0$

$234=2*10^2+3*10^1+4*10^0$

从$123$到$234$的滚动哈希计算过程：

本次循环结束时：shash = (shash - source.charAt(i - target.length() + 1) * power % mod + mod) % mod

此时 shash从$123$变为$23$

下次循环开始时：shash = (shash * base + source.charAt(i)) % mod;

此时shash从$23$变为$234$

```java
class Solution {
    public int strStr(String source, String target) {
        if(source.length() < target.length()) {
            return -1;
        }
        long thash = 0, mod = 10000_00007, base = 256;
        for(int i = 0; i < target.length(); i ++) {  // 计算target哈希值
            thash = (thash * base + target.charAt(i)) % mod;
        }
        long shash = 0, power = 1;
        for(int i = 0; i < target.length() - 1; i ++) {
            power = power * base % mod;
        }
        for(int i = 0; i < target.length() - 1; i ++) {
            shash = (shash * base + source.charAt(i)) % mod;
        }
        for(int i = target.length() - 1; i < source.length(); i ++) {
            shash = (shash * base + source.charAt(i)) % mod;
            if(shash == thash && equals(source, target, i - target.length() + 1)) {
                return i - target.length() + 1;
            }
            shash = (shash - source.charAt(i - target.length() + 1) * power % mod + mod) % mod;  // 防止取模出现负数
        }
        return -1;
    }

    private boolean equals(String source, String target, int start) {
        for(int i = 0; i < target.length(); i ++) {
            if(target.charAt(i) != source.charAt(i + start)) {
                return false;
            }
        }
        return true;
    }
}
```

> KMP算法

KMP算法核心是在于求解next数组，next[i]表示字符串[0..i]的最长border，定义为原字符串中既是前缀也是后缀的字符串（非空且不包含自身）。

例题：[1392. 最长快乐前缀](https://leetcode.cn/problems/longest-happy-prefix/)

```java
class Solution {
    public String longestPrefix(String s) {
        int n = s.length();
        int[] next = new int[n];
        next[0] = 0;
        for(int i = 1; i < n; i ++) {
            int a = next[i - 1];
            while(a > 0 && s.charAt(i) != s.charAt(a)) {
                a = next[a - 1];
            }
            if(s.charAt(i) == s.charAt(a)) {
                next[i] = a + 1;
            }
        }
        return s.substring(0, next[n-1]);
    }
}
```

KMP算法：假设判断字符串target是否在字符串source中出现。用指针i遍历source串，指针j遍历target串，当source[i]不等于target[j]时，此时j需要根据next数组计算的值进行回退，而i则不需要回退。

例题：[28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

```java
class Solution {
    public int strStr(String haystack, String needle) {
        int[] next = getNext(needle);
        int i = 0, j = 0;
        while(i < haystack.length()) {
            if(haystack.charAt(i) == needle.charAt(j)) {
                i ++;
                j ++;
                if(j == needle.length()) {
                    return i - j;
                }
            } else if(j > 0) {
                j = next[j - 1];
            } else {
                i ++;
            }
        }
        return -1;
    }

    private int[] getNext(String s) {
        int n = s.length();
        int[] next = new int[n];
        next[0] = 0;
        for(int i = 1; i < n; i ++) {
            int a = next[i - 1];
            while(a > 0 && s.charAt(i) != s.charAt(a)) {
                a = next[a - 1];
            }
            if(s.charAt(i) == s.charAt(a)) {
                next[i] = a + 1;
            }
        }
        return next;
    }
}
```

时间复杂度：$O(n+m)$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [187. 重复的DNA序列](https://leetcode.cn/problems/repeated-dna-sequences/) | 中等 |

#### 1.4.2 Palindrome String

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

>  双向扩展法

双向扩展法是一种能够快速计算回文子串的算法。其核心思想是从一个中心向左右两边扩展，当左右两边字符相同时。

例题：[647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)

示例：$i$的循环次数为$2*n - 1$，会依次遍历如下子串

| i    | l    | r    | 对应子串                        |
| ---- | ---- | ---- | ------------------------------- |
| 0    | 0    | 0    | 下标从0开始中心扩展的偶数回文串 |
| 1    | 0    | 1    | 下标从0开始中心扩展的奇数回文串 |
| 2    | 1    | 1    | 下标从1开始中心扩展的偶数回文串 |
| 3    | 1    | 2    | 下标从1开始中心扩展的奇数回文串 |

```java
class Solution {
    public int countSubstrings(String s) {
        int n = s.length();
        int count = 0;
        for(int i = 0; i < 2 * n - 1; i++) {
            int l = i / 2;
            int r = l + i % 2;
            while(l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                count ++;
                r ++;
                l --;
            }
        }
        return count;
    }
}
```

时间复杂度：$O(n^2)$

中心扩展法还可以结合动态规划进行考察

例题：[132. 分割回文串 II](https://leetcode.cn/problems/palindrome-partitioning-ii/)

```java
class Solution {
    public int minCut(String s) {
        int n = s.length();
        int[] dp = new int[n];
        Arrays.fill(dp, n);
        for(int i = 0; i < 2 * n - 1; i++) {
            int l = i / 2;
            int r = l + i % 2;
            while(l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                dp[r] = Math.min(dp[r], l == 0 ? 0 : dp[l-1] + 1);
                l --;
                r ++;
            }
        }
        return dp[n-1];
    }
}
```

练习题单：

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/) | 中等 |
| [2472. 不重叠回文子字符串的最大数目](https://leetcode.cn/problems/maximum-number-of-non-overlapping-palindrome-substrings/) | 困难 |

> 马拉车算法

马拉车算法是一种能在$O(n)$时间复杂度内查找一个字符串的最长回文子串的算法，在面试中几乎不可能涉及。

1. 解决回文串奇数和偶数的问题 

在每个字符串中间插入#，在两端插入^和$(不可能在字符串中出现的字符即可)，为了使扩展过程中到边界自动结束，得到字符串t。

 如果原字符串⻓度为偶数，则插入的#数量为奇数，总⻓度为奇数。

如果原字符串⻓度为奇数，则插入的#数量为偶数，总⻓度为偶数。

2. 记录中心扩展最大个数

|      | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   | 17   | 18   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| t    | ^    | #    | c    | #    | b    | #    | c    | #    | b    | #    | c    | #    | c    | #    | d    | #    | e    | #    | $    |
| p    | 0    | 0    | 1    | 0    | 3    | 0    | 5    | 0    | 3    | 0    | 1    | 2    | 1    | 0    | 1    | 0    | 1    | 0    | 0    |

数组p保存每个字符从中心能够扩展的最大个数，其取值刚好又对应着最长回文串的长度。例如p[6]=5，表示从c字符串可以向左扩展5个字符，向右扩展5个字符，即"#c#b#c#b#c#"，对应的回文串为"cbcbc"，长度也为5。偶数情况同样成立。

3. 求原字符串下标

假设p中取得最大中心扩展个数的下标为$i$，对应原字符串的开始下标为$\frac{i-p[i]}{2}$

4. 求解数组p

参考一下例题

例题：[5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

```java
class Solution {
    public String longestPalindrome(String s) {
        String t = preprocess(s);
        int n = t.length(), c = 0, r = 0;
        int[] p = new int[n];
        for(int i = 1; i < n - 1; i ++) {
            int iMirror = 2 * c - i;  // i关于中心c对称的点
            if(r > i) {
                p[i] = Math.min(r - i, p[iMirror]);
            } else {
                p[i] = 0;
            }
            while(t.charAt(i + 1 + p[i]) == t.charAt(i - 1 - p[i])) {
                p[i] ++;
            }
            if(i + p[i] > r) {
                c = i;
                r = i + p[i];
            }
        }
        int maxLen = 0, centerIndex = 0;
        for(int i = 1; i < n - 1; i ++) {
            if(p[i] > maxLen) {
                maxLen = p[i];
                centerIndex = i;
            }
        }
        int start = (centerIndex - maxLen) / 2;
        return s.substring(start, start + maxLen);
    }

    private String preprocess(String s) {
        StringBuilder sb = new StringBuilder("^");
        for(char c : s.toCharArray()) {
            sb.append("#").append(c);
        }
        sb.append("#$");
        return sb.toString();
    }
}
```

#### 1.4.3 Repeated Substring

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 低       |

例题：[459. 重复的子字符串](https://leetcode.cn/problems/repeated-substring-pattern/)

```java
class Solution {
    public boolean repeatedSubstringPattern(String s) {
        return (s + s).indexOf(s, 1) != s.length();
    }
}
```

以下例题是结合**区间DP**和上述例题知识点综合的题目。

例题：[471. 编码最短长度的字符串](https://leetcode.cn/problems/encode-string-with-shortest-length/)

分析：利用459题的思想，快速找到重复子串。

```java
class Solution {
    private String[][] dp;
    public String encode(String s) {
        int n = s.length();
        dp = new String[n][n]; 
        for(int i = n-1; i >= 0; i--) {
            for(int j = i; j < n; j++) {
                dp[i][j] = concat(s, i, j);
                if(j - i <= 3) {
                    continue;
                }
                for(int k = i; k < j; k++) {
                    String com = dp[i][k] + dp[k+1][j];
                    if(dp[i][j].length() > com.length()) {
                        dp[i][j] = com;
                    }
                }
            }
        }   
        return dp[0][n-1];
    }
    private String concat(String s, int i, int j) {
        s = s.substring(i, j+1);
        if(s.length() < 5) {
            return s;
        }
        int index = (s + s).indexOf(s, 1);
        if(index != s.length()) {
            int count = s.length() / index;
            return count + "[" + dp[i][i+index-1] + "]";
        }
        return s;
    }
}
```

给定一个字符串，如何求解其最长的重复子串？

例题：[1044. 最长重复子串](https://leetcode.cn/problems/longest-duplicate-substring/)

分析：可以采用二分+滚动哈希的思想进行求解

```java
class Solution {
    public String longestDupSubstring(String s) {
        int l = 0, r = s.length();
        String ret = "";
        while(l < r) {
            int mid = l + r + 1 >> 1;
            int start = check(s, mid);
            if(start == -1) {
                r = mid - 1;
            }else {
                l = mid;
                ret = s.substring(start, start + l);
            }
        }
        return ret;
    }

    private int check(String s, int len) {
        long hash = 0, base = 26, mod = 10000_00007, power = 1;
        for(int i = 0; i < len - 1; i ++) {
            power = power * base % mod;
        }
        for(int i = 0; i < len - 1; i ++) {
            hash = (hash * base + s.charAt(i) - 'a') % mod;
        }
        Map<Long, Integer> visited = new HashMap<>();   // 每一个hash值及其开始下标
        for(int i = len - 1; i < s.length(); i ++) {
            hash = (hash * base + s.charAt(i) - 'a') % mod;
            int start = i - len + 1;
            if(visited.containsKey(hash)) {
                int j = visited.get(hash);
                if(equals(s, start, j, len)) {
                    return start;
                }
            }
            visited.put(hash, start);
            hash = ((hash - (s.charAt(start) - 'a') * power) % mod + mod) % mod;
        }
        return -1;
    }

    private boolean equals(String s, int i, int j, int len) {
        for(int k = 0; k < len; k ++) {
            if(s.charAt(k+i) != s.charAt(k+j)) {
                return false;
            }
        }
        return true;
    }
}
```

## Part 2 Data Structure

### 2.1 Linked List

#### 2.1.1 Linked List and Recursion

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 低       |

例题：[24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null) {
            return head;
        }
        ListNode ret = head.next;
        head.next = swapPairs(ret.next);
        ret.next = head;
        return ret;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/) | 中等 |

#### 2.1.2 Linked List Inversion

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 低       |

例题：[206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

> 基于递归实现

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        ListNode ret = reverseList(next);
        next.next = head;
        head.next = null;
        return ret;
    }
}
```

> 基于循环实现

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, cur = head, next = null;
        while(cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```

思考：如果只反转链表前$n$个节点，需要怎么写代码？

```java
private ListNode reverseN(ListNode head, int n) {
    if(head == null) {
        return null;
    }
    if(n == 1) {
        succssor = head.next;
        return head;
    }
    ListNode next = head.next;
    ListNode ret = reverseN(next, n - 1);
    next.next = head;
    head.next = successor;
    return ret;
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/) | 中等 |
| [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/) | 困难 |

#### 2.1.3 Merge Linked List

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 低       |

例题：[21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

```java
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if(list1 == null || list2 == null) {
            return list1 == null ? list2 : list1;
        }
        if(list1.val <= list2.val) {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
    }
}
```

若要合并K个升序链表，如何解决？

例题：[23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

分析：根据数组归并的知识，采用优先队列实现

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        Queue<ListNode> queue = new PriorityQueue<>((a, b) -> a.val - b.val);
        for(ListNode node : lists) {
            if(node != null) {
                queue.offer(node);
            }
        }
        ListNode ret = new ListNode();
        ListNode p = ret;
        while(!queue.isEmpty()) {
            p.next = queue.poll();
            if(p.next.next != null) {
                queue.offer(p.next.next);
            }
            p = p.next;
        }
        return ret.next;
    }
}
```

时间复杂度：$O(kn\log k)$

合并K个升序链表还可以基于分治算法实现。

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return merge(lists, 0, lists.length - 1);
    }

    private ListNode merge(ListNode[] lists, int l, int r) {
        if(l >= r) {
            return l == r ? lists[l] : null;
        }
        int mid = l + r >> 1;
        return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    private ListNode mergeTwoLists(ListNode first, ListNode second) {
        if(first == null || second == null) {
            return first == null ? second : first;
        }
        if(first.val <= second.val) {
            first.next = mergeTwoLists(first.next, second);
            return first;
        }else {
            second.next = mergeTwoLists(first, second.next);
            return second;
        }
    }
}
```

时间复杂度：$O(kn\log k)$

#### 2.1.4 Fast and Slow Pointers

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 低       |

利用快慢指针一次扫描得到链表倒数第$k$个节点。

例题：[19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy, slow = dummy;
        for(int i = 0; i < n; i ++) {
            fast = fast.next;
        }
        while(fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        if(slow != null) {
            slow.next = slow.next.next;
        }
        return dummy.next;
    }
}
```

使用快慢指针寻找环

例题：[142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

分析：

设两指针fast，slow，一开始指向链表头。fast每次走两步，slow每次走一步。若循环过程中fast和slow未相遇，说明链表无环。

若链表有环，假设链表$a+b$个节点，其中链表头到链表入口有$a$个节点，链表环$b$个节点。第一次相遇时，假设fast指针走了$f$步，slow指针走了$s$步。有等式成立：

$f=2s$

$f=s+nb$

两式相减，得$s=nb$

让fast回到head，和slow再走$a$步，即到达环入口。$a$的值未知，但fast和slow下一次相遇时一定是在环入口。

```java
ublic class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode intersect = getIntersection(head);
        if(intersect == null) {
            return null;
        }
        ListNode p1 = head, p2 = intersect;
        while(p1 != p2) {
            p1 = p1.next;
            p2 = p2.next;
        }
        return p1;
    }

    private ListNode getIntersection(ListNode head) {
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow) {
                return slow;
            }
        }
        return null;
    }
}
```

使用快慢指针快速定位链表中部节点

奇数个节点的链表：返回正中间的节点

偶数个节点的链表：返回中间靠左的节点

```java
public ListNode firstHalf(ListNode head) {
    if(head == null) {
        return null;
    }
    ListNode fast = head, slow = head;
    while(fast.next != null && fast.next.next != null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    return slow;
}
```

例题：[143. 重排链表](https://leetcode.cn/problems/reorder-list/)

```java
lass Solution {
    public void reorderList(ListNode head) {
        if(head == null || head.next == null) {
            return;
        }
        ListNode mid = findMedium(head);
        ListNode head2 = reverse(mid.next);
        mid.next = null;
        while(head2 != null) {
            ListNode next1 = head.next;
            ListNode next2 = head2.next;
            head.next = head2;
            head2.next = next1;
            head = next1;
            head2 = next2;
        }
    }

    private ListNode findMedium(ListNode head) {
        ListNode fast = head, slow = head;
        while(fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    private ListNode reverse(ListNode head) {
        if(head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        ListNode ret = reverse(next);
        next.next = head;
        head.next = null;
        return ret;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/) | 简单 |

#### 2.1.5 Sort Linked List

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 低       |

归并排序是最适合链表排序的算法。

例题：[148. 排序链表](https://leetcode.cn/problems/sort-list/)

```java
class Solution {
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null) {
            return head;
        }
        ListNode fast = head, slow = head;
        while(fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode mid = slow.next;
        slow.next = null;
        ListNode first = sortList(head);
        ListNode second = sortList(mid);
        return merge(first, second);
    }

    private ListNode merge(ListNode first, ListNode second) {
        if(first == null || second == null) {
            return first == null ? second : first;
        }
        if(first.val <= second.val) {
            first.next = merge(first.next, second);
            return first;
        } else {
            second.next = merge(first, second.next);
            return second;
        }
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [147. 对链表进行插入排序](https://leetcode.cn/problems/insertion-sort-list/) | 简单 |

### 2.2 Stack 

#### 2.2.1 Classic Problems of Stack

##### 2.2.1.1 Calculator

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

例题：[772. 基本计算器 III](https://leetcode.cn/problems/basic-calculator-iii/)

本题是实现一个支持+，-，*，/和括号运算的基本计算器。

实现思路：采用两个栈，一个为数栈，一个为操作符栈。

遍历表达式，根据不同字符类型进行如下操作：

- 若当前表达式的字符为数字类型，则向后遍历，直到非数字类型字符或到达表达式末尾。将数字添加到数栈中。
- 若当前表达式的字符为操作符，即+，-，*，/，判断操作符栈顶的优先级和当前操作符优先级谁更高。若栈顶操作符优先级大于等于当前操作符优先级，进行计算操作。循环上述操作，直到条件不满足或操作数栈为空或栈顶操作符为左括号。将当前操作符入栈。+、-的优先级相同，\*，/的优先级比+，-更高。
- 若当前表达式的字符为左括号，直接入栈。
- 若当前表达式的字符为右括号，进行计算操作，直到操作数栈顶为左括号。

在计算操作时，每次从数栈中取出两个元素，从操作符栈中取出一个操作符，将计算后的结果放入栈中。

遍历完表达式后，只要操作数栈不为空，继续执行计算操作，返回数栈的栈顶元素即为最终计算结果。

当表达式的开头为负号时，或者左括号后有负号时，可以预先往数栈中加入元素0，避免进入异常分支。

```java
class Solution {
    private Deque<Integer> numStack = new ArrayDeque<>();
    private Deque<Character> opStack = new ArrayDeque<>();
    private static Map<Character, Integer> priority = new HashMap<>();
    static {
        priority.put('+', 1);
        priority.put('-', 1);
        priority.put('*', 2);
        priority.put('/', 2);
    }
    public int calculate(String s) {
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(Character.isDigit(c)) {
                int num = 0;
                while(i < s.length() && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    i ++;
                }
                i--;
                numStack.offerLast(num);
            }else if(isSign(c)) {
                while(!opStack.isEmpty() && opStack.peekLast() != '(' && priority.get(opStack.peekLast()) >= priority.get(c)) {
                    eval();
                }
                opStack.offerLast(c);
            }else if(c == '(') {
                opStack.offerLast(c);
            }else if(c == ')') {
                while(!opStack.isEmpty()) {
                    char op = opStack.peekLast();
                    if(op == '(') {
                        opStack.pollLast();
                        break;
                    }else {
                        eval();
                    }
                }
            }
        }
        while(!opStack.isEmpty()) {
            eval();
        }
        return numStack.pollLast();
    }
    private void eval() {
        int num1 = numStack.pollLast();
        int num2 = numStack.pollLast();
        char op = opStack.pollLast();
        if(op == '+') {
            numStack.offerLast(num1+num2);
        }else if(op == '-') {
            numStack.offerLast(num2-num1);
        }else if(op == '*') {
            numStack.offerLast(num1*num2);
        }else if(op == '/') {
            numStack.offerLast(num2/num1);
        }
    }
    private boolean isSign(char c) {
        return c == '+' || c == '-' || c == '*' || c == '/';
    }
}
```

再给出一道例题，体会这两道题的共同点。

例题：[1096. 花括号展开 II](https://leetcode.cn/problems/brace-expansion-ii/)

```java
class Solution {
    Deque<Character> opStack = new ArrayDeque<>();
    Deque<Set<String>> setStack = new ArrayDeque<>();
    public List<String> braceExpansionII(String expression) {
        for(int i = 0; i < expression.length(); i++) {
            char c = expression.charAt(i);
            Character pre = null;
            if(i > 0) {
                pre = expression.charAt(i-1);
            }
            if(Character.isLowerCase(c)) {
                Set<String> set = new TreeSet<>();
                set.add(c + "");
                setStack.add(set);
                if(pre != null && (Character.isLowerCase(pre) || pre == '}')) {
                    opStack.offerLast('*');
                }
            }else if(c == ',') {
                while(!opStack.isEmpty() && opStack.peekLast() == '*') {
                    eval();
                }
                opStack.offerLast('+');
            }else if(c == '{') {
                if(pre != null && (Character.isLowerCase(pre) || pre == '}')) {
                    opStack.offerLast('*');
                }
                opStack.offerLast('{');
            }else if(c == '}') {
                while(!opStack.isEmpty()) {
                    char op = opStack.peekLast();
                    if(op == '{') {
                        opStack.pollLast();
                        break;
                    }else {
                        eval();
                    }
                }
            }
        }
        while(!opStack.isEmpty()) {
            eval();
        }
        return new ArrayList<>(setStack.pollLast());
    }
    private void eval() {
        Set<String> second = setStack.pollLast();
        Set<String> first =  setStack.pollLast();
        char op = opStack.pollLast();
        if(op == '+') {
            first.addAll(second);
            setStack.offerLast(first);
        }else if(op == '*') {
            Set<String> set = new TreeSet<>();
            for(String f : first) {
                for(String s : second) {
                    set.add(f + s);
                }
            }
            setStack.offerLast(set);
        }
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [224. 基本计算器](https://leetcode.cn/problems/basic-calculator/) | 困难 |
| [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/) | 中等 |

##### 2.2.1.2 Parenthesis Matching

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |
例题：[32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

```java
class Solution {
    public int longestValidParentheses(String s) {
        int n = s.length(), ans = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for(int i = 0; i < n; i ++) {
            if(s.charAt(i) == '(') {
                stack.push(i);
            }else {
                stack.pop();
                if(stack.isEmpty()) {
                    stack.push(i);
                } else {
                    ans = Math.max(ans, i - stack.peek());
                }
            }
        }
        return ans;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1541. 平衡括号字符串的最少插入次数](https://leetcode.cn/problems/minimum-insertions-to-balance-a-parentheses-string/) | 中等 |
|                                                              |      |



#### 2.2.2 Monotonic Stack

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |

先从一个简单的问题出发，学习单调栈。

例题：https://www.nowcoder.com/practice/2a2c00e7a88a498693568cef63a4b7bb

采用暴力的做法，时间复杂度$O(n^2)$，显然不是最优的。使用单调栈，能够在$O(n)$时间内求解每一个元素左侧和右侧第一个比当前元素更小的位置。

对于本问题，首先维护一个严格单调递减栈（从栈顶往栈底看）。其中，栈中单调性的维护满足如下几点要求。

1. 所有元素都会入栈一次。
2. 若栈内没有元素，直接入栈。
3. 若栈内有元素，且当前元素严格大于栈顶元素，则入栈，否则弹出栈顶元素，直至栈为空或者当前元素严格大于栈顶元素。在出栈的过程中进行结算。
4. 栈中的元素为数组的下标而非数组的元素值。

给出单调栈的模版。

```java
// 返回二维数组ret，其中
// ret[i][0]表示第i个元素左侧第一个比arr[i]小的元素下标
// ret[i][1]表示第i个元素右侧第一个比arr[i]小的元素下标
public int[][] findFirstSmallerElement(int[] arr) {
    Stack<Integer> stack = new Stack<>();
    int[] left = new int[n], right = new int[n];
    for(int i = 0; i < n; i ++) {
        while(!stack.isEmpty() && arr[stack.peek()] >= arr[i]) {
            int j = stack.pop();
            // 栈顶元素严格大于栈底，所以左侧第一个比arr[j]小的元素下标一定为当前栈顶元素。
            left[j] = stack.isEmpty() ? -1 : stack.peek();
            // 此时arr[i] >= arr[j]，非严格大于，所以还需要答案修复阶段
            right[j] = i;
        }
        stack.push(i);  // 所有元素都需要入队一次
    }
    // 栈中还未结算的元素
    while(!stack.isEmpty()) {
        int j = stack.pop();
        left[j] = stack.isEmpty() ? -1 : stack.peek();
        right[j] = -1;  
    }
    // 答案修复
    for(int i = n - 2; i >= 0; i --) {
        if(right[i] != -1 && arr[i] == arr[right[i]]) {
            right[i] = right[right[i]];
        }
    }
}
```
时间复杂度：O(n)，采用一次遍历即可完成统计。

对于不包含重复元素的数组，可以不需要答案修复的过程，另外，对于有重复元素的数组，需要根据实际情况分析是否需要答案修复的过程。

模版代码，由于栈中保证了严格递减，所以left[j]的答案不需要修复，而right[j]需要答案修复。如果只需要求解右侧严格小于的第一个元素下标，可以将循环条件的等号去掉，即栈中的元素非严格递减，在结算时，right[j]的答案就无需修复了。

例题：[84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

分析：使用模版，找到每个元素向左向右能够扩展的最大宽度，乘上当前节点的高度即可。在实际代码书写时有一些技巧和细节。

1. 循环i==n，用于处理栈中还未出栈的元素的结算逻辑。
2. 栈提前填入元素-1，避免结算过程中对栈是否为空的判断。注意循环的条件为stack.size() > 1而非stack.isEmpty()。

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int n = heights.length, ans = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for(int i = 0; i <= n; i ++) {
            while(stack.size() > 1 && (i == n || heights[stack.peek()] >= heights[i])) {
                int j = stack.pop();
                int width = i - stack.peek() - 1;
                int height = heights[j];
                ans = Math.max(ans, width * height);
            }
            stack.push(i);
        }
        return ans;
    }
}
```

思考：对于相同元素，为何没有答案修复的过程？

看如下示例：[6,7,6,8,6,2]

在计算下标为0的6时，由于弹出规则，此时结算的面积为6*2=12，因为下标为2的6会让下标为0的6出栈并进行结算，宽度计算为2-(-1)-1=2。

在计算下标为2的6时，此时结算的面积为6*4=24，因为下标为4的6会让下标为2的6出栈并进行结算，下标为2的6左侧为-1，宽度计算为4-(-1)-1=4。

在计算下标为4的6时，此时结算的面积为6*5=30，因为下标为5的2会让下标为4的6出栈并进行结算，下标为4的6左侧为-1，宽度计算为5-(-1)-1=5。

单调栈还可以结合贡献法进行考察：

例题：[907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)

分析：使用模版计算以arr[j]为最小值最大能向左向右扩展多远的距离。对于j位置，以arr[j]为最小值向左的子数组共有j - stack.peek()个，以arr[j]为最小值向右的子数组有i - j个，根据乘法原理，arr[j]对答案的贡献即为arr[j]\*(j-stack.peek())\*(i-j)。

```java
class Solution {
    public int sumSubarrayMins(int[] arr) {
        long ans = 0;
        Stack<Integer> stack = new Stack<>();
        int n = arr.length, mod = 10000_00007;
        stack.push(-1);
        for(int i = 0; i <= n; i ++) {
            while(stack.size() > 1 && (i == n || arr[stack.peek()] >= arr[i])) {
                int j = stack.pop();
                ans += (long) arr[j] * (j - stack.peek()) * (i - j);
            }
            stack.push(i);
        }
        return (int) (ans % mod);
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点                         |
| ------------------------------------------------------------ | ---- | ------------------------------ |
| [496. 下一个更大元素 I](https://leetcode.cn/problems/next-greater-element-i/) | 简单 | 右侧更大                         |
| [503. 下一个更大元素 II](https://leetcode.cn/problems/next-greater-element-ii/) | 中等 | 右侧更大+环形数组                  |
| [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/) | 中等 | 右侧更大           
| [1019. 链表中的下一个更大节点](https://leetcode.cn/problems/next-greater-node-in-linked-list/) | 中等 | 右侧更大 + 链表             |          |
| [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/) | 困难 | 左右第一个更大
| [85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/) | 困难 | 左右第一个更小，建模为84题                     |
| [1856. 子数组最小乘积的最大值](https://leetcode.cn/problems/maximum-subarray-min-product/) | 中等 | 前缀和、单调栈、贡献法         |
| [2104. 子数组范围和](https://leetcode.cn/problems/sum-of-subarray-ranges/) | 中等 | 单调栈、贡献法                 |
| [1950. 所有子数组最小值中的最大值](https://leetcode.cn/problems/maximum-of-minimum-values-in-all-subarrays/) | 中等 | 单调栈、贡献法、思维           |
| [2281. 巫师的总力量和](https://leetcode.cn/problems/sum-of-total-strength-of-wizards/) | 困难 | 前缀和的前缀和、单调栈、贡献法 |

单调栈还有一种用法，用于维持求解答案的可能性。单调栈中所有对象按照一定单调性进行组织，某个对象进入单调栈时，会从栈顶开始依次淘汰栈中对后续答案求解没有帮助的对象。

例题：[962. 最大宽度坡](https://leetcode.cn/problems/maximum-width-ramp/)

分析：下标$i,j$，满足$i<j,s[i]<s[j]$，要最大化$j-i$的值。

对于左端点$j$，从左到右遍历下标，维护一个单调递减的栈（仅入队），因为靠右且更大的元素对答案没有贡献。

对于右端点$i$，倒序遍历，一旦$s[j]>s[i]$，则更新答案并出栈，因为栈顶元素已经计算出了最大宽度，后续的答案不会更优。

```java
class Solution {
    public int maxWidthRamp(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < nums.length; i ++) {
            if(stack.isEmpty() || nums[stack.peek()] > nums[i]) {
                stack.push(i);
            }
        }
        int ans = 0;
        for(int r = nums.length - 1; r > 0; r --) {
            while(!stack.isEmpty() && nums[r] >= nums[stack.peek()]) {
                ans = Math.max(ans, r - stack.pop());
            }
        }
        return ans;
    }
}
```

时间复杂度：$O(n)$

思考：本题还有一种非常巧妙的思路，将数组按照值排序，转换为买卖股票问题。时间复杂度：$O(n \log n)$，读者可以尝试求解。

练习题单

| 题号                                                         | 难度 | 知识点 |
| ------------------------------------------------------------ | ---- | ------ |
| [1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/) | 中等 | 前缀和 |

从以下例题，学习如何用数组模拟栈和单调栈。

例题：[321. 拼接最大数](https://leetcode.cn/problems/create-maximum-number/)

本题采用了用数组模拟栈的写法，其中，维护了一个大小为$k$的单调递减栈。
```java
class Solution {
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int[] ret = new int[k];
        int m = nums1.length, n = nums2.length;
        for(int i = Math.max(0, k - n); i <= k && i <= m; i ++) {
            // 枚举两个数组的长度
            int[] arr = merge(maxArr(nums1, i), maxArr(nums2, k-i), k);
            if(greaterThan(arr, 0, ret, 0)) {
                ret = arr;
            }
        }
        return ret;
    }

    private int[] maxArr(int[] nums, int k) {
        int n = nums.length;
        int[] ret = new int[k];
        // j表示当前栈中有多少元素，获取栈顶元素: ret[j-1]
        for(int i = 0, j = 0; i < nums.length; i ++) {
            while(n - i + j > k && j > 0 && nums[i] > ret[j-1]) {  // 能够凑够k个元素，栈非空，当前元素比栈顶更大，出栈
                j --;
            }
            if(j < k) {
                ret[j++] = nums[i];
            }
            /** 等效写法
            while(stack.size() + n - i > k && !stack.isEmpty() && stack.peek() < nums[i]) {
                stack.pop();
            }
            if(stack.size() < k) {
                stack.push(nums[i]);
            }
            **/
        }
        return ret;
    }

    private int[] merge(int[] nums1, int[] nums2, int k) {
        int[] ret = new int[k];
        for(int i = 0, j = 0, t = 0; t < k; t ++) {
            ret[t] = greaterThan(nums1, i, nums2, j) ? nums1[i++] : nums2[j++];
        }
        return ret;
    }

    private boolean greaterThan(int[] nums1, int i, int[] nums2, int j) {
        while(i < nums1.length && j < nums2.length && nums1[i] == nums2[j]) {
            i ++;
            j ++;
        }
        return j == nums2.length || (i < nums1.length && nums1[i] > nums2[j]);
    }
}
```


### 2.3 Queue

#### 2.3.1 Monotonic Queue

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- |  -------- |
| 低       | 低       | 了解 |

单调队列一个最经典的应用就是求滑动窗口的最大值。

例题：[239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

分析：本题中，维护一个单调不增的队列，队列左侧出队（窗口过期时），右侧入队。单调队列中，既可以存储元素值，又可以存储下标值。

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> queue = new ArrayDeque<>();
        int n = nums.length;
        int[] ret = new int[n - k + 1];
        for(int i = 0; i < k - 1; i ++) {
            while(!queue.isEmpty() && nums[queue.peekLast()] < nums[i]) {
                queue.pollLast();
            }
            queue.offerLast(i);
        }
        for(int l = 0, r = k - 1; r < n; r ++, l ++) {
            while(!queue.isEmpty() && nums[queue.peekLast()] < nums[r]) {
                queue.pollLast();
            }
            queue.offerLast(r);
            ret[l] = nums[queue.peekFirst()];
            if(l == queue.peekFirst()) {
                queue.pollFirst();
            }
        }
        return ret;
    }
}
```
时间复杂度：$O(n)$

练习题单

| 题号                                                         | 难度 | 知识点          |
| ------------------------------------------------------------ | ---- | --------------- |
| [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/) | 中等 | 滑动窗口+单调队列维护最大最小值 |

和单调栈一样，单调队列也可以用于淘汰掉对答案没有贡献的元素。

例题：[862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

分析：设原数组的前缀和数组为$s$，遍历到$s[i]$时，队首下标$j$，满足$s[i]-s[j] \ge k$，记录答案后，无论$s[i]$右侧数字是大是小，得到的子数组都不可能比$i-j$更短，此时$s[j]$需要弹出(pollFirst)。

队尾下标$j$，若$s[i]\le s[j]$，若存在$s[t] - s[j] \ge k,(t>i>j)$，则必然有$s[t] - s[j] \ge k$，且$t-i$更小。所以$s[j]$对答案没有贡献，需要弹出(pollLast)。

```java
class Solution {
    public int shortestSubarray(int[] nums, int k) {
        int n = nums.length, ans = n + 1;
        long[] pre = new long[n+1];
        for(int i = 0; i < n; i ++) {
            pre[i+1] = pre[i] + nums[i];
        }
        Deque<Integer> queue = new ArrayDeque<>();
        for(int i = 0; i <= n; i ++) {
            long sum = pre[i];
            while(!queue.isEmpty() && sum - pre[queue.peekFirst()] >= k) {
                ans = Math.min(ans, i - queue.pollFirst());
            }
            while(!queue.isEmpty() && pre[queue.peekLast()] >= sum) {
                queue.pollLast();
            }
            queue.offerLast(i);
        }
        return ans < n + 1 ? ans : -1;
    }
}
```

思考：如果原数组中没有负数，是否有更简单的做法？

提示：学习**滑动窗口**章节的知识点。

回顾单调栈962最大宽度坡问题，再深入理解单调栈与单调队列。

#### 2.3.2 Priority Queue

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 高       |

优先队列底层基于堆数据结构实现，可以在$\log k$时间让元素按照定义的优先级入队、出队。

以下列举了优先队列的一些经典应用：

| 应用                                       |
| ------------------------------------------ |
| TOP K问题                                  |
| 数据流的中位数（见数组中位数章节）         |
| 多路归并（见数组归并章节）                 |
| 最小生成树Prim算法（见图论最小生成树章节） |
| 最短路径Dijkstra算法（见图论最短路径章节） |
| 贪心策略                                   |

优先队列可以在$O(\log k)$时间范围内求解**TOP K**问题。

对于**TOP K**大问题，应该选择最小堆，而非最大堆。

例题：[347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Long> freqMap = Arrays.stream(nums).boxed().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        Queue<Integer> queue = new PriorityQueue<>(Comparator.comparingLong(freqMap::get));
        for(int num : freqMap.keySet()) {
            queue.offer(num);
            if(queue.size() > k) {
                queue.poll();
            }
        }
        return queue.stream().mapToInt(e -> e).toArray();
    }
}
```

时间复杂度：$O(n\log k)$

思考如下问题。

1. 基于优先队列实现**TOP K**问题相比于排序算法的优势？

排序算法时间复杂度为$O(n\log n)$，$k$较小时，基于优先队列实现的性能更优。

2. 和基于划分实现**SELECT K**的区别？

基于划分实现**SELECT K**，能在$O(n)$时间内寻找第$K$小/大的元素，但返回的**TOP K**是无序的。

基于优先队列实现**TOP K**问题返回的**TOP K**是有序的（逆序），并且能处理数据流的**TOP K**问题。

除了**TOP K**问题，在探究一下优先队列的其他应用。

例题：[407. 接雨水 II](https://leetcode.cn/problems/trapping-rain-water-ii/)

分析：$water[i][j]$表示$(i,j)$位置接水后的高度。

能接水的方块满足两个条件：

1. 该方块不为最外层方块
2. 该方块比四周相邻方块要低

$water[i][j]=\max(height[i][j],\min(water[i-1][j],water[i+1][j],water[i][j-1],water[i][j+1]))$

使用优先队列，记录结算的$water[i][j]$。由于最外层方块不能接水，所以$water[i][j]=height[i][j]$，将最外层方块全部放入优先队列。$water[i][j]$越小，优先级越高。

可以从动态规划的角度理解该问题，通过优先队列，确保当前状态依赖的前置状态都被计算过。外层方块相当于初始状态。

```java
class Solution {
    public int trapRainWater(int[][] heightMap) {
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
        int m = heightMap.length, n = heightMap[0].length;
        boolean[][] visited = new boolean[m][n];
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                if(i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    queue.offer(new int[]{i, j, heightMap[i][j]});
                    visited[i][j] = true;
                }
            }
        }
        int res = 0;
        int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        while(!queue.isEmpty()) {
            int[] cur = queue.poll();
            for(int[] d : dir) {
                int x = cur[0] + d[0];
                int y = cur[1] + d[1];
                if(0 <= x && x < m && 0 <= y && y < n && !visited[x][y]) {
                    res += Math.max(cur[2], heightMap[x][y]) - heightMap[x][y];
                    queue.offer(new int[]{x, y, Math.max(cur[2], heightMap[x][y])});
                    visited[x][y] = true;
                }
            }
        }
        return res;
    }
}
```

时间复杂度：$O(mn\log mn)$

例题：[1383. 最大的团队表现值](https://leetcode.cn/problems/maximum-performance-of-a-team/)

分析：先按照效率降序排序，遍历数组efficiency，枚举效率值中的最小值。维护一个speed数组的最大堆，其容量最大为k，累加最大堆中的元素，二者相乘即可。

```java
class Solution {
    public int maxPerformance(int n, int[] speed, int[] efficiency, int k) {
        int[][] pair = new int[n][2];
        for(int i = 0; i < n; i ++) {
            pair[i][0] = speed[i];
            pair[i][1] = efficiency[i];
        }
        Arrays.sort(pair, (a, b) -> Integer.compare(b[1], a[1]));
        Queue<Integer> queue = new PriorityQueue<>();
        long sum = 0, ans = 0;
        for(int i = 0; i < n; i ++) {
            queue.offer(pair[i][0]);
            int min = pair[i][1];
            sum += pair[i][0];
            ans = Math.max(ans, sum * min);
            if(queue.size() == k) {
                sum -= queue.poll();
            }
        }
        return (int)(ans % 10000_00007);
    }
}
```

思考：如果最多$k$名工程师，改为恰好$k$名工程师，又该如何解决？

提示：最大堆的容量必须为$k$。

| 题号                                                         | 难度 | 知识点          |
| ------------------------------------------------------------ | ---- | --------------- |
| [2542. 最大子序列的分数](https://leetcode.cn/problems/maximum-subsequence-score/) | 中等 | 排序，TOP K问题 |



### 2.4 Tree

#### 2.4.1 Tree and Recursion

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 高       |

从一道例题深入理解树的递归结构

例题：[110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

分析：自顶向下的递归，先计算左右子树的高度，再递归判断左右子树是否平衡。

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        if(root == null) {
            return true;
        }
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);
        if(Math.abs(leftHeight - rightHeight) > 1) {
            return false;
        }
        return isBalanced(root.left) && isBalanced(root.right);
    }

    private int height(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);
        return Math.max(leftHeight, rightHeight) + 1;
    }
}
```

时间复杂度：最坏$O(n^2),n$为二叉树节点的个数，此时二叉树退化为链表。平均时间复杂度为$O(n\log n)$，求解树的高度需要$\log n$ 的时间。上述解法的缺陷在于，对于同一个节点，获取高度的方法会被重复调用。

自底向上的递归（后序遍历），每个节点height方法只会调用一次。

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return height(root) >= 0;
    }

    private int height(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);
        if(leftHeight == -1 || rightHeight == -1 || Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        }
        return Math.max(leftHeight, rightHeight) + 1;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [100. 相同的树](https://leetcode.cn/problems/same-tree/)     | 简单 |
| [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/) | 简单 |
| [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/) | 简单 |
| [111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/) | 简单 |
| [222. 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/) | 中等 |

现讨论树的路径和问题。

例题：[437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)

在树中写算法，最重要的是定义清晰递归语义。请仔细理解pathSum和rootSum的递归语义。

```java
lass Solution {
    public int pathSum(TreeNode root, int targetSum) {  // 以root为根节点的二叉树中节点值之和等于targetSum的路径数目，root可以不选。 
        if(root == null) {
            return 0;
        }
        int ans = rootSum(root, targetSum);
        ans += pathSum(root.left, targetSum);
        ans += pathSum(root.right, targetSum);
        return ans;
    }

    public int rootSum(TreeNode root, long targetSum) { // 以root为根节点往下的路径中路径和等于targetSum的路径数目，root必选。 
        if(root == null) {
            return 0;
        }
        int ans = root.val == targetSum ? 1 : 0;
        ans += rootSum(root.left, targetSum - root.val);
        ans += rootSum(root.right, targetSum - root.val);
        return ans;
    }

}
```

时间复杂度$O(n^2)$

还可以结合前缀和知识对时间复杂度进行优化。

```java
class Solution {
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> map = new HashMap<>(); // key为Long，因为有恶心的测试用例
        map.put(0L, 1);
        return dfs(root, targetSum, 0, map);
    }

    private int dfs(TreeNode root, int targetSum, long sum, Map<Long, Integer> map) {
        if(root == null) {
            return 0;
        }
        sum += root.val;
        int ans = 0;
        ans += map.getOrDefault(sum - targetSum, 0);
        map.put(sum, map.getOrDefault(sum, 0) + 1);
        ans += dfs(root.left, targetSum, sum, map);
        ans += dfs(root.right, targetSum, sum, map);
        map.put(sum, map.get(sum) - 1);  // 访问完节点后回撤
        return ans;
    }
}
```

时间复杂度：$O(n)$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [112. 路径总和](https://leetcode.cn/problems/path-sum/)      | 简单 |
| [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/) | 中等 |

#### 2.4.2 Properties of Trees
| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

树的性质：

- 树是联通无环图
- 树的节点数 = 树的边数 + 1
- 去掉任何一条边，树会变得不连通
- 给定$n$个节点，能够构成$\frac{C_{2n}^n}{n+1}$种不同的二叉树

例题：[95. 不同的二叉搜索树 II](https://leetcode.cn/problems/unique-binary-search-trees-ii/)

```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int left, int right) {
        List<TreeNode> list = new ArrayList<>();
        if(left == right) {
            TreeNode node = new TreeNode(left);
            list.add(node);
            return list;
        } else if(left > right) {
            list.add(null);
            return list;
        }
        for(int i = left; i <= right; i ++) {
            List<TreeNode> leftList = generateTrees(left, i-1);
            List<TreeNode> rightList = generateTrees(i+1, right);
            for(TreeNode l : leftList) {
                for(TreeNode r : rightList) {
                    TreeNode root = new TreeNode(i);
                    root.left = l;
                    root.right = r;
                    list.add(root);
                }
            }
        }
        return list;
    }
}
```

根据95题的解题过程，思考如下问题：给定$n$个节点，能够构成多少种不同的二叉树？如何总结公式？

例题：[96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int j = 2; j <= n; j ++) {
            for(int i = 1; i <= j; i ++) {
            // 对应95题
            // left => 1, right => j
            //  List<TreeNode> leftList = generateTrees(left, i-1); => dp[i-1]
            //  List<TreeNode> rightList = generateTrees(i+1, right); => dp[j-i]
                dp[j] += dp[i - 1] * dp[j - i];
            }
        }
        return dp[n];
    }
}
```

也可以用组合数计算

```java
class Solution {
    public int numTrees(int n) {
        return (int)(comb(2 * n, n)/ (n + 1));
    }

    private long[][] res = new long[67][67];

    public long comb(int n, int m) {
        if(m == 0 || m == n) {
            return 1;
        }
        if(res[n][m] != 0) {
            return res[n][m];
        }
        return res[n][m] = comb(n-1,m) + comb(n-1,m-1);
    }
}
```

设$G(n)=\sum_{i=1}^nG(i-1)·G(n-i)$，$G(n)$成为卡塔兰数，其递推公式如下：

$\begin{cases}C_0=1\\ C_{n+1}=\frac{2(2n+1)}{n+1}C_n \end{cases}$

下题利用完全二叉树的性质解答。

例题：[222. 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/)

```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        if(leftHeight == rightHeight) {  // 左右子树高度相同，左子树一定是满二叉树
            return (1 << leftHeight) + countNodes(root.right);
        }  // 左子树比右子树高1，右子树一定是满二叉树
        return (1 << rightHeight) + countNodes(root.left);
    }

    private int getHeight(TreeNode root) {
        if(root == null) {
            return 0;
        }
        return 1 + getHeight(root.left);
    }
}
```

#### 2.4.3 Tree Traversal

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 高       |

> 有根树的遍历

对于以节点形式给出的树，正常遍历即可。

先序遍历：基于递归实现，先遍历根节点，再遍历左子树、右子树。(根左右)

中序遍历：基于递归实现，先遍历左子树，再遍历根节点、右子树。(左根右)

后序遍历：基于递归实现，先遍历左子树，再遍历右子树、根节点。(左右根)

层序遍历：基于队列实现，按层从左到右遍历。

探究1：非递归实现

先序遍历的非递归实现：

```java
public List<Integer> preOrder(TreeNode root) {
    Stack<TreeNode> stack = new Stack<>();
    List<Integer> ret = new ArrayList<>();
    if(root == null) {
        return ret;
    }
    stack.push(root);
    while(!stack.isEmpty()) {
        TreeNode cur = stack.pop();
        if(cur.right != null) {
            stack.push(cur.right);
        }
        if(cur.left != null) {
            stack.push(cur.left);
        }
        ret.add(cur.val);
    }
    return ret;
}
```

中序遍历的非递归实现：

```java
public List<Integer> inOrder(TreeNode root) {
    Stack<TreeNode> stack = new Stack<>();
    TreeNode cur = root;
    List<Integer> ret = new ArrayList<>();
    while(!stack.isEmpty() || cur != null) {
        while(cur != null) {
            stack.push(cur);
            cur = cur.left;
        }
        cur = stack.pop();
        ret.add(cur.val);
        cur = cur.right;
    }
    return ret;
}
```

后序遍历的非递归实现：

```java
public List<Integer> postOrder(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    if(root != null) {
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode c = null;
        while(!stack.isEmpty()) {
            c = stack.peek();
            if(c.left != null && root != c.left && root != c.right) {
                stack.push(c.left);
            }else if(c.right != null && root != c.right) {
                stack.push(c.right);
            }else {
                ret.add(stack.pop().val);
                root = c;
            }
        }
        return ret;
    }
}
```

探究2：还原二叉树

先序遍历和中序遍历的序列能够唯一确定一颗二叉树。

后序遍历和中序遍历的序列能够唯一确定一颗二叉树。

先序遍历和后序遍历的结果不能确定一颗二叉树。

例题：[105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return dfs(preorder, inorder, 0, preorder.length, 0, inorder.length);
    }

    private TreeNode dfs(int[] preorder, int[] inorder, int lp, int rp, int li, int ri) {
        if(lp >= rp) {
            return null;
        }
        int value = preorder[lp];
        TreeNode root = new TreeNode(value);
        int i;
        for(i = li; i <= ri; i ++) {
            if(inorder[i] == value) {
                break;
            }
        }
        root.left = dfs(preorder, inorder, lp + 1, lp + i - li + 1, li, i);
        root.right = dfs(preorder, inorder, lp + i - li + 1, rp, i + 1, ri);
        return root;
    }
}
```

探究3：序列化与反序列化

例题：[449. 序列化和反序列化二叉搜索树](https://leetcode.cn/problems/serialize-and-deserialize-bst/)

```java
public class Codec {

    public String serialize(TreeNode root) {
        if(root == null) {
            return "$";
        }
        return root.val + "#" + serialize(root.left) + "#" + serialize(root.right);
    }

    public TreeNode deserialize(String data) {
        String[] split = data.split("#");
        Queue<String> queue = new LinkedList<>();
        for(String s : split) {
            queue.offer(s);
        }
        return deserialize(queue);
    }

    private TreeNode deserialize(Queue<String> queue) {
        if(queue.isEmpty()) {
            return null;
        }
        String str = queue.poll();
        if(str.equals("$")) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.valueOf(str));
        root.left = deserialize(queue);
        root.right = deserialize(queue);
        return root;
    }
}
```

探究4：基于递归实现层序遍历

例题：[102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        dfs(root, 0);
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    private void dfs(TreeNode root, int level) {
        if(root == null) {
            return;
        }
        if(level == ret.size()) {
            ret.add(new ArrayList<>());
        }
        ret.get(level).add(root.val);
        dfs(root.left, level + 1);
        dfs(root.right, level + 1);
    }
}
```

探究5：前序遍历序列判断

例题：[255. 验证前序遍历序列二叉搜索树](https://leetcode.cn/problems/verify-preorder-sequence-in-binary-search-tree/)

分析：使用单调递减栈。

```java
class Solution {
    public boolean verifyPreorder(int[] preorder) {
        Stack<Integer> stack = new Stack<>();
        int pre = Integer.MIN_VALUE;
        for(int i = 0; i < preorder.length; i ++) {
            if(preorder[i] < pre) {
                return false;
            }
            while(!stack.isEmpty() && preorder[i] > stack.peek()) {
                // 数组元素大于栈顶元素，说明往右子树走了。找到右子树根节点，之前左子树全部弹出。
                pre = stack.pop();
            }
            stack.push(preorder[i]);
        }
        return true;
    }
}
```

探究6：二分搜索树中序遍历性质

二分搜索树中序遍历的序列是有序的。可以根据这一性质求解一系列问题。

例题：[272. 最接近的二叉搜索树值 II](https://leetcode.cn/problems/closest-binary-search-tree-value-ii/)

```java
class Solution {
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        List<Integer> list = new ArrayList<>();
        dfs(root, target, k, list);
        return list;
    }

    private void dfs(TreeNode root, double target, int k, List<Integer> list) {
        if(root == null) {
            return;
        }
        dfs(root.left, target, k, list);
        if(list.size() < k) {
            list.add(root.val);
        }else if(Math.abs(list.get(0) - target) > Math.abs(root.val - target)) {
            list.remove(0);
            list.add(root.val);
        }else {
            return;
        }
        dfs(root.right, target, k, list);
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [501. 二叉搜索树中的众数](https://leetcode.cn/problems/find-mode-in-binary-search-tree/) | 简单 |
| [173. 二叉搜索树迭代器](https://leetcode.cn/problems/binary-search-tree-iterator/) | 中等 |
| [99. 恢复二叉搜索树](https://leetcode.cn/problems/recover-binary-search-tree/) | 中等 |

> 无根树的遍历

对于以边形式给出的树，通常是先建立图。由于树的连通性，可以从任意节点遍历，同时用一个变量parent记录当前节点的父节点，避免遍历相邻节点时重复遍历。

```java
private void dfs(int v, int p, List<Integer>[] graph) {
    for(int u : graph[v]) {
        if(u != p) {
            dfs(u, v, graph);
        }
    }
}
```

#### 2.4.4 The Diameter of the Tree

树的直径，定义为树中任意两个节点之间最长路径的长度。

先从一个简单的例题入手。

例题：[543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

分析：本题代码量虽然少，且难度标为简单，但实际上有一定思维量。注意递归语义。

```java
class Solution {
    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return ans - 1;
    }

    private int ans = 0;

    private int dfs(TreeNode root) {  // 递归语义：以root为根节点的最大深度
        if(root == null) {
            return 0;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);
        ans = Math.max(ans, left + right + 1);
        return Math.max(left, right) + 1;
    }
}
```

思考1：如果要寻找最大路径和，应该怎么求解？

例题：[124. 二叉树中的最大路径和
](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

分析：其本质就是带权的二叉树直径问题。

```java
class Solution {
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }

    private int ans = Integer.MIN_VALUE;

    private int dfs(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int left = Math.max(dfs(root.left), 0);
        int right = Math.max(dfs(root.right), 0);
        ans = Math.max(ans, left + right + root.val);
        return Math.max(left, right) + root.val;
    }
}
```

思考2：如果不是二叉树，而是一颗无根树，如何找出路径最长的两个叶子节点？

1. 以任意节点p出发，利用DFS或BFS寻找最长路径的终点x。
2. 从x出发，找到最长路径的终点y。
3. x到y之间的路径即为最长路径。

证明参考算法导论习题解答9-1。

有了以上知识的铺垫，可以求解以下例题。

例题：[310. 最小高度树
](https://leetcode.cn/problems/minimum-height-trees/)

分析：找出树中距离最远的两个节点，求出路径。路径的中点即为树的根节点。

```java
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> res = new ArrayList<>();
        if(n == 1) {
            res.add(0);
            return res;
        }
        List<Integer>[] g = new List[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for(int[] edge : edges) {
            g[edge[0]].add(edge[1]);
            g[edge[1]].add(edge[0]);
        }
        int[] pre = new int[n];
        Arrays.fill(pre, -1);
        int x = findLongestNode(0, pre, g);
        int y = findLongestNode(x, pre, g);
        List<Integer> path = new ArrayList<>();
        pre[x] = -1;
        while(y != -1) {
            path.add(y);
            y = pre[y];
        }
        int m = path.size();
        if(m % 2 == 0) {
            res.add(path.get(m / 2 - 1));
        }
        res.add(path.get(m / 2));
        return res;
    }

    public int findLongestNode(int u, int[] pre, List<Integer>[] g) {
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[g.length];
        queue.offer(u);
        visited[u] = true;
        int ans = -1;
        while(!queue.isEmpty()) {
            ans = queue.poll();
            for(int v : g[ans]) {
                if(!visited[v]) {
                    visited[v] = true;
                    pre[v] = ans;
                    queue.offer(v);
                }
            }
        }
        return ans;
    }
}
```

在动态规划章节，还会继续介绍**换根DP**的解法

在图章节，还会继续介绍**拓扑排序**的解法。


#### 2.4.5 Multiple Tree

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |
例题：[440. 字典序的第K小数字](https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/)

分析：数字的字典序，可以看作是一颗十叉树。

```java
class Solution {
    public int findKthNumber(int n, int k) {
        int cur = 1;
        k --;
        while(k > 0) {
            int steps = getSteps(cur, n);
            if(steps <= k) {
                k -= steps;
                cur ++;     // 跳到兄弟节点
            } else {
                cur *= 10;  // 跳到孩子节点
                k --;
            }
        }
        return cur;
    }

    private int getSteps(int cur, long n) {  // 计算当前节点为根节点的孩子个数
        int steps = 0;
        long first = cur;
        long last = cur;
        while(first <= n) {
            steps += Math.min(last, n) - first + 1;
            first = first * 10;
            last = last * 10 + 9;
        }
        return steps;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [386. 字典序排数](https://leetcode.cn/problems/lexicographical-numbers/) | 中等 |

#### 2.4.6 Lowest Common Ancestor

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

例题：[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null) {
            return root;
        }
        return left == null ? right : left;
    }
}
```
时间复杂度：$O(n)$

如果是二分搜索树，时间复杂度可以优化到$O(h)$，$h$在最坏情况下等于$n$。

例题：[235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int x = root.val;
        if(p.val < x && q.val < x) {
            return lowestCommonAncestor(root.left, p, q);
        }
        if(p.val > x && q.val > x) {
            return lowestCommonAncestor(root.right, p, q);
        }
        return root;
    }
}
```

如果树的结构未知，只有边的信息，则需要使用树上倍增求解最近公共祖先问题。

#### 2.4.7 Tree Doubling

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

例题：[1483. 树节点的第 K 个祖先](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/)

分析：

$pa[x][i]$表示节点$x$的第$2^i$个祖先节点，若祖先节点不存在，则$pa[x][i]=-1$。

$pa[x][0]=parent[x]$：父节点

$pa[x][1]=pa[pa[x][0]][0]$：爷爷节点

$pa[x][i+1]=pa[pa[x][i]][i]$

```java
class TreeAncestor {
    private int[][] pa;

    public TreeAncestor(int n, int[] parent) {
        int m = 32 - Integer.numberOfLeadingZeros(n);
        pa = new int[n][m];
        for(int i = 0; i < n; i ++) {
            pa[i][0] = parent[i];
        }
        for(int i = 0; i < m - 1; i ++) {
            for(int x = 0; x < n; x ++) {
                int p = pa[x][i];
                pa[x][i + 1] = p < 0 ? -1 : pa[p][i];
            }
        }
    }
    
    public int getKthAncestor(int node, int k) {
        for(; k > 0 && node != -1; k &= k - 1) {
            node = pa[node][Integer.numberOfTrailingZeros(k)];
        }
        return node;
    }
}
```
时间复杂度：$O(n\log n)$预处理，$O(\log k)$回答每次询问。

模版：利用树上倍增求解最近公共祖先

设节点$i$深度为$depth[i]$，通过一次DFS预处理计算出来。

假设$depth[x]\le depth[y]$，若不满足，可以将$y$更新为$y$的$depth[y]-depth[x]$个祖先节点。

如果$x=y$，则$x$即为LCA，否则$x,y$一起向上跳。

```java
class TreeAncestor {
    private int[] depth;
    private int[][] pa;

    public TreeAncestor(int[][] edges) {
        int n = edges.length + 1;
        int m = 32 - Integer.numberOfLeadingZeros(n);
        List<Integer> g[] = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int[] e : edges) { 
            int x = e[0], y = e[1];
            g[x].add(y);
            g[y].add(x);
        }

        depth = new int[n];
        pa = new int[n][m];
        dfs(g, 0, -1);

        for (int i = 0; i < m - 1; i++) {
            for (int x = 0; x < n; x++) {
                int p = pa[x][i];
                pa[x][i + 1] = p < 0 ? -1 : pa[p][i];
            }
        }
    }

    private void dfs(List<Integer>[] g, int x, int fa) {
        pa[x][0] = fa;
        for (int y : g[x]) {
            if (y != fa) {
                depth[y] = depth[x] + 1;
                dfs(g, y, x);
            }
        }
    }

    public int getKthAncestor(int node, int k) {
        for (; k > 0; k &= k - 1) {
            node = pa[node][Integer.numberOfTrailingZeros(k)];
        }
        return node;
    }

    public int getLCA(int x, int y) {
        if (depth[x] > depth[y]) {
            int tmp = y;
            y = x;
            x = tmp;
        }
        // x, y位于同一高度
        y = getKthAncestor(y, depth[y] - depth[x]);
        if (y == x) {
            return x;
        }
        for (int i = pa[x].length - 1; i >= 0; i--) {
            int px = pa[x][i], py = pa[y][i];
            if (px != py) {  // LCA在pa[x][i]上面
                x = px;
                y = py;
            }
        }
        return pa[x][0];
    }
}
```
例题：[2836. 在传球游戏中最大化函数值](https://leetcode.cn/problems/maximize-value-of-function-in-a-ball-passing-game/)

```java
class Solution {
    public long getMaxFunctionValue(List<Integer> receiver, long k) {
        int n = receiver.size();
        int m = 64 - Long.numberOfLeadingZeros(k);
        int[][] pa = new int[m][n];
        long[][] sum = new long[m][n];
        for(int i = 0; i < n; i ++) {
            pa[0][i] = receiver.get(i);
            sum[0][i] = receiver.get(i);
        }
        for(int i = 0; i < m - 1; i ++) {
            for(int x = 0; x < n; x ++) {
                int p = pa[i][x];
                pa[i+1][x] = pa[i][p];
                sum[i+1][x] = sum[i][x] + sum[i][p];
            }
        }
        long ans = 0;
        for(int i = 0; i < n; i ++) {
            long s = i;
            int x = i;
            for(long j = k; j > 0; j &= j - 1) {
                int ctz = Long.numberOfTrailingZeros(j);
                s += sum[ctz][x];
                x = pa[ctz][x];
            }
            ans = Math.max(ans, s);
        }
        return ans;
    }
}
```


#### 2.4.8 Tree Data Structure

##### 2.4.8.1 Segment Tree

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |

线段树，如果区间有$n$个元素，数组来表示需要多少节点？

如果$n=2^k$，只需要$2n$的空间。

如果$n=2^k+1$，则需要$4n$的空间。

$n=4$，此时需要$7$个空间，开辟$2n=8$个空间。

| 0        | 1        | 2        | 3    | 4    | 5    | 6    |
| -------- | -------- | -------- | ---- | ---- | ---- | ---- |
| A[0...3] | A[0...1] | A[2...3] | A[0] | A[1] | A[2] | A[3] |

$n=5$，由于采用满二叉树的方式存储，此时需要$15$个空间，开辟$4n=20$个空间。

| 0       | 1        | 2        | 3        | 4    | 5    | 6    | 7    | 8    | 10~14 |
| ------- | -------- | -------- | -------- | ---- | ---- | ---- | ---- | ---- | ------- |
| A[0...4] | A[0...2] | A[3...4] | A[0...1] | A[2] | A[3] | A[4] | A[0] | A[1] | null |

如果$n$很大，可以通过离散化的方式解决，将值域映射到较小的空间去。这一前提是提前知晓所有询问。

否则，无法进行离散化，只能采用动态开点的方式，在插入操作和查询操作时进行开点。

本节内容参考题解：https://leetcode.cn/problems/range-module/solutions/1612955/by-lfool-eo50/

线段树可以解决以下三类经典问题：

- 数字之和：总数字之和 = 左区间数字之和 + 右区间数字之和

- 最大公因数gcd：总区间gcd = gcd(左区间gcd, 右区间gcd)

- 最大值max：总最大值max = max(左区间max, 右区间max)

线段树是一颗近似的完全二叉树，可以基于数组和链表实现。基于数组实现的线段树空间浪费比较大。本节介绍基于链表实现的线段树。

定义Node节点
```java
class Node {
    Node left, right;
    int val;
    int add; // 懒惰更新标记
}
```

若具体区间范围给定，可以以如下方式构建线段树。
```java
public void buildTree(Node node, int start, int end) { 
    if(start == end) {
        node.val = arr[start];
        return;
    }
    int mid = start + end >> 1;
    buildTree(node.left, start, mid);
    buildTree(node.right, mid + 1, end);
    pushUp(node);
}
private void pushDown(Node node) {
    node.val = node.left.val + node.right.val;
}
```

若题目未给出具体范围，只有数据的取值范围，通常采用动态开点。动态开点通常在查询或更新的时候建立。

应用一：维护区间最大值（动态开点+覆盖）

例题：[699. 掉落的方块](https://leetcode.cn/problems/falling-squares/)

```java
class Solution {
    public List<Integer> fallingSquares(int[][] positions) {
        List<Integer> ret = new ArrayList<>();
        for(int[] p : positions) {
            int cur = query(root, 1, n, p[0], p[0] + p[1] - 1);
            update(root, 1, n, p[0], p[0] + p[1] - 1, cur + p[1]);
            ret.add(root.val);
        }
        return ret;
    }

    class Node {
        Node left, right;
        int val, add;
    }

    int n = (int)1e9;
    Node root = new Node();

    public void update(Node node, int start, int end, int l, int r, int val) {
        if(l <= start && end <= r) {
            node.val = val;
            node.add = val;
            return;
        }
        pushDown(node);
        int mid = (start + end) >> 1;
        if (l <= mid) {
            update(node.left, start, mid, l, r, val);
        }
        if (r > mid) {
            update(node.right, mid + 1, end, l, r, val);
        }
        node.val = Math.max(node.left.val, node.right.val);
    }

    public int query(Node node, int start, int end, int l, int r) {
        if(l <= start && end <= r) {
            return node.val;
        }
        pushDown(node);
        int mid = start + end >> 1, ans = 0;
        if(l <= mid) {
            ans = Math.max(ans, query(node.left, start, mid, l, r));
        }
        if(r > mid) {
            ans = Math.max(ans, query(node.right, mid + 1, end, l, r));
        }
        return ans;
    }

    private void pushDown(Node node) {
        if(node.left == null) {
            node.left = new Node();
        }
        if(node.right == null) {
            node.right = new Node();
        }
        if(node.add == 0) {
            return;
        }
        node.left.val = node.add;
        node.right.val = node.add;
        node.left.add = node.add;
        node.right.add = node.add;
        node.add = 0;
    }

}
```
应用二：维护区间最大值（动态开点+加减）

例题：[732. 我的日程安排表 III](https://leetcode.cn/problems/my-calendar-iii/)

```java
class MyCalendarThree {

    public MyCalendarThree() {
        root = new Node();
    }

    class Node {
        Node left, right;
        int val, add;
    }

    private int n = (int)1e9;
    private Node root;
    
    public int book(int startTime, int endTime) {
        update(root, 0, n, startTime, endTime - 1, 1);
        return root.val;
    }

    public void update(Node node, int start, int end, int l, int r, int val) {
        if(l <= start && end <= r) {
            node.val += val;
            node.add += val;
            return;
        }
        pushDown(node);
        int mid = start + end >> 1;
        if(l <= mid) {
            update(node.left, start, mid, l, r, val);
        }
        if(r > mid) {
            update(node.right, mid + 1, end, l, r, val);
        }
        node.val = Math.max(node.left.val, node.right.val);
    }

    private void pushDown(Node node) {
        if(node.left == null) {
            node.left = new Node();
        }
        if(node.right == null) {
            node.right = new Node();
        }
        if(node.add == 0) {
            return;
        }
        node.left.val += node.add;
        node.right.val += node.add;
        node.left.add += node.add;
        node.right.add += node.add;
        node.add = 0;
    }
}
```

练习题单

| 题单                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [731. 我的日程安排表 II](https://leetcode.cn/problems/my-calendar-ii/) | 中等 |
| [729. 我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/) | 中等 |

应用三：区间覆盖（动态开点+布尔值覆盖）

例题：[715. Range 模块](https://leetcode.cn/problems/range-module/)

```java
class RangeModule {

    public RangeModule() {
        root = new Node();
    }

    class Node {
        Node left, right;
        boolean cover;
        int add;
    }

    private Node root;
    private int n = (int)1e9;

    private void update(Node node, int start, int end, int l, int r, int val) {
        if(l <= start && end <= r) {
            node.cover = val == 1;
            node.add = val;
            return;
        }
        int mid = start + end >> 1;
        pushDown(node, mid - start + 1, end - mid);
        if(l <= mid) {
            update(node.left, start, mid, l, r, val);
        }
        if(r > mid) {
            update(node.right, mid + 1, end, l, r, val);
        }
        node.cover = node.left.cover && node.right.cover;
    }

    private void pushDown(Node node, int leftNum, int rightNum) {
        if(node.left == null) {
            node.left = new Node();
        }
        if(node.right == null) {
            node.right = new Node();
        }
        if(node.add == 0) {
            return;
        }
        node.left.cover = node.add == 1;
        node.right.cover = node.add == 1;
        node.left.add = node.add;
        node.right.add = node.add;
        node.add = 0;
    }

    private boolean query(Node node, int start, int end, int l, int r) {
        if(l <= start && end <= r) {
            return node.cover;
        }
        int mid = start + end >> 1;
        pushDown(node, mid - start + 1, end - mid);
        boolean ans = true;
        if(l <= mid) {
            ans &= query(node.left, start, mid, l, r);
        }
        if(r > mid) {
            ans &= query(node.right, mid + 1, end, l, r);
        }
        return ans;
    }
    
    public void addRange(int left, int right) {
        update(root, 1, n, left, right - 1, 1);
    }
    
    public boolean queryRange(int left, int right) {
        return query(root, 1, n, left, right - 1);
    }
    
    public void removeRange(int left, int right) {
        update(root, 1, n, left, right - 1, -1);
    }
}
```

应用四：区间和（数组+点更新覆盖）

例题：[307. 区域和检索 - 数组可修改](https://leetcode.cn/problems/range-sum-query-mutable/)

```java
class NumArray {

    private int[] arr;
    private int[] tree;

    public NumArray(int[] nums) {
        int n = nums.length;
        arr = new int[n];
        tree = new int[4*n];
        for(int i = 0; i < n; i ++) {
            arr[i] = nums[i];
        }
        buildTree(0, 0, n - 1);
    }

    private void buildTree(int treeIndex, int l, int r) {
        if(l == r) {
            tree[treeIndex] = arr[l];
            return;
        }
        int mid = l + r >> 1;
        int leftTree = 2 * treeIndex + 1, rightTree = 2 * treeIndex + 2;
        buildTree(leftTree, l, mid);
        buildTree(rightTree, mid + 1, r);
        tree[treeIndex] = tree[leftTree] + tree[rightTree];
    }
    
    public void update(int index, int val) {
        arr[index] = val;
        update(0, 0, arr.length - 1, index, val);
    }

    private void update(int treeIndex, int l, int r, int index, int val) {
        if(l == r) {
            tree[treeIndex] = val;
            return;
        }
        int mid = l + r >> 1, left = treeIndex * 2 + 1, right = treeIndex * 2 + 2;
        if(index > mid) {
            update(right, mid + 1, r, index, val);
        }else {
            update(left, l, mid, index, val);
        }
        tree[treeIndex] = tree[left] + tree[right];
    }
    
    public int sumRange(int left, int right) {
        return sumRange(0, 0, arr.length - 1, left, right);
    }

    private int sumRange(int treeIndex, int l, int r, int left, int right) {
        if(l == left && r == right) {
            return tree[treeIndex];
        }
        int mid = l + r >> 1;
        int leftTree = 2 * treeIndex + 1, rightTree = 2 * treeIndex + 2;
        if(left >= mid + 1) {
            return sumRange(rightTree, mid + 1, r, left, right);
        } else if(right <= mid) {
            return sumRange(leftTree, l, mid, left, right);
        } else {
            return sumRange(leftTree, l, mid, left, mid) + sumRange(rightTree, mid + 1, r, mid + 1, right);
        }
    }
}
```

应用五：区间和（动态开点+区间加减）

模版

```java
public class SegmentTreeDynamic {
    class Node {
        Node left, right;
        int val, add;
    }
    private int N = (int) 1e9;
    private Node root = new Node();
    public void update(Node node, int start, int end, int l, int r, int val) {
        if (l <= start && end <= r) {
            node.val += (end - start + 1) * val;
            node.add += val;
            return;
        }
        int mid = (start + end) >> 1;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) {
            update(node.left, start, mid, l, r, val);
        }
        if (r > mid) {
            update(node.right, mid + 1, end, l, r, val);
        }
        pushUp(node);
    }
    public int query(Node node, int start, int end, int l, int r) {
        if (l <= start && end <= r) {
            return node.val;
        }
        int mid = (start + end) >> 1, ans = 0;
        pushDown(node, mid - start + 1, end - mid);
        if (l <= mid) {
            ans += query(node.left, start, mid, l, r);
        }
        if (r > mid) {
            ans += query(node.right, mid + 1, end, l, r);
        }
        return ans;
    }
    private void pushUp(Node node) {
        node.val = node.left.val + node.right.val;
    }
    private void pushDown(Node node, int leftNum, int rightNum) {
        if (node.left == null) {
            node.left = new Node();
        }
        if (node.right == null) {
            node.right = new Node();
        }
        if (node.add == 0) {
            return ;
        }
        node.left.val += node.add * leftNum;
        node.right.val += node.add * rightNum;
        node.left.add += node.add;
        node.right.add += node.add;
        node.add = 0;
    }
}
```



##### 2.4.8.2 Trie
| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 低       |

从一道例题学习字典树。

例题：[211. 添加与搜索单词 - 数据结构设计](https://leetcode.cn/problems/design-add-and-search-words-data-structure/)

```java
class WordDictionary {

    public WordDictionary() {
        root = new Node();
    }

    class Node {
        Map<Character, Node> next;
        boolean isWord;
        Node() {
            next = new HashMap<>();
        }
    }

    Node root;
    
    public void addWord(String word) {
        Node cur = root;
        for(char c : word.toCharArray()) {
            cur.next.putIfAbsent(c, new Node());
            cur = cur.next.get(c);
        }
        cur.isWord = true;
    }
    
    public boolean search(String word) {
        return dfs(word, root, 0);
    }

    private boolean dfs(String word, Node node, int start) {
        if(start == word.length()) {
            return node.isWord;
        }
        char c = word.charAt(start);
        if(c == '.') {
            for(char ch : node.next.keySet()) {
                if(dfs(word, node.next.get(ch), start + 1)) {
                    return true;
                }
            }
            return false;
        } else {
            if(!node.next.containsKey(c)) {
                return false;
            }
            return dfs(word, node.next.get(c), start + 1);
        }
    }
}

```

利用字典树可以进行剪枝：

例题：[212. 单词搜索 II](https://leetcode.cn/problems/word-search-ii/)
```java
class Solution {

    private Node root = new Node();
    private Set<String> set = new HashSet<>();
    private int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    private int m, n;
    private boolean[][] visited;

    public List<String> findWords(char[][] board, String[] words) {
        m = board.length;
        n = board[0].length;
        visited = new boolean[m][n];
        for(String word : words) {
            addWord(word);
        }
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                if(!visited[i][j]) {
                    dfs(board, root, i, j);
                }
            }
        }
        return new ArrayList<>(set);
    }

    private void dfs(char[][] board, Node node, int i, int j) {
        if(!node.next.containsKey(board[i][j])) {
            return;
        }
        char ch = board[i][j];
        Node cur = node.next.get(ch);
        if(!cur.word.equals("")) {
            set.add(cur.word);
        }
        visited[i][j] = true;
        for(int[] d : dir) {
            int x = d[0] + i, y = d[1] + j;
            if(0 <= x && x < m && 0 <= y && y < n && !visited[x][y]) {
                dfs(board, cur, x, y);
            }
        }
         visited[i][j] = false;
    }

    class Node {
        String word;
        Map<Character, Node> next;
        Node() {
            next = new HashMap<>();
            word = "";
        }
    }
    
    private void addWord(String word) {
        Node cur = root;
        for(char c : word.toCharArray()) {
            cur.next.putIfAbsent(c, new Node());
            cur = cur.next.get(c);
        }
        cur.word = word;
    }
}
```
时间复杂度：$O(m·n·4^{10})$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/) | 中等 |
| [1032. 字符流](https://leetcode.cn/problems/stream-of-characters/) | 困难 |
| [677. 键值映射](https://leetcode.cn/problems/map-sum-pairs/) | 中等 |
| [472. 连接词](https://leetcode.cn/problems/concatenated-words/) | 困难 |
| [745. 前缀和后缀搜索](https://leetcode.cn/problems/prefix-and-suffix-search/) | 困难 |



##### 2.4.8.3 Union Find

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 中       | 中       | 建议掌握 |

并查集是一种简单高效的数据结构，用于判断两个节点是否连接，核心是两个操作：

- union(p, q) : 将p节点和q节点进行合并
- isConnected(p, q) : 判断p节点和q节点是否连接

先给出最基本的模版代码。

模版

```java
public class UnionFind {
    // 假设有若干连通分量，每一个连通分量都用一个唯一ID表示。可以用连通分量中的根节点的ID表示
    private int[] parent;  // 节点的父节点ID
    private int count;     // 联通分量的个数
    public UnionFind(int size) {
        parent = new int[size];
        for(int i = 0; i < size; i ++) {
            parent[i] = i;  // 初始化，每一个节点的父节点指向自身，每个节点均为根节点
        }
        count = size;
    }
    private int find(int p) {  // 查找p节点的根节点ID
        if(p != parent[p]) {  // 只要不指向自身，说明不是根节点
            parent[p] = find(parent[p]);  // 路径压缩，降低树高
        }
        return parent[p];
    }
    public boolean isConnected(int p, int q) {
        // 两个节点如果集合ID相同，则说明属于同一个连通分量
        return find(p) == find(q);
    }
    public int getCount() {  // 计算连通分量
        return count;
    }
    public void union(int p, int q) {
        int pRoot = find(p);
        int qRoot = find(q);
        if(pRoot == qRoot) {
            return;
        }
        count --;  // 连通后，连通分量减一
        // pRoot指向qRoot
        parent[pRoot] = qRoot;
    }
}
```
用一个例子解释find时的路径压缩：

find之前的parent数组情况：

| parent | 0    | 1    | 2    | 3    | 4    |
| ------ | ---- | ---- | ---- | ---- | ---- |
|        | 0    | 0    | 1    | 2    | 3    |

$0$为根节点，$4\to3\to2\to1\to0$

find之后

| parent | 0    | 1    | 2    | 3    | 4    |
| ------ | ---- | ---- | ---- | ---- | ---- |
|        | 0    | 0    | 0    | 0    | 0    |

0为根节点，$1,2,3,4$统一指向0

时间复杂度：$O(\log^* n)$，可以近乎当作$O(1)$。

并查集有如下经典应用。

- 求解联通分量个数
- 维护每个连通分量中节点个数
- 动态判断两个节点是否属于同一集合
- 环检测

例题：[947. 移除最多的同行或同列石头](https://leetcode.cn/problems/most-stones-removed-with-same-row-or-column/)

分析：遍历石头，合并每个石头所在的行和列。最终每个连通分量中保留一个石头即可。注意：在计算连通分量时，采用set统计不同的find值的个数，而非模版中给出的count每次合并后减一的方式。思考一下为什么。

```java
class Solution {
    public int removeStones(int[][] stones) {
        parent = IntStream.range(0, 20002).toArray();  // 取决于数据范围，行列最大值不超过10000。
        for(int[] stone : stones) { 
            union(stone[0], stone[1] + 10001);
        }
        Set<Integer> set = new HashSet<>();
        for(int[] stone : stones) {
            set.add(find(stone[0]));
        }
        return stones.length - set.size();
    }

    private int[] parent;

    private int find(int p) {
        if(parent[p] != p) {
            parent[p] = find(parent[p]);
        }
        return parent[p];
    }

    public void union(int p, int q) {
        int pRoot = find(p);
        int qRoot = find(q);
        if(pRoot != qRoot) {
            parent[pRoot] = qRoot;
        }
    }
}
```
模版中只维护了parent数组，如果需要知道每个连通分量中节点的个数，还可以维护一个size数组，见例题。

例题：[924. 尽量减少恶意软件的传播](https://leetcode.cn/problems/minimize-malware-spread/)

分析：在initial的两个节点，如果属于同一个连通分量，则删除其中任意一个节点都无法避免感染。使用并查集维护节点连通性，同时用size记录每个连通分量的大小。尽量删除size较大的节点。
```java
class Solution {
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int m = graph.length;
        UnionFind uf = new UnionFind(m);
        for(int i = 0; i < m; i ++) {
            for(int j = i + 1; j < m; j ++) {
                if(graph[i][j] == 1) {
                    uf.union(i, j);
                }
            }
        }
        int[] count = new int[m];
        for(int i : initial) {
            count[uf.find(i)] ++;
        }
        int ans = -1, maxSize = -1;
        for(int i : initial) {
            int root = uf.find(i);
            if(count[root] == 1) { 
                int size = uf.getSize(root);
                if(size > maxSize) {
                    maxSize = size;
                    ans = i;
                } else if(size == maxSize && i < ans) {
                    ans = i;
                }
            }
        }
        if(ans == -1) {
            ans = Arrays.stream(initial).min().getAsInt();
        }
        return ans;
    }

    class UnionFind {
        private int[] parent;
        private int[] size;
        public UnionFind(int n) {
            parent = new int[n];
            size = new int[n];
            for(int i = 0; i < n; i ++) {
                parent[i] = i;
                size[i] = 1;
            }
        }
        private int find(int p) {
            if(p != parent[p]) {
                parent[p] = find(parent[p]);
            }
            return parent[p];
        }
        public boolean isConnected(int p, int q) {
            return find(p) == find(q);
        }
        public int getSize(int p) {
            return size[find(p)];
        }
        public void union(int p, int q) {
            int pRoot = find(p);
            int qRoot = find(q);
            if(pRoot == qRoot) {
                return;
            }
            // size较小的根指向size更大的根
            if(size[pRoot] < size[qRoot]) {
                parent[pRoot] = qRoot;
                size[qRoot] += size[pRoot];
            }else {
                parent[qRoot] = pRoot;
                size[pRoot] += size[qRoot];
            }
        }
    }
}
```
除了可以维护size数组，还可以根据实际问题维护不同的信息，如下例题是并查集又一巧妙的应用。

例题：[399. 除法求值](https://leetcode.cn/problems/evaluate-division/)

```java
class Solution {
    private int[] parent;
    private double[] weight;
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, Integer> map = new HashMap<>();
        // 用map为每一个字符串赋予一个从0开始递增的ID
        for(List<String> equation : equations) {
            String first = equation.get(0), second = equation.get(1);
            map.putIfAbsent(first, map.size());
            map.putIfAbsent(second, map.size());
        }
        int n = map.size();
        parent = new int[n];
        weight = new double[n];
        // weight[i] = values(i) / values(parent[i]);
        for(int i = 0; i < n; i++) {
            parent[i] = i;
            weight[i] = 1.;
        }
        for(int i = 0; i < equations.size(); i++) {
            List<String> equation = equations.get(i);
            String first = equation.get(0), second = equation.get(1);
            union(map.get(first), map.get(second), values[i]);
        }
        double[] ret = new double[queries.size()]; 
        for(int i = 0; i < queries.size(); i++) {
            List<String> query = queries.get(i);
            String first = query.get(0), second = query.get(1);
            if(!map.containsKey(first) || !map.containsKey(second)) {
                ret[i] = -1.0; 
            }else {
                ret[i] = divide(map.get(first), map.get(second));
            }
        }
        return ret;
    }
    private double divide(int p, int q) {
        if(find(p) != find(q)) {
            return -1.0;
        }
        // weight[p] = values(p) / values(pRoot);
        // weight[q] = values(q) / values(qRoot);
        return weight[p] / weight[q];
    }
    private int find(int p) {  // 因为有路径压缩，所以需要更新weight
        if(p != parent[p]) {
            int origin = parent[p];
            parent[p] = find(origin);
            // weight[p] = values(p) / values(origin)
            // weight[origin] = values(origin) / values(pRoot);
            weight[p] *= weight[origin];
            // weight[p] = values(p) / values(pRoot);
        }
        return parent[p];
    }
    private void union(int p, int q, double v) {
        int pRoot = find(p);
        int qRoot = find(q);
        if(pRoot == qRoot) {
            return;
        }
        parent[pRoot] = qRoot;
        // weight[q] = values(q) / values(qRoot);
        // weight[p] = values(p) / values(pRoot);
        // v = values(p) / values(q);
        weight[pRoot] = weight[q] * v / weight[p];
        // weight[pRoot] = values(pRoot) / values(qRoot);
    }
}
```
真题链接：字节跳动20230903笔试 https://codefun2000.com/p/P1537/

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/) | 中等 |
| [959. 由斜杠划分区域](https://leetcode.cn/problems/regions-cut-by-slashes/) | 中等 |
| [2503. 矩阵查询可获得的最大分数](https://leetcode.cn/problems/maximum-number-of-points-from-grid-queries/) | 困难 |
| [305. 岛屿数量 II](https://leetcode.cn/problems/number-of-islands-ii/) | 困难 |
| [547. 省份数量](https://leetcode.cn/problems/number-of-provinces/) | 中等 |
| [1697. 检查边长度限制的路径是否存在](https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/) | 困难 |
| [765. 情侣牵手](https://leetcode.cn/problems/couples-holding-hands/) | 困难 |
| [839. 相似字符串组](https://leetcode.cn/problems/similar-string-groups/) | 困难 |
| [2092. 找出知晓秘密的所有专家](https://leetcode.cn/problems/find-all-people-with-secret/) | 困难 |

##### 2.4.8.4 Huffman Tree

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

讲解哈夫曼树之前，先定义如下两个概念：

- 节点的带权路径长度：从树的根到该节点的路径长度(经过的边数)与该节点上权值的乘积
- 树的带权路径长度：树中所有叶子结点的带权路径长度之和

在含有$n$个带权叶子结点的二叉树中带权路径长度最小的二叉树称为哈夫曼树，即最优二叉树。

假设有节点，权值分别是1，2，2，3，7。哈弗曼树的构造过程如下：

每次选择两颗节点权值最小的树作为新节点的左右子树，新节点的权值置为左右子树上根节点的权值之和。

依次类推，直到产生根节点。

每个初始节点最终都成为叶子节点，且权值越小的节点到根节点的路径长度越长。

例题：[1167. 连接棒材的最低费用](https://leetcode.cn/problems/minimum-cost-to-connect-sticks/)

```java
class Solution {
    public int connectSticks(int[] sticks) {
        int total = 0;
        Queue<Integer> pq = new PriorityQueue<>();
        for (int stick : sticks) {
            pq.offer(stick);
        }
        while (pq.size() > 1) {
            int stick1 = pq.poll();
            int stick2 = pq.poll();
            int cost = stick1 + stick2;
            total += cost;
            pq.offer(stick1 + stick2);
        }
        return total;
    }
}
```

哈夫曼树的一大经典应用是哈夫曼编码。

固定长度编码：每个字符用相等长度的二进制位表示

$A – 00$

$B – 01$

$C – 10$

$D – 11$

假设有$10$个$A$，$8$个$B$，$80$个$C$，$2$个$D$，

需要二进制长度为$(80+10+8+2)*2=200$bit

哈夫曼编码是一种可变长度编码，将字符出现的频数看作权值，构造哈夫曼树。

根据哈夫曼编码的结果

A - 10

B - 111

C - 0

D - 110

需要二进制长度为$80*1+10*2+2*3+8*3=130$bit

##### 2.4.8.5 Treap

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

树堆是一种比较简易的平衡树实现，可以收下模版。

例题：[1649. 通过指令创建有序数组](https://leetcode.cn/problems/create-sorted-array-through-instructions/)

```java
class Solution {
    public int createSortedArray(int[] instructions) {
        Treap treap = new Treap();
        long price = 0;
        for(int i : instructions) {
            treap.insert(i);
            int[] rank = treap.rank(i);
            price = (price + Math.min(rank[0] - 1, treap.getSize() - rank[1])) % 10000_00007;
        }
        return (int)price;
    }
}
class Treap {
    private static class TreeNode {
        long value;
        int priority;
        int count;
        int size;
        TreeNode left;
        TreeNode right;
        TreeNode(long value, int priority) {
            this.value = value;
            this.priority = priority;
            this.count = 1;
            this.size = 1;
        }
        TreeNode leftRotate() {
            int preSize = size;
            int curSize = (left == null ? 0 : left.size) + (right.left == null ? 0 : right.left.size) + count;
            TreeNode root = right;
            right = root.left;
            root.left = this;
            this.size = curSize;
            root.size = preSize;
            return root;
        }
        TreeNode rightRotate() {
            int preSize = size;
            int curSize = (right == null ? 0 : right.size) + (left.right == null ? 0 : left.right.size) + count;
            TreeNode root = left;
            left = root.right;
            root.right = this;
            this.size = curSize;
            root.size = preSize;
            return root;
        }
    }
    private TreeNode root;
    private final Random random;
    public Treap() {
        this.random = new Random();
    }
    public int getSize() {
        return root == null ? 0 : root.size;
    }
    public void insert(long x) {
        root = insert(root, x);
    }
    private TreeNode insert(TreeNode root, long x) {
        if (root == null)
            return new TreeNode(x, random.nextInt());
        root.size ++;
        if (x < root.value) {
            root.left = insert(root.left, x);
            if (root.left.priority > root.priority) {
                root = root.rightRotate();
            }
        } else if (x > root.value) {
            root.right = insert(root.right, x);
            if (root.right.priority > root.priority) {
                root = root.leftRotate();
            }
        } else {
            root.count ++;
        }
        return root;
    }
    public long lowerBound(long x) { //第一个大于等于x的数(从小到大排序)
        long ret = Long.MAX_VALUE;
        TreeNode node = root;
        while (node != null) {
            if (node.value == x) {
                return x;
            } else if (node.value > x) {
                ret = node.value;
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return ret;
    }
    public long upperBound(long x) { //第一个大于x的数(从小到大排序)
        long ret = Long.MAX_VALUE;
        TreeNode node = root;
        while (node != null) {
            if (node.value > x) {
                ret = node.value;
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return ret;
    }
    public int[] rank(long x) { //返回x的排名，从1开始。返回数组ret，ret[0]表示第一个x的rank，ret[1]表示最后一个x的rank。
        TreeNode node = root;
        int ans = 0;
        while (node != null) {
            if (node.value > x) {
                node = node.left;
            } else {
                ans += (node.left == null ? 0 : node.left.size) + node.count;
                if (x == node.value) {
                    return new int[]{ans - node.count + 1, ans};
                }
                node = node.right;
            }
        }
        return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE};
    }
    public void delete(int val) {
        root = delete(root, val);
    }
    private TreeNode delete(TreeNode root, int value) {
        if (root == null)
            return null;
        if (root.value > value) {
            root.left = delete(root.left, value);
        } else if (root.value < value) {
            root.right = delete(root.right, value);
        } else {
            if (root.count > 1) {
                root.count --;
                root.size --;
                return root;
            }
            if (root.left == null || root.right == null) {
                root.size --;
                return root.left == null ? root.right : root.left;
            } else if (root.left.priority > root.right.priority) {
                root = root.rightRotate();
                root.right = delete(root.right, value);

            } else {
                root = root.leftRotate();
                root.left = delete(root.left, value);
            }
        }
        root.size = (root.left == null ? 0 : root.left.size) + (root.right == null ? 0 : root.right.size) + root.count;
        return root;

    }
    public boolean contains(long value) {
        return contains(root, value);
    }
    private boolean contains(TreeNode root, long value) {
        if (root == null)
            return false;
        if (root.value == value)
            return true;
        else if (root.value > value) {
            return contains(root.left, value);
        }
        else {
            return contains(root.right, value);
        }
    }
}
```

### 2.5 Graph Theroy

本章规定，用$m$表示图中边的数量，$n$表示图中节点的数量，后文不再赘述。

> 图的建立

学习图论时，图有两种表示方式，分别是邻接矩阵，邻接表。

其中，邻接矩阵需要$O(n^2)$空间，节点数量较多时，非常容易超出空间限制。

邻接表是最常用的方式。

在笔试/面试过程中，采用邻接表建图即可。偏竞赛的笔试中，如果对于空间要求苛刻，则需要采用链式前向星法建图。

有向带权图的建立：
```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Graph {
    int n;  // 最大点数
    int m;  // 最大边数，仅链式前向星需要
    int[][] g1;  // 邻接矩阵
    List<int[]>[] g2;  // 邻接表  如果不带权，可以用List<Integer>[] 表示
    int[] head;   // 链式前向星
    int[] next;   // 链式前向星
    int[] to;     // 链式前向星
    int[] weight; // 链式前向星，针对带权图
    int count;    // 链式前向星，边的编号

    public Graph(int n, int[][] edges) {
        this.n = n;
        this.m = edges.length * 2 + 1;  // 无向图需要2*edges

        this.g1 = new int[n][n];  // 邻接矩阵初始化

        this.g2 = new List[n];   // 邻接表初始化
        Arrays.setAll(g2, k -> new ArrayList<>());

        this.head = new int[n];  // 链式前向星初始化
        // head长度为n，记录每一个节点的头编号。
        this.next = new int[this.m];
        // next长度为m，记录每一条边的下一条边的编号。其中，编号为0代表边不存在。
        this.to = new int[this.m];
        // to长度为m，记录当前边去往的点。
        this.weight = new int[this.m];
        // weight长度为m，记录当前边的权重。
        this.count = 1;

        for(int[] edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            g1[u][v] = w;
            // 无向图：g1[v][u] = w;
            // 无权图：g1[u][v] = 1;

            g2[u].add(new int[]{v, w});
            // 无向图：g2[v].add(new int[]{u, w});

            next[count] = head[u];
            to[count] = v;
            weight[count] = w;
            head[u] = count ++;
            // 无向图：u, v 反过来
        }
    }
}
```

示例：

edges数组为[[0,3,1],[0,4,5],[1,2,7],[1,3,6],[1,4,4],[2,3,2],[2,4,4]]，建立无向带权图。

邻接矩阵

|      | 0    | 1    | 2    | 3    | 4    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 1    | 5    |
| 1    | 0    | 0    | 7    | 6    | 4    |
| 2    | 0    | 7    | 0    | 2    | 4    |
| 3    | 1    | 6    | 2    | 0    | 0    |
| 4    | 5    | 4    | 4    | 0    | 0    |

邻接表

|      |       |       |       |
| ---- | ----- | ----- | ----- |
| 0    | (3,1) | (4,5) |       |
| 1    | (2,7) | (3,6) | (4,4) |
| 2    | (1,7) | (3,2) | (4,4) |
| 3    | (1,6) | (2,2) |       |
| 4    | (0,5) | (1,4) | (2,4) |

链式前向星

head : [3, 9, 13, 12, 14]

next : [0, 0, 0, 1, 0, 0, 0, 5, 2, 7, 4, 6, 8, 11, 10]

to : [0, 3, 0, 4, 0, 2, 1, 3, 1, 4, 1, 3, 2, 4, 2]

weight: [0, 1, 1, 5, 5, 7, 7, 6, 6, 4, 4, 2, 2, 4, 4]

链式前向星的遍历

```java
for(int ei = head[i]; ei > 0; ei = next[ei]) {
    int v = to[ei];
    int w = weight[ei];
}
```
例如，遍历节点1的所有邻边。

head = 9, next[9] = 7, to[9] = 4, weight[9] = 4, 找到(4,4)

head = 7, next[7] = 5, to[7] = 3, weight[7] = 6, 找到
(3,6)

head = 5, next[5] = 0, to[5] = 2, weight[5] = 7, 找到
(2,7)


#### 2.5.1 Graph Traversal

##### 2.5.1.1 Depth First Search

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 高       | 高       | 必须掌握 |
应用一：环检测

例题：[261. 以图判树](https://leetcode.cn/problems/graph-valid-tree/)

```java
class Solution {
    private List<Integer>[] graph;
    private int[] visited;

    public boolean validTree(int n, int[][] edges) {
        if(edges.length + 1 != n) {  // 树的性质
            return false;
        }
        graph = new List[n];
        Arrays.setAll(graph, k -> new ArrayList<>());
        for(int[] edge : edges) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }
        visited = new int[n];
        for(int i = 0; i < n; i ++) {
            if(visited[i] == 0 && !dfs(i, -1)) {
                return false;
            }
        }
        return true;
    }

    private boolean dfs(int u, int p) {  // 环检测
        visited[u] = 1;
        for(int v : graph[u]) {
            if(v == p) {
                continue;
            }
            if(visited[v] == 0) {
                if(!dfs(v, u)) {
                    return false;
                }
            }else if(visited[v] == 1) {
                return false;
            }
        }
        visited[u] = 2;
        return true;
    }
}
```

应用二：二分图检测

例题：[785. 判断二分图](https://leetcode.cn/problems/is-graph-bipartite/)

```java
class Solution {
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        int[] colors = new int[n];
        Arrays.fill(colors, -1);
        for(int i = 0; i < n; i ++) {
            if(colors[i] == -1 && !dfs(graph, 0, i, colors)) {
                return false;
            }
        }
        return true;
    }

    private boolean dfs(int[][] graph, int c, int i, int[] colors) {
        colors[i] = c;
        for(int w : graph[i]) {
            if(colors[w] == -1 && !dfs(graph, 1 - c, w, colors)) {
                return false;
            }else if(colors[w] == colors[i]) {
                return false;
            }
        }
        return true;
    }
}
```

应用三：求解联通分量

例题：[323. 无向图中连通分量的数目](https://leetcode.cn/problems/number-of-connected-components-in-an-undirected-graph/)

```java
class Solution {
    private List<Integer>[] graph;
    private int[] count;
    public int countComponents(int n, int[][] edges) {
        graph = new List[n];
        Arrays.setAll(graph, k -> new ArrayList<>());
        for(int[] edge : edges) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }
        count = new int[n];
        int ans = 0;
        for(int i = 0; i < n; i ++) {
            if(count[i] == 0) {
                ans ++;
                dfs(i, ans);
            }
        }
        return ans;
    }

    private void dfs(int u, int id) {
        count[u] = id;
        for(int v : graph[u]) {
            if(count[v] == 0) {
                dfs(v, id);
            }
        }
    }
}
```
本题中，$count[i]$相同的元素属于同一个联通分量。

应用四：寻找单源路径

例题：[797. 所有可能的路径](https://leetcode.cn/problems/all-paths-from-source-to-target/)

```java
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        dfs(0, graph, new ArrayList<>(), graph.length - 1);
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    private void dfs(int start, int[][] graph, List<Integer> list, int target) {
        list.add(start);
        if(start == target) {
            ret.add(new ArrayList<>(list));
        }
        for(int adj : graph[start]) {
            dfs(adj, graph, list, target);
        }
        list.remove(list.size() - 1);
    }
}
```
思考：如果只需要找到一条单源路径即返回，应该怎么写代码？

答案：为dfs增加返回值，标识是否寻找到单源路径。
```java
class Solution {

    public List<Integer> singlePathSourceTarget(int[][] graph) {
        dfs(0, graph, new ArrayList<>(), graph.length - 1);
        return path;
    }

    private List<Integer> path = new ArrayList<>();

    private boolean dfs(int start, int[][] graph, List<Integer> list, int target) {
        list.add(start);
        if(start == target) {
            path = list;
            return true;
        }
        for(int adj : graph[start]) {
            if(dfs(adj, graph, list, target)) {
                return true;
            }
        }
        list.remove(list.size() - 1);
        return false;
    }
}
```
应用五：floodfill问题

例题：[529. 扫雷游戏](https://leetcode.cn/problems/minesweeper/)

```java
class Solution {
    public char[][] updateBoard(char[][] board, int[] click) {
        return dfs(board, click[0], click[1]);
    }

    private char[][] dfs(char[][] board, int x, int y) {
        if(board[x][y] == 'M') {
            board[x][y] = 'X';
            return board;
        }
        int m = board.length, n = board[0].length;
        if(board[x][y] == 'E') {
            int count = 0;
            for(int[] d : dir) {
                int nx = x + d[0], ny = y + d[1];
                if(0 <= nx && nx < m && 0 <= ny && ny < n && board[nx][ny] == 'M') {
                    count ++;
                }
            }
            if(count == 0) {
                board[x][y] = 'B';
                for(int[] d : dir) {
                    int nx = x + d[0], ny = y + d[1];
                    if(0 <= nx && nx < m && 0 <= ny && ny < n && board[nx][ny] == 'E') {
                        dfs(board, nx, ny);
                    }
                }
            }else {
                board[x][y] = (char)(count + '0');
            }
        }
        return board;
    }

    private int[][] dir = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
}
```
练习题单
| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1466. 重新规划路线](https://leetcode.cn/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/) | 中等 |
| [802. 找到最终的安全状态](https://leetcode.cn/problems/find-eventual-safe-states/) | 中等 |
| [417. 太平洋大西洋水流问题](https://leetcode.cn/problems/pacific-atlantic-water-flow/) | 中等 |

##### 2.5.1.2 Breadth First Search

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 高       | 高       | 必须掌握 |

深度优先遍历能实现的功能，广度优先遍历基本上都能实现。除此之外，广度优先遍历还能实现求解无权图的最短路径。

广度优先遍历的实现通常是基于队列，代码模版如下

```java
Queue<Integer> queue = new LinkedList<>();
boolean[] visited = new boolean[n];  // 标记，防止重复访问
queue.offer(s); // 单源，多源则将所有源点入队
visited[s] = true;  
int level = 0;
while(!queue.isEmpty()) {
    int cur = queue.poll();   // 单点弹出
    // 遍历cur的邻边，将其入队

    int size = queue.size();  // 整层弹出，常用于层序遍历
    for(int i = 0; i < size; i ++) {
        int cur = queue.poll();   
        // 遍历cur的邻边，将其入队
    }
    level ++;   // 层数 + 1
}
```

应用一：无权图最短路径

例题：[847. 访问所有节点的最短路径](https://leetcode.cn/problems/shortest-path-visiting-all-nodes/)

```java
class Solution {
    public int shortestPathLength(int[][] graph) {
        Queue<int[]> queue = new LinkedList<>();
        int n = graph.length;
        int[][] dist = new int[1<<n][n];
        for(int[] d : dist) {
            Arrays.fill(d, Integer.MAX_VALUE);
        }
        for(int i = 0; i < n; i ++) {
            queue.offer(new int[]{1<<i, i});
            dist[1<<i][i] = 0;
        }
        while(!queue.isEmpty()) {
            int[] cur = queue.poll();
            int mask = cur[0], u = cur[1], d = dist[mask][u];
            if(mask == (1 << n) - 1) {
                return d;
            }
            for(int adj : graph[u]) {
                int next = mask | (1 << adj);
                if(d + 1 < dist[next][adj]) {
                    dist[next][adj] = d + 1;
                    queue.offer(new int[]{next, adj});
                }
            }
        }
        return 0;
    }
}
```
时间复杂度：$O(n^2 2^n)$。可以看作$2^n$次广度优先遍历，每次时间复杂度$O(n+m)$。当为完全图时，$m\simeq n^2$。

有的题目没有显式的图结构，但可以通过描述状态间的转移，建模为无权图最短路径问题，例如以下题目。

例题：[773. 滑动谜题](https://leetcode.cn/problems/sliding-puzzle/)

```java
class Solution {
    public int slidingPuzzle(int[][] board) {
        Queue<String> queue = new LinkedList<>();
        String s = board2String(board), finals = "123450";
        if(finals.equals(s)) {
            return 0;
        }
        queue.offer(s);
        Map<String, Integer> visited = new HashMap<>();
        visited.put(s, 0);
        while(!queue.isEmpty()) {
            String cur = queue.poll();
            int zero = cur.indexOf("0"), x = zero / 3, y = zero % 3;
            int[][] b = string2board(cur);
            for(int[] d : dir) {
                int nx = x + d[0], ny = y + d[1];
                if(0 <= nx && nx < 2 && 0 <= ny && ny < 3) {
                    swap(b, x, y, nx, ny);
                    String next = board2String(b);
                    if(finals.equals(next)) {
                        return visited.get(cur) + 1;
                    }
                    if(!visited.containsKey(next)) {
                        queue.offer(next);
                        visited.put(next, visited.get(cur) + 1);
                    }
                    swap(b, x, y, nx, ny);
                }
            }
        }
        return -1;
    }

    private void swap(int[][] board, int x, int y, int nx, int ny) {
        int temp = board[x][y];
        board[x][y] = board[nx][ny];
        board[nx][ny] = temp;
    }

    private int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1,0}};

    private String board2String(int[][] board) {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < board.length; i ++) {
            for(int j = 0; j < board[i].length; j ++) {
                sb.append(board[i][j]);
            }
        }
        return sb.toString();
    }

    private int[][] string2board(String s) {
        int[][] board = new int[2][3];
        for(int i = 0; i < s.length(); i++) {
            board[i/3][i%3] = s.charAt(i) - '0';
        }
        return board;
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点 | 
| ------------------------------------------------------------ | ---- | ---- |
| [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/) | 中等 | 最短路径/动态规划 |
| [752. 打开转盘锁](https://leetcode.cn/problems/open-the-lock/) | 中等 | 最短路径 |
| [1162. 地图分析](https://leetcode.cn/problems/as-far-from-land-as-possible/) | 中等 | 多源最短路径+分层 |

应用二：状态空间搜索

例题：[139. 单词拆分](https://leetcode.cn/problems/word-break/)

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        boolean[] visited = new boolean[s.length()];
        while(!queue.isEmpty()) {
            int start = queue.poll();
            if(start == s.length()) {
                return true;
            }
            if(visited[start]) {
                continue;
            }
            visited[start] = true;
            for(int end = start + 1; end <= s.length(); end ++) {
                String sub = s.substring(start, end);
                if(wordSet.contains(sub)) {
                    queue.offer(end);
                }
            }
        }
        return false;
    }
}
```

以下例题综合了广度优先遍历和深度优先遍历，是一道好题。

例题：[126. 单词接龙 II](https://leetcode.cn/problems/word-ladder-ii/)

分析：本题的测试用例经过了多次加强，时间复杂度较高的算法会超时。

1. 先用BFS（虽然代码采用递归写法，但本质是BFS）寻找最短转换路径，map中存储每个节点的前一个节点。
2. DFS构建最短路序列，一定从end到begin开始构建。

```java
class Solution {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> res = new ArrayList<>();
        Set<String> wordDict = new HashSet<>(wordList);
        if(!wordDict.contains(endWord)) {
            return res;
        }
        Map<String, Set<String>> map = new HashMap<>();
        Set<String> begin = new HashSet<>(), end = new HashSet<>();
        begin.add(beginWord);
        end.add(endWord);
        if(search(wordDict, begin, end, map)) {
            buildPath(res, map, endWord, beginWord, new LinkedList<>());
        }
        return res;
    }

    private boolean search(Set<String> wordDict, Set<String> begin, Set<String> end, Map<String, Set<String>> map) {
        if(begin.size() == 0) {
            return false;
        }
        wordDict.removeAll(begin);
        boolean isMeet = false;
        Set<String> next = new HashSet<>();
        for(String word : begin) {
            char[] chars = word.toCharArray();
            for(int i = 0; i < chars.length; i ++) {
                char temp = chars[i];
                for(char ch = 'a'; ch <= 'z'; ch ++) {
                    chars[i] = ch;
                    if(ch == temp) {
                        continue;
                    }
                    String str = String.valueOf(chars);
                    if(wordDict.contains(str)) {
                        next.add(str);
                        if(end.contains(str)) {
                            isMeet = true;
                        }
                        map.computeIfAbsent(str, k -> new HashSet<>()).add(word);
                    }
                }
                chars[i] = temp;
            }
        }
        if(isMeet) {
            return true;
        }
        return search(wordDict, next, end, map);
    }

    private void buildPath(List<List<String>> res, Map<String, Set<String>> map, String begin, String end, LinkedList<String> list) {
        list.addFirst(begin);
        if(begin.equals(end)) {
            res.add(new ArrayList<>(list));
        }
        if(map.containsKey(begin)) {
            for(String next : map.get(begin)) {
                buildPath(res, map, next, end, list);
            }
        }
        list.removeFirst();
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [140. 单词拆分 II](https://leetcode.cn/problems/word-break-ii/) | 困难 |

BFS可以用于求解无权图的最短路径问题，如果是带权图，则需要使用带权图最短路径的算法。特别的，如果图中边的权值只有0和1，求源点到目标点的最短距离，可以利用BFS+双端队列实现。

例题：[1368. 使网格图至少有一条有效路径的最小代价](https://leetcode.cn/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)

分析：01BFS算法的实现步骤如下

1. dist[i]表示起点到i的最短距离，初始化为Integer.MAX_VALUE.
2. 将起点入队，dist[s] = 0.
3. 从队列头部弹出x，如果x为终点，直接返回dist[x].
4. 否则遍历x的相邻节点，假设为y，边权重为x。
若dist[y] > dist[x] + w，则更新dist[y] = dist[x] + w。若w == 0,则将y从队列头部入队；否则将y从队列尾部入队。
1. 队列为空时停止。

```java
class Solution {
    public int minCost(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dist = new int[m][n], dir = {{}, {0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for(int[] d : dist) {
            Arrays.fill(d, Integer.MAX_VALUE);
        }
        dist[0][0] = 0;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.offerLast(0);
        while(!queue.isEmpty()) {
            int cur = queue.poll();
            int x = cur / n, y = cur % n;
            if(x == m - 1 && y == n - 1) {
                return dist[x][y];
            }
            for(int i = 1; i <= 4; i ++) {
                int nx = x + dir[i][0], ny = y + dir[i][1];
                if(0 <= nx && nx < m && 0 <= ny && ny < n) {
                    int w = i == grid[x][y] ? 0 : 1;
                    if(dist[nx][ny] > dist[x][y] + w) {
                        dist[nx][ny] = dist[x][y] + w;
                        if(w == 0) {
                            queue.offerFirst(nx * n + ny);
                        }else {
                            queue.offerLast(nx * n + ny);
                        }
                    }
                }
            }
        }
        return -1;
    }
}
```
时间复杂度：$O(mn)$

练习题单


|题号                                                         | 难度 | 知识点 |
| ------------------------------------------------------------ | ---- | ---- |
| [2290. 到达角落需要移除障碍物的最小数目](https://leetcode.cn/problems/minimum-obstacle-removal-to-reach-corner/) | 困难 | 01BFS模版


#### 2.5.2 Minimum Spanning Tree

| 面试概率 | 笔试概率 | 学习建议 
| -------- | -------- | -------- |
| 中       | 中       |  建议掌握 |

最小生成树，是连通带权无向图中权值最小的生成树。假设图有$n$个节点，则最小生成树的$n-1$条边的权值和最小。

最小生成树不一定唯一，例如图中每一条边权值都相同时，图的所有生成树都是最小生成树。如果图的每一条边权值都不相同，则最小生成树只有一个。

如果图中具有最小权值的边只有一条，那么这条边包含在任意一个最小生成树中，可以假设该边不在最小生成树中，加入该边后，用边替换任意一条其他的边，都会形成权值更小的生成树。

基于一道例题，讲解最小生成树的两种算法。

例题：[1584. 连接所有点的最小费用](https://leetcode.cn/problems/min-cost-to-connect-all-points/)

##### 2.5.2.1 Kruskal
Kruskal算法是最常用的求解最小生成树的算法，只需要对边进行排序，而无需建图。在实现过程中，需要依赖并查集进行快速环检测。

1. 将所有边根据权值从小到大排序。
2. 如果连接当前的边不会产生环(基于并查集快速判断)，则将该边加入集合。
3. 遍历完所有边后，得到的边构成最小生成树。

时间复杂度：$O(m \log m)$，主要时间开销为对边进行排序。

```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        List<int[]> edges = new ArrayList<>();
        int n = points.length;
        for(int i = 0; i < n; i ++) {
            for(int j = i + 1; j < n; j ++) {
                edges.add(new int[]{i, j, getDistance(points, i, j)});
            }
        }
        UnionFind uf = new UnionFind(n);
        Collections.sort(edges, Comparator.comparingInt(a -> a[2]));
        int ans = 0;
        for(int[] edge : edges) {
            int i = edge[0], j = edge[1], d = edge[2];
            if(!uf.connected(i, j)) {
                uf.union(i, j);
                ans += d;
            }
        }
        return ans;
    }

    private int getDistance(int[][] points, int i, int j) {
        return Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]);
    }

    class UnionFind {
        int[] parent;
        public UnionFind(int size) {
            parent = IntStream.range(0, size).toArray();
        }

        public int find(int p) {
            if(p != parent[p]) {
                parent[p] = find(parent[p]);
            }
            return parent[p];
        }

        public boolean connected(int p, int q) {
            return find(p) == find(q);
        }

        public void union(int p, int q) {
            int pRoot = find(p), qRoot = find(q);
            if(pRoot != qRoot) {
                parent[pRoot] = qRoot;
            }
        }
    }
}
```
##### 2.5.2.2 Prim

Prim算法更为少见，需要建图，并且借助于优先队列这一数据结构。

1. 从任意点出发，将开始点设为已访问，开始点的邻边加入优先队列中。边权重越低，优先级越高。
2. 查看优先队列当前的边(from, to)的to端点，如果to已经访问到，则丢弃该边，否则该边树与最小生成树，将to标记为已访问。
3. 重复步骤2，直到堆为空。

时间复杂度：$O(m \log m),m$为图的边数。

```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
        int n = points.length, ans = 0;
        boolean[] visited = new boolean[n];
        visited[0] = true;
        for(int i = 1; i < n; i ++) {
            queue.offer(new int[]{0, i, getDistance(points, 0, i)});
        }
        while(!queue.isEmpty()) {
            int[] cur = queue.poll();
            int to = cur[1], w = cur[2];
            if(visited[to]) {
                continue;
            }
            ans += w;
            visited[to] = true;
            for(int i = 0; i < n; i ++) {
                if(i != to && !visited[i]) {
                    queue.offer(new int[]{to, i, getDistance(points, to, i)});
                }
            }
        }
        return ans;
    }

    private int getDistance(int[][] points, int i, int j) {
        return Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]);
    }

}
```

练习题单

| 题号                                                         | 难度 | 知识点                |
| ------------------------------------------------------------ | ---- | --------------------- |
| [1168. 水资源分配优化](https://leetcode.cn/problems/optimize-water-distribution-in-a-village/) | 困难 | 最小生成树+思维       |
| [1697. 检查边长度限制的路径是否存在](https://leetcode.cn/problems/checking-existence-of-edge-length-limited-paths/) | 困难 | 并查集+最小生成树思想 |

如果读者在并查集章节做了1697号问题，学习完最小生成树后，应该有更深的体会。1697号的本质也是在求解最小生成树，不过权值是取最大值，而非求和。

进一步思考，最小生成树生成的树还有一个性质，即使得图连通后最大边的权值在所有生成树中最小，该生成树又称为最小瓶颈树。最小生成树一定是最小瓶颈树，但最小瓶颈树不一定是最小生成树。

#### 2.5.3 Shortest Path of Weighted Graph

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

| 算法            | 时间复杂度                  | 特点                             |
| --------------- | --------------------------- | -------------------------------- |
| Dijkstra算法    | $O((m+n)\log m)$ 基于二叉堆 | 不能含负权边，单源最短路径       |
| BellmanFord算法 | $O(m*n)$                    | 可以检测负权环，单源最短路径     |
| SPFA算法        | $O(k*m)$，$k$为平均入队次数 | 可以检测负权环，单源最短路径     |
| Floyed算法      | $O(n^3)$                    | 可以检测负权环，所有点对最短路径 |

通过一道例题来学习最短路径算法。

例题：[743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/)

##### 2.5.3.1 Dijkstra

分析：dijkstra算法用于求解不包含负权边的单源最短路径问题，其思想如下。

定义dist[v]表示从起点s到v的最短路径。

初始化dist数组Integer.MAX_VALUE, 然后dist[s]赋值为0。

每一轮，遍历dist数组中未访问过的最小距离，假设找到节点v，此时dist[v]一定是从s到v的最短路径（不包含负权边，从其他节点出发的权值一定大于dist[v]）。

将dist[v]设置为已访问，遍历v的相邻节点u，如果从v到u的距离更短，即(dist[v] + d < dist[u])，则更新dist[u]。

所有节点都更新后，退出循环。

```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[--k] = 0;
        boolean[] visited = new boolean[n];
        List<int[]>[] g = new List[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for(int[] t : times) {
            g[t[0]-1].add(new int[]{t[1]-1, t[2]});
        }
        while(true) {
            int minDis = Integer.MAX_VALUE, minV = -1;
            for(int i = 0; i < n; i ++) {
                if(!visited[i] && dist[i] < minDis) {
                    minDis = dist[i];
                    minV = i;
                }
            }
            if(minV == -1) {
                break;
            }
            visited[minV] = true;
            for(int[] adj : g[minV]) {
                int u = adj[0], d = adj[1];
                if(!visited[u] && dist[minV] + d < dist[u]) {
                    dist[u] = dist[minV] + d;
                }
            }
        }
        int max = Arrays.stream(dist).max().getAsInt();
        return max == Integer.MAX_VALUE ? -1 : max;
    }
}
```
时间复杂度：$O(V^2)$

继续分析，上述代码的性能瓶颈在于，每次需要遍历出dist数组未访问过的最小dist值，可以通过优先队列进行优化。

```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer, Integer>[] graph = new Map[n];
        Arrays.setAll(graph, e -> new HashMap<>());
        for(int[] t : times) {
            graph[t[0] - 1].put(t[1] - 1, t[2]);
        }
        k --;
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        boolean[] visited = new boolean[n];
        Queue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        queue.offer(new int[]{k, 0});
        dist[k] = 0;
        while(!queue.isEmpty()) {
            int cur = queue.poll()[0];
            if(visited[cur]) {  // 每一个顶点可能会入队多次
                continue;
            }
            visited[cur] = true;
            for(Map.Entry<Integer, Integer> entry : graph[cur].entrySet()) {
                int v = entry.getKey(), w = entry.getValue();
                if(dist[cur] + w < dist[v]) {
                    dist[v] = dist[cur] + w;
                    queue.offer(new int[]{v, dist[v]});
                }
            }
        }
        int max = Arrays.stream(dist).max().getAsInt();
        return max == Integer.MAX_VALUE ? -1 : max;
    }
}
```
时间复杂度：$O(E\log E)$，优先队列中最多有E个元素。

如果图为稠密图，采用未优化过的Dijstra算法性能可能更高。

例题：[2662. 前往目标的最小代价](https://leetcode.cn/problems/minimum-cost-of-a-path-with-special-roads/)

分析：由于是二维，采用位运算压缩的方式，使用哈希表来记录最短路径。对于坐标(x, y)，用一个long类型数据x << 32 | y 表示。

```java
class Solution {
    public int minimumCost(int[] start, int[] target, int[][] specialRoads) {
        long end = (long) target[0] << 32 | target[1];
        Map<Long, Integer> dis = new HashMap<>();
        dis.put(end, Integer.MAX_VALUE);
        dis.put((long) start[0] << 32 | start[1], 0);
        Set<Long> visited = new HashSet<>();
        while(true) {
            long v = -1;
            int minD = Integer.MAX_VALUE;
            for(Map.Entry<Long, Integer> entry : dis.entrySet()) {
                if(!visited.contains(entry.getKey()) && (entry.getValue() < minD)) {
                    v = entry.getKey();
                    minD = entry.getValue();
                }
            }
            if(v == end) {
                return minD;
            }
            visited.add(v);
            int vx = (int) (v >> 32), vy = (int) (v & Integer.MAX_VALUE);
            dis.merge(end, minD + Math.abs(target[0] - vx) + Math.abs(target[1] - vy), Math::min);
            for(int[] road : specialRoads) {
                int d = minD + Math.abs(road[0] - vx) + Math.abs(road[1] - vy) + road[4];
                long u = (long) road[2] << 32 | road[3];
                if(d < dis.getOrDefault(u, Integer.MAX_VALUE)) {
                    dis.put(u, d);
                }
            }
        }
    }
}
```
时间复杂度：$O(n^2)$，$n$为specialRoads的长度。

思考：体会BFS算法和Dijkstra算法的异同。
##### 2.5.3.2 BellmanFord

Bellan核心是松弛操作。

假设初始节点为$s$，$dis[v]$是从$s$到$v$经过边数不超过$k$的最短距离

```
if(dis[a] + ab < dis[b])
	dis[b] = dis[a] + ab
```

找到从$s$到$a$经过边数不超过$k+1$的最短距离。

初始$dis[s]=0$，其余$dis$值设置为Integer.MAX_VALUE。

对所有边进行一次松弛操作，求出到所有点经过的边数最多为$1$的最短路。

松弛$V-1$次则求出所有点经过的边数最多为$V-1$的最短路。

本题中，采用BellmandFord算法，可以不用建图，直接遍历边。

```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        k --;
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[k] = 0;
        for(int i = 0; i < n - 1; i ++) {
            for(int[] t : times) {
                int from = t[0] - 1, to = t[1] - 1, w = t[2];
                if(dist[from] != Integer.MAX_VALUE && dist[from] + w < dist[to]) {
                    dist[to] = dist[from] + w;
                }
            }
        }
        int max = Arrays.stream(dist).max().getAsInt();
        return max == Integer.MAX_VALUE ? -1 : max;
    }
}
```
对于负权环的检测：在松弛完V-1轮后，如果还能松弛，说明存在负权环。
```java
for(int v = 0; v < n; v ++) {
    for(int w : graph[v]) {
        if(dis[v] != Integer.MAX_VALUE && dis[v] + graph.getWeight(v, w) < dis[w]) {
            // 存在负权环
        }
    }
}
```

在该模板的实现下，计算的是$s$到任一顶点最少经过$k$步的最小路径。

考虑一个线性的图，$0 \to 1 \to 2 \to 3$，每条边的权重都为$1$。

第一轮松弛：

松弛$0\to1$边：$dis[0] = 0, dis[1] = 1, dis[2] = inf, dis[3] = inf$。

松弛$1\to2$边：$dis[0] = 0, dis[1] = 1, dis[2] = 2, dis[3] = inf$。

松弛$2\to3$边：$dis[0] = 0, dis[1] = 1, dis[2] = 2, dis[3] = 3$。

可以看出，只用了$1$轮松弛就得到了从$0\to3$的最短路径，原因在于松弛$1\to2$时，$0\to1$刚好松弛过，此时$dis[1]$不再是$inf$。$dis[j]=dis[i] + w[i][j]$时的$dis[j]$语义变为了经过($dis[i]$最短路径边数$+1$)条边到达$j$的最短路径。

该松弛保证了$V-1$松弛后得出的答案一定是正确的。

若要满足松弛$V-1$次求出所有点经过的边数最多为$V-1$的最短路，参考如下例题，使用一个clone数组记录上一次的最短路径的值，避免对第$i$条边进行更新之后，第$i+1$条边在上次更新之后的值基础之上更新。

例题：[787. K 站中转内最便宜的航班](https://leetcode.cn/problems/cheapest-flights-within-k-stops/)

```java
class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        int[] d = new int[n];
        Arrays.fill(d, Integer.MAX_VALUE);
        d[src] = 0;
        for(int i = 0; i < k+1; i++) {
            int[] clone = d.clone();
            for(int[] f : flights) {
                int from = f[0], to = f[1], w = f[2];
                if(clone[from] != Integer.MAX_VALUE && clone[from] + w < d[to]) {
                    d[to] = clone[from] + w;
                }
            }
        }
        return d[dst] == Integer.MAX_VALUE ? -1 : d[dst];
    }
}
```

##### 2.5.3.3 SPFA

SPFA是一种优化后的BellmanFord算法，类似于BFS，但每个节点可能入队出队多次。

```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        graph = new Map[n];
        Arrays.setAll(graph, e -> new HashMap<>());
        dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        visited = new boolean[n];
        this.k = k - 1;
        for(int[] time : times) {
            int from = time[0] - 1, to = time[1] - 1, w = time[2];
            graph[from].put(to, w);
        }
        spfa();
        int max = Arrays.stream(dist).max().getAsInt();
        return max == Integer.MAX_VALUE ? -1 : max;
    }

    private int k;
    private boolean[] visited;
    private int[] dist;
    private Map<Integer, Integer>[] graph;

    private void spfa() {
        dist[k] = 0;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.offerLast(k);
        visited[k] = true;
        /**
        int[] count = new int[n];
        count[start] = 1;
        **/
        while(!queue.isEmpty()) {
            int cur = queue.pollFirst();
            visited[cur] = false;
            for(Map.Entry<Integer, Integer> entry : graph[cur].entrySet()) {
                int adj = entry.getKey(), w = entry.getValue();
                if(dist[cur] + w < dist[adj]) {
                    dist[adj] = dist[cur] + w;
                    if(!visited[adj]) {
                        visited[adj] = true;
                        /**
                        count[adj] ++;
                        if(count[adj] > n) {
                            return false;   存在负权环
                        }
     					**/
                        if(!queue.isEmpty() && dist[queue.peek()] > dist[adj]) {
                            queue.offerFirst(adj);
                        }else {
                            queue.offerLast(adj);
                        }
                    }
                }
            }
        }
        // return true;
    }
}
```

以上代码，如果图中存在负权环，则while循环可能一直无法退出。解决方案，用一个count数组记录每个节点入队次数，若入队次数大于$n$，说明存在负权环（见被注释的代码）。

##### 2.5.3.4 Floyed
```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        int[][] dp = new int[n][n];
        for(int i = 0; i < n; i ++) {
            for(int j = 0; j < n; j ++) {
                dp[i][j] = i == j ? 0 : Integer.MAX_VALUE;
            }
        }
        for(int[] t : times) {
            dp[t[0] - 1][t[1] - 1] = t[2];
        }
        k --;
        for(int t = 0; t < n; t ++) {
            for(int i = 0; i < n; i ++) {
                for(int j = 0; j < n; j ++) {
                    if(dp[i][t] != Integer.MAX_VALUE && dp[t][j] != Integer.MAX_VALUE) {
                        dp[i][j] = Math.min(dp[i][j], dp[i][t] + dp[t][j]);
                    }
                }
            }
        }
        int max = Arrays.stream(dp[k]).max().getAsInt();
        return max == Integer.MAX_VALUE ? -1 : max;
    }

}
```
Floyed算法检测负权环：松弛完毕后，若发现$dis[v][v]<0$，说明存在负权环。

#### 2.5.4 Topological Sort

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 中       | 中       | 建议掌握 |

拓扑排序是有向图的算法，是针对节点进行排序，排序后每个节点的前置节点都在该节点之前。拓扑排序的结果可能不唯一。

从一道例题中学习拓扑排序。

例题：[210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)

有两种算法可以实现拓扑排序，其中广度优先遍历更为常用。

> 广度优先遍历

1. 找出图中入度为0的点，入队。
2. 删除图中入度为0的点，即出队，将相邻节点的入度减一。若入度变为0，则入队。
3. 节点出队的顺序即为拓扑排序的结果。
4. 如果无法将所有的点删掉，说明有向图中存在环。

```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<Integer>[] graph = new List[numCourses];
        Arrays.setAll(graph, k -> new ArrayList<>());
        int[] indegree = new int[numCourses];
        for(int[] pre : prerequisites) {
            indegree[pre[0]] ++;
            graph[pre[1]].add(pre[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < indegree.length; i ++) {
            if(indegree[i] == 0) {
                queue.offer(i);
            }
        }
        List<Integer> ret = new ArrayList<>();
        while(!queue.isEmpty()) {
            int cur = queue.poll();
            ret.add(cur);
            for(int adj : graph[cur]) {
                indegree[adj] --;
                if(indegree[adj] == 0) {
                    queue.offer(adj);
                }
            }
        }
        return ret.size() == numCourses ? ret.stream().mapToInt(e -> e).toArray() : new int[0];
    }
}
```

思考：如果要求字典序最小的拓扑排序结果，应该怎么实现？

提示：将队列换为优先队列。

> 深度优先遍历逆序
```java
class Solution {
    private boolean valid = true;
    private int[] result;
    private int[] visited;
    private int index;
    private List<Integer>[] graph;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        graph = new List[numCourses];
        Arrays.setAll(graph, k -> new ArrayList<>());
        visited = new int[numCourses];
        result = new int[numCourses];
        index = numCourses - 1;
        for(int[] pre : prerequisites) {
            graph[pre[1]].add(pre[0]);
        }
        for(int i = 0; i < numCourses && valid; i ++) {
            if(visited[i] == 0) { // 未搜索
                dfs(i);
            }
        }
        return valid ? result : new int[0];
    }

    private void dfs(int u) {
        visited[u] = 1;
        for(int v : graph[u]) {
            if(visited[v] == 0) { // 未搜索
                dfs(v);
                if(!valid) {
                    return;
                }
            }else if(visited[v] == 1) { // 搜索中，存在环
                valid = false;
                return;
            }
        }
        visited[u] = 2; // 搜索完毕
        result[index --] = u;
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点            |
| ------------------------------------------------------------ | ---- | ----------------- |
| [269. 火星词典](https://leetcode.cn/problems/alien-dictionary/) | 困难 | 拓扑排序+细节处理 |


更难一些的题目，需要分析出为何拓扑排序是正确的。

例题：[310. 最小高度树
](https://leetcode.cn/problems/minimum-height-trees/)

```java
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> res = new ArrayList<>();
        if(n == 1) {
            res.add(0);
            return res;
        }
        List<Integer>[] g = new List[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        int[] degree = new int[n];
        for(int[] edge : edges) {
            g[edge[0]].add(edge[1]);
            g[edge[1]].add(edge[0]);
            degree[edge[0]] ++;
            degree[edge[1]] ++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < n; i ++) {
            if(degree[i] == 1) {
                queue.offer(i);
            }
        }
        List<Integer> ret = new ArrayList<>();
        while(!queue.isEmpty()) {
            ret = new ArrayList<>();
            int size = queue.size();
            for(int i = 0; i < size; i ++) {
                int cur = queue.poll();
                ret.add(cur);
                for(int v : g[cur]) {
                    degree[v] --;
                    if(degree[v] == 1) {
                        queue.offer(v);
                    }
                }
            }
        }
        return ret;
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点                              |
| ------------------------------------------------------------ | ---- | ----------------------------------- |
| [2603. 收集树中金币](https://leetcode.cn/problems/collect-coins-in-a-tree/) | 中等 | 拓扑排序，310题基础上再深入思考一点 |

探究拓扑排序与动态规划的关系（读者可以先学习**动态规划**章节后再看本小节)

我们知道，动态规划中，原问题依赖于子问题的解。而拓扑排序能够保障依赖问题求解的顺序。

例题：[851. 喧闹和富有](https://leetcode.cn/problems/loud-and-rich/)

```java
class Solution {
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length, m = richer.length + 1, count = 1;
        int[] dp = IntStream.range(0, n).toArray(), head = new int[n], to = new int[m+1], next = new int[m+1], deg = new int[n];
        for(int[] edge : richer) {
            int u = edge[0], v = edge[1];
            next[count] = head[u];
            to[count] = v;
            head[u] = count ++;
            deg[v] ++; 
        }
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < n; i ++) {
            if(deg[i] == 0) {
                queue.offer(i);
            }
        }
        while(!queue.isEmpty()) {
            int cur = queue.poll();
            for(int ei = head[cur]; ei > 0; ei = next[ei]) {
                int u = to[ei];
                deg[u] --;
                dp[u] = quiet[dp[cur]] < quiet[dp[u]] ? dp[cur] : dp[u];
                if(deg[u] == 0) {
                    queue.offer(u);
                }
            }

        }
        return dp;
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点            |
| ------------------------------------------------------------ | ---- | ----------------- |
| [2050. 并行课程 III](https://leetcode.cn/problems/parallel-courses-iii/) | 困难 | 拓扑排序+动态规划 |

#### 2.5.5 Cycle Detection

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

对于无向图的环检测，在深度优先遍历中已经进行了介绍。同时，也可以采用并查集进行解决。

对于有向图的环检测，可以基于拓扑排序实现，采用深度优先遍历，则需要额外记录遍历的信息。

例题：[207. 课程表](https://leetcode.cn/problems/course-schedule/)

分析：新增一个onPath数组，用于标识当前节点是否已经在访问路径中出现，若出现，说明存在环，此时无法完成所有课程的学习。

```java
class Solution {
    private List<Integer>[] g;
    private boolean[] visited;
    private boolean[] onPath;

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        g = new List[numCourses];
        visited = new boolean[numCourses];
        onPath = new boolean[numCourses];
        Arrays.setAll(g, k -> new ArrayList<>());
        for(int[] pre : prerequisites) {
            g[pre[1]].add(pre[0]);
        }
        for(int i = 0; i < numCourses; i ++) {
            if(!visited[i] && dfs(i)) {
                return false;
            }
        }
        return true;
    }

    private boolean dfs(int u) {  // 以u出发是否存在环
        visited[u] = true;
        onPath[u] = true;   // visited[u] = 1
        for(int v : g[u]) {
            if(!visited[v]) {  // visited[v] == 0
                if(dfs(v)) {
                    return true;
                }
            }else if(onPath[v]) {  // visited[v] == 1
                return true;
            }
        }
        onPath[u] = false;  // visited[u] = 2;
        return false;
    }
}
```

也可以优化visited数组和onPath数组为一个visited的整型数组。

visited[u] = 0，说明u节点未曾访问。

visited[u] = 1，说明u节点正在访问中(对应onPath[u]=true)。 

visited[u] = 2，说明u节点已经访问完毕(对应onPath[u]=false)。

#### 2.5.6 Bridge and Cutting Point

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

桥：对于无向图，如果删除一条边后，图的连通分量数量发生了变化，则这一条边是桥。

割点：对于无向图，如果删除了一个顶点（顶点邻边也删除），图的连通分量数量发生了变化，则这个点是割点。

求解桥的算法为Tarjan算法，其核心思想如下：

判断u到v的边是不是桥？看能否通过v从另一条路回到u，或者u之前的顶点。若存在路径，说明u-v边不是桥。

使用数组ord[v]，表示顶点v在DFS中的访问顺序。

使用数组low[v]表示DFS过程中顶点v能到达的最小ord值。

例题：[1192. 查找集群内的关键连接](https://leetcode.cn/problems/critical-connections-in-a-network/)

```java
class Solution {
    private List<Integer>[] g;
    private boolean[] visited;
    private int[] ord;
    private int[] low;
    private int id;
    private List<List<Integer>> ret;

    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        g = new List[n];
        visited = new boolean[n];
        ord = new int[n];
        low = new int[n];
        ret = new ArrayList<>();
        Arrays.setAll(g, k -> new ArrayList<>());
        for(List<Integer> con : connections) {
            g[con.get(0)].add(con.get(1));
            g[con.get(1)].add(con.get(0));
        }
        for(int i = 0; i < n; i ++) {
            if(!visited[i]) {
                dfs(i, i);
            }
        }
        return ret;
    }

    private void dfs(int u, int p) {
        visited[u] = true;
        ord[u] = id ++;
        low[u] = ord[u];
        for(int v : g[u]) {
            if(!visited[v]) {
                dfs(v, u);
                low[u] = Math.min(low[u], low[v]);
                if(low[v] > ord[u]) {
                    ret.add(Arrays.asList(u, v));
                }
            }else if(v != p){
                low[u] = Math.min(low[u], low[v]);
            }
        }
    }
}
```

如果点u有一个孩子节点v，满足low[v] >= ord[u]，则u是割点。对于根节点，需要特殊判断。根节点如果有1个以上的孩子，则根节点是割点。其中孩子定义为DFS遍历树上的节点。

```java
public void dfs(int u, int p, List<Integer>[] g) {
    visited[u] = true;
    low[u] = ord[u] = count++;
    int child = 0;
    for(int v : g[u]) {
        if(!visited[v]) {
            dfs(v, u, g);
            low[u] = Math.min(low[u], low[v]);
            if(v != p && low[v] >= ord[u]) {
                ret.add(u);
            }
            child ++;
            if(u == p && child > 1) {
                ret.add(u);
            }
        }else if(v != p) {
            low[u] = Math.min(low[u], low[v]);
        }
    }
}
```


#### 2.5.7 Connected Component

无向图的连通分量，可以采用DFS/BFS/并查集进行求解。

有向图的连通分量，可以基于Tarjan算法进行求解。Tarjan算法的思想在桥和割点章节已经进行了介绍。

当ord[u] == low[u]时，u的子树可以构成一个强连通分量。

例题：[2360. 图中的最长环](https://leetcode.cn/problems/longest-cycle-in-a-graph/)

分析：图中的每一个环即为一个强连通分量。使用Tarjan算法求出所有连通分量即可。

```java
class Solution {
    private int[] ord;
    private int[] low;
    private int[] visited;
    private int[] edges;
    private int id;
    private int ans;
    private LinkedList<Integer> path;

    public int longestCycle(int[] edges) {
        int n = edges.length;
        visited = new int[n];
        ord = new int[n];
        low = new int[n];
        this.edges = edges;
        path = new LinkedList<>();
        for(int i = 0; i < n; i ++) {
            if(visited[i] == 0) {
                dfs(i);
            }
        }
        return ans == 1 ? -1 : ans;
    }

    private void dfs(int u) {
        ord[u] = low[u] = id ++;
        visited[u] = 1;
        path.addLast(u);
        if(edges[u] != -1) {
            int v = edges[u];
            if(visited[v] == 0) {
                dfs(v);
                low[u] = Math.min(low[u], low[v]);
            }else if(visited[v] == 1) {
                low[u] = Math.min(low[u], low[v]);
            }
        }
        if(ord[u] == low[u]) {
            int count = 1;
            int v = path.removeLast();
            visited[v] = 2;
            while(v != u) {
                count ++;
                v = path.removeLast();
                visited[v] = 2;
            }
            ans = Math.max(ans, count);
        }
    }
}
```





#### 2.5.8 Hamiltonian Path
| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

哈密尔顿回路：从一个点出发，沿着边行走，经过每一个顶点恰好一次，最后回到出发点。

哈密尔顿路径：从一个点出发，沿着边行走，经过每一个顶点恰好一次，最后回到出发点。

哈密尔顿路径求解是一个NP难问题。

例题：[980. 不同路径 III](https://leetcode.cn/problems/unique-paths-iii/)

```java
class Solution {
    public int uniquePathsIII(int[][] grid) {
        m = grid.length; 
        n = grid[0].length;
        int startX = -1, startY = -1, mask = 0;
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                if(grid[i][j] == 1) {
                    startX = i;
                    startY = j;
                }else if(grid[i][j] < 0) {
                    mask |= (1 << (i * n + j));
                }
            }
        }
        memo = new int[m*n][1<<(m*n)];
        for(int[] arr : memo) {
            Arrays.fill(arr, -1);
        }
        return dfs(grid, startX, startY, mask);
    }

    private int m;
    private int n;
    private int[][] memo;

    private int dfs(int[][] grid, int x, int y, int mask) {
        int id = x * n + y;
        // 越界判断 + 重复访问判断
        if(x < 0 || x >= m || y < 0 || y >= n || (mask >> id & 1) > 0) {
            return 0;
        }
        // 设置已访问
        mask |= 1 << id;
        // 终止条件判断
        if(grid[x][y] == 2) {
            return mask == (1 << m * n) - 1 ? 1 : 0;
        }
        // 重复计算判断
        if(memo[id][mask] != -1) {
            return memo[id][mask];
        }
        // 四方向递归
        int ans = dfs(grid, x - 1, y, mask) + dfs(grid, x + 1, y, mask) + dfs(grid, x, y - 1, mask) + dfs(grid, x, y + 1, mask);
        return memo[id][mask] = ans;
    }
}
```
时间复杂度：$O(mn\times 2^{mn})$

可以采用哈希表进行记忆化：


```java
int key = (id << m * n) | mask;  // mask最多m*n个比特
if(memo.containsKey(key)) {
    return memo.get(key);
}
```

也可采用回溯算法求解：

```java
class Solution {
    public int uniquePathsIII(int[][] grid) {
        m = grid.length; 
        n = grid[0].length;
        int startX = -1, startY = -1, count = 0;
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                if(grid[i][j] == 0) {
                    count ++;
                }else if(grid[i][j] == 1) {
                    startX = i;
                    startY = j;
                }
            }
        }
        return dfs(grid, startX, startY, count + 1);
    }

    private int m;
    private int n;

    private int dfs(int[][] grid, int x, int y, int left) {
        if(x < 0 || x >= m || y < 0 || y >= n || grid[x][y] < 0) {
            return 0;
        }
        if(grid[x][y] == 2) {
            return left == 0 ? 1 : 0;
        }
        grid[x][y] = -1;
        int ans = dfs(grid, x - 1, y, left - 1) + dfs(grid, x + 1, y, left - 1) + dfs(grid, x, y - 1, left - 1) + dfs(grid, x, y + 1, left - 1);
        grid[x][y] = 0;
        return ans;
    }
}
```

时间复杂度：$O(3^{mn})$，由于不能重复访问同一个格子，实际执行效率比记忆化搜索更快。
#### 2.5.9 Eulerian Path

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

欧拉回路：从一个点出发，沿着边行走，经过每一个边恰好一次，最后回到出发点。

欧拉路径：从一个点出发，沿着边行走，经过每一个边恰好一次，起始点和终止点可以不同。

著名的七桥问题本质就是寻找是否存在欧拉回路。

对于无向连通图，欧拉回路存在的充分必要条件是每个点的度是偶数。

对于无向连通图，欧拉路径存在的充分必要条件是除了两个点每个点的度是偶数。

Hierholzer算法用于在连通图中寻找欧拉路径，其流程如下：
1. 从起点出发进行深度优先遍历。
2. 每次沿着某条边从某个顶点移动到另一个顶点时，删除该边。
3. 如果没有可以移动的路径，则将所在节点加入到栈中并返回。

对于步骤3，如果顺序思考，很难判断当前节点哪一个分支会走入死胡同。

采用逆向思维，遍历完一个节点相连的所有节点后才将该节点入栈。对于当前节点，从它每一个非死胡同分支出发进行深度优先遍历都能搜回到当前节点，从它每一个死胡同分支出发则不会搜回当前节点，会先于非死胡同分支入栈。最后的路径将栈中节点逆序即可。

例题：[332. 重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/)

```java
class Solution {
    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String, Queue<String>> g = new HashMap<>();
        for(List<String> ticket : tickets) {
            String from = ticket.get(0), to = ticket.get(1);
            g.computeIfAbsent(from, k -> new PriorityQueue<>()).offer(to);
        }
        dfs("JFK", g);
        Collections.reverse(ret);
        return ret;
    }

    private List<String> ret = new ArrayList<>();

    public void dfs(String str, Map<String, Queue<String>> g) {
        while((g.get(str) != null && !g.get(str).isEmpty())) {
            String next = g.get(str).poll();
            dfs(next, g);
        }
        ret.add(str);
    }
}
```
时间复杂度：$O(m\log m),m$为边的数量。

从下一道例题看实际问题怎么用欧拉回路问题建模。

例题：[753. 破解保险箱](https://leetcode.cn/problems/cracking-the-safe/)

分析：将所有$n-1$位数看作节点，共有$k^{n-1}$个节点，每个节点有$k$跳入边和出边。
例如，如果当前节点对应$a_1a_2...a_{n-1}$，则其第$x$条出边连接$a_2...a_{n-1}x$节点，相当于输入了$x$。

每个节点都能用这样方式形成$k$个$n$位数，总共$k^{n-1}*k=k^n$个$n$位数，正好对应所有可能的密码。

由于图中每个节点都有$k$条入边和出边，一定存在一个欧拉回路，用Hierholzer算法求解即可。

```java
class Solution {
    public String crackSafe(int n, int k) {
        this.k = k;
        this.mod = (int)Math.pow(10, n - 1);
        dfs(0);
        ans.append("0".repeat(n - 1));  // 从n-1个0出发，最后放入，逆不逆序均可
        return ans.toString();
    }

    private Set<Integer> visited = new HashSet<>();
    private StringBuilder ans = new StringBuilder();
    private int k, mod;

    private void dfs(int u) {
        for(int i = 0; i < k; i ++) {
            int v = u * 10 + i;
            if(visited.contains(v)) {
                continue;
            }
            visited.add(v);
            dfs(v % mod);
            ans.append(i);
        }
    }
}
```
时间复杂度：$O(n\times k^n)$

#### 2.5.10 Base Ring Tree

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |

对于有$n$个节点的树，必定有$n-1$条边；反之由$n-1$条无向边组成的连通图必定是一颗树。若有$n$个节点$n$条边的无向连通图，则必定是在一棵树上的任意两点直接连接一条边构成的。$n$个节点$n$条边的无向连通图，称之为基环树。

基环树森林可以看作是许多基环树的集合，同样是$n$个节点$n$条边，但不一定保证连通。

内向树和外向树：可以视作有向图的基环树，同样是$n$个节点$n$条边，对于内向树，每个点仅有一条出边；对于外向树，每个点仅有一条入边。

例题：[2127. 参加会议的最多员工数](https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/)

分析：可以分析出，每个连通块必定有且仅有一个环。若环大小大于2，则圆桌的最大员工数即为最大的基环大小。若环的大小等于2，则沿着成环的两个节点反向进行深度优先遍历，找出最长的链，拼成一个圆。

遍历基环，可以从拓扑排序后入度为1的节点出发在图上搜索。

遍历树枝，可以从基环与树枝的连接处为起点，顺着反图搜索树枝（入度为0的点）。

```java
class Solution {
    private List<Integer>[] rg;  // 反图
    private int[] deg;

    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        rg = new List[n];
        deg = new int[n];
        Arrays.setAll(rg, e -> new ArrayList<>());
        for(int i = 0; i < n; i ++) {
            int v = favorite[i];
            rg[v].add(i);
            ++deg[v]; 
        }
        // 拓扑排序，剪掉树枝
        Queue<Integer> queue = new ArrayDeque<>();
        for(int i = 0; i < n; i ++) {
            if(deg[i] == 0) {
                queue.offer(i);
            }
        }
        while(!queue.isEmpty()) {
            int u = queue.poll();
            int v = favorite[u];
            deg[v] --;
            if(deg[v] == 0) {
                queue.offer(v);
            }
        }
        int maxRingSize = 0, sumChainSize = 0;
        for(int i = 0; i < n; i ++) {
            if(deg[i] <= 0) {
                continue;
            }
            // 遍历基环上的点(拓扑排序后入度大于0)
            deg[i] = -1;
            int ringSize = 1;
            for(int v = favorite[i]; v != i; v = favorite[v]) {
                deg[v] = -1;  // 避免重复访问
                ringSize ++;
            }
            if(ringSize == 2) {
                sumChainSize += rdfs(i) + rdfs(favorite[i]); // 累加两条最长链
            }else{
                maxRingSize = Math.max(maxRingSize, ringSize);
            }
        }
        return Math.max(maxRingSize, sumChainSize);
    }

    private int rdfs(int u) {
        int depth = 1;
        for(int v : rg[u]) {
            if(deg[v] == 0) {
                depth = Math.max(depth, rdfs(v) + 1);
            }
        }
        return depth;
    }
}
```
对于基环为2的情况，我们采用沿着反图进行dfs的方式求解最长链。根据拓扑排序章节介绍的知识，可以在拓扑排序时同时求解最长链。

```java
class Solution {
    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        int[] deg = new int[n], dp = new int[n];
        for(int i = 0; i < n; i ++) {
            int v = favorite[i];
            deg[v] ++; 
        }
        Queue<Integer> queue = new ArrayDeque<>();
        for(int i = 0; i < n; i ++) {
            if(deg[i] == 0) {
                queue.offer(i);
            }
        }
        while(!queue.isEmpty()) {
            int u = queue.poll();
            int v = favorite[u];
            deg[v] --;
            dp[v] = Math.max(dp[v], dp[u] + 1);
            if(deg[v] == 0) {
                queue.offer(v);
            }
        }
        int maxRingSize = 0, sumChainSize = 0;
        for(int i = 0; i < n; i ++) {
            if(deg[i] <= 0) {
                continue;
            }
            deg[i] = -1;
            int ringSize = 1;
            for(int v = favorite[i]; v != i; v = favorite[v]) {
                deg[v] = -1;
                ringSize ++;
            }
            if(ringSize == 2) {
                sumChainSize += dp[i] + dp[favorite[i]] + 2;
            }else{
                maxRingSize = Math.max(maxRingSize, ringSize);
            }
        }
        return Math.max(maxRingSize, sumChainSize);
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点                     |
| ------------------------------------------------------------ | ---- | -------------------------- |
| [2359. 找到离给定两个节点最近的节点](https://leetcode.cn/problems/find-closest-node-to-given-two-nodes/) | 中等 | 基环树+枚举距离            |
| [2360. 图中的最长环](https://leetcode.cn/problems/longest-cycle-in-a-graph/) | 困难 | 基环树+求环/Tarjan连通分量 |
| [2836. 在传球游戏中最大化函数值](https://leetcode.cn/problems/maximize-value-of-function-in-a-ball-passing-game/) | 困难 | 基环树/树上倍增            |
| [2876. 有向图访问计数](https://leetcode.cn/problems/count-visited-nodes-in-a-directed-graph/) | 困难 | 基环树                     |



#### 2.5.11 Network Flow

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |

最小费用最大流

例题：[2850. 将石头分散到网格图的最少移动次数](https://leetcode.cn/problems/minimum-moves-to-spread-stones-over-grid/)

```java
public class Solution {

    public int minimumMoves(int[][] g) {
        List<int[]> gOne = new ArrayList<>();
        List<int[]> zero = new ArrayList<>();
        int n = g.length;
        for(int i = 0; i < n; i ++) {
            for(int j = 0; j < n; j ++) {
                if(g[i][j] > 1) {
                    gOne.add(new int[]{i, j});
                }else if(g[i][j] == 0) {
                    zero.add(new int[]{i, j});
                }
            }
        }
        int m = gOne.size(), k = zero.size(), start = m + k, end = start + 1;
        MinCostMaxFlowEK mc = new MinCostMaxFlowEK(start + 2, start, end);
        for(int i = 0; i < m; i ++) {
            int[] pos = gOne.get(i);
            mc.addEdge(start, i, g[pos[0]][pos[1]] - 1, 0);
        }
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < k; j ++) {
                int[] pos1 = gOne.get(i), pos2 = zero.get(j);
                int x1 = pos1[0], y1 = pos1[1], x2 = pos2[0], y2 = pos2[1];
                mc.addEdge(i, m + j, 1, Math.abs(x1 - x2) + Math.abs(y1 - y2));
            }
        }
        for(int j = 0; j < k; j ++) {
            mc.addEdge(m + j, end, 1, 0);
        }
        return mc.work()[1];
    }

    class Edge {
        int fromV;
        int toV;
        int cap;
        int cost;
        int flow;

        public Edge(int fromV, int toV, int cap, int cost, int flow) {
            this.fromV = fromV;
            this.toV = toV;
            this.cap = cap;
            this.cost = cost;
            this.flow = flow;
        }
    }

    class MinCostMaxFlowEK {
        int n;
        int start;
        int end;
        List<Edge> edges;
        List<Integer>[] reverseGraph;
        int[] dist;
        int[] pre;
        int[] flow;

        public MinCostMaxFlowEK(int n, int start, int end) {
            this.n = n;
            this.start = start;
            this.end = end;
            this.edges = new ArrayList<>();
            this.reverseGraph = new List[n];
            Arrays.setAll(reverseGraph, k -> new ArrayList<>());
            this.dist = new int[n];
            this.pre = new int[n];
            this.flow = new int[n];
        }

        public void addEdge(int fromV, int toV, int cap, int cost) {
            edges.add(new Edge(fromV, toV, cap, cost, 0));
            edges.add(new Edge(toV, fromV, 0, -cost, 0));
            this.reverseGraph[fromV].add(edges.size() - 2);
            this.reverseGraph[toV].add(edges.size() - 1);
        }

        public int[] work() {
            int maxFlow = 0, minCost = 0;
            while(spfa()) {
                int delta = flow[end];
                minCost += delta * dist[end];
                maxFlow += delta;
                int cur = end;
                while(cur != start) {
                    int edgeIndex = pre[cur];
                    edges.get(edgeIndex).flow += delta;
                    edges.get(edgeIndex ^ 1).flow -= delta;
                    cur = this.edges.get(edgeIndex).fromV;
                }
            }
            return new int[]{maxFlow, minCost};
        }

        private boolean spfa() {
            Arrays.fill(flow, 0);
            Arrays.fill(pre, -1);
            Arrays.fill(dist, Integer.MAX_VALUE);
            dist[start] = 0;
            flow[start] = Integer.MAX_VALUE;
            boolean[] inQueue = new boolean[n];
            inQueue[start] = true;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.offerFirst(start);
            while(!queue.isEmpty()) {
                int cur = queue.pollFirst();
                inQueue[cur] = false;
                for(int edgeIndex : reverseGraph[cur]) {
                    Edge edge = edges.get(edgeIndex);
                    int cost = edge.cost, remain = edge.cap - edge.flow, next = edge.toV;
                    if(remain > 0 && dist[cur] + cost < dist[next]) {
                        dist[next] = dist[cur] + cost;
                        pre[next] = edgeIndex;
                        flow[next] = Math.min(remain, flow[cur]);
                        if(!inQueue[next]) {
                            inQueue[next] = true;
                            if(!queue.isEmpty() && dist[queue.peek()] > dist[next]) {
                                queue.offerFirst(next);
                            }else {
                                queue.offerLast(next);
                            }
                        }
                    }
                }
            }
            return pre[end] != -1;
        }
    }
}
```

#### 2.5.12 Bipartite Matching

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

> 匈牙利算法

从左侧的一个非匹配点出发。

从右向左的边，永远走匹配边。

匹配边和非匹配边交替出现：交替路

终止于另外一个非匹配点：增广路径

有增广路径，意味着最大匹配数可以加一。

对左侧每一个尚未匹配的点，不断地寻找可以增广的交替路。

例题：[LCP 04. 覆盖](https://leetcode.cn/problems/broken-board-dominoes/)

```java
class Solution {
    public int domino(int n, int m, int[][] broken) {
        int[][] board = new int[n][m];
        for(int[] b : broken) {
            board[b[0]][b[1]] = 1;
        }
        List<Integer>[] graph = new List[n*m];
        Arrays.setAll(graph, k -> new ArrayList<>());
        for(int i = 0; i < n; i ++) {
            for(int j = 0; j < m; j ++) {
                if(j + 1 < m && board[i][j] == 0 && board[i][j+1] == 0) {
                    graph[i*m+j].add(i*m+j+1);
                    graph[i*m+j+1].add(i*m+j);
                }
                if(i + 1 < n && board[i][j] == 0 && board[i+1][j] == 0) {
                    graph[i*m+j].add((i+1)*m+j);
                    graph[(i+1)*m+j].add(i*m+j);
                }
            }
        }
        matching = new int[n*m];
        visited = new boolean[n*m];
        Arrays.fill(matching, -1);
        int ans = 0;
        for(int i = 0; i < n; i ++) {
            for(int j = 0; j < m; j ++) {
                Arrays.fill(visited, false);
                if(board[i][j] == 0 && (i + j) % 2 == 0) {
                    if(dfs(i*m+j, graph)) {
                        ans ++;
                    }
                }
            }
        }
        return ans;
    }
    private boolean[] visited;
    private int[] matching;

    private boolean dfs(int v, List<Integer>[] graph) {
        visited[v] = true;
        for(int adj : graph[v]) {
            if(!visited[adj]) {
                visited[adj] = true;
                if(matching[adj] == -1 || dfs(matching[adj], graph)) {
                    matching[adj] = v;
                    matching[v] = adj;
                    return true;
                }
            } 
        }
        return false;
    }
}
```
#### 2.5.13 Summarization

| 算法/问题      | 无向/有向  | 无权/带权  | 思想/方法                      | 时间复杂度   |
| -------------- | ---------- | ---------- | ------------------------------ | ------------ |
| 无向图连通分量 | 无向       | 无影响     | DFS/BFS/并查集                 | $O(m+n)$     |
| 无向图环检测   | 无向       | 无影响     | DFS/BFS/并查集                 | $O(m+n)$     |
| 二分图检测     | 无向       | 无影响     | DFS/BFS                        | $O(m+n)$     |
| 单源路径       | 均可       | 无影响     | DFS/BFS(最短)                  | $O(m+n)$     |
| 桥和割点       | 无向       | 无影响     | Tarjan算法：ord,low数组        | $O(m+n)$     |
| 哈密尔顿路径   | 无向       | 无影响     | DFS+状态压缩                   | $O(n·2^n)$   |
| 欧拉路径       | 均可       | 无影响     | DFS+删边                       | $O(m+n)$     |
| 洪水填充       | 无向       | 无影响     | DFS+BFS                        | $O(m+n)$     |
| 最小生成树     | 无向       | 带权       | Kruskal(并查集)/Prim(优先队列) | $O(m\log m)$ |
| 无权图最短路径 | 均可       | 无权       | BFS                            | $O(m+n)$     |
| 带权图最短路径 | 均可       | 带权       | Dijkstra+优先队列              | $O(m\log m)$ |
|                |            |            | BellmanFord                    | $O(mn)$      |
|                |            |            | SPFA                           | $O(km)$      |
|                |            |            | Floyed                         | $O(n^3)$     |
| 有向图环检测   | 有向       | 无影响     | DFS+onPath/拓扑排序            | $O(m+n)$     |
| 拓扑排序       | 有向       | 无影响     | BFS/DFS逆序                    | $O(m+n)$     |
| 有向图连通分量 | 有向       | 无影响     |                                |              |
| 网络流         | 有向       | 权值为流量 |                                |              |
| 匹配           | 有向二分图 | 权值为1    | 匈牙利算法                     |              |
|                |            |            |                                |              |
|                |            |            |                                |              |
|                |            |            |                                |              |
|                |            |            |                                |              |


以一道综合题，串联图论知识。

例题：[1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/)

解法一：基于最短路径Dijkstra算法

解法二：基于最小生成树Kruskal算法

解法三：基于二分答案+DFS寻路

解法四：基于备忘录+BFS

参考题解：https://leetcode.cn/problems/path-with-minimum-effort/solutions/460667/javasi-chong-jie-fa-zui-duan-lu-zui-xiao-sheng-che/

### 2.6 Data Structure Design
#### 2.6.1 LRU
例题：[146. LRU 缓存](https://leetcode.cn/problems/lru-cache)

分析：

使用双向链表 + 哈希表，实现O(1)时间复杂度。

get操作，存在缓存中，使用HashMap获取，同时更新key的最近使用，将其移动到链表头部。

get操作，不存在缓存中，返回-1即可。

put操作，存在缓存中，使用HashMap更新，同时更新key的最近使用，将其移动到链表头部。

put操作，不存在缓存中，缓存未满，往链表头部添加节点；缓存已满，删除链表尾部节点，往链表头部添加节点。

辅助方法：

1. 删除链表尾部节点removeTail()
2. 移动节点到头部move2head()，本质是删除该节点，再调用add2head
3. 添加节点到头部add2head()

```java
public class LRUCache {
    private class ListNode {
        int key;
        int value;
        ListNode pre;
        ListNode post;
        public ListNode(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private int capacity;
    private Map<Integer, ListNode> map;
    private ListNode head;
    private ListNode tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new HashMap<>();
        this.head = new ListNode(-1, -1);
        this.tail = new ListNode(-1, -1);
        head.post = tail;
        tail.pre = head;
    }

    public int get(int key) {
        if (map.containsKey(key)) {
            ListNode node = map.get(key);
            move2head(node);
            return node.value;
        }
        return -1;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            ListNode node = map.get(key);
            node.value = value;
            move2head(node);
        } else {
            if (map.size() == capacity) {
                ListNode node = removeTail();
                map.remove(node.key);
            }
            ListNode node = new ListNode(key, value);
            map.put(key, node);
            add2head(node);
        }
    }
    
    private ListNode removeTail() {
        ListNode oldTail = tail.pre;
        ListNode newTail = oldTail.pre;
        newTail.post = tail;
        tail.pre = newTail;
        oldTail.pre = null;
        oldTail.post = null; 
        return oldTail;
    }

    private void move2head(ListNode node) {
        node.pre.post = node.post;
        node.post.pre = node.pre;
        node.pre = null;
        node.post = null;
        add2head(node);
    }
    
    private void add2head(ListNode node) {
        ListNode headPost = head.post;
        headPost.pre = node;
        node.post = headPost;
        node.pre = head;
        head.post = node;
    }
}
```
#### 2.6.2 LFU

例题：[460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/)

分析：

增加一个Node节点，记录key和value，以及访问的频率。

get操作，key不存在，返回-1。

get操作，key存在，返回value，更新频率。用一个Map记录每一个频率下包含哪些Node，使用LinkedHashSet，既能高效删除节点，又能维持节点的插入顺序（利用了Java现成的集合，也可以采用LRU中双端链表+哈希表的方法）。

更新频率时，将节点频率加一，同时从旧频率下的HashSet中移除，添加到新频率下的HashSet。特别的，用一个变量记录全局最低频率。

put操作，key存在，返回value，更新频率。

put操作，key不存在，进行添加。如果缓存容量已满，则根据最低频率找到LinkedHashSet，去除头节点。添加节点后，最低频率更新为1。

```java
class LFUCache {

    private Map<Integer, Node> map;
    private Map<Integer, LinkedHashSet<Node>> freqMap;

    private int capacity;
    private int minFreq;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        this.map = new HashMap<>();
        this.freqMap = new HashMap<>();
    }

    class Node {
        int value;
        int freq;
        int key;
        Node(int key, int value, int freq) {
            this.value = value;
            this.freq = freq;
            this.key = key;
        }
    }
    
    public int get(int key) {
        if(map.containsKey(key)) {
            Node node = map.get(key);
            updateFreq(node);
            return node.value;
        }else {
            return -1;
        }
    }

    private void updateFreq(Node node) {
        int freq = node.freq;
        LinkedHashSet<Node> set = freqMap.get(freq);
        set.remove(node);
        if(freq == minFreq && set.size() == 0) {
            minFreq ++;
        }
        node.freq ++;
        freqMap.computeIfAbsent(freq + 1, k -> new LinkedHashSet<>()).add(node);
    }

    private void discard() {
        LinkedHashSet<Node> set = freqMap.get(minFreq);
        Node removeNode = set.iterator().next();
        set.remove(removeNode);
        map.remove(removeNode.key);
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)) {
            Node node = map.get(key);
            node.value = value;
            updateFreq(node);
        }else {
            if(map.size() == capacity) {
                discard();
            }
            Node node = new Node(key, value, 1);
            minFreq = 1;
            map.put(key, node);
            freqMap.computeIfAbsent(minFreq, k -> new LinkedHashSet<>()).add(node);
        }
    }
}
```

## Part 3 Algorithm

### 3.1 Data Scale and Time Complexity

| 时间复杂度     | 数据规模      | 常见算法                  |
| -------------- | ------------- | ------------------------- |
| $O(n!)$        | $\le 11$      | 全排列                    |
| $O(3^n)$       | $\le 20$      | 枚举子集                  |
| $O(2^n)$       | $\le 25$      | 递归与回溯                |
| $O(n^3)$       | $\le 500$     | 三重循环，如Floyed算法    |
| $O(n^2\log n)$ | $\le 1000$    | 二分答案，BellmanFord算法 |
| $O(n^2)$       | $\le 5000$    | 二重循环                  |
| $O(n\log n)$   | $\le 10^6$    | 排序，优先队列，线段树    |
| $O(\sqrt n)$   | $\le 10^9$    | 判断质数                  |
| $O(\log n)$    | $\le 10^{18}$ | 二分、快速幂、数位DP      |
|                |               |                           |

有时候，我们需要根据数据规模反向推断需要选用什么时间复杂度的算法。

例题：[6988. 统计距离为 k 的点对](https://leetcode.cn/problems/count-pairs-of-points-with-distance-k/)

分析：朴素的做法，枚举点计算距离，时间复杂度$O(n^2)$，在该题的数据规模下会超时。

观察题目数据规模，由于$k$很小，可以枚举$k$的值，时间复杂度为$O(nk)$
```java
class Solution {
    public int countPairs(List<List<Integer>> coordinates, int k) {
        int ans = 0;
        Map<Long, Integer> map = new HashMap<>(); // 用Long表示(x, y)的组合
        for(List<Integer> coor : coordinates) {
            int x = coor.get(0), y = coor.get(1);
            for(int i = 0; i <= k; i ++) {
                long targetX = i ^ x;
                long targetY = (k - i) ^ y;
                long key = (targetX << 32) | targetY;
                ans += map.getOrDefault(key, 0);
            }
            long key = ((long) x << 32) | y;
            map.merge(key, 1, Integer::sum);
        }
        return ans;
    }
}
```

#### 3.1 Optimization Algorithm

##### 3.1.1 Using Monotonic Stack/Queue
例题：[2866. 美丽塔 II](https://leetcode.cn/problems/beautiful-towers-ii/)

分析：采用暴力，枚举山顶的位置，时间复杂度$O(n^2)$，会超时，采用单调栈优化。

```java
class Solution {
    public long maximumSumOfHeights(List<Integer> maxHeights) {
        int n = maxHeights.size();
        long[] suffix = new long[n+1];  // suffix[i]表示以i最为山顶，i+1往后的最大高度值。
        Stack<Integer> stack = new Stack<>();
        stack.push(n);  // 哨兵，避免对栈的空判断。
        long sum = 0, pre = 0, ans = 0;
        // pre[i]表示以i为山顶，区间[0...i]的最大高度值，包含i，优化为一个变量。
        for(int i = n - 1; i >= 0; i --) {
            while(stack.size() > 1 && maxHeights.get(stack.peek()) >= maxHeights.get(i)) { //注意不是stack.isEmpty()，栈底有元素n。
                int j = stack.pop();
                // 假设j右侧有索引k，且h[k]是右侧第一个小于h[j]的下标，之前j为山顶时，区间[k+1...j]的值全为h[j]。
                sum -= (long)maxHeights.get(j) * (stack.peek() - j);
            }
            // 此时栈顶元素假设为j，h[j] < h[i]。区间[i...j-1]的元素都更新为h[i]，总共j-1-i+1=j-i个。
            sum += (long)maxHeights.get(i) * (stack.peek() - i);
            suffix[i] = sum;
            stack.push(i);
        }
        stack.clear();
        stack.push(-1);
        for(int i = 0; i < n; i ++) {
            while(stack.size() > 1 && maxHeights.get(stack.peek()) >= maxHeights.get(i)) {
                int j = stack.pop();
                pre -= (long)maxHeights.get(j) * (j - stack.peek());
            }
            pre += (long)maxHeights.get(i) * (i - stack.peek());
            ans = Math.max(ans, pre + suffix[i+1]);
            stack.push(i);
        }
        return ans;
    }
}
```
时间复杂度：$O(n)$
##### 3.1.2 Using Binary Search
##### 3.1.3 Using Segment Tree
##### 3.1.4 Using Hashtable

### 3.2 Two Pointers 

#### 3.2.1 Fast and Slow Pointers

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 低       |

在链表章节，我们介绍过快慢指针用于找环，该技巧在数组中依然适用。

例题：[287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int fast = nums[nums[0]], slow = nums[0];
        while(slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        fast = 0;
        while(slow != fast) {
            fast = nums[fast];
            slow = nums[slow];
        }
        return slow;
    }
}
```

#### 3.2.2 Head and Tail Pointers

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

首尾指针可以方便地求解两数之和及其变体问题。

两数之和：假设数据有序，在数组$arr$中寻找下标$i,j$，使得$arr[i]+arr[j]=target$

```java
int l = 0, r = arr.length - 1;
while(l < r) {
    if(arr[l] + arr[r] == target) {
    	// 找到，进行相应的业务处理
        l ++;
        r --;
    }else if(arr[l] + arr[r] > target) {
        r --;
    }else {
        l ++;
    }
}
```

例题：[18. 四数之和](https://leetcode.cn/problems/4sum/)

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> ret = new ArrayList<>();
        for(int i = 0; i < nums.length; i ++) {
            if(i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            for(int j = i+1; j < nums.length; j++) {
                if(j > i+1 && nums[j] == nums[j-1]) {
                    continue;
                }
                int k = j + 1, m = nums.length - 1;
                while(k < m) {
                    long sum = (long)nums[i] + nums[j] + nums[k] + nums[m];
                    if(sum == target) {
                        ret.add(Arrays.asList(nums[i], nums[j], nums[k], nums[m]));
                        while(k < nums.length - 1 && nums[k+1] == nums[k]) {
                            k ++;
                        }
                        while(m > 0 && nums[m-1] == nums[m]) {
                            m --;
                        }
                        k ++;
                        m --;
                    } else if(sum < target) {
                        k ++;
                    } else {
                        m --;
                    }
                }
            }
        }
        return ret;
    }
}
```

时间复杂度：$O(n^3)$

如果需要求有多少个四元组，满足相加和为target又该怎么处理呢？

练习题单

| 题号                                                         | 难度 | 知识点 |
| ------------------------------------------------------------ | ---- | ---- |
| [15. 三数之和](https://leetcode.cn/problems/3sum/)           | 中等 | 双指针
| [16. 最接近的三数之和](https://leetcode.cn/problems/3sum-closest/) | 中等 | 双指针
| [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/) | 中等 | 双指针+贪心
| [923. 三数之和的多种可能](https://leetcode.cn/problems/3sum-with-multiplicity/) | 中等 | 双指针+计数/组合数

#### 3.2.3 Sliding Window

滑动窗口的核心：两个不回退的指针顺序遍历数组。

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 中       | 中       | 建议掌握 |

不定长滑动：

```java
int[] arr = new int[]{...};
int l = 0, r = 0;
while(r < arr.length) {  // l和r指针不会回退，时间复杂度O(n)
    while(condition) {  // 当满足某个条件时，收缩左边界
        l ++;
    }
    // 执行到这段代码时，condition条件一定不满足
    // 根据实际情况，判断是在满足条件时进行计数/更新答案还是在不满足条件时计数/更新答案。
    r ++;
}
```

由于指针不回退，虽然有两重循环，但时间复杂度为$O(n)$。

例题：[76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

```java
class Solution {
    public String minWindow(String s, String t) {
        int[] count = new int[256];
        int[] window = new int[256];
        int match = 0, l = 0, r = 0, cnt = 0;   // match表示s中匹配的数量，cnt表示t中不同字符的数量。
        for(char c : t.toCharArray()) {
            count[c] ++;
            if(count[c] == 1) {
                cnt ++;
            }
        }
        String ret = "";
        while(r < s.length()) {
            char c = s.charAt(r);
            window[c] ++;
            if(window[c] == count[c]) {
                match ++;
            }
            while(match == cnt) {  // 对应模版中的condition条件
                if(ret.equals("")) {   // 计数 & 更新答案
                    ret = s.substring(l, r + 1);
                } else if(ret.length() > r - l + 1) {
                    ret = s.substring(l, r + 1);
                }
                char ch = s.charAt(l);
                window[ch] --;
                if(window[ch] < count[ch]) {
                    match --;
                }
                l ++;  // 收缩左边界
            }
            r ++;
        }
        return ret;
    }
}
```

注意：引入cnt和match是为了避免比较数组相等，因为Java中数组用等号判断相等，只会比较地址是否相同。例题438采用了数组比较的写法，可以对比学习。

定长滑动：当窗口长度确定时，可以采用滚动的方式更新。滚动的思想在之前介绍**字符串匹配**算法的Rabin Karp算法中有体现。

例题：[438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ret = new ArrayList<>();
        int m = s.length(), n = p.length();
        if(m < n) {
            return ret;
        }
        int[] need = new int[256], window = new int[256];
        for(int i = 0; i < n; i ++) {
            need[p.charAt(i)] ++;
        }
        for(int i = 0; i < n - 1; i ++) {
            char c = s.charAt(i);
            window[c] ++;
        }
        for(int i = n - 1; i < m; i ++) {
            char c = s.charAt(i);
            window[c] ++;
            if(equals(window, need)) {
                ret.add(i - n + 1);
            }
            window[s.charAt(i - n + 1)] --;
        }
        return ret;
    }

    private boolean equals(int[] window, int[] need) {
        for(int i = 0; i < window.length; i ++) {
            if(window[i] != need[i]) {
                return false;
            }
        }
        return true;
    }
}
```

滑动窗口算法的特点在于指针不会回退，指针不回退的前提要求数组具有单调性，从以下例题深入理解这一点。

例题：[209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

分析：先从0作为左端点，由于不存在负数，所以区间和具有单调性，假设指针到达i时，区间和[0...r] $\ge$ target。此时指针r向后的所有子数组一定满足条件。由于需要求最短子数组，此时记录答案即可。

现考虑1作为左端点，由于单调性的存在，r指针一定不会回退，因为r是从0开始的子数组中大于等于target中最小的右端点。在满足条件的前提下，移动l指针，更新答案。当移动l不能满足条件后，则需要继续移动r指针。每次满足条件后(while循环)都需要更新答案。

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        int[] presum = new int[n+1];
        for(int i = 0; i < n; i ++) {
            presum[i+1] = presum[i] + nums[i];
        }
        int l = 0, r = 0, ans = n + 1;
        while(r < nums.length) {
            int sum = presum[r + 1] - presum[l];
            while(sum >= target) {
                ans = Math.min(ans, r - l + 1);
                sum -= nums[l];
                l ++;
            }
            r ++;
        }
        return ans == n + 1 ? 0 : ans;
    }
}
```

如果存在负数，则区间和不具有单调性，此时无法用滑动窗口求解。

例如，数组[84,-37,32,40,95]中寻找167，按照滑动窗口的做法，l指针全程不会移动，计算出的长度为5，但实际值为3，即选择[32,40,95]子数组。在收缩左端点时，会出现满足条件(l=0,r=4,sum>=k) -> 不满足条件(l=1,r=4,sum < k) -> 满足条件(l=2,r=4,sum >=k)的状态，因此滑动窗口的解法是错误的。

下面变式题即为存在负数的情况，需要借助于**单调队列**来求解，该部分题目解析放在单调队列章节。

变式：[862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)

练习题单

| 题号                                                         | 难度 | 知识点 |
| ------------------------------------------------------------ | ---- | ---- |
| [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)           | 中等 | 滑动窗口 + 哈希表 |
| [992. K 个不同整数的子数组](https://leetcode.cn/problems/subarrays-with-k-different-integers/) | 困难 | 滑动窗口 + 思维 | 
| [632. 最小区间](https://leetcode.cn/problems/smallest-range-covering-elements-from-k-lists/) | 困难 | 滑动窗口+哈希表(类似76题)/贪心+堆

#### 3.2.4 Pointer Counting

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

例题：[259. 较小的三数之和](https://leetcode.cn/problems/3sum-smaller/)

分析：结合首尾指针的思想进行计数即可。

```java
class Solution {
    public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int ans = 0;
        for(int i = 0; i < nums.length; i ++) {
            int j = i + 1, k = nums.length - 1;
            while(j < k) {
                if(nums[i] + nums[j] + nums[k] < target) {
                    ans += k - j;
                    j ++;
                } else {
                    k --;
                }
            }
        }
        return ans;
    }
}
```

时间复杂度：$O(n^2)$

### 3.3 Sorting Algorithm

| 面试概率 | 笔试概率 | 学习建议 |
| -------- | -------- | -------- |
| 高       | 高       | 必须掌握 |

十大排序算法大比较

| 排序算法 | 时间复杂度   | 稳定性 | 特点                                                         |
| -------- | ------------ | ------ | ------------------------------------------------------------ |
| 选择排序 | $O(n^2)$     | 不稳定 |                                                              |
| 插入排序 | $O(n^2)$     | 稳定   | 完全有序时，时间复杂度$O(n)$                                 |
| 冒泡排序 | $O(n^2)$     | 稳定   | 完全有序时，时间复杂度$O(n)$                                 |
| 希尔排序 | $<O(n^2)$    | 不稳定 |                                                              |
| 快速排序 | $O(n\log n)$ | 不稳定 | 时间复杂度可能退化为$O(n^2)$，应用：划分解决SELEC K问题      |
| 归并排序 | $O(n\log n)$ | 稳定   | 非原地，完全有序时，时间复杂度$O(n)$，应用：归并解决逆序对问题 |
| 堆排序   | $O(n\log n)$ | 不稳定 | 应用：堆解决TOP K问题                                        |
| 计数排序 |              | 稳定   |                                                              |
| 基数排序 |              | 稳定   | LSD适用于等长字符串，数字需要补零对齐；MSD适用于不等长字符串，无法用于数字 |
| 桶排序   |              |        | 应用：桶思想解决最大间隔问题                                 |

> 选择排序

循环不变量： $arr[0...i)$是有序的，$arr[i...n)$是无序的

```java
public void selectionSort(int[] arr) {
    for (int i = 0; i < arr.length; i++) {
        // 选择 arr[i...n)中的最小值的索引
        int minIndex = i;
        for (int j = i; j < arr.length; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        // arr[i] 放置 arr[i...n) 最小值
        swap(arr, i, minIndex);
    }
}
private static void swap(int[] arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}
```

> 插入排序

循环不变量： $arr[0...i)$是有序的，$arr[i...n)$是无序的

```java
public void insertSort(int[] arr) {
    for (int i = 0; i < arr.length; i++) {
        int temp = arr[i];
        int j;
        for (j = i; j >= 1 && arr[j-1] > temp; j --) {  // 完全有序时，内存循环不会执行，时间复杂度为O(n) 若条件写为arr[j-1] >= temp，则排序算法不稳定
            arr[j] = arr[j-1];
        }
        arr[j] = temp;
    }
}
```

插入排序和选择排序的区别？

选择排序：每次遍历下标$i$后，$arr[i]$的位置在后续遍历过程中不会再改变。

插入排序：每次遍历下标$i$后，$arr[i]$的位置在后续遍历过程中还可能向后移动。

> 冒泡排序

```java
public void bubbleSort(int[] arr) {
    for (int i = 0; i < arr.length - 1; i++) {
        boolean isSwapped = false;
        // arr[n-i, n) 已经排好序
        // 冒泡在arr[n-i-1]位置放置合适元素
        for (int j = 0; j < arr.length - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {  // 检测是否存在逆序对
                swap(arr, j, j + 1);
                isSwapped = true;
            }
        }
        if (!isSwapped) {   // 完全有序时，时间复杂度O(n)
            break;
        }
    }
}
```

> 希尔排序

希尔排序结合了冒泡排序和插入排序的思想

```java
public void shellSort2(int[] arr) {
    int h = arr.length / 2;
    while (h >= 1) {
        for (int i = h; i < arr.length; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j - h >= 0 && arr[j - h] > temp; j -= h) {
                arr[j] = arr[j - h];
            }
            arr[j] = temp;
        }
        h /= 2;
    }
}
```

> 归并排序

```java
public void mergeSort(int[] arr) {
    temp = new int[arr.length];
    mergeSort(arr, 0, arr.length - 1);
}
private void mergeSort(int[] arr, int l, int r) {
    if (l >= r) {
        return;
    }
    int mid = (l + r) / 2;
    mergeSort(arr, l, mid);
    mergeSort(arr, mid + 1, r);
    if (arr[mid] > arr[mid + 1]) {   // 有序
        merge(arr, l, mid, r);
    }
}
private int[] temp;
private void merge(int[] arr, int l, int mid, int r) {
	for (int i = l; i <= r; i++) {
        temp[i] = arr[i];
    }
    int i = l;
    int j = mid + 1;
    for (int k = l; k <= r; k++) {
        if (i > mid) {
            arr[k] = temp[j];
            j ++;
        } else if (j > r) {
            arr[k] = temp[i];
            i ++;
        } else if (temp[i] < temp[j]) {
            arr[k] = temp[i];
            i ++;
        } else {
            arr[k] = temp[j];
            j ++;
        }
    }
}
// 自底向上归并排序
public void mergeSortBottomUp(int[] arr) {
	int n = arr.length;
    temp = new int[n];
    for (int size = 1; size < n; size += size) {
        for (int i = 0; i + size < n; i += 2 * size) {
            // 合并[i, i + size - 1] 和 [i + size, Math.min(i + size + size - 1, n - 1)]
            if (arr[i + size - 1] > arr[i + size]) {
                merge(arr, i, i + size - 1, Math.min(i + size + size - 1, n - 1));
            }
        }
    }
}
```

> 快速排序

经典快速排序

```java
public void quickSort(int[] arr) {
    quickSort(arr, 0, arr.length - 1);
}
private void quickSort(int[] arr, int l, int r) {
    if (l >= r) {
        return;
    }
    int p = partition(arr, l, r);
    quickSort(arr, l, p - 1);
    quickSort(arr, p + 1, r);
}
Random random = new Random();
private int partition(int[] arr, int l, int r) {
    int randomIndex = l + random.nextInt(r - l + 1);
    swap(arr, l, randomIndex);
    int j = l;
    for (int i = l + 1; i <= r; i++) {
        if (arr[i] < arr[l]) {
            j ++;
            swap(arr, i, j);
        }
    }
    swap(arr, l, j);
    return j;
}
```

经典快速排序的两大缺陷：

- 当数组完全有序(正序/逆序)时，时间复杂度会退化到$O(n^2)$。解决思路：随机pivot。
- 当数组元素完全相同时，时间复杂度会退化到$O(n^2)$。解决思路：二路快速排序/三路快速排序。

二路快速排序

略，可参考**数组划分**章节中partition的相关实现。

三路快速排序

```java
public void quickSort3ways(int[] arr) {
    quickSort3ways(arr, 0, arr.length - 1);
}
public static void quickSort3ways(int[] arr, int l, int r) {
    if (l >= r) {
        return;
    }
    int randomIndex = l + random.nextInt(r - l + 1);
    swap(arr, l, randomIndex);
    int lt = l;
    int i = l + 1;
    int gt = r + 1;
    while (i < gt) {
        if (arr[i] < arr[l]) {
            lt ++;
            swap(arr, lt, i);
            i ++;
        } else if(arr[i] > arr[l]) {
            gt --;
            swap(arr, i, gt);
        } else {
            i ++;
        }
    }
    swap(arr, l, lt);
    quickSort3ways(arr, l, lt - 1);
    quickSort3ways(arr, gt, r);
}
```

> 堆排序

```java
public void heapSort(int[] data) {
   	if (data.length <= 1) {
        return;
    }
    for (int i = (data.length - 2) / 2; i >= 0; i --) {
        siftDown(data, i, data.length);
    }
    for (int i = data.length - 1; i >= 0; i --) {
        swap(data, 0, i);
        siftDown(data, 0, i);
    }
}
public void siftDown(int[] data, int k, int n) {
    while (2 * k + 1 < n) {
        int j = 2 * k + 1;
        if (j + 1 < n && data[j] < data[j + 1]) {
            j ++;
        }
        if (data[k] >= data[j]) {
            break;
        }
        swap(data, k, j);
        k = j;
    }
}
public void swap(int[] data, int i, int j) {
    int temp = data[i];
    data[i] = data[j];
    data[j] = temp;
}
```

> 计数排序

当已知数据范围时，可以采用计数排序。

假设元素取值范围为$[0, R)$，用一个$count[R]$数组记录取值为$[0, R)$的每个取值出现的次数。用一个$index[R+1]$数组记录值为$i$的元素出现的区间为$[index[i], index[i+1])$。

以元素$0,1,2$举例：

```java
int[] count = new int[3];
for(int num : nums) {
    count[num] ++;
}
for(int i = 0; i < count[0]; i++) {
    nums[i] = 0;
}
for(int i = count[0]; i < count[0] + count[1]; i++) {
    nums[i] = 1;
}
for(int i = count[0] + count[1]; i < count[0] + count[1] + count[2]; i++) {
    nums[i] = 2;
}
```

可见，在区间$[0, count[0])$全是$0$；$[count[0], count[0]+count[1])$全是$1$，$[count[0]+count[1], count[0]+count[1]+count[2])$全是$2$。

$index$数组则为$[0, count[0], count[0]+count[1], count[0]+count[1]+count[2]]$，即为$count$数组的前缀和数组。

更一般化：

```java
int[] count = new int[R];
for(int num : nums) {
    count[num] ++;
}
int[] index = new int[R+1];
for(int i = 0; i < R; i++) {
    index[i+1] = index[i] + count[i];
}
for(int i = 0; i+1 < index.length; i++) {
    for(int j = index[i]; j < index[i+1]; j++) {
    	nums[j] = i;
    }
}
```

上述代码仍然不够一般化，因为最后是将索引$i$赋值给$nums$数组，倘若需要排序的不是整型数组，则需要辅助数组，并根据$index[i]$将其放在对应的位置。定义循环不变量$index[i]$表示$i$的起始索引，每当将其放置在对应位置之后，$index[i]$进行自增。可以看出，计数算法是稳定的。该算法的具体实现，参见下一节LSD基数排序。

例题：[2653. 滑动子数组的美丽值](https://leetcode.cn/problems/sliding-subarray-beauty/)

```java
class Solution {
    public int[] getSubarrayBeauty(int[] nums, int k, int x) {
        int offset = 50, n = nums.length;
        int[] count = new int[offset], ret = new int[n - k + 1];
        for(int i = 0; i < k - 1; i ++) {
            if(nums[i] < 0) {
                count[nums[i] + offset] ++;
            }
        }
        for(int i = k - 1; i < n; i ++) {
            if(nums[i] < 0) {
                count[nums[i] + offset] ++;
            }
            int order = x;
            for(int j = 0; j < offset; j ++) {
                order -= count[j];
                if(order <= 0) {
                    ret[i - k + 1] = j - offset;
                    break;
                }
            }
            if(nums[i - k + 1] < 0) {
                count[nums[i - k + 1] + offset] --;
            }
        }
        return ret;
    }
}
```


> LSD基数排序

LSD(Least Significant Digit)基数排序对字符串排序，要求字符串长度相等，从末位向前排序。

```java
public void lsdSort(String[] arr, int n) {   // n = str.length()
    int r = 256;
    int[] cnt = new int[r];
    String[] temp = new String[arr.length];
    int[] index = new int[r + 1];
    for (int ri = n - 1; ri >= 0; ri --) {
        Arrays.fill(cnt, 0);
        for (String s : arr) {
            cnt[s.charAt(r)] ++;
        }
        for (int i = 0; i < r; i++) {
            index[i + 1] = index[i] + cnt[i];
        }
        for (String s : arr) {
            temp[index[s.charAt(r)]] = s;
            index[s.charAt(r)] ++;
        }
        for (int i = 0; i < arr.length; i++) {
            arr[i] = temp[i];
        }
    }
}
```

> MSD基数排序

MSD(Most Significant Digit)基数排序，对于字符串排序，不需要字符串长度相等。

现给出一个例子，有字符串BCA，CBAA，AC，BADFE，ABC，CBA，运用MSD基数排序。

加粗字体表示当前轮次比较的字符。空（长度不够）对应的“字符的值”最小。

| 初始  | 第一轮    | 第二轮    | 第四轮   |
| ----- | --------- | --------- | -------- |
| BCA   | **A**C    | A**B**C   |          |
| CBAA  | **A**BC   | A**C**    |          |
| AC    | **B**CA   | B**A**DFE |          |
| BADFE | **B**ADFE | B**C**A   |          |
| ABC   | **C**BAA  | C**B**AA  | CBA      |
| CBA   | **C**BA   | C**B**A   | CBA**A** |

```java
public void msdSort(String[] arr) {
    int n = arr.length;
    String[] temp = new String[n];
    msdSort(arr, 0, n - 1, 0, temp);
}
private void msdSort(String[] arr, int left, int right, int r, String[] temp) {
    if (left >= right) {
        return;
    }
    int n = 256;
    int[] cnt = new int[n + 1];
    int[] index = new int[n + 2];
    for (int i = left; i <= right; i++) {
        cnt[r >= arr[i].length() ? 0 : (arr[i].charAt(r) + 1)] ++;
    }
    for (int i = 0; i < n + 1; i++) {
        index[i + 1] = index[i] + cnt[i];
    }
    for (int i = left; i <= right; i++) {
        temp[index[r >= arr[i].length() ? 0 : (arr[i].charAt(r) + 1)] + left] = arr[i];
        index[r >= arr[i].length() ? 0 : (arr[i].charAt(r) + 1)] ++;
    }
    for (int i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
    for (int i = 0; i < n; i++) {
        msdSort(arr, left + index[i], left + index[i + 1] - 1, r + 1, temp);
    }
}
```

> 桶排序

```java
public void bucketSort(int[] nums, int interval) {
    int minValue = Arrays.stream(nums).min().getAsInt();
    int maxValue = Arrays.stream(nums).max().getAsInt();
    int bucketNum = (maxValue - minValue) / interval + 1;
    ArrayList<Integer>[] buckets = new ArrayList[bucketNum];
    for(int i = 0; i < bucketNum; i++) {
        buckets[i] = new ArrayList<>();
    }
    for(int num : nums) {
        buckets[(num - minValue) / interval].add(num);
    }
    for(int i = 0; i < bucketNum; i++) {
        Collections.sort(buckets[i]);
    }
    int index = 0;
    for(int i = 0; i < bucketNum; i++) {
        for(int num : buckets[i]) {
            nums[index++] = num;
        }
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [912. 排序数组](https://leetcode.cn/problems/sort-an-array/) | 中等 |

### 3.4 Recursive, Backtracking and Divide and Conquer

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 高       |

> 递归经典问题之嵌套问题

例题：[726. 原子的数量](https://leetcode.cn/problems/number-of-atoms/)

```java
class Solution {
    public String countOfAtoms(String formula) {
        Map<String, Integer> map = dfs(formula.toCharArray(), 0);
        StringBuilder sb = new StringBuilder();
        for(Map.Entry<String, Integer> entry : map.entrySet()) {
            sb.append(entry.getKey());
            if(entry.getValue() > 1) {
                sb.append(entry.getValue());
            }
        }
        return sb.toString();
    }

    private int index = 0;

    public Map<String, Integer> dfs(char[] c, int i) {
        Map<String, Integer> map = new TreeMap<>();
        Map<String, Integer> sub = null;
        int cnt = 0;
        StringBuilder name = new StringBuilder();
        while(i < c.length && c[i] != ')') {
            if(Character.isLowerCase(c[i])) {
                name.append(c[i]);
                i ++;
            }else if(Character.isDigit(c[i])) {
                cnt = cnt * 10 + c[i] - '0';
                i ++;
            }else {
                fill(map, sub, name, cnt);
                cnt = 0;
                name.setLength(0);
                sub = null;
                if(Character.isUpperCase(c[i])) {
                    name.append(c[i]);
                    i ++;
                } else {
                    sub = dfs(c, i + 1);
                    i = index + 1;
                }
            }
        }
        index = i;
        fill(map, sub, name, cnt);
        return map;
    }

    private void fill(Map<String, Integer> map, Map<String, Integer> sub, StringBuilder name, int count) {
        count = count == 0 ? 1 : count;
        if(name.length() > 0) {
            map.merge(name.toString(), count, Integer::sum);
        }
        if(sub != null) {
            for(Map.Entry<String, Integer> entry : sub.entrySet()) {
                map.merge(entry.getKey(), entry.getValue() * count, Integer::sum);
            }
        }
    }
}
```

练习题单


| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [394. 字符串解码](https://leetcode.cn/problems/decode-string/) | 中等 |
| [772. 基本计算器 III](https://leetcode.cn/problems/basic-calculator-iii/) | 困难 |

> 回溯经典问题之子集问题

例题：[78. 子集](https://leetcode.cn/problems/subsets/)

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        dfs(nums, 0, new ArrayList<>());
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    public void dfs(int[] nums, int i, List<Integer> list) {
        if(i == nums.length) {
            ret.add(new ArrayList<>(list));
            return;
        }
        list.add(nums[i]);
        dfs(nums, i+1, list);
        list.remove(list.size() - 1);
        dfs(nums, i+1, list);
    }
}
```

时间复杂度：$O(n \times 2^n)$

在位运算章节，还讲解了基于二进制掩码枚举子集的算法。

如果nums中有重复元素，应该怎样去重呢？

变式题：[90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

不选时，增加如下逻辑：

```java
while(i + 1 < nums.length && nums[i + 1] == nums[i]) {
	i ++;
}
```

> 回溯经典问题之排列问题

例题：[46. 全排列](https://leetcode.cn/problems/permutations/)

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        dfs(nums, new ArrayList<>());
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    public void dfs(int[] nums, List<Integer> onPath) {
        if(onPath.size() == nums.length) {
            ret.add(new ArrayList<>(onPath));
            return;
        }
        for(int n : nums) {
            if(!onPath.contains(n)) {
                onPath.add(n);
                dfs(nums, onPath);
                onPath.remove(onPath.size() - 1);
            }
        }
    }
}
```

基于交换的解法

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        dfs(nums, 0);
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    public void dfs(int[] nums, int i) {
        if(i == nums.length) {
            ret.add(Arrays.stream(nums).boxed().toList());
            return;
        }
        for(int j = i; j < nums.length; j ++) {
            swap(nums, i, j);
            dfs(nums, i + 1);
            swap(nums, i, j);
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

思考：如果包含重复数字，应该怎么解决？

例题：[47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

```java
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        dfs(nums, 0, new boolean[nums.length], new ArrayList<>());
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    public void dfs(int[] nums, int start, boolean[] visited, List<Integer> onPath) {
        if(start == nums.length) {
            ret.add(new ArrayList<>(onPath));
            return;
        }
        for(int i = 0; i < nums.length; i ++) {
            if(visited[i] || (i > 0 && nums[i] == nums[i-1] && !visited[i-1])) {
                continue;
            }
            onPath.add(nums[i]);
            visited[i] = true;
            dfs(nums, start + 1, visited, onPath);
            visited[i] = false;
            onPath.remove(start);
        }
    }
}
```

思考：如果把(i > 0 && nums[i] == nums[i-1] && !visited[i-1])这行判断逻辑改为(i > 0 && nums[i] == nums[i-1] && visited[i-1])，代码仍然能通过，为什么？这两种写法的效率哪个更高？

思考：基于交换的方式，如何书写代码？

时间复杂度：$O(n·n!)$

> 回溯经典问题之组合问题

例题：[39. 组合总和](https://leetcode.cn/problems/combination-sum/)

分析：组合问题核心在于方案不能重复，因此，增加一个变量start，后续选择不能选择start之前的方案。

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        dfs(candidates, target, 0, new ArrayList<>());
        return ret;
    }

    private List<List<Integer>> ret = new ArrayList<>();

    public void dfs(int[] candidates, int target, int start, List<Integer> list) {
        if(target == 0) {
            ret.add(new ArrayList<>(list));
            return;
        }
        for(int i = start; i < candidates.length; i ++) {
            int candidate = candidates[i];
            if(candidate > target) {
                return;
            }
            list.add(candidate);
            dfs(candidates, target - candidate, i, list);
            list.remove(list.size() - 1);
        }
    }
}
```

时间复杂度：$O(n·2^n)$，实际由于剪枝的原因，很难达到此上界。

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/) | 中等 |

其他经典回溯问题，参见题单

| 题号                                                       | 难度 |
| ---------------------------------------------------------- | ---- |
| [51. N 皇后](https://leetcode.cn/problems/n-queens/)       | 困难 |
| [52. N 皇后 II](https://leetcode.cn/problems/n-queens-ii/) | 困难 |
| [37. 解数独](https://leetcode.cn/problems/sudoku-solver/)  | 困难 |

> 分治算法

快速排序算法、归并排序算法很好地体现了分治的思想。

例题：[241. 为运算表达式设计优先级](https://leetcode.cn/problems/different-ways-to-add-parentheses/)

```java
class Solution {
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> list = new ArrayList<>();
        int len = expression.length();
        int start;
        for (start = 0; start < len; start ++) {
            if (!Character.isDigit(expression.charAt(start))) {
                break;
            }
        }
        if (start == len) {
            list.add(Integer.parseInt(expression));
        }
        for (int i = start; i < len; i++) {
            if (Character.isDigit(expression.charAt(i))) {
                continue;
            }
            char op = expression.charAt(i);
            List<Integer> left = diffWaysToCompute(expression.substring(0,i));
            List<Integer> right = diffWaysToCompute(expression.substring(i+1,len));
            for (int j : left){
                for (int k : right){
                    if (op == '+') {
                        list.add(j+k);
                    } else if(op == '-') {
                        list.add(j-k);
                    } else if(op == '*') {
                        list.add(j*k);
                    }
                }
            }
        }
        return list;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/) | 中等 |
| [169. 多数元素](https://leetcode.cn/problems/majority-element/) | 简单 |
| [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/) | 困难 |

### 3.5 Binary Search

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 高       | 高       |

先从一个经典问题进行引入：给定升序数组$arr[i]$，查找目标元素target，返回其下标。不存在返回-1。

三段式二分查找：

```java
public int binarySearch(int[] arr, int target) {
    int l = 0, r = arr.length - 1;
    while(l <= r) {
        int mid = l + r >> 1;
        if(arr[mid] == target) {
            return mid;
        }
        if(arr[mid] < target) {
            l = mid + 1;
        }else {
            r = mid - 1;
        }
    }
    return -1;
}
```

现考虑如下问题：

1. l和r表示的是开区间还是闭区间？开闭区间的代码写法有什么不同？
2. mid的更新方式是否会有问题？
3. 循环的退出条件为什么是l <= r

对于问题1，代码表示的是在arr[l...r]前闭后闭区间进行搜索。如果是前闭后开区间，代码应该做如下改写：

```java
public int binarySearch(int[] arr, int target) {
    int l = 0, r = arr.length;
    while(l < r) {
        int mid = l + r >> 1;
        if(arr[mid] == target) {
            return mid;
        }
        if(arr[mid] < target) {
            l = mid + 1;
        }else {
            r = mid;
        }
    }
    return -1;
}
```

对于问题2，现总结四种计算mid的方式。

| 计算方式          | 特点                                      | 举例                              |
| ----------------- | ----------------------------------------- | --------------------------------- |
| $ l + r >> 1$     | 算术右移，向下取整，可能溢出              | $-4-5 >> 1 = -5$                  |
| $ (l + r) / 2$    | 向下取整，可能溢出，l + r为负数时向上取整 | $(-4-5)/2=-4$                     |
| $l + r >>> 1$     | 逻辑右移，向下取整，不会溢出              | $1 + 2147483647 >>> 1=1073741824$ |
| $(r - l) / 2 + l$ | 不会溢出                                  |                                   |
| $l+r+1>>1$        | 向上取整                                  | $-4-5+1 >> 1 = -4$                |

对于问题3，当$l \le r$时，区间$[l..r]$始终有效，退出循环时，说明未查找到元素，返回-1。

总结：对于二分查找，在写代码时，需要考虑如下四个问题。

- 左右边界的初始取值
- while循环退出条件
- mid的计算方式
- 区间更新方式

三段式的写法比较繁琐，更简洁且更通用的写法是二段式。

> 二段式模版一

```java
public int binarySearch(int[] arr, int target) {
    int l = 0, r = arr.length - 1;
    while(l < r) {
        int mid = l + r >> 1;
        if(arr[l] < target) {
            l = mid + 1;
        }else {
            r = mid;
        }
    }
    return arr[l] == target ? l : -1;
}
```

在二段式的写法中，我们只考虑前闭后闭区间的写法。

思考第一个问题：左右边界取值。左端点l取值为0，右端点为arr.length - 1。

考虑第二个问题：while循环退出条件。两段式中，统一写为$l < r$。当循环退出时，$l = r$。

考虑第三个问题：mid的计算方式。根据是否可能溢出选用合适的即可。

考虑第四个问题：区间的更新方式。

二段式之包含两部分，一个是if段，一个是else段。

if段：用于排除非法区间。如$arr[l] < target$时，说明$[l...mid]$区间不合法，此时$l$一定从$mid+1$开始搜索。

else段：用于缩小区间。如$arr[l] \ge target$时，此时$[l...mid]$区间可能包含了目标索引，此时$r$更新为mid(非$mid-1$)，搜索区间缩小。

当$ l == r$时，此时搜索区间达到最小，只有1个元素，返回前还需要一段逻辑判断，因为$arr[l]$这个元素并没有在循环中判断和target的关系。

该二段式的代码查找元素下标还有一个特点，如果数组中存在多个target，会返回最左边target的下标。因为在else分支时right边界会不断压缩，最终缩小到只有一个元素。

现考虑这个问题，如果数组中存在多个target，想返回最右侧target的下标，需要怎么改写代码呢？

```java
// 错误代码
public int binarySearch(int[] arr, int target) {
    int l = 0, r = arr.length - 1;
    while(l < r) {
        int mid = l + r >> 1;
        if(arr[l] > target) {
            r = mid - 1;
        }else {
            l = mid;
        }
    }
    return arr[l] == target ? l : -1;
}
```

上述代码看似正确，但会出现死循环问题。

现深入探究这个问题。

先考虑为什么会产生死循环。while循环退出的前提是区间不断缩小，要么$l$增加，要么$r$减小。

若某次对于区间的更新，$l$和$r$的值都没有发生变化，则后续无法缩小区间，导致死循环。

首先，导致死循环时，一定存在关系$ l + 1 = r$，即$l$和$r$相邻。

若不相邻，则一定有$r - l > 1$，此时计算出的$mid = l + r >> 1$ 一定满足 $l < mid < r$。

无论走到if分支还是else分支，都会导致$l$或者$r$的值变化。

if分支会减小$mid$，else分支会增大$l$，因为先前$l < mid$。

当$l + 1 = r$时，此时计算出的$mid$因为下取整的原因，会等于$l$。

若走到if分支，$r = mid - 1$，下一次循环时$l = r$会退出循环。

若走到else分支，因为$l = mid$，此时$l$值没有变化，导致死循环。

因此，在计算$mid$时，需要采用向上取整的写法：

>  二段式模版二

```java
public int binarySearch(int[] arr, int target) {
    int l = 0, r = arr.length - 1;
    while(l < r) {
        int mid = l + r + 1 >> 1;
        if(arr[l] < target) {
            r = mid - 1;
        }else {
            l = mid;
        }
    }
    return arr[l] == target ? l : -1;
}
```

当$l + 1 = r$时，此时计算出的$mid$因为上取整的原因，会等于$r$。

若走到if分支，$r = mid - 1=l$，下一次循环时$l = r$会退出循环。

若走到else分支，因为$l = mid=r$，下一次循环时$l = r$会退出循环。

结论：

对于$l = mid + 1$的写法，采用向下取整。

对于$r = mid - 1$的写法，采用向上取整。

二段式除了精确查找元素下标之外，还可以用于解决查找大于某个元素的最小值等问题。

例题：[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

分析：查找元素的第一个位置。根据先前的分析，采用模版一即可实现。

查找元素的最后一个位置，采用模版二可以实现。

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        if(nums.length == 0) {
            return new int[]{-1, -1};
        }
        return new int[]{searchFirstPosition(nums, target), searchLastPosition(nums, target)};
    }

    private int searchFirstPosition(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = l + r >> 1;
            if(nums[mid] < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return nums[l] == target ? l : -1;
    }

    private int searchLastPosition(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = l + r + 1 >> 1;
            if(nums[mid] > target) {
                r = mid - 1;
            } else {
                l = mid;
            }
        }
        return nums[l] == target ? l : -1;
    }
}
```

对于查找最后一个元素的位置，还可以采用一下方式实现。

首先查找到严格大于target的下标，该下标前一个位置一定是target的最后一个位置(前提是搜索第一个位置时target存在)。

```java
private int searchLastPosition(int[] nums, int target) {
    int l = 0, r = nums.length;
    while(l < r) {
        int mid = l + r >> 1;
        if(nums[mid] <= target) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    if(l - 1 >= 0 && nums[l - 1] == target) {
        return l - 1;
    }
    return -1;
}
```

注意细节：此时$r$的取值不再是nums.length - 1，而是nums.length。$r$取值到nums.length是有意义的，说明当前target是数组中最大的元素。$l$取值一直增大，直到和$r$相等时退出循环。因为向下取值的缘故，$mid$不会取到nums.length这个值，故不会下标越界。最终搜寻到的值需要减去1，因为是其前面一个元素。当$l = $nums.length时，$l - 1$刚好取到最后一个元素的值。

> 二分答案

二分答案是搜索一个区间范围内的答案，在某个临界点前的答案不满足题意，在临界点后的答案满足题意，题目通常是需要找出该临界点，此时可以通过二分的方式进行求解。

例题：[719. 找出第 K 小的数对距离](https://leetcode.cn/problems/find-k-th-smallest-pair-distance/)

分析：之前介绍过多路归并的算法可以用于求解本题，由于时间复杂度$O(k \log n)$过高，会超时。因为$k$最坏能到达$n^2$，时间复杂度为$O(n^2\log n)$。

本题可以采用二分查找，结合双指针的思想进行优化。

```java
class Solution {
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int l = 0, n = nums.length, r = nums[n - 1] - nums[0];
        while(l < r) {
            int mid = l + r >> 1;
            int cnt = 0;
            for(int i = 0, j = 0; j < n; j ++) {
                while(nums[j] - nums[i] > mid) {
                    i ++;
                }
                cnt += j - i;
            }
            if(cnt < k) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
}
```

再从这道例题中深入理解二分答案

例题：[378. 有序矩阵中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/)

分析：搜索的不是矩阵的下标，而是可能的答案。答案的搜索区间中，左边界是矩阵的最小值，右边界是矩阵中的最大值。搜索过程中的答案不一定是矩阵中的元素，但搜索结束后返回的$l$一定是矩阵中的元素。

需要格外注意，mid的计算不能写作$(l + r)/2$，因为$l+r$可能为负数，负数采用上取整后，会导致死循环。

```java
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int l = matrix[0][0], r = matrix[n-1][n-1];
        while(l < r) {
            int mid = l + r >> 1;  // mid一定不能写作(l + r) / 2 ！！！
            if(countSmaller(matrix, mid, n) < k) {
                l = mid + 1;
            }else {
                r = mid;
            }
        }
        return l;
    }

    private int countSmaller(int[][] matrix, int mid, int n) {
        int i = 0, j = n - 1, count = 0;
        while(i < n && j >= 0) {
            if(matrix[i][j] <= mid) {
                count += j + 1;
                i ++;
            }else {
                j --;
            }
        }
        return count;
    }
}
```

时间复杂度：$O(n\log(r-l))$

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [875. 爱吃香蕉的珂珂](https://leetcode.cn/problems/koko-eating-bananas/) | 中等 |
| [410. 分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/) | 困难 |
| [69. x 的平方根](https://leetcode.cn/problems/sqrtx/)        | 简单 |
| [658. 找到 K 个最接近的元素](https://leetcode.cn/problems/find-k-closest-elements/) | 中等 |
| [1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/) | 中等 |
| [1552. 两球之间的磁力](https://leetcode.cn/problems/magnetic-force-between-two-balls/) | 中等 |

### 3.6 Greedy Algorithm

贪心算法在求解时，总是做出在当前看来是最好的选择，把求解的问题分成若干个子问题。对每个子问题求解，得到子问题的局部最优解。子问题的局部最优解合成原来问题的一个解。贪心算法正确性的关键在于证明贪心选择性质，即一个问题的整体最优解可以通过一系列局部最优解的选择达到。

#### 3.6.1 Interval Arrangement

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

区间安排问题是一类最经典的贪心问题，其目的是尽可能多地选择区间，使得区间之间不重叠。

例如，有如下区间：$(s_1,e_1),(s_2,e_2),...(s_n,e_n)$。区间不重叠，当且仅当$s_i\ge e_j$ 或者$e_i\le s_j$。

考虑如下三种策略：

1. 开始区间更小的优先。

反例：[1,15]。选择[1,15]显然没有选择[2,6] + [7,15]更优。

2. 区间间隔小的优先。

反例：[1,7],[7,15],[6,8]。区间[6,8]间隔更小，但选择[6,8]后无法再选择[1,7]和[7,15]。

3. 结束区间更小的优先。

此贪心策略是正确的。可以通过数学归纳法进行证明。

例题：[452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)

```java
class Solution {
    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> Integer.compare(a[1], b[1]));
        int end = points[0][1], count = 1;
        for(int i = 1; i < points.length; i ++) {
            if(points[i][0] > end) {
                count ++;
                end = points[i][1];
            }
        }
        return count;
    }
}
```
练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [646. 最长数对链](https://leetcode.cn/problems/maximum-length-of-pair-chain/) | 中等 |
| [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/) | 中等 |

对于区间安排问题，再深入进行思考。

如果每个区间的权重不一样，此时为了保证选择的区间权重和尽可能大，应该如何求解该问题？参考问题：[1235. 规划兼职工作](https://leetcode.cn/problems/maximum-profit-in-job-scheduling/)。
该问题将在动态规划章节介绍。

根据本题，对区间类问题做一个专题总结。

假设两个区间$A,B$，区间范围分别是$[x_1,y_1],[x_2,y_2]$。

判断两个区间是否有交集：
```java
if(y1 >= x2 && y2 >= x1) {
    return true;
}
```
取区间交集：
```java
int left = Math.max(x1, x2);
int right = Math.min(y1, y2);
return new int[]{left, right};
```
取区间并集（相交前提下）：
```java
int left = Math.min(x1, x2);
int right = Math.max(y1, y2);
return new int[]{left, right};
```
判断不相交：
```java
if(x1 > y2 || x2 > y1) {
    return true;
}
```
例题：[986. 区间列表的交集](https://leetcode.cn/problems/interval-list-intersections/)
```java
class Solution {
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> ret = new ArrayList<>();
        int i = 0, j = 0;
        while(i < firstList.length && j < secondList.length) {
            int x1 = firstList[i][0], y1 = firstList[i][1];
            int x2 = secondList[j][0], y2 = secondList[j][1];
            if(y1 >= x2 && y2 >= x1) {
                ret.add(new int[]{Math.max(x1, x2), Math.min(y1, y2)});
            }
            if(y1 < y2) {  // 不能写作 x1 < x2，左边界小不代表后续没有相交区间了
                i ++;
            }else {
                j ++;
            }
        }
        return ret.toArray(new int[0][0]);
    }
}
```
现证明根据右边界移动指针的正确性：

first: [x1,y1], [x3,y3]

second: [x2,y2], [x4,y4]

满足 x3 > y1, x4 > y2

若 y1 < y2，则 y1 < x4，区间[x1, y1]不可能和[x2, y2]后面的任何区间相交，因此移动first指针。

若 y1 > y2，则 y1 和 x4 的大小关系未知，后续还需要进一步判断，fist指针不能移动，second指针移动。

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [56. 合并区间](https://leetcode.cn/problems/merge-intervals/) | 中等 |
| [759. 员工空闲时间](https://leetcode.cn/problems/employee-free-time/) | 困难 |
| [57. 插入区间](https://leetcode.cn/problems/insert-interval/) | 中等 |

#### 3.6.2 Covering Problem

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

覆盖问题是一系列问题：给定一个区间范围和若干个子区间，选择尽可能少的区间，使得区间能够完全覆盖整个区间。

可以采用动态规划的解法，假设$dp[i]$表示将区间$[0...i)$覆盖所需的最小子区间数量，区间$[a_j,b_j)$满足$a_j<i\le b_j$，则有：

$dp[i]=\min\{dp[j]\}+1$，其中$(a_j<i\le b_j)$

时间复杂度：$O(m·n)$，$m$为区间长度，$n$为子区间个数。

该问题采用贪心算法求解更优。对于左端点相同的子区间，右端点越远越有利。用一个数组记录每个位置的左端点能到达的最远右端点，从以下例题中学习贪心解法。

例题：[1024. 视频拼接](https://leetcode.cn/problems/video-stitching/)

```java
class Solution {
    public int videoStitching(int[][] clips, int time) {
        int[] rightMost = new int[time];
        for(int[] clip : clips) {
            if(clip[0] < time) {
                rightMost[clip[0]] = Math.max(rightMost[clip[0]], clip[1]);
            }
        }
        int last = 0, ans = 0, pre = 0;
        for(int i = 0; i < time; i ++) {  // 只遍历到time - 1
            last = Math.max(last, rightMost[i]);  // last表示当前能覆盖到的最远右端点
            if(i == last) {
                return -1;
            }
            if(i == pre) {   // pre表示上一个被使用的子区间结束位置
                ans ++;
                pre = last;
            }
        }
        return ans;
    }
}
```
时间复杂度：$O(m+n)$

练习题单

| 题号                                                         | 难度 | 知识点           |
| ------------------------------------------------------------ | ---- | ---------------- |
| [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/) | 中等 | 同例题           |
| [134. 加油站](https://leetcode.cn/problems/gas-station/)     | 中等 | 需要一定思维转化 |
| [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)      | 中等 | 同例题           |
| [1326. 灌溉花园的最少水龙头数目](https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/) | 困难 | 同例题           |

如果$m$的取值特别大，此时得采用优先队列进行优化。见如下例题。

例题：[871. 最低加油次数](https://leetcode.cn/problems/minimum-number-of-refueling-stops/)

```java
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        Queue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        int cnt = 0, dist = startFuel;
        for(int[] station : stations) {
            while(!queue.isEmpty() && dist < station[0]) {
                dist += queue.poll();
                cnt ++;
            }
            if(dist < station[0]) {
                return -1;
            }
            queue.offer(station[1]);
        }
        while(!queue.isEmpty() && dist < target) {
            dist += queue.poll();
            cnt ++;
        }
        return dist >= target ? cnt : -1;
    }
}
```

时间复杂度：$O(n\log n)$

思考：如何用优先队列的方法解决例题1024？

#### 3.6.3 Maximum Number Problem

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

例题：[402. 移掉 K 位数字](https://leetcode.cn/problems/remove-k-digits/)

```java
class Solution {
    public String removeKdigits(String num, int k) {
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < num.length(); i ++) {
            int n = num.charAt(i) - '0';
            while(!stack.isEmpty() && stack.peek() > n && k > 0) {
                stack.pop();
                k --;
            }
            if(!stack.isEmpty() || n != 0) {
                stack.push(n);
            }
        }
        while(!stack.isEmpty() && k > 0) {
            stack.pop();
            k --;
        }
        StringBuilder sb = new StringBuilder();
        for (int i : stack) {
            sb.append(i);
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [316. 去除重复字母](https://leetcode.cn/problems/remove-duplicate-letters/) | 中等 |

#### 3.6.4 Pairing Problem
例题：[2576. 求出最多标记下标](https://leetcode.cn/problems/find-the-maximum-number-of-marked-indices/)

分析：若$2*nums[i]\le nums[j]$，称$nums[i]$和$nums[j]$匹配。将数组从小到大排序后，如果存在$k$对匹配，一定能让最小的$k$个数和最大的$k$个数匹配。

从小到大排序后，$nums[0]$肯定需要匹配$nums[n-k]$，若匹配了$nums[n-k]$右侧的数，则一定有一个更大的数$nums[i](i>0)$要匹配$nums[n-k]$，此时不一定能匹配上。

假设找到了$k$（可以用二分的方式寻找），则一定可以有如下匹配方式：

$nums[0]\sim nums[n-k]$

$nums[1]\sim nums[n-k+1]$

$nums[i]\sim nums[n-k+i]$

$nums[k-1]\sim nums[n-1]$

基于二分：
```java
class Solution {
    public int maxNumOfMarkedIndices(int[] nums) {
        Arrays.sort(nums);
        int l = 0, r = nums.length / 2;
        while(l < r) {
            int mid = l + r + 1 >> 1;
            if(!check(nums, mid)) {
                r = mid - 1;
            }else {
                l = mid;
            }
        }
        return l * 2;
    }

    private boolean check(int[] nums, int k) {
        for(int i = 0; i < k; i ++) {
            if(nums[i] * 2 > nums[nums.length - k + i]) {
                return false;
            }
        }
        return true;
    }
}
```

基于双指针：

在先前的分析中也可以看出，如果要尽可能多的配对，则左侧的数需要和右侧的数进行匹配。
```java
class Solution {
    public int maxNumOfMarkedIndices(int[] nums) {
        Arrays.sort(nums);
        int i = 0, n = nums.length;
        for(int j = (n + 1) / 2; j < n; j ++) {
            if(nums[i] * 2 <= nums[j]) {
                i ++;
            }
        }
        return i * 2;
    }
}
```

真题链接：美团20230826笔试 https://codefun2000.com/p/P1496

提示：$1 \le a_i + b_i \le m$，将$a$数组升序排序，$b$数组降序排序，判断是否所有的都满足$1 \le a[i]+b[i] \le m$。

练习题单

| 题号                         | 难度 | 知识点 |
| ---------------------------- | ---- | ---------------------------- |
| [2856. 删除数对后的最小数组长度](https://leetcode.cn/problems/minimum-array-length-after-pair-removals/) | 中等 | 例题结论运用 |
| [881. 救生艇](https://leetcode.cn/problems/boats-to-save-people/) | 中等 |例题结论运用 |


#### 3.6.5 Regret-based Greedy

在例题[122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)中，我们采用了贪心算法，由于可以在同一天出售，且限制了最多同时持有一只股票，贪心策略为，只要$arr[i]>arr[i-1]$，我们就进行卖出。

如果去掉最多同时持有一只股票的限制，每天依然只能买进一只股票，卖出一只股票，如果要求解能获得的最大利润，此时的贪心策略应该是怎样的？

分析：考虑$i<j<k,p_i<p_j<pk$，怎样交易利润最大？

如果在第$i$天买入，第$j$天卖出，此时利润$p_j-p_i$;

如果在第$i$天买入，第$k$天卖出，此时利润$p_k-p_i$更大。

所以可以采用贪心+反悔策略。如果当天有利可图，就卖出，同时买入当天股票，后面再涨再卖出。

$(p_j-p_i)+(p_k-p_j)=p_k-p_i$，等价于第$j$天没有交易。

从前往后遍历$p_i$，用小根堆枚举每天的股票价格，每个$p_i$入堆（买入），以便将来交易。若满足$p_i>$堆顶元素，则计算利润，同时把$p_i$再次入栈（买入），以便将来反悔。

```java
class Solution {
    public int maxProfits(int[] arr) {
        Queue<Integer> queue = new PriorityQueue<>();
        int ans = 0;
        for(int i = 0; i < arr.length; i ++) {
            if(!queue.isEmpty() && queue.peek() < arr[i]) {
                ans += arr[i] - queue.poll();
                queue.offer(arr[i]);  // 为了反悔
            }
            queue.offer(arr[i]);  // 为了交易
        }
        return ans;
    }
}
```
时间复杂度：$O(n\log n)$

示例：

$[1,2,5,8,3,1,6]$

最佳策略：-1 - 2 + 5 + 8 - 1 + 6 = 15

q = {1}

ans += 2 - 1; q = {2,2}

ans += 5 - 2; q = {2,5,5}

ans += 8 - 2; q = {5,5,8,8}

q = {3,5,5,8,8}

q = {1,3,5,5,8,8}

ans += 6 - 1; q = {3,5,5,6,6,8,8}

思考：深入理解本题和122问题的异同。

例题：[630. 课程表 III](https://leetcode.cn/problems/course-schedule-iii)

分析：假设已完成前$n-1$门课程，则有：

$t_1+t_2+...t_{n-1} < d_{n-1}$

如果无法学习第$n$门课程，则有：

$t_1+t_2+...+t_{n-1}+t_n>d_n$

移除掉$\max\{t_1,t_2,...t_{n-1}\}$中最大值$t_i,(t_i>t_n)$，则有：

$t_1+t_2+...+t_{n-1}+t_n-t_i<d_{n-1}<d_n$，
此时课程$n$一定能学习。(前提：已经按照结束时间$d_i$排序)

```java
class Solution {
    public int scheduleCourse(int[][] courses) {
        Arrays.sort(courses, Comparator.comparingInt(a -> a[1]));  // 按照截止日期排序
        Queue<Integer> queue = new PriorityQueue<>((a, b) -> Integer.compare(b, a));
        int day = 0;
        for(int[] c : courses) {
            int duration = c[0], lastDay = c[1];
            if(day + duration <= lastDay) {
                day += duration;
                queue.offer(duration);
            }else if(!queue.isEmpty() && duration < queue.peek()) {
                day -= queue.poll() - duration;  // 反悔，选择duration小的课程
                queue.offer(duration);
            }
        }
        return queue.size();
    }
}
```

练习题单

| 题号                                                         | 难度 | 知识点        |
| ------------------------------------------------------------ | ---- | ------------- |
| [1642. 可以到达的最远建筑](https://leetcode.cn/problems/furthest-building-you-can-reach/) | 中等 | 反悔贪心+优先队列 |

### 3.7 Dynamic Programming

#### 3.7.1 Memory-based Search

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 高       |

记忆化搜索是一种自顶向下的求解方法，通常是先写出递归代码，对于递归代码中会多次重复计算的部分，采用一个备忘录数组，将第一次计算的结果进行存储。后续需要计算时直接使用备忘录数组中的值。

动态规划是一种自底向上的求解方法，通常采用递推的方式实现。对于某些问题，解空间可能非常大，自底向上动态规划需要遍历整个解空间。在这种情况下，记忆化搜索可以通过存储中间结果来避免重复的搜索，性能可能更高。

从一道题目体会二者的异同：

例题：[1553. 吃掉 N 个橘子的最少天数](https://leetcode.cn/problems/minimum-number-of-days-to-eat-n-oranges/)

分析：本题的递推式很好想，但是由于$n\le 2*10^9$，直接采用自底向上的代码会超出时间限制和内存限制。

$dp[i] = dp[i-1]+1$

$dp[i] = \min(dp[i], dp[i/2]+1),(i \% 2 = 0)$

$dp[i] = \min(dp[i], dp[i/3]+1),(i \% 3 = 0)$

此时需要采用记忆化搜索的方式，备忘录使用哈希表而非数组，避免不必要的空间开销。

```java
class Solution {
    Map<Integer, Integer> memo = new HashMap<Integer, Integer>();

    public int minDays(int n) {
        if (n <= 1) {
            return n;
        }
        if (memo.containsKey(n)) {
            return memo.get(n);
        }
        memo.put(n, Math.min(n % 2 + 1 + minDays(n / 2), n % 3 + 1 + minDays(n / 3)));
        return memo.get(n);
    }
}
```
时间复杂度：$O(\log^2n)$，取决于需要记忆化的状态数。

思考：受题目[279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)的启发，你能否采用最短路径的思路解题？

```java
class Solution {
    public int minDays(int n) {
        Queue<int[]> queue = new PriorityQueue<>((a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        Set<Integer> visited = new HashSet<>();
        queue.offer(new int[]{0, n});
        int ans = 0;
        while(true) {
            int[] cur = queue.poll();
            int days = cur[0], rest = cur[1];
            if(visited.contains(rest)) {
                continue;
            }
            visited.add(rest);
            if(rest == 1) {
                ans = days + 1;
                break;
            }
            queue.offer(new int[]{days + rest % 2 + 1, rest / 2});
            queue.offer(new int[]{days + rest % 3 + 1, rest / 3});
        }
        return ans;
    }
}
```

#### 3.7.2 Linear Dynamic Programming

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 高       |


##### 3.7.2.1 Maximum Sum of Subarray

例题：[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

分析：$dp[i]$表示从$[0...i]$包含元素$i$的最大子数组和，需要变量单独记录遍历到的最大值。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length, ans = nums[0];
        int[] dp = new int[n];
        dp[0] = nums[0];
        for(int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i-1] + nums[i], nums[i]);
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}
```

时间复杂度：$O(n)$

空间复杂度优化：由于$dp[i]$只依赖$dp[i-1]$的结果，可以用一个变量来记录。

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int ans = Integer.MIN_VALUE, sum = 0;
        for(int i = 0; i < nums.length; i ++) {
            sum = sum < 0 ? nums[i] : nums[i] + sum;
            ans = Math.max(ans, sum);
        }
        return ans;
    }
}
```

本题还可以使用分治算法求解，时间复杂度$O(n \log n)$。

例题：[面试题 17.24. 最大子矩阵](https://leetcode.cn/problems/max-submatrix-lcci/)

```java
class Solution {
    public int[] getMaxMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length, max = Integer.MIN_VALUE;
        int[] ret = new int[4];
        for(int i = 0; i < m; i ++) {
            int[] arr = new int[n];
            for(int j = i; j < m; j ++) {
                for(int k = 0; k < n; k ++) {
                    arr[k] += matrix[j][k];
                }
                int[] ans = maxSubArray(arr);
                if(ans[2] > max) {
                    max = ans[2];
                    ret[0] = i;
                    ret[1] = ans[0];
                    ret[2] = j;
                    ret[3] = ans[1];
                }
            }
        }
        return ret;
    }
    // 数组返回三个值，分别为左端点，右端点和最大子数组和
    private int[] maxSubArray(int[] arr) { 
        int[] ret = new int[]{0, 0, arr[0]};
        int dp = arr[0], begin = 0;
        for(int i = 1; i < arr.length; i ++) {
            if(dp > 0) {
                dp += arr[i];
            } else {
                dp = arr[i];
                begin = i;
            }
            if(dp > ret[2]) {
                ret[0] = begin;
                ret[1] = i;
                ret[2] = dp;
            }
        }
        return ret;
    }
}
```

时间复杂度：$O(m^2n)$

##### 3.7.2.2 Series of Problems Related to Buying and Selling Stocks

> 只能买卖一次

例题：[121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

分析：用一变量记录买股票的最低价格$minPrice$，初始化为$price[0]$。一变量记录遍历过程中$price[i]-minPrice$的最大值，初始化为0。

进阶：在本题基础上增加难度，请看例题。

例题：[2874. 有序三元组中的最大值 II](https://leetcode.cn/problems/maximum-value-of-an-ordered-triplet-ii/)

分析：

枚举$j$，预处理$j$左侧元素的最大值和右侧元素的最大值。

枚举$k$，则需要维护$k$左侧$nums[i]-nums[j]$的最大值。

```java
class Solution {
    public long maximumTripletValue(int[] nums) {
        long ans = 0, maxDiff = 0, preMax = 0;
        for(int num : nums) {
            ans = Math.max(ans, maxDiff * num);
            maxDiff = Math.max(maxDiff, preMax - num);
            preMax = Math.max(preMax, num);
        }
        return ans;
    }
}
```

> 无限次买卖股票

例题：[122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

分析：贪心，只要$price[i]>price[i-1]$，则卖出。

考虑$i<j<k,p_i<p_j<pk$，怎样交易利润最大？

如果在第$i$天买入，第$j$天卖出，此时利润$p_j-p_i$;

如果在第$i$天买入，第$k$天卖出，此时利润$p_k-p_i$更大。

$(p_j-p_i)+(p_k-p_j)=p_k-p_i$，等价于第$j$天没有交易。

注意：贪心算法只能用于计算最大利润，计算过程并不是实际的交易过程。

该题变式，参见**反悔贪心**的章节

> 最多k次交易

例题：[188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)

分析：定义$dp[i][j][1]$表示前$i$天，最多交易$k$次，当前持有股票的最大收益。定义$dp[i][j][0]$表示前$i$天，最多交易$k$次，当前未持有股票的最大收益。

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int[][][] dp = new int[n][k+1][2];
        for(int i = 1; i <= k; i ++) {
            dp[0][i][1] = - prices[0];
        }
        for(int i = 1; i < n; i ++) {
            for(int j = 1; j <= k; j ++) {
                dp[i][j][0] = Math.max(dp[i-1][j][0], dp[i-1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i]);
            }
        }
        return dp[n-1][k][0];
    }
}
```

注意：当规定最多交易$k$次时，在初始化时，需要初始化$dp[0][i][1]=-prices[0],i\in[1...k]$。

若规定恰好交易$k$次，在初始化时，只能初始化$dp[0][1][1]=-prices[0]$。

无限次：省略$k$所在的维度，也可以采用贪心算法求解。

当$k=2$时，即为123题。

例题：[123. 买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][][] dp = new int[n][3][2]; 
        dp[0][1][1] = - prices[0];
        dp[0][2][1] = - prices[0];
        for(int i = 1; i < n; i ++) {
            dp[i][1][0] = Math.max(dp[i-1][1][1] + prices[i], dp[i-1][1][0]);
            dp[i][1][1] = Math.max(- prices[i], dp[i-1][1][1]);
            dp[i][2][0] = Math.max(dp[i-1][2][1] + prices[i], dp[i-1][2][0]);
            dp[i][2][1] = Math.max(dp[i-1][1][0] - prices[i], dp[i-1][2][1]);
        }
        return dp[n-1][2][0];
    }
}
```

由于$k$较小，可以直接枚举。现通过该题讲解如何优化空间复杂度。

将$dp[i][1][0]$优化为$dp10$

将$dp[i][1][1]$优化为$dp11$

将$dp[i][2][0]$优化为$dp20$

将$dp[i][2][1]$优化为$dp21$。由于$dp[i][2][1]$依赖$dp[i-1][1][0]$的值会被$dp11$覆盖，所以用临时变量存储一下$dp[i-1][1][0]$的值。

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int dp10 = 0, dp11 = Integer.MIN_VALUE, dp20 = 0, dp21 = Integer.MIN_VALUE;
        for(int i = 0; i < n; i ++) {
            int temp = dp10;
            dp10 = Math.max(dp11 + prices[i], dp10);
            dp11 = Math.max(dp11, - prices[i]);
            dp20 = Math.max(dp21 + prices[i], dp20);
            dp21 = Math.max(dp21, temp - prices[i]);
        }
        return dp20;
    }
}
```

采用空间优化后的代码可读性非常差，在内存充足的情况下不建议进行空间复杂度的优化。

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [714. 买卖股票的最佳时机含手续费](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) | 中等 |
| [309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | 中等 |
| [552. 学生出勤记录 II](https://leetcode.cn/problems/student-attendance-record-ii/) | 困难 |

##### 3.7.2.3 Series of Problems Related to House Robber


例题：[198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

分析：$dp[i]$表示前$i$间房屋能偷窃到的最高金额，$dp[i]=\max(dp[i-2]+nums[i], dp[i-1])$

初始化

$\begin{cases}dp[0]=nums[0]\\dp[1]=\max(nums[0],nums[1])\end{cases}$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/) | 中等 |
| [740. 删除并获得点数](https://leetcode.cn/problems/delete-and-earn/) | 中等 |

打家劫舍体现了一种选与不选的思想。该思想在很多题目中也有体现。

例题：[1235. 规划兼职工作](https://leetcode.cn/problems/maximum-profit-in-job-scheduling/)

分析：

先按照结束时间进行排序。

设$dp[i]$表示按照结束时间排序后，前$i$个工作的最大报酬。

$dp[i]=\max(dp[i-1], dp[j]+profit[i])$，其中$j$为满足$endTime[j]\le startTime[i]$的最大的$j$。由于已经按照结束时间排序，可以用二分查找进行优化。

```java
class Solution {
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        int[][] pair = new int[n][3];
        for(int i = 0; i < n; i ++) {
            pair[i][0] = startTime[i];
            pair[i][1] = endTime[i];
            pair[i][2] = profit[i];
        }
        Arrays.sort(pair, Comparator.comparingInt(a -> a[1]));
        int[] dp = new int[n + 1];
        for(int i = 0; i < n; i ++) {
            int j = search(pair, pair[i][0]);
            dp[i + 1] = Math.max(dp[i], dp[j + 1] + pair[i][2]);
        }
        return dp[n];
    }

    private int search(int[][] pair, int start) {
        int l = -1, r = pair.length - 1;
        while(l < r) {
            int mid = l + r + 1 >> 1;
            if(pair[mid][1] > start) {
                r = mid - 1;
            } else {
                l = mid;
            }
        }
        return l;
    }
}
```

时间复杂度：$O(n\log n)$

还可以增加选择次数的限定。

例题：[1751. 最多可以参加的会议数目 II](https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended-ii/)

```java
class Solution {
    public int maxValue(int[][] events, int k) {
        Arrays.sort(events, Comparator.comparingInt(a -> a[1]));
        int n = events.length;
        int[][] dp = new int[n+1][k+1];
        for(int i = 0; i < n; i ++) {
            int index = search(events, events[i][0]);
            for(int j = 0; j < k; j ++) {
                dp[i+1][j+1] = Math.max(dp[i][j+1], dp[index+1][j] + events[i][2]);
            }
        }
        return dp[n][k];
    }

    private int search(int[][] events, int start) {
        int l = -1, r = events.length - 1;
        while(l < r) {
            int mid = l + r + 1 >> 1;
            if(events[mid][1] >= start) {
                r = mid - 1;
            } else {
                l = mid;
            }
        }
        return l;
    }
}
```

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [2054. 两个最好的不重叠活动](https://leetcode.cn/problems/two-best-non-overlapping-events/) | 中等 |
| [2008. 出租车的最大盈利](https://leetcode.cn/problems/maximum-earnings-from-taxi/) | 中等 |
| [2830. 销售利润最大化](https://leetcode.cn/problems/maximize-the-profit-as-the-salesman/) | 中等 |

#### 3.7.3 Sequence Dynamic Programming

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |
##### 3.7.3.1 Longest Increasing Subsequence

例题：[300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length, ans = 0;
        int[] dp = new int[n];
        for(int i = 0; i < n; i ++) {
            dp[i] = 1;
            for(int j = 0; j < i; j ++) {
                if(nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}
```

时间复杂度：$O(n^2)$

当数据规模到达$10^5$时，需要用贪心+二分查找进行优化

以测试用例[10,9,2,5,3,7,101,18]为例，解释top数组的元素

| 当前元素target | top         |
| -------------- | ----------- |
| 10             | [10]        |
| 9              | [9]         |
| 2              | [2]         |
| 5              | [2,5]       |
| 3              | [2,3]       |
| 7              | [2,3,7]     |
| 101            | [2,3,7,101] |
| 18             | [2,3,7,18]  |

贪心性质：假设当前数为$b$，找到大于等于$b$的最小的数为$a$，用$b$替换$a$，得到的最长上升子序列一定不会变短。

设$b$后面一个数为$c$。

若$c \le b$，则最长上升子序列无变化。

若$c > b $且$c > a$，则最长子序列可以是$[a,c]$或$[b,c]$

若$c > b$且$c < a$，则最长上升子序列为$[b,c]$。

无论如何，用$b$替换$a$都不会使得结果变得更差。

注意：top数组的实际长度piles为最长上升子序列的长度，但top数组中的序列并不一定为最长上升子序列的实际序列。

例如，数组[4,6,2]遍历完后对应的top数组是[2,6]，但实际上升子序列为[4,6]。

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length, piles = 0;  // piles表示top数组的实际长度，即最长上升子序列的长度
        int[] top = new int[n]; 
        for(int i = 0; i < n; i ++) {
            int target = nums[i];
            int l = 0, r = piles;
            while(l < r) {   // 二分查找目标：大于等于target的最小下标
                int mid = l + r >> 1;
                if(top[mid] < target) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }
            if(l == piles) {  
                piles ++;
            }
            top[l] = target;
        }
        return piles;
    }
}
```

时间复杂度：$O(n\log n)$

思考：若求解的是非严格递增的子序列，应该怎样修改代码？

思考：如何反推最长上升子序列？如何求解最长递增子序列的个数？

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [354. 俄罗斯套娃信封问题](https://leetcode.cn/problems/russian-doll-envelopes/) | 困难 |
| [368. 最大整除子集](https://leetcode.cn/problems/largest-divisible-subset/) | 中等 |
| [1027. 最长等差数列](https://leetcode.cn/problems/longest-arithmetic-subsequence/) | 中等 |
| [873. 最长的斐波那契子序列的长度](https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/) | 中等 |
| [673. 最长递增子序列的个数](https://leetcode.cn/problems/number-of-longest-increasing-subsequence/) | 中等 |

##### 3.7.3.2 Longest Common Subsequence

例题：[1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m+1][n+1];
        for(int i = 1; i <= m; i ++) {
            for(int j = 1; j <= n; j ++) {
                if(text1.charAt(i-1) == text2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                }
            }
        }
        return dp[m][n];
    }
}
```

时间复杂度：$O(mn)$

空间复杂度：$O(mn)$

现思考如下问题：

- 当$s[i]=t[j]$时(if代码段)，此时为什么不考虑$dp[i-1][j]$和$dp[i][j-1]$？

假设s为abcd<font color="red">c</font>

t为ab<font color="red">c</font>

设$x = dp[i-1][j-1]$，代表子串abcd和ab的最长公共子序列

若$dp[i-1][j] > x + 1$，即abcd和abc的最长公共子序列大于$x + 1$

去掉abcd和abc的公共元素c，说明abd和ab的最长公共子序列大于$x$

又因为abd和ab是abcd和ab的子串，说明abd和ab的最长公共子序列应该小于等于$x$，此时产生矛盾。

- 当$s[i]\ne t[j]$时(else代码段)，此时为什么不考虑$dp[i-1][j-1]$？

分析可以得知：$dp[i-1][j] \ge dp[i-1][j-1]$

$dp[i][j-1] \ge dp[i-1][j-1]$

- 空间复杂度能否优化？

每个状态只依赖左侧、上侧和左上侧三个状态，左上侧的状态可能被覆盖，用临时变量记录pre即可。

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[] dp = new int[n+1];
        for(int i = 1; i <= m; i ++) {
            int pre = dp[0];
            for(int j = 1; j <= n; j ++) {
                int temp = dp[j];
                if(text1.charAt(i-1) == text2.charAt(j-1)) {
                    dp[j] = pre + 1;
                }else {
                    dp[j] = Math.max(dp[j-1], dp[j]);
                }
                pre = temp;
            }
        }
        return dp[n];
    }
}
```

空间复杂度能够优化到$O(\min(m,n))$

- 如果题目要求输出所有最长公共子序列，应该如何求解？

```java
public void traceback(int i, int j, int[][] dp, String s, String first, String second) {  // 初始值：i = first.length(), j = second.length()
    while (i > 0 && j > 0) {
        if (first.charAt(i - 1) == second.charAt(j - 1)) {
            s = s + first.charAt(i - 1);
            i --;
            j --;
        } else {
            if (dp[i-1][j] > dp[i][j-1]) {
                i --;
            } else if (dp[i-1][j] < dp[i][j-1]) {
                j --;
            } else {
                traceback(i - 1, j, dp, s, first, second);
                traceback(i, j - 1, dp, s, first, second);
                return;
            }
        }
    }
    lcs.add(new StringBuilder(s).reverse().toString());
}
private List<String> lcs = new ArrayList<>();
```

例题：[1092. 最短公共超序列](https://leetcode.cn/problems/shortest-common-supersequence/)

```java
class Solution {
    public String shortestCommonSupersequence(String str1, String str2) {
        int m = str1.length(), n = str2.length();
        int[][] dp = new int[m+1][n+1];
        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                if(str1.charAt(i-1) == str2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        int i = m, j = n;
        StringBuilder sb = new StringBuilder();
        while(i > 0 || j > 0) {
            if(i == 0) {
                sb.append(str2.charAt(--j));
            }else if(j == 0) {
                sb.append(str1.charAt(--i));
            }else {
                if(str1.charAt(i-1) == str2.charAt(j-1)) {
                    sb.append(str1.charAt(--i));
                    -- j;
                }else if(dp[i][j] == dp[i][j-1]) {
                    sb.append(str2.charAt(--j));
                }else {
                    sb.append(str1.charAt(--i));
                }
            }
        }
        return sb.reverse().toString();
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)  | 困难 |
| [516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/) | 中等 |
| [1216. 验证回文字符串 III](https://leetcode.cn/problems/valid-palindrome-iii/) | 困难 |
|                                                              |      |

##### 3.7.3.3 Pattern Matching

例题：[10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/)

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        for(int j = 1; j <= n; j ++) {
            if(p.charAt(j-1) == '*') {
                dp[0][j] |= dp[0][j-2];
            }
        }
        for(int i = 1; i <= m; i ++) {
            for(int j = 1; j <= n; j ++) {
                if(s.charAt(i-1) == p.charAt(j-1) || p.charAt(j-1) == '.') {
                    dp[i][j] |= dp[i-1][j-1];
                }
                if(p.charAt(j-1) == '*') {
                    dp[i][j] |= dp[i][j-2];
                    if(s.charAt(i-1) == p.charAt(j-2) || p.charAt(j-2) == '.') {
                        dp[i][j] |= dp[i-1][j];
                    } 
                }
            }
        }
        return dp[m][n];
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [44. 通配符匹配](https://leetcode.cn/problems/wildcard-matching/) | 困难 |

#### 3.7.4 Path 

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

例题：[174. 地下城游戏](https://leetcode.cn/problems/dungeon-game/)

```java
class Solution {
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length, n = dungeon[0].length;  
        int[][] dp = new int[m][n];
        dp[m-1][n-1] = Math.max(1, 1 - dungeon[m-1][n-1]);
        for(int i = m - 2; i >= 0; i --) {
            dp[i][n-1] = Math.max(dp[i+1][n-1] - dungeon[i][n-1], 1);
        }
        for(int i = n - 2; i >= 0; i --) {
            dp[m-1][i] = Math.max(dp[m-1][i+1] - dungeon[m-1][i], 1);
        }
        for(int i = m - 2; i >= 0; i --) {
            for(int j = n - 2; j >= 0; j --) {
                int min = Math.min(dp[i+1][j], dp[i][j+1]);
                dp[i][j] = Math.max(1, min - dungeon[i][j]);
            }
        }
        return dp[0][0];
    }
}
```

例题：[1289. 下降路径最小和 II](https://leetcode.cn/problems/minimum-falling-path-sum-ii/)

```java
class Solution {
    public int minFallingPathSum(int[][] grid) {
        int n = grid.length;
        int[][] dp = new int[n][n];
        for(int i = 0; i < n; i++) {
            dp[0][i] = grid[0][i];
        }
        for(int i = 1; i < n; i ++) {
            for(int j = 0; j < n; j ++) {
                dp[i][j] = Integer.MAX_VALUE;
                for(int k = 0; k < n; k ++) {
                    if(j != k) {
                        dp[i][j] = Math.min(dp[i][j], dp[i-1][k] + grid[i][j]);
                    }
                }
            }
        }
        return Arrays.stream(dp[n-1]).min().getAsInt();
    }
}
```

时间复杂度$O(n^3)$

现考虑如何优化时间复杂度：事实上，$dp[i][j]$只依赖上一行的最小值，如果最小值对应的$k=j$的话，则需要依赖上一行的次小值。用两个变量维护每一行的最小值和次小值即可。

```java
class Solution {
    public int minFallingPathSum(int[][] grid) {
        int min = 0, secondMin = 0, color = -1;
        for(int[] g : grid) {
            int preMin = min, preSecondMin = secondMin, preColor = color;
            min = Integer.MAX_VALUE;
            secondMin = Integer.MAX_VALUE;
            for(int i = 0; i < g.length; i ++) {
                int cost = (i == preColor ? preSecondMin : preMin) + g[i];
                if(cost < secondMin) {
                    if(cost < min) {
                        secondMin = min;
                        min = cost;
                        color = i;
                    }else {
                        secondMin = cost;
                    }
                }
            }
        }
        return min;
    }
}
```
时间复杂度：$O(n^2)$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [62. 不同路径](https://leetcode.cn/problems/unique-paths/)   | 中等 |
| [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/) | 中等 |
| [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/) | 中等 |
| [265. 粉刷房子 II](https://leetcode.cn/problems/paint-house-ii/) | 困难 |
| [931. 下降路径最小和](https://leetcode.cn/problems/minimum-falling-path-sum/) | 中等 |


#### 3.7.5 Tree Dynamic Programming

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |
现讨论更复杂的一种打家劫舍问题。

例题：[337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)

```java
class Solution {
    public int rob(TreeNode root) {
        int[] ret = dfs(root);
        return Math.max(ret[0], ret[1]);
    }

    private int[] dfs(TreeNode root) {
        if(root == null) {
            return new int[]{0, 0};
        }
        int[] left = dfs(root.left);
        int[] right = dfs(root.right);
        int[] ret = new int[2];
        // ret[0]表示当前节点不偷的最大收益，ret[1]偷的最大收益
        ret[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        ret[1] = root.val + left[0] + right[0];
        return ret;
    }
}
```
时间复杂度：$O(n)$

例题：[333. 最大 BST 子树](https://leetcode.cn/problems/largest-bst-subtree/)

```java
class Solution {

    class Result {
        TreeNode node;
        int size;
        int max;
        int min;
    }

    public int largestBSTSubtree(TreeNode root) {
        Result result = dfs(root);
        return result == null ? 0 : result.size;
    }

    public Result dfs(TreeNode node) {
        if(node == null) {
            return null;
        }
        Result left = null, right = null;
        if(node.left != null) {
            left = dfs(node.left);
        }
        if(node.right != null) {
            right = dfs(node.right);
        }
        boolean leftValid = (left == null || left.node == node.left && left.max < node.val);
        boolean rightValid = (right == null || right.node == node.right && right.min > node.val);
        if(leftValid && rightValid) {
            Result result = new Result();
            result.node = node;
            result.max = right == null ? node.val : right.max;
            result.min = left == null ? node.val : left.min;
            result.size = (left == null ? 0 : left.size) + (right == null ? 0 : right.size) + 1;
            return result;
        }
        if(left != null && right != null) {
            return left.size > right.size ? left : right;
        }
        return left == null ? right : left;
    }

}
```
时间复杂度：$O(n)$

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1373. 二叉搜索子树的最大键值和](https://leetcode.cn/problems/maximum-sum-bst-in-binary-tree/) | 困难 |
| [979. 在二叉树中分配硬币](https://leetcode.cn/problems/distribute-coins-in-binary-tree/) | 中等 |

#### 3.7.6 Reroot Dynamic Programming

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |

例题：[834. 树中距离之和](https://leetcode.cn/problems/sum-of-distances-in-tree/description/)

分析：暴力解法，以$i$为根进行DFS，单次DFS时间复杂度为$O(n)$，总共$n$次，需要$O(n^2)$时间复杂度，会超时。

换根DP的思想：从0出发进行DFS，计算0到每个节点的距离ans[0]，DFS同时记录每个子树的大小size[i]。然后再从0出发DFS，假设$y$是$x$的儿子，$ans[y]=ans[x]+n-2*size[y]$。

将根从$x$换成$y$，所有$y$的子孩子节点到根节点距离减少1，共$size[y]$个，非$y$的子孩子节点到根节点的距离增加1，共$n-size[y]$个。

```java
class Solution {
    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        List<Integer>[] graph = new List[n];
        for(int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for(int[] edge : edges) {
            int from = edge[0], to = edge[1];
            graph[from].add(to);
            graph[to].add(from);
        }
        ans = new int[n];
        size = new int[n];
        Arrays.fill(size, 1);
        dfs(0, graph, 0, 0);
        reroot(0, graph, 0);
        return ans;
    }

    private int[] ans;
    private int[] size;
    private int n;

    private void dfs(int start, List<Integer>[] graph, int parent, int depth) {
        ans[0] += depth;
        for(int adj : graph[start]) {
            if(adj != parent) {
                dfs(adj, graph, start, depth + 1);
                size[start] += size[adj];
            }
        }
    }

    private void reroot(int start, List<Integer>[] graph, int parent) {
        for(int adj : graph[start]) {
            if(adj != parent) {
                ans[adj] = ans[start] + graph.length - 2 * size[adj];
                reroot(adj, graph, start);
            }
        }
    }
}
```
时间复杂度：$O(n)$

变式：[310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/)

```java
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        ds = new int[n];
        ds2 = new int[n];
        g = new List[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for(int[] edge : edges) {
            g[edge[0]].add(edge[1]);
            g[edge[1]].add(edge[0]);
        }
        dfs(0, 0);
        reroot(0, 0);
        int min = Arrays.stream(ds2).min().getAsInt();
        return IntStream.range(0, n).filter(e -> ds2[e] == min).boxed().toList();
    }

    private int[] ds;  // 记录每个节点为根往下的最大深度
    private int[] ds2; // 记录每个节点为根的最小高度
    private List<Integer>[] g;

    private int dfs(int u, int p) {
        for(int v : g[u]) {
            if(v != p) {
                ds[u] = Math.max(ds[u], dfs(v, u) + 1);
            }
        }
        return ds[u];
    }

    private void reroot(int u, int p) {
        int first = -1, second = -1;  // 子树中最高的高度和次高的高度。
        for(int v : g[u]) {
            if(ds[v] > first) {
                second = first;
                first = ds[v];
            }else if(ds[v] > second) {
                second = ds[v];
            }
        }
        ds2[u] = first + 1;
        for(int v : g[u]) {
            if(v != p) {
                ds[u] = (ds[v] == first ? second : first) + 1;
                reroot(v, u);
            }
        }
    }
}
```


练习题单

| 题号 | 难度 |
| -------- | -------- |
| [2858. 可以到达每一个节点的最少边反转次数](https://leetcode.cn/problems/minimum-edge-reversals-so-every-node-is-reachable/) |困难|
| [2581. 统计可能的树根数目](https://leetcode.cn/problems/count-number-of-possible-root-nodes/) |困难|
#### 3.7.7 Interval Dynamic Programming

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

例题：[312. 戳气球](https://leetcode.cn/problems/burst-balloons/)

为了方便计算，对原数组$nums$进行处理，在其前后添加$1$，得到新数组$arr$，则$arr[1...n]=nums$

定义$dp[i][j]$为填满开区间$(i,j)$能得到的最大硬币数，边界条件为$i\ge j-1$。

最终返回$dp[0][n+1]$。开区间$(0,n+1)$，即闭区间$[1...n]$，和$nums$对应。

根据依赖顺序分析循环遍历方向：

$dp[i][j]$依赖$dp[i][k],dp[k][j]$，其中$i<k<j$，即当前元素依赖其左侧元素和下方元素。

依赖下方元素：$i$从大到小遍历。

依赖左侧元素：$j$从小到大遍历。

```java
class Solution {
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[][] dp = new int[n+2][n+2];
        int[] arr = new int[n+2];
        arr[0] = 1;
        arr[n+1] = 1;
        for(int i = 1; i <= n; i ++) {
            arr[i] = nums[i-1];
        }
        for(int i = n - 1; i >= 0; i --) {  // 左边界
            for(int j = i + 2; j <= n + 1; j ++) {  // 右边界
                for(int k = i + 1; k < j; k ++) { // 枚举中间边界
                    // i < k < j (严格小于)
                    int score = arr[i] * arr[k] * arr[j] + dp[i][k] + dp[k][j];
                    dp[i][j] = Math.max(dp[i][j], score);
                }
            }
        }
        return dp[0][n+1];
    }
}
```

时间复杂度：$O(n^3)$

上述写法依然有一些细节需要注意。

1. 分析出$i$需要逆序，$i$的初始值怎么确定？$i$可以初始化为$n+1$，但是由于$j=i+2$（至少保证区间有三个元素），$j \le n + 1$，因此若$i=n+1$，前两次循环内层循环不会执行。
2. $j$的初始化方式为$j=i+2$，正向遍历。
3. $k$需要枚举从$[i+1...j-1]$的所有取值。需要根据具体题目具体分析。
4. $dp$数组的初始化，无需初始化，默认值为0。

深入理解三个变量的初始化值，请看下一题。

例题：[375. 猜数字大小 II](https://leetcode.cn/problems/guess-number-higher-or-lower-ii/)

$dp[i][j]$表示猜测区间$[i...j]$能获胜的最小现金数。

由于题目中取值为$[1...n]$，很容易想到的点是初始化一个大小为$(n+1)*(n+1)$的$dp$数组。本题为什么要初始化大小为$(n+2)*(n+2)$的数组？

当玩家猜测数字$k$时，如果猜测不正确，此时要么从$[i...k-1]$猜测，要么从$[k+1...j]$猜测。为了确保获胜，需要取这二者的最大值。

$k$代表的是猜测的数值范围，所以$k$的取值为$[i...j]$，区别上一题中$k$不能取到边界两个端点。

$k=i$时，由于$i$最小取到1，$j-1=0$，此时不会越界。

$k=j$时，由于$j$最大取到$n$，$j+1=n+1$，如果申请大小为$(n+1)*(n+1)$的$dp$数就会越界，因此$dp$数组需要为$(n+2)*(n+2)$的数组。

$i$和$j$的取值范围较为容易分析。

对于边界区间，如$dp[i][i],dp[i][i-1]$，默认值为0，无需再赋值。

```java
class Solution {
    public int getMoneyAmount(int n) {
        int[][] dp = new int[n+2][n+2];
        for(int i = n; i >= 1; i --) {
            for(int j = i + 1; j <= n; j ++) {
                dp[i][j] = Integer.MAX_VALUE;
                for(int k = i; k <= j; k ++) {
                    int money = k + Math.max(dp[i][k-1], dp[k+1][j]);
                    dp[i][j] = Math.min(dp[i][j], money);
                }
            }
        }
        return dp[1][n];
    }
}
```

当然，如果觉得开辟$(n+2)*(n+2)$大小的数组别扭，可以采用如下写法，对于$k=j$的结果在循环外提前计算。

```java
class Solution {
    public int getMoneyAmount(int n) {
        int[][] dp = new int[n+1][n+1];
        for(int i = n; i >= 1; i --) {
            for(int j = i + 1; j <= n; j ++) {
                dp[i][j] = j + dp[i][j-1];
                for(int k = i; k < j; k ++) {
                    int money = k + Math.max(dp[i][k-1], dp[k+1][j]);
                    dp[i][j] = Math.min(dp[i][j], money);
                }
            }
        }
        return dp[1][n];
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1039. 多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/) | 中等 |
| [1547. 切棍子的最小成本](https://leetcode.cn/problems/minimum-cost-to-cut-a-stick/) | 困难 |
| [1000. 合并石头的最低成本](https://leetcode.cn/problems/minimum-cost-to-merge-stones/) | 困难 |
| [664. 奇怪的打印机](https://leetcode.cn/problems/strange-printer/) | 困难 |
| [471. 编码最短长度的字符串](https://leetcode.cn/problems/encode-string-with-shortest-length/) | 困难 |

#### 3.7.8 State Compression Dynamic Programming

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |

状态压缩DP问题主要是借助位运算，用二进制表示集合当前状态。

假设集合有$n$个元素，则可以用$n$位二进制数来表示集合当前的状态。

> 状态压缩之枚举子集类问题

例题：[2305. 公平分发饼干](https://leetcode.cn/problems/fair-distribution-of-cookies/)

分析：用mask表示当前cookies数组的分发状态，若mask$ = (1 << n) - 1$，此时mask的二进制全为1，表示所有cookies已经分发完毕。

$dp[i][j]$表示前$i$个孩子，分得饼干集合为$j$时的不公平程度最小值。

首先需要预计算集合的元素和sum。

给第$i$个孩子分配饼干集合$s$后，前$i$个孩子不公平程度为$\max(dp[i-1][j\textbackslash s]),sum[s]$

枚举子集的代码模版：

```java
for(int sub = mask; sub > 0; sub = (sub - 1) & mask) {
    
}
```

解法

```java
class Solution {
    public int distributeCookies(int[] cookies, int k) {
        int n = cookies.length;
        int total = 1 << n;
        int[] sum = new int[total];
        for(int mask = 1; mask < total; mask ++) {
            for(int i = 0; i < n; i ++) {
                if((mask & (1 << i)) != 0) {
                    sum[mask] += cookies[i];
                }
            }
        }
        int[][] dp = new int[k][total];
        dp[0] = sum;
        for(int i = 1; i < k; i ++) {
            for(int mask = total - 1; mask > 0; mask --) {
                dp[i][mask] = sum[mask];
                for(int subset = mask; subset > 0; subset = (subset - 1) & mask) {
                    dp[i][mask] = Math.min(dp[i][mask], Math.max(dp[i-1][subset], sum[mask^subset]));
                }
            }
        }
        return dp[k-1][total-1];
    }
}
```

时间复杂度：$O(k·3^n)$

元素个数为$i$的集合有$C(n,i)$个，子集个数有$2^i$，根据二项式定理$\sum C(n,i)·2^i=(2+1)^n=3^n$

优化点：在求解集合元素和时，可以如下优化

```java
int[] sum = new int[1 << n];
for(int mask = 1; mask < total; mask ++) {
	int x = Integer.numberOfTrailingZeros(mask);
	int y = mask - (1 << x);
	sum[mask] = sum[y] + cookies[x];
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1986. 完成任务的最少工作时间段](https://leetcode.cn/problems/minimum-number-of-work-sessions-to-finish-the-tasks/) | 中等 |
| [1723. 完成所有工作的最短时间](https://leetcode.cn/problems/find-minimum-time-to-finish-all-jobs/) | 困难 |

> 状态压缩之匹配问题

例题：[1947. 最大兼容性评分和](https://leetcode.cn/problems/maximum-compatibility-score-sum/)

分析：定义$dp[i]$表示将集合$i$中的导师分配给前$bitcount(i)$位学生。

```java
class Solution {
    public int maxCompatibilitySum(int[][] students, int[][] mentors) {
        int m = students.length, total = 1 << m;
        int[] dp = new int[total];
        for(int mask = 1; mask < total; mask ++) {
            int i = Integer.bitCount(mask) - 1;
            for(int j = 0; j < m; j ++) {
                if((mask >> j & 1) != 0) {
                    dp[mask] = Math.max(dp[mask], dp[mask ^ (1 << j)] + getScore(students[i], mentors[j]));
                }
            }
        }
        return dp[total-1];
    }

    private int getScore(int[] student, int[] mentor) {
        int score = 0;
        for(int i = 0; i < student.length; i ++) {
            score += student[i] == mentor[i] ? 1 : 0;
        }
        return score;
    }
}
```

时间复杂度：$O(m·2^m)$

现对匹配问题做一个专题总结。

| 算法     | 时间复杂度 | 数据规模   |
| -------- | ---------- | ---------- |
| 枚举排列 | $O(m·m!)$  | $m\le 9$   |
| 状态压缩 | $O(m·2^m)$ | $m \le 14$ |

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [1879. 两个数组最小的异或值之和](https://leetcode.cn/problems/minimum-xor-sum-of-two-arrays/) | 困难 |
| [2172. 数组的最大与和](https://leetcode.cn/problems/maximum-and-sum-of-array/) | 困难 |
| [1066. 校园自行车分配 II](https://leetcode.cn/problems/campus-bikes-ii/)| 中等 |
| [526. 优美的排列](https://leetcode.cn/problems/beautiful-arrangement/) | 中等 |
| [1947. 最大兼容性评分和](https://leetcode.cn/problems/maximum-compatibility-score-sum/) | 中等 |

> 状态压缩之球装桶问题

匹配问题描述的是一对一关系，而求装桶问题，则是将$m$个球装入$n$个桶。

可以根据$m$和$n$的大小关系，考虑到底是将球放入桶中，还是每个桶选择球。

采用暴力回溯的算法：

遍历每个球，放入$n$个桶中，每个球有$n$种选择方式，时间复杂度为$n^m$。

遍历每个桶，选择$m$个球，每个桶有$m$种选择方式，时间复杂度为$m^n$。

采用状态压缩+动态规划的算法：

当$m$较小时，可以考虑压缩球的状态，时间复杂度为$O(m·2^m)$。

当$n$较小时，可以考虑压缩桶的状态，时间复杂度为$O(n·2^n)$。

例题：[698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

```java
class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for(int n : nums) {
            sum += n;
        }
        if(sum % k != 0) {
            return false;
        }
        int avg = sum / k, n = nums.length, total = 1 << n;
        int[] dp = new int[total];   
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for(int mask = 1; mask < total; mask ++) {
            for(int i = 0; i < n; i ++) {
                if((mask & (1 << i)) == 0) {
                    continue;
                }
                int pre = mask ^ (1 << i);
                if(dp[pre] >= 0 && dp[pre] + nums[i] <= avg) {
                    dp[mask] = (dp[pre] + nums[i]) % avg;  // 通过取余操作，桶装满后，dp[mask] = 0
                }
            }
        }
        return dp[total-1] == 0;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [473. 火柴拼正方形](https://leetcode.cn/problems/matchsticks-to-square/) | 中等 |

> 状态压缩之最值类型

例题：[1125. 最小的必要团队](https://leetcode.cn/problems/smallest-sufficient-team/)

```java
class Solution {
    public int[] smallestSufficientTeam(String[] req_skills, List<List<String>> people) {
        Map<String, Integer> map = new HashMap<>();
        int m = req_skills.length;
        for(int i = 0; i < m; i ++) {
            map.put(req_skills[i], i);
        }
        int k = 1 << m;
        int n = people.size();
        int[] dp = new int[k];
        Arrays.fill(dp, n + 1);
        dp[0] = 0;
        int[] preSkill = new int[k];
        int[] prePeople = new int[k];
        for(int i = 0; i < n; i ++) {
            List<String> skill = people.get(i);
            int mask = getMask(skill, map);
            for(int j = 0; j < k; j ++) {
                int next = j | mask;
                if(dp[j] + 1 < dp[next]) {
                    dp[next] = dp[j] + 1;
                    preSkill[next] = j;
                    prePeople[next] = i;
                }
            }
        }
        List<Integer> ret = new ArrayList<>();
        int i = k - 1;
        while(i > 0) {
            ret.add(prePeople[i]);
            i = preSkill[i];
        }
        return ret.stream().mapToInt(j -> j).toArray();
    }

    private int getMask(List<String> skill, Map<String, Integer> map) {
        int mask = 0;
        for(String s : skill) {
            mask += 1 << map.get(s);
        }
        return mask;
    }
}
```

#### 3.7.9 Maximize-Minimize Problem

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |

例题：[486. 预测赢家](https://leetcode.cn/problems/predict-the-winner/)

$dp[i][j]$表示先手玩家选择区间$dp[i...j]$的最大收益。由于玩家都会选择最优方案，所以下一次轮到自己选的时候，对方肯定会让自己的收益尽可能小。

```java
class Solution {
    public boolean predictTheWinner(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        int n = nums.length;
        int[][] dp = new int[n][n];
        for(int i = n - 1; i >= 0; i --) {
            dp[i][i] = nums[i];
            if(i + 1 < n) {
                dp[i][i+1] = Math.max(nums[i], nums[i+1]);
            }
            for(int j = i + 2; j < n; j ++) {
                dp[i][j] = Math.max(nums[i] + Math.min(dp[i+1][j-1], dp[i+2][j]), 
                                    nums[j] + Math.min(dp[i+1][j-1], dp[i][j-2]));
            }
        }
        return 2 * dp[0][n-1] >= sum;
    }
}
```

时间复杂度：$O(n^2)$

$dp[i][j]$表示先手玩家选择区间$[i...j]$与后手玩家分数的最大差值。

```java
class Solution {
    public boolean predictTheWinner(int[] nums) {
        int n = nums.length;
        int[][] dp = new int[n][n];
        for(int i = n - 1; i >= 0; i --) {
            dp[i][i] = nums[i];
            for(int j = i + 1; j < n; j ++) {
                dp[i][j] = Math.max(nums[i] - dp[i+1][j], nums[j] - dp[i][j-1]);
            }
        }
        return dp[0][n - 1] >= 0;
    }
}
```

空间复杂度优化：根据依赖顺序，可以直接去掉dp数组的第一个维度。

例题：[887. 鸡蛋掉落](https://leetcode.cn/problems/super-egg-drop/)

分析：$dp[k][n]$表示用$k$个鸡蛋$n$层楼层的最少操作数。

在第$x$楼扔鸡蛋，如果鸡蛋碎了，问题变为$dp[k-1][n-1]$。

如果鸡蛋没碎，则问题变为$dp[k][n-x]$

可以得出状态转移方程：$dp[k][n]=\min_{0\le i\le n}{\max(dp[k-1][i-1], dp[k][n-i])}+1$。

直接枚举，时间复杂度为$O(kn^2)$，在本题的数据规模下会超时。

注意到$dp[k-1][i-1]$关于$i$单调递增，$dp[k][n-i]$关于$i$单调递减，可以采用二分查找进行优化。

```java
class Solution {
    public int superEggDrop(int k, int n) {
        int[][] dp = new int[k+1][n+1];
        for(int i = 1; i <= n; i ++) {
            dp[1][i] = i;
        }
        for(int i = 1; i <= k; i ++) {
            dp[i][1] = 1;
        }
        for(int i = 2; i <= k; i ++) {
            for(int j = 2; j <= n; j ++) {
                int l = 1, r = j, res = Integer.MAX_VALUE;
                while(l < r) {
                    int mid = l + r >> 1;
                    if(dp[i-1][mid-1] < dp[i][j-mid]) {
                        l = mid + 1;
                        res = Math.min(res, dp[i][j-mid] + 1);
                    }else {
                        r = mid;
                        res = Math.min(res, dp[i-1][mid-1] + 1);
                    }
                }
                dp[i][j] = res;
            }
        }
        return dp[k][n];
    }
}
```
时间复杂度：$O(kn\log n)$

记忆化搜索解法
```java
class Solution {
    private int[][] memo;

    public int superEggDrop(int k, int n) {
        memo = new int[k+1][n+1];
        return dfs(k, n);
    }

    private int dfs(int k, int n) {
        if(k == 1) {
            return n;
        }
        if(n <= 1) {
            return n;
        }
        if(memo[k][n] != 0) {
            return memo[k][n];
        }
        int res = Integer.MAX_VALUE;
        int l = 1, r = n;
        while(l < r) {
            int mid = l + r >> 1;
            int broken = dfs(k - 1, mid - 1);
            int unbroken = dfs(k, n - mid);
            if(broken < unbroken) {
                l = mid + 1;
                res = Math.min(res, unbroken + 1);
            }else {
                r = mid;
                res = Math.min(res, broken + 1);
            }
        }
        return memo[k][n] = res;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [464. 我能赢吗](https://leetcode.cn/problems/can-i-win/)     | 中等 |
| [294. 翻转游戏 II](https://leetcode.cn/problems/flip-game-ii/) | 中等 |
| [877. 石子游戏](https://leetcode.cn/problems/stone-game/)    | 中等 |
| [1406. 石子游戏 III](https://leetcode.cn/problems/stone-game-iii/) | 困难 |

#### 3.7.10 Knapsack Problem

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 中       | 中       |

完全背包类型：求最大价值、求组合数、求排列数。

> 完全背包之求排列数

特征：给定物品可以选择无限次，总价值为target，顺序不同的序列视作不同的组合。

例题：[377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1]; 
        dp[0] = 1;
        for(int j = 1; j <= target; j ++) {
            for(int num : nums) {
                dp[j] += num <= j ? dp[j - num] : 0;
            }
        }
        return dp[target];
    }
}
```

结论：求排列数，外层循环遍历背包容量，内层循环遍历物品。

> 完全背包之求组合数

例题：[518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

特征：给定物品可以选择无限次，总价值为target，选择序列需要去重。

```java
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for(int coin : coins) {
            for(int j = coin; j <= amount; j ++) {  // 由于j - coin需要大于等于0，直接从coin开始遍历
                dp[j] += dp[j - coin];
            }
        }
        return dp[amount];
    }
}
```

和377题解法最大的不同，在于两个for循环的先后顺序。

由于先选1再选2凑成3和先选2再选1凑成3是同一种方案，故在循环的时候，物品的循环放在外层for循环，此时则不会出现先选2再选1的情况。

结论：求组合数，外层循环遍历物品，内层循环遍历背包容量。

> 完全背包之求最大价值

例题：[322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount+1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for(int i = 1; i <= amount; i ++) {
            for(int coin : coins) {
                if(i >= coin) {
                    dp[i] = Math.min(dp[i], dp[i-coin] + 1);
                }
            }
        }
        // for(int coin : coins) {
        //     for(int i = coin; i <= amount; i ++) {
        //         dp[i] = Math.min(dp[i], dp[i - coin] + 1);
        //     }
        // } 两种写法均可
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
}
```

结论：求最大价值类问题，内外层循环遍历顺序均可。

> 零一背包之最大价值

零一背包的特点是每件物品只能选择最多一次。

假设物品的重量为$w$，物品的价值为$v$，物品数量为$n$，背包容量为$m$

其状态转移方程定义为$dp[i][j] = \max(dp[i-1][j], dp[i][j-w[i]]+v[i])$

表示前$i$件商品，剩余容量为$j$能获取的最大价值。

最终结果返回$dp[n][m]$。

例题：[474. 一和零](https://leetcode.cn/problems/ones-and-zeroes/)

分析，该题中，物品的价值恒为1，背包的容量有2个，分别为0的个数和1的个数。

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int k = strs.length;
        int[][][] dp = new int[k+1][m+1][n+1];
        for(int i = 1; i <= k; i ++) {
            int zero = getZeros(strs[i-1]), one = strs[i-1].length() - zero;
            for(int j = 0; j <= m; j ++) {
                for(int t = 0; t <= n; t ++) {
                    dp[i][j][t] = dp[i-1][j][t];
                    if(j >= zero && t >= one) {
                        dp[i][j][t] = Math.max(dp[i][j][t], dp[i-1][j-zero][t-one] + 1);
                    }
                }
            }
        }
        return dp[k][m][n];
    }

    private int getZeros(String str) {
        int zero = 0;
        for(char c : str.toCharArray()) {
            if(c == '0') {
                zero ++;
            }
        }
        return zero;
    }
}
```

dp数组的第一个纬度可以进行优化，但内层循环的遍历顺序需要改为逆序，防止计算覆盖。

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m+1][n+1];
        for(int i = 0; i < strs.length; i ++) {
            int zero = getZeros(strs[i]), one = strs[i].length() - zero;
            for(int j = m; j >= zero; j --) {
                for(int t = n; t >= one; t --) {
                    dp[j][t] = Math.max(dp[j][t], dp[j-zero][t-one] + 1);
                }
            }
        }
        return dp[m][n];
    }

    private int getZeros(String str) {
        int zero = 0;
        for(char c : str.toCharArray()) {
            if(c == '0') {
                zero ++;
            }
        }
        return zero;
    }
}
```

> 零一背包之能否装满

为了方便，后续零一背包的题解都采用优化空间复杂度的写法。

例题：[1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)

分析：可以证明，无论按照何种顺序粉碎石头，最后一块石头的重量总是可以表示成如下等式：

$\sum_{i=0}^{n-1} k_i \times stones_i, k_i\in\{-1,1\}$

可以将$k_i=1$和$k_i=-1$的石头分成两堆。设$k_i=-1$的石头堆重量之和为$neg$，石头堆总重量为$sum$，则$k_i=1$的石头堆质量和为$sum-neg$。
要使最后一块石头的重量尽可能地小，
$\sum_{i=0}^{n-1} k_i \times stones_i, k_i\in\{-1,1\}=(sum-neg)-neg=sum-2*neg$，则$neg$需要在不超过$sum/2$的前提下尽可能大。定义$dp[i][j]$表示前$i$个物品能否恰好装满容量为$j$的背包。

$dp[i+1][j]=\begin{cases} dp[i][j] & j < stones[i] \\ 
dp[i][j] \cup dp[i][j-stones[i]] & j \ge stones[i] \end{cases}$

采用逆序遍历，将dp数组维度降为一维。

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = Arrays.stream(stones).sum();
        int target = sum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for(int i = 0; i < stones.length; i ++) {
            for(int j = target; j >= stones[i]; j --) {
                dp[j] = dp[j] || dp[j - stones[i]];
            }
        }
        for(int j = target; ; j --) {
            if(dp[j]) {
                return sum - 2 * j;
            }
        }
    }
}
```
时间复杂度：$O(n·sum)$ 

基于动态规划解决零一背包问题的时间复杂度为伪多项式时间复杂度，因为复杂度和$sum$的实际值相关。

也可以转换$dp$数组的定义，定义$dp[j]$表示容量为$j$的背包实际能装下的最大质量。

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = Arrays.stream(stones).sum();
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for(int i = 0; i < stones.length; i ++) {
            for(int j = target; j >= stones[i]; j --) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[target];
    }
}
```

思考：在1049题中，如果还要求解具体的粉碎顺序（不唯一），应该如何求解？

提示：用List<Integer>[] path记录每个容量装入的物品下标列表。

真题链接：字节跳动20230903笔试 https://codefun2000.com/p/P1538
```java
int target = sum / 2;
List<Integer>[] path = new List[target + 1];
Arrays.setAll(path, k -> new ArrayList<>());
int[] dp = new int[target + 1];
for(int i = 0;i < n; i ++) {
    for(int j = target; j >= people[i]; j --) {
        if (dp[j] < dp[j - people[i]] + people[i]) {
            path[j] = new ArrayList<>(path[j - people[i]]);
            path[j].add(i);
            dp[j] = dp[j - people[i]] + people[i];
        }
    }
}
Deque<Integer> group1 = new ArrayDeque<>(path[target]);
Deque<Integer> group2 = IntStream.range(0, n).filter(e -> !path[target].contains(e)).boxed().collect(Collectors.toCollection(ArrayDeque::new));
```


练习题单：

| 题号                                                         | 难度 | 提示                 |
| ------------------------------------------------------------ | ---- | -------------------- |
| [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/) | 中等 | 模版题               |
| [494. 目标和](https://leetcode.cn/problems/target-sum/)      | 中等 | 如何建模成背包问题？ |

> 分组背包

分组背包，物品被分为若干组，每组最多只能选一个/恰好选择一个，根据题意而定。

该部分较为容易，读者可以选择以下题目练习。

| 题号                                                         | 难度 | 提示                 |
| ------------------------------------------------------------ | ---- | -------------------- |
| [2585. 获得分数的方法数](https://leetcode.cn/problems/number-of-ways-to-earn-points/) | 困难 | 分组背包               |
| [1981. 最小化目标值与所选元素的差](https://leetcode.cn/problems/minimize-the-difference-between-target-and-chosen-elements/)      | 中等 | 分组背包 |
| [2218. 从栈中取出 K 个硬币的最大面值和](https://leetcode.cn/problems/maximum-value-of-k-coins-from-piles/)      | 困难 | 分组背包，前缀和 |


#### 3.7.11 Digital Dynamic Planning

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |
例题：[1012. 至少有 1 位重复的数字](https://leetcode.cn/problems/numbers-with-repeated-digits/)

```java
class Solution {
    private char[] s;
    private int[][] memo;

    public int numDupDigitsAtMostN(int n) {
        s = String.valueOf(n).toCharArray();
        int m = s.length;
        memo = new int[m][1<<10];
        for(int[] arr : memo) {
            Arrays.fill(arr, -1);
        }
        return n - dfs(0, 0, true, false);
    }

    private int dfs(int i, int mask, boolean isLimit, boolean isNum) {
        if(i == s.length) {
            return isNum ? 1 : 0;  // 填了数字才为一种合法方案
        }
        if(!isLimit && isNum && memo[i][mask] >= 0) {
            return memo[i][mask];
        }
        int up = isLimit ? s[i] - '0' : 9;
        int ans = 0;
        if(!isNum) {  // 前面没有填数字，isLimit和isNum都为false
            ans = dfs(i + 1, mask, false, false);
        }
        // isNum为true，说明前面有数字，当前选择的数字可以从0开始。
        for(int d = isNum ? 0 : 1; d <= up; d ++) {
            if((mask >> d & 1) == 0) {  // d数字未使用
                ans += dfs(i + 1, mask | (1 << d), isLimit && d == up, true);
            }
        }
        if(!isLimit && isNum) {
            memo[i][mask] = ans;
        }
        return ans;
    }
}
```

时间复杂度：$O(mD2^D)$，$m$为s的长度，即$\log n$，$D=10$

从本题中总结数位dp的模版：

s为数字转化为的字符数组。记录s的长度为$m$。

把所有递归可能遍历到的数字都视作长度为$m$的字符串，如果长度不为$m$，则可以通过补零的方式补齐长度。该部分逻辑通过isNum变量控制。

memo为记忆化数组，维度由实际情况决定。本题需要记录数字的使用情况，所以有第二个维度。

i表示当前遍历到s的下标。

isLimit用于标识当前位是否受到最大值约束。递归开始时为true。

isNum用于i之前是否填写数字。需要根据题意判断是否需要考虑前导0。

在记忆化时，理应该记忆化i，isLimit, isNum和mask的状态。其中mask根据具体问题具体定义。实际编程中，可以只记忆化isLimit为False和isNum为True的情况。因为isLimit为True的场景全局只会递归计算一次。

例题：[233. 数字 1 的个数](https://leetcode.cn/problems/number-of-digit-one/)

本题不需要isNum变量，因为前导零的影响无关紧要。

例如，若s的长度为3，长度不足3的数字可以通过补零补齐。如007，补零不影响对1的计数。

和上一题的区别是，上一题前导零是有影响的，通过补零的方式，007会被判定为有重复数字，而事实上007是没有重复数字的。

$memo[i][j]$数组用于记录从左到右遍历到第$i$位，1的个数为$j$的个数。

```java
class Solution {
    private char[] s;
    private int[][] memo;

    public int countDigitOne(int n) {
        s = String.valueOf(n).toCharArray();
        int m = s.length;
        memo = new int[m][m];
        for(int[] arr : memo) {
            Arrays.fill(arr, -1);
        }
        return dfs(0, 0, true);
    }

    private int dfs(int i, int cnt, boolean isLimit) {
        if(i == s.length) {
            return cnt;
        }
        if(!isLimit && memo[i][cnt] >= 0) {
            return memo[i][cnt];
        }
        int ans = 0;
        for(int d = 0, up = isLimit ? s[i] - '0' : '9'; d <= up; d ++) {
            ans += dfs(i + 1, d == 1 ? cnt + 1 : cnt, isLimit && d == up);
        }
        if(!isLimit) {
            memo[i][cnt] = ans;
        }
        return ans;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [面试题 17.06. 2出现的次数](https://leetcode.cn/problems/number-of-2s-in-range-lcci/) | 困难 |
| [600. 不含连续1的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/) | 困难 |
| [902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/) | 困难 |
| [788. 旋转数字](https://leetcode.cn/problems/rotated-digits/) | 中等 |
| [1067. 范围内的数字计数](https://leetcode.cn/problems/digit-count-in-range/) | 困难 |

#### 3.7.12 Square And Rectangle Problem

本章为专题，探究一系列特殊的正方形/矩形问题。部分问题可以通过动态规划解决，部分问题需要结合前面的知识点。

类型1：只包含1的最大正方形

例题：[221. 最大正方形](https://leetcode.cn/problems/maximal-square/)

分析：动态规划，定义dp[i][j]为以i，j作为右下端点的最大正方形边长。

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        int side = 0;
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                if(matrix[i][j] == '1') {
                    if(i == 0 || j == 0) {
                        dp[i][j] = 1;
                    }else {
                        dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1])) + 1;
                    }
                    side = Math.max(side, dp[i][j]);
                }
            }
        }
        return side * side;
    }
}
```
时间复杂度：$O(m·n)$

类型2：边界为1的最大正方形

例题：[1139. 最大的以 1 为边界的正方形](https://leetcode.cn/problems/largest-1-bordered-square/)

思路：二维前缀和数组，前缀和章节有题解，代码略。

类型3：只包含1的正方形计数

例题：[1277. 统计全为 1 的正方形子矩阵](https://leetcode.cn/problems/count-square-submatrices-with-all-ones/)

思路：动态规划，类似221题，定义dp[i][j]为以i，j作为右下端点的正方形数量，代码略。

$dp[i][j]=\min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1,(matrix[i][j]=1)$

类型4：只包含1的最大矩形

例题：[85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/)

分析：单调栈。

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length, ans = 0;
        int[] height = new int[n];
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                height[j] = matrix[i][j] == '0' ? 0 : height[j] + 1;
            }
            ans = Math.max(ans, maxArea(height));
        }
        return ans;
    }

    private int maxArea(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int ans = 0, m = height.length;
        stack.push(-1);
        for(int i = 0; i <= m; i ++) {
            while(stack.size() > 1 && (i == m || height[i] < height[stack.peek()])) {
                int j = stack.pop();
                ans = Math.max(ans, height[j] * (i - stack.peek() - 1));
            }
            stack.push(i);
        }
        return ans;
    }
}
```

时间复杂度：$O(mn)$

类型5：只包含1的矩形计数

例题：[1504. 统计全 1 子矩形](https://leetcode.cn/problems/count-submatrices-with-all-ones/)

分析：单调栈，01矩阵的处理同85题，将矩阵在列方向累加，计算每个元素左侧和右侧第一个严格小于当前元素的下标。

举例：[4,9,7,2]

以7为例，7右侧第一个最小元素为2，下标为3；左侧第一个最小元素为4，下标为0，区间长度为 3 - 0 - 1 = 2，对应len变量。

取4和2的最大值，即为4，统计高度为7，6，5的子矩形数量，长为5,6,7，宽为1的矩形2\*3个；长为5,6,7，宽为2的矩形1\*3=个，总共9个，计算公式：(7-4)\*(3\*2)/2。

下面考虑存在重复元素的情况：

距离：[4,9,9,2]

栈依然要求严格单增，只是在计算时，第一个9先不参与计算，第二个9入栈后，由2出栈时，第二个9的左侧下标为0，此时可以正确进行结算。

```java
class Solution {
    public int numSubmat(int[][] mat) {
        int m = mat.length, n = mat[0].length, ans = 0;
        int[] height = new int[n];
        for(int i = 0; i < m; i ++) {
            for(int j = 0; j < n; j ++) {
                height[j] = mat[i][j] == 0 ? 0 : height[j] + 1;
            }
            ans += countRectangle(height);
        }
        return ans;
    }

    private int countRectangle(int[] height) {  // 以最后一行为底的矩形个数
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        int ans = 0, m = height.length;
        for(int r = 0; r <= m; r ++) {
            // 单调栈严格递增
            while(stack.size() > 1 && (r == m || height[stack.peek()] >= height[r])) {
                int i = stack.pop();
                if(r == m || height[i] > height[r]) {  // 严格大于时才计数
                    int l = stack.peek(), len = r - l - 1;
                    int leftHeight = l == -1 ? 0 : height[l], rightHeight = r == m ? 0 : height[r];
                    int bottom = Math.max(leftHeight, rightHeight), top = height[i];
                    ans += (top - bottom) * len * (len + 1) / 2;
                }
            }
            stack.push(r);
        }
        return ans;
    }
}
```
时间复杂度：$O(mn)$



###  3.8 State Machine

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 中       |
例题：[65. 有效数字](https://leetcode.cn/problems/valid-number/)

分析：

| 状态            | 空   | +/-  | 0-9  | .    | e    | 其他 |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0（初始态）     | 0    | 1    | 6    | 2    | -1   | -1   |
| 1（带符号）     | -1   | -1   | 6    | 2    | -1   | -1   |
| 2（小数）       | -1   | -1   | 3    | -1   | -1   | -1   |
| 3（小数终止）   | 8    | -1   | 3    | -1   | 4    | -1   |
| 4（指数）       | -1   | 7    | 5    | -1   | -1   | -1   |
| 5（指数终止）   | 8    | -1   | 5    | -1   | -1   | -1   |
| 6（整数终止）   | 8    | -1   | 6    | 3    | 4    | -1   |
| 7（指数后符号） | -1   | -1   | 5    | -1   | -1   | -1   |
| 8（空白终止）   | 8    | -1   | -1   | -1   | -1   | -1   |

```java
class Solution {
    public boolean isNumber(String s) {
        int state = 0, finals = 0b101101000;
        int[][] transfer = {{0,1,6,2,-1},{-1,-1,6,2,-1},{-1,-1,3,-1,-1},{8,-1,3,-1,4},
            {-1,7,5,-1,-1},{8,-1,5,-1,-1},{8,-1,6,3,4},{-1,-1,5,-1,-1},{8,-1,-1,-1,-1}};
        for(char c : s.toCharArray()) {
            int id = getIndex(c);
            if(id < 0) {
                return false;
            }
            state = transfer[state][id];
            if(state < 0) {
                return false;
            }
        }
        return (finals & (1 << state)) > 0;
    }

    private int getIndex(char c) {
        switch(c) {
            case ' ': return 0;
            case '+':
            case '-': return 1;
            case '.': return 3;
            case 'e': 
            case 'E': return 4;
            default:
                if(c >= '0' && c <= '9') {
                    return 2;
                }
        }
        return -1;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/) | 中等 |

### 3.9 Random Algorithm

| 面试概率 | 笔试概率 |
| -------- | -------- |
| 低       | 低       |
#### 3.9.1 Random Sampling

例题：[528. 按权重随机选择](https://leetcode.cn/problems/random-pick-with-weight/)

```java
class Solution {

    private int[] presum;
    private Random random;

    public Solution(int[] w) {
        int n = w.length;
        presum = new int[n + 1];
        for(int i = 0; i < n; i ++) {
            presum[i+1] = presum[i] + w[i];
        }
        random = new Random();
    }
    
    public int pickIndex() {
        int n = presum[presum.length - 1];
        int index = random.nextInt(n);
        return search(index);
    }

    private int search(int target) {
        int l = 0, r = presum.length - 2;
        while(l < r) {
            int mid = l + r + 1 >> 1;
            if(presum[mid] > target) {
                r = mid - 1;
            }else {
                l = mid;
            }
        }
        return l;
    }
}
```

练习题单

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [497. 非重叠矩形中的随机点](https://leetcode.cn/problems/random-point-in-non-overlapping-rectangles/) | 中等 |

#### 3.9.2 Rejection Sampling

给你一个函数rand10，可以生成1～10之间的随机数，写一个函数rand7，生成1～7之间的随机数。

拒绝采样：调用rand10，如果结果是1～7，直接返回，否则重新调用rand10。

例题：[470. 用 Rand7() 实现 Rand10()](https://leetcode.cn/problems/implement-rand10-using-rand7/)

分析：调用两次Rand7，生成大于10种可能的结果。

$t=(a-1)*7+(b-1)$

$t\in[0...39]$，接受，返回$t%10+1$

$t\in[40...48]$，拒绝

```java
class Solution extends SolBase {
    public int rand10() {
        while(true) {
            int a = rand7() - 1, b = rand7() - 1;
            int t = a * 7 + b;
            if(t >= 40) {
                continue;
            }
            return t % 10 + 1;
        }
    }
}
```

| 题号                                                         | 难度 |
| ------------------------------------------------------------ | ---- |
| [478. 在圆内随机生成点](https://leetcode.cn/problems/generate-random-point-in-a-circle/) | 中等 |

#### 3.9.3 Fisher–Yates Shuffle

例题：[384. 打乱数组](https://leetcode.cn/problems/shuffle-an-array/)

```java
class Solution {
    private int[] nums;
    private Random random;

    public Solution(int[] nums) {
        this.nums = nums.clone();
        this.random = new Random();
    }
    
    public int[] reset() {
        return nums.clone();
    }
    
    public int[] shuffle() {
        int[] data = nums.clone();
        for(int i = data.length - 1; i >= 0; i --) {
            int j = random.nextInt(i + 1);
            swap(data, i, j);
        }
        return data;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

等概率抽取三名不同的获奖用户？执行上述代码三步，取最后3个元素。

#### 3.9.4 Reservoir Sampling

从数据流中抽取$k$个元素，使得数据流中每一个元素能够被等概率抽取。

先从数据流中取出$k$个元素，放入一个蓄水池中。

处理数据流中索引为$i$的元素，随机生成$[0...i]$之间的索引$j$。

若$j$在$[0...k-1]$之间，则将新元素替换蓄水池下标$j$的元素，否则丢弃。

正确性：

处理索引为$k$的元素。

对于新元素，放入蓄水池的概率为$\frac{k}{k+1}$；

对于原本存在蓄水池中的每个元素，被替换的概率为$\frac{1}{k+1}$。

每个元素都以$\frac{k}{k+1}$的概率出现在蓄水池中。

假设处理完索引为$i-1$的元素成立，此时每个元素以$\frac{k}{i}$概率出现在蓄水池中。

处理索引为$i$的元素。

对于新元素，放入蓄水池的概率为$\frac{k}{i+1}$；

对于原本存在蓄水池中的每个元素，继续保留的概率为$\frac{i}{i+1}*\frac{k}{i}=\frac{k}{i+1}$。

所有元素以$\frac{k}{i+1}$的概率出现在蓄水池中，根据数学归纳法得证。

例题：[382. 链表随机节点](https://leetcode.cn/problems/linked-list-random-node/)

```java
class Solution {
    private ListNode node;
    private Random random;

    public Solution(ListNode head) {
        node = head;
        random = new Random();
    }
    
    public int getRandom() {
        int res = node.val;
        int index = 1;
        for(ListNode cur = node.next; cur != null; cur = cur.next, index ++) {
            int j = random.nextInt(index + 1);
            if(j == 0) {
                res = cur.val;
            }
        }
        return res;
    }
}
```
