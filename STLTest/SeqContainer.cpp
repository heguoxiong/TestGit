/*
 * @Author: mango 2965531503@qq.com
 * @Date: 2022-05-04 15:23:11
 * @LastEditors: mango 2965531503@qq.com
 * @LastEditTime: 2022-05-06 09:28:23
 * @FilePath: \VScodePragram\STLTest\main.cpp
 * @Description: 这是默??设置,请??置`customMade`, 打开koroFileHeader查看配置 进??设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

/*
顺序容器包括：vector deque list forward_list array string
最大最小堆（优先队列）：priority_queue
*/

// vector：可变大小数组，????????随机??????在尾部插入、删除元素比较快??
// deque：双????列，????????随机??????在头尾插入、删除元素比较快??
// list : 双向链表，只????双向顺序访问。在list????何位????入、删除都比较????
// forward_list：单向链????????持单向顺序??????在链表任何位置进??插入、删除都比较????
// array：固定大小数组，????随机访问。不能添加或者删除元素??
// string：与vector相同的??????但专门用于保存字符。在尾部插入、删除比较快??

#include "SeqContainer.h"

using namespace std;
/**
 * @description: 测试vector的构造方??
 * @param {*}
 * @return {*}
 */
void TestVectorInitial()
{
    vector<int> vecInt1;                                 // 默??构造函??
    vector<int> vecInt2{1, 2, 3, 4, 5};                  //列表初????
    vector<int> vecInt3(vecInt2);                        //拷贝构??
    vector<int> vecInt4(vecInt2.begin(), vecInt2.end()); //使用????器指定范围的元素构造（array不支持）
    PrintVecor(vecInt1);
    PrintVecor(vecInt2);
    PrintVecor(vecInt3);
    PrintVecor(vecInt4);
}

/**
 * @description: 打印容器nums????每一??????
 * @param {vector<int>} &nums 传入待打印的容器
 * @return {*}
 */
void PrintVecor(const vector<int> &nums)
{
    if (nums.empty())
    {
        cout << "容器为空器" << endl;
        return;
    }
    cout << "开始打印容器中的元素：" << endl;
    for (auto item : nums)
    {
        cout << item << endl;
    }
}