class Node:
    def __init__(self, value=None, next=None):
        self.value = value
        self.next = next

    def __str__(self):
        return "Node:{}".format(self.value)


class LinkedList:
    def __init__(self):
        self.root = Node()
        self.next = None
        self.size = 0

    def append(self, value):
        node = Node(value)
        if not self.next:               # 如果没有节点时
            self.root.next = node       # 将新节点挂到root后面
        else:
            self.next.next = node       # 将新节点挂到最后一个节点上
        self.next = node
        self.size += 1

    def append_first(self, value):
        node = Node(value)
        if not self.next:
            self.root.next = node
            self.next = node
        else:
            temp = self.root.next       # 获取原来root后面的那个节点
            self.root.next = node       # 将新的节点挂到root上
            node.next = temp            # 新的节点的下一个节点是原来的root后的节点
        self.size += 1

    def __iter__(self):
        current = self.root.next
        if current:
            while current is not self.next:
                yield current
                current = current.next
            yield current


if __name__ == '__main__':
    list1 = LinkedList()
    list1.append(1)
    list1.append(2)
    list1.append(3)
    list1.append_first(4)
    for v in list1:
        print(v)
