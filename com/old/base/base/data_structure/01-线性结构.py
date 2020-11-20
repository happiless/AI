class Array:
    def __init__(self, size=4):
        self.__size = size
        self.__item = [None] * size
        self.__len = 0

    def __setitem__(self, key, value):
        self.__item[key] = value
        self.__len += 1

    def __getitem__(self, item):
        return self.__item[item]

    def __len__(self):
        return self.__len

    def __iter__(self):
        for value in self.__item:
            yield value


if __name__ == '__main__':
    a1 = Array()
    a1[0] = '孙悟空'
    a1[1] = '猪八戒'
    for v in a1:
        print(v)

    print(len(a1))
