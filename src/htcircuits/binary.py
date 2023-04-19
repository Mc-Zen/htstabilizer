from numpy import int8

class Binary(int8):
    def __new__(cls, value):
        return super(cls, cls).__new__(cls, int8(value != 0))
    

    def __add__(self, other):
        return self.__class__(self, other)