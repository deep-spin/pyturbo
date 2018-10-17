class Reader(object):
    '''An object that creates readers of a specialized class.'''
    def __init__(self, reader_class=None):
        """
        Initialize the reader manager with the class of the specialized reader.

        Once created, this reader can be used in with blocks by calling the
        open() method:

        ```
        reader = Reader(SomeClass)
        with reader.open(file1) as r:
            for item in r:
                ...
        ```

        :param reader_class: a concrete subclass of AuxiliaryReader.
        """
        assert issubclass(reader_class, AuxiliaryReader)
        self.reader_class = reader_class

    def open(self, path):
        return self.reader_class(path)


class AuxiliaryReader(object):
    """
    Auxiliary reader used internally as a context manager.

    It allows the usage of `Reader.open()` in with blocks.
    """
    def __init__(self, path):
        self.file = open(path, 'r', encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def close(self):
        return self.file.close()
