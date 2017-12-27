import unittest

from utils.pascal_voc import pascal_voc


class MyTestCase(unittest.TestCase):
    def test_getdata(self):
        pascal = pascal_voc('train')
        images, labels = pascal.get()

        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
