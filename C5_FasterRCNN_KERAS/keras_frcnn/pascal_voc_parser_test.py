import pprint
import unittest

from keras_frcnn.pascal_voc_parser import get_data


class MyTestCase(unittest.TestCase):
    def test_something(self):
        input_path = '/home/syh/dl/dlfive/C5_FasterRCNN_KERAS/data/'
        all_imgs, classes_count, class_mapping = get_data(input_path)

        print("class_mapping:")
        pprint.pprint(class_mapping)

        print("classes_count:")
        pprint.pprint(classes_count)
        self.assertEqual(len(all_imgs), 17125)


if __name__ == '__main__':
    unittest.main()
