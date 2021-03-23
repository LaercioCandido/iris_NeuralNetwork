import unittest 


class TestSample(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')