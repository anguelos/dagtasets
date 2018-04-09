import unittest
from dagtasets.wi2013 import WI2013


class Tester(unittest.TestCase):

    def test_download(self):
        trds = WI2013('/tmp/wi2013', train=True, transform=None, target_transform=None, download=True)
        tstds = WI2013('/tmp/wi2013', train=False, transform=None, target_transform=None, download=True)

    def test_reuse(self):
        trds = WI2013('/tmp/wi2013', train=True, transform=None, target_transform=None, download=False)
        tstds = WI2013('/tmp/wi2013', train=False, transform=None, target_transform=None, download=False)
        assert len(trds) == 400
        assert len(tstds) == 1000

    def test_transforms(self):
        trds = WI2013('/tmp/wi2013', train=True, transform=None, target_transform=None, download=False)
        assert any([sample[0].size(1) == 512 for sample in trds])


if __name__ == '__main__':
    unittest.main()
