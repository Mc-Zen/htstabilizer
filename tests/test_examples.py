# This test file is designated for running the example files
# (and checking that they run without error). The example files
# lie in the example folder and thus a call to for example
#    import htstabilizer.graph
# results in an import error. 
# To solve this, we add the src/ directory to the sys path here

import unittest
import sys

sys.path.insert(0, "src/") # simulate the package being installed, so that


class TestExamples(unittest.TestCase):

    def test_readout_example(self):
        import examples.readout_circuit
        
    def test_tomography_example(self):
        import examples.state_tomography
        
    def test_compress_preparation_circuit(self):
        import examples.compress_preparation_circuit
        