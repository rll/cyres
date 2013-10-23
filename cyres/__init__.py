from cyres import *

def get_cython_include():
    import os
    include_path = os.path.dirname(__file__)
    include_path = os.path.join(include_path, "src")
    return include_path
