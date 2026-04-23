import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def test_import():
    import litcluster as lc
    assert hasattr(lc, 'LitCluster')

def test_paper():
    import litcluster as lc
    assert hasattr(lc, 'Paper')

def test_cluster():
    import litcluster as lc
    assert hasattr(lc, 'Cluster')

def test_tokenise():
    import litcluster as lc
    tokens = lc._tokenise('hello world test foo bar')
    assert isinstance(tokens, list) and len(tokens) > 0
