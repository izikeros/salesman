from salesman.route import sort_routes
from salesman.darwin import (
    NaturalSelection)


class TestNaturalselection:
    def test_1(self):
        n_keep = 3
        routes = ['a', 'b', 'c', 'd']
        routes_len = [7, 4, 1, 5]
        routes_sorted, routes_len = sort_routes(routes, routes_len)
        routes, routes_len = NaturalSelection.keep_n_best(routes_sorted,
                                                          routes_len,
                                                          n_keep=n_keep)
        assert routes == ['c', 'b', 'd']
        assert routes_len == [1, 4, 5]
