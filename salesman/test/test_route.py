import numpy as np

from salesman.city import CityFactory
from salesman.route import Route, generate_rand_routes, sort_routes, \
    calculate_capacity, divide_route_before_after, \
    generate_nearest_neighbour_route


class TestRoute:
    def test_generate(self):
        n_cities = 10
        cities = CityFactory.random(n_cities=n_cities)
        route = Route(cities=cities)
        route.generate()
        assert len(route.tour) == n_cities

    def test_calc_length_initial(self):
        xy = np.array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]
                       ])
        cities = CityFactory.custom(xy=xy)
        n_cities = cities['xy'].shape[0]
        route = Route(cities=cities)
        route.tour = list(range(n_cities))
        len = route.calc_length()
        assert len == 4

    def test_calc_length_recalc_outdated(self):
        pass

    def test_get_length(self):
        xy = np.array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]
                       ])
        cities = CityFactory.custom(xy=xy)
        n_cities = cities['xy'].shape[0]
        route = Route(cities=cities)
        route.tour = list(range(n_cities))
        len = route.get_length()
        assert len == 4

    def test_plot(self):
        pass


def test_tour_swap_2_next():
    pass


def test_tour_swap_2_random():
    pass


def test_tour_reverse_triple():
    pass


def test_tour_compare():
    pass


def test_generate_rand_routes():
    cities = CityFactory.random(n_cities=8)
    generate_rand_routes(cities, n_routes=10)


def test_sort_routes():
    routes = ['a', 'b', 'c', 'd']
    routes_len = [7, 4, 1, 5]
    routes, routes_len = sort_routes(routes, routes_len)
    assert routes == ['c', 'b', 'd', 'a']
    assert routes_len == [1, 4, 5, 7]


def test_calculate_capacity():
    capacity = calculate_capacity(routes_sorted=['a', 'b', 'c', 'd'])
    assert list(capacity) == [1, 0.75, 0.5, 0.25]


def test_divide_route_before_after_first():
    tour = ['a', 'b', 'c', 'd', 'e', 'f']
    before, after = divide_route_before_after(tour, start_idx=0)
    assert before == []
    assert after == ['b', 'c', 'd', 'e', 'f']


def test_divide_route_before_after_last():
    tour = ['a', 'b', 'c', 'd', 'e', 'f']
    before, after = divide_route_before_after(tour, start_idx=5)
    assert after == []
    assert before == ['a', 'b', 'c', 'd', 'e']


def test_divide_route_before_after_middle():
    tour = ['a', 'b', 'c', 'd', 'e', 'f']
    before, after = divide_route_before_after(tour, start_idx=2)
    assert before == ['a', 'b']
    assert after == ['d', 'e', 'f']

def test_generate_nearest_neighbour_route():
    cities = CityFactory.circle(r=10, n_cities=4)
    generate_nearest_neighbour_route(cities)
