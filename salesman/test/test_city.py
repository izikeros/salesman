import numpy as np
import pytest

from salesman.city import (
    cities_on_circle, cities_on_line, cities_random,
    gen_distance_lookup, get_cities_labels, CityFactory)


def test_circle():
    n = 12
    cities = CityFactory.circle(r=1, n_cities=n)
    assert cities['xy'].shape[0] == n
    assert cities['xy'].shape[1] == 2
    assert cities['dist'][0][1] == cities['dist'][1][0]


def test_line():
    n = 12
    cities = CityFactory.line(n_cities=n)
    assert cities['xy'].shape[0] == n
    assert cities['xy'].shape[1] == 2
    assert cities['dist'][0][1] == cities['dist'][1][0]


def test_random():
    n = 12
    cities = CityFactory.random(n_cities=n)
    assert cities['xy'].shape[0] == n
    assert cities['xy'].shape[1] == 2
    assert cities['dist'][0][1] == cities['dist'][1][0]


def test_cities_on_circle():
    r = 10
    n = 8
    xy = cities_on_circle(r=r, n_cities=n)
    assert xy.shape[0] == n
    assert xy.shape[1] == 2


def test_cities_on_line():
    n = 8
    xy = cities_on_line(n_cities=n)
    assert xy.shape[0] == n
    assert xy.shape[1] == 2


def test_cities_random():
    n = 8
    xy = cities_random(n_cities=n)
    assert xy.shape[0] == n
    assert xy.shape[1] == 2


def test_gen_distance_lookup():
    """test if distance is equal to diagonal of square with edge len=1"""
    xy = np.array([[1, 1],
                   [2, 2]])
    d = gen_distance_lookup(xy=xy)
    idx_1 = 0
    idx_2 = 1

    assert d[idx_1][idx_2] == pytest.approx(1.41, 0.01)

    # is distance matrix symetrical?
    assert d[idx_1][idx_2] == d[idx_1][idx_2]
    pass


def test_get_cities_labels():
    n_cities = 10
    labels = get_cities_labels(n_cities)
    assert len(labels) == n_cities


def test_dist_one():
    xy = np.array([[1, 0],
                   [2, 0]])
    cities = CityFactory.custom(xy=xy)
    dist = cities['dist']
    idx_1 = 0
    idx_2 = 1
    d = dist[idx_1][idx_2]
    assert d == 1


def test_dist_diagonal():
    """test if distance is equal to diagonal of square with edge len=1"""
    xy = np.array([[1, 1],
                   [2, 2]])
    cities = CityFactory.custom(xy=xy)
    dist = cities['dist']
    idx_1 = 0
    idx_2 = 1
    d = dist[idx_1][idx_2]
    assert d == pytest.approx(1.41, 0.01)
