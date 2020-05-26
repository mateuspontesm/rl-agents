#!/usr/bin/env python

"""Tests for `rl_agents` package."""

import pytest

from rl_agents import example


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/johanvergeer/'+
    # 'cookiecutter-poetry')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    assert example.some_function(1,2) == 1 + 2
