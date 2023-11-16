# spatialGRN
# __package__ = 'spagrn'

__all__ = ['regulatory_network', 'plot'] #, 'spa_logger']

# deprecated to keep older scripts who import this from breaking
from .regulatory_network import InferNetwork
# from spagrn.spa_logger import GetLogger

COLORS = [
        '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
        '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
        '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
        '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
        '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
        '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
    ]
