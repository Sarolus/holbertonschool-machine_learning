#!/usr/bin/env python3
"""
    From dict panda conversion module
"""

import pandas as pd


my_dict = {"First": [i / 2 for i in range(0, 4)],
           "Second": ["one", "two", "three", "four"]}

dataframe = pd.DataFrame(my_dict, index=["A", "B", "C", "D"])
