"""
    FuzzyDict is a dictionary object that finds
    the closest match
"""

from collections import defaultdict

from fuzzywuzzy import process


class FuzzyDict(defaultdict):
    """ FuzzyDict attempts to pair to find a good key match
        before resorting to the passed default_factory
    """

    def __init__(self, default_factory, threshold=85, process_fn=process.extractOne):
        """

        :default_factory: the factory function that outputs the values for
            keys not in the dictionary
        :params threshold: is the score that the process function should
            output to be accepted as a key
        """
        self.default_factory = default_factory
        self.threshold = threshold
        self.process_fn = process_fn

    def __missing__(self, key):
        """ Handle a key that does not exist in the FuzzyDict """

        if len(self) > 0:
            best_choice, score = self.process_fn(key, self.keys())
            if score > self.threshold:
                return self[best_choice]

        return self.default_factory()
