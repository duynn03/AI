from scipy.spatial import distance as dist


class Searcher:
    def __init__(self, index):
        # store the index that we will be searching over
        self.index = index

    def search(self, pokemonFeatures):
        # initialize our dictionary of results
        results = {}

        # loop over the images in our index
        for (i, features) in self.index.items():
            # compute the distance between the pokemon features and features in our index, then update the results
            distance = dist.euclidean(pokemonFeatures, features)
            results[i] = distance

        # sort our results, where a smaller distance indicates higher similarity
        results = sorted([(v, i) for (i, v) in results.items()])

        # return the results
        return results
