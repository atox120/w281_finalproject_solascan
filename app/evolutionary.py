import math
import random
import time
import numpy as np
import pandas as pd
from app.utils import parallelize


class Feature:

    def __init__(self, feature_name, low_range, high_range, feat_cast):

        self.name = feature_name
        self.low_range = low_range
        self.high_range = high_range
        self.feat_cast = feat_cast
        self.mask = 1 if feat_cast == int else 0

    def create_sample(self):
        """

        :return:
        """

        if self.feat_cast == int:
            sample = random.randint(self.low_range, self.high_range)
        else:
            sample = self.low_range + (random.random() * (self.high_range - self.low_range))

        return sample


class Evolver:

    def __init__(self, optimize_class, optimize_params, mutation_rate=0.1, num_jobs=1):
        """

        :param optimize_class: The class to optimize
        :param optimize_params: Initialization parameters for the class
        :param mutation_rate: The mutation rate for random mutations
        :param num_jobs: Number of jobs to split into.
        """

        self.mutation_rate = mutation_rate
        self.optimize_class = optimize_class
        self.optimize_params = optimize_params

        self.features = []
        self.feature_names = None
        self.parent_gen = pd.DataFrame()
        self.num_jobs = num_jobs

    def add_feature(self, feature_name, low_range, high_range, feat_cast):
        """

        :param feature_name: Name of feature, this will be the dictionary key
        :param low_range: Low range for feature
        :param high_range: High range for feature
        :param feat_cast: Type cast 'int' or 'float'
        :return:
        """

        # Add the feature to list
        self.features.append(Feature(feature_name, low_range, high_range, feat_cast))

    def generate_sample(self, num_samples=1):
        """

        :param num_samples: Number of samples to collect response on
        :return:
        """

        # List out all features
        if self.feature_names is None:
            self.feature_names = [x.name for x in self.features]

        samples = []
        for i in range(num_samples):
            # Create the samples
            this_sample = {x.name: x.create_sample() for x in self.features}
            samples.append(this_sample)

        return samples

    def collect_response(self, samples):
        """

        :param samples: These are samples to collect response
        :return:
        """

        # Number of chunks to break the list down to num of jobs
        num_chunks = math.ceil(len(samples)/self.num_jobs)
        chunks = [samples[i:i+self.num_jobs] for i in range(0, num_chunks, self.num_jobs)]

        print('    Starting response collection')
        start = time.perf_counter()
        results = []
        for chunk in chunks:
            funcs = []
            args = []
            chunk_start = time.perf_counter()
            for arg in chunk:
                # Create an object of the class to be optimized
                data_obj = self.optimize_class(**self.optimize_params)
                funcs.append(data_obj.get)
                args.append(arg)

            if self.num_jobs > 1:
                # Run the object in parallel and append the results
                result = parallelize(funcs=funcs, args=args)
            else:
                result = [func(**arg) for func, arg in zip(funcs, args)]

            # Append the args to the results
            for ar, r in zip(chunk, result):
                ar['response'] = r
            results += chunk

            print(f'        Completed chunk in {time.perf_counter() - chunk_start}s')

        # Convert it into a DataFrame and order in order of features
        responses_df = pd.DataFrame(results)[self.feature_names + ['response']]
        print(f"    Collected response in {time.perf_counter() - start}s")
        return responses_df

    def create_gen0(self, num_samples):
        """

        :param num_samples: Number of samples for generation 0
        :return:
        """

        start = time.perf_counter()
        print('Starting generation 0')

        # Create samples to be measured
        samples = self.generate_sample(num_samples)

        # Measure the response for the samples
        responses_df = self.collect_response(samples)

        print(f'Completed generation 0 in {time.perf_counter() - start}s')
        return responses_df

    def _crossover(self):
        """
        Let's consider we are optimizing four params(A, B, C, D) with 2 samples [4, 7, 11, 2] and [55, 4, 97, 8]
        Then one crossover child would be a combination of these two:
            ex: [4, 7, 97, 8] for integers or random weighted sum for floats

        :return:
        """

        # These are the masks to use for crossover
        # Integers get the one of the parent value
        # Floats get a combination of parents
        feature_mask = np.array([x.mask for x in self.features])[:, np.newaxis].T
        feature_mask = np.repeat(feature_mask, self.parent_gen.shape[0], axis=0)

        # All the integers are set to 1
        # Adding a random number will make them greater than 1
        # Now pegging it at 1 should do the trick
        feature_mask = np.random.random(feature_mask.shape) + feature_mask
        feature_mask[feature_mask > 1] = 1

        # Cross over mask determines the point in the array where the crossover occurs
        crossover_point = np.random.randint(0, feature_mask.shape[1], size=(feature_mask.shape[0], ))
        crossover_mask = np.zeros(feature_mask.shape)
        crossover_mask[np.arange(crossover_mask.shape[0]), crossover_point] = 1
        # Anything to the left of the crossover point is 0 and anything to the right is one
        crossover_mask = (np.cumsum(crossover_mask, axis=1) > 0).astype(int)

        # Eliminate all points on the left of the feature mask
        feature_mask = feature_mask * crossover_mask

        # Convert the parent array to numpy
        parent = self.parent_gen[self.feature_names].to_numpy()
        another_parent = parent.copy()

        # Shuffle the second parent
        index = list(range(another_parent.shape[0]))
        random.shuffle(index)
        another_parent = another_parent[index, :]

        # Create a child
        children = (1 - feature_mask) * parent + feature_mask * another_parent

        return children

    def _mutate(self, children):
        """
        Randomly chnage the value of one gene

        :return:
        """

        # Create an array for the mutation
        mutations = np.random.random(children.shape)
        mutations = mutations < self.mutation_rate

        # Generate samples
        samples = self.generate_sample(num_samples=mutations.shape[0])
        samples = pd.DataFrame(samples)[self.feature_names].to_numpy()

        # The children are replaced with the mutated value
        children[mutations] = samples[mutations]

        return children

    def run(self, num_samples, generations):
        """

        :param num_samples: Number of samples per iteration
        :param generations: Number of generations
        :return:
        """

        self.parent_gen = self.create_gen0(num_samples)

        for gen in range(generations):
            print(f'Starting generation {gen+1}')
            # Do crossover
            children = self._crossover()

            # Do mutation
            children = self._mutate(children)

            # Collect response
            children = pd.DataFrame(children, columns=self.feature_names).to_dict('records')

            # Typecast to the desired datatype
            cast_dict = {x.name: x.feat_cast for x in self.features}
            for child in children:
                for key in child:
                    child[key] = cast_dict[key](child[key])

            response = self.collect_response(children)

            # Parent or child
            self.parent_gen = pd.concat((self.parent_gen, response))
            self.parent_gen = self.parent_gen.sort_values(by='response')[:num_samples]
            self.parent_gen = self.parent_gen.reset_index(drop=True)
            self.parent_gen.to_csv(f'generation_{gen+1}.csv')
            print(f"Completed generation {gen+1} with response {self.parent_gen.iloc[0]['response']}")

        return self.parent_gen


class Rastrigin:

    def __init__(self, dimensions=5):
        """

        :param dimensions: Number of dimensions in the Rastrigin function
        """
        self.d = dimensions

    def get(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        val = 10*self.d
        for i in range(self.d):
            key = f'x{i}'
            val += kwargs[key]**2 - 10*math.cos(2*math.pi*kwargs[key])

        return val


if __name__ == '__main__':
    do_rastrigin = True
    if do_rastrigin:
        evo = Evolver(Rastrigin, {}, num_jobs=10)
        dims = 5
        for d in range(dims):
            cast = int if random.random() < 0.5 else float
            # cast = float
            evo.add_feature(f'x{d}', cast(-5.12), cast(5.12), cast)

        evo.run(num_samples=500, generations=100)
