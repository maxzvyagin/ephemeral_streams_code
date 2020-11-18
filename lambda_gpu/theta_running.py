"""Code specifically to faciliate runs on ThetaGPU"""

from argparse import ArgumentParser
from hyperspace import create_hyperspace
import ray
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
import pickle
import sys
from pytorch_unet import segmentation_pt_objective

# 1) Construct the scikit optimize spaces and save as pickled objects
def construct_spaces(args):
    # set up hyperparameter spaces
    hyperparameters = [(0.00001, 0.1),  # learning_rate
                       (10, 100),  # epochs
                       (1, 64)]  # batch size
    space = create_hyperspace(hyperparameters)
    space_name = args.out.split(".csv")[0]
    space_name += "spaces.pkl"
    f = open("/tmp/mzvyagin/"+space_name, "wb")
    pickle.dump(space, f)
    print("Created pickled hyperspaces.")

# 2) For each space, run the
def run_space(args):
    s = int(args.space)
    space_name = args.out.split(".csv")[0]
    space_name += "spaces.pkl"
    f = open("/tmp/mzvyagin/" + space_name, "rb")
    spaces = pickle.load(f)
    current_space = spaces[s]
    optimizer = Optimizer(current_space)
    search_algo = SkOptSearch(optimizer, ['learning_rate', 'epochs', 'batch_size'],
                              metric='average_res', mode='max')
    analysis = tune.run(tune_unet, search_alg=search_algo, num_samples=int(args.trials),
                        resources_per_trial={'cpu': 25, 'gpu': 1},
                        local_dir="/tmp/ray_results/")
    df = analysis.results_df
    df_name = args.out.split('.csv')[0]
    df_name += (s+".csv")
    df.to_csv(df_name)
    print("Finished space "+s)


def tune_unet(config):
    pt_test_acc, pt_model = segmentation_pt_objective(config)
    search_results = {'test_acc': pt_test_acc}
    tune.report(**search_results)
    return search_results

# 3) For each space, concatenate results
def concat_results(args):
    pass


if __name__ == "__main__":
    #ray.init()
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", required=True)
    parser.add_argument("-o", "--out")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-s", "--space")
    args = parser.parse_args()

    if args.mode == "create":
        construct_spaces(args)

    elif args.mode == "run":
        run_space(args)

    elif args.mode == "concat":
        concat_results(args)

    else:
        print("Unknown mode error. Please try again.")
        sys.exit()
