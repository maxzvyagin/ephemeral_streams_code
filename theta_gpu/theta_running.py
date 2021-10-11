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
import pandas as pd

# 1) Construct the scikit optimize spaces and save as pickled objects
def construct_spaces(args):
    # set up hyperparameter spaces
    hyperparameters = [(0.00001, 0.1),  # learning_rate
                       (10, 100),  # epochs
                       (1, 250),  # batch size
                       (1, .00000001)] # epsilon for Adam optimizer
    space = create_hyperspace(hyperparameters)
    space_name = args.out.split(".csv")[0]
    space_name += "spaces.pkl"
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/"+space_name, "wb")
    pickle.dump(space, f)
    print("Created pickled hyperspaces.")

# 2) For each space, run the space
def run_space(args):
    s = int(args.space)
    space_name = args.out.split(".csv")[0]
    space_name += "spaces.pkl"
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/" + space_name, "rb")
    spaces = pickle.load(f)
    current_space = spaces[s]
    optimizer = Optimizer(current_space)
    search_algo = SkOptSearch(optimizer, ['learning_rate', 'epochs', 'batch_size', 'adam_epsilon'],
                              metric='test_acc', mode='max')
    analysis = tune.run(tune_unet, search_alg=search_algo, num_samples=int(args.trials),
                        resources_per_trial={'cpu': 25, 'gpu': 1},
                        local_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results")
    df = analysis.results_df
    df_name = "/home/mzvyagin/ephemeral_streams_code/theta_gpu/"
    split_name = args.out.split('.csv')[0]
    df_name += split_name
    df_name += "_space"
    df_name += args.space
    df_name += ".csv"
    df.to_csv(df_name)
    print("Finished space "+args.space)


def tune_unet(config):
    pt_test_acc, pt_model = segmentation_pt_objective(config)
    search_results = {'test_acc': pt_test_acc}
    tune.report(**search_results)
    return search_results

# 3) For each space, concatenate results
def concat_results(args):
    df_name = "/home/mzvyagin/ephemeral_streams_code/theta_gpu/"
    split_name = args.out.split('.csv')[0]
    df_name += split_name
    df_name += "_space"
    df_name += str(0)
    df_name += ".csv"
    results = pd.read_csv(df_name)
    for i in list(range(1, 16)):
        df_name = "/home/mzvyagin/ephemeral_streams_code/theta_gpu/"
        split_name = args.out.split('.csv')[0]
        df_name += split_name
        df_name += "_space"
        df_name += str(i)
        df_name += ".csv"
        df = pd.read_csv(df_name)
        results = pd.concat([results, df], ignore_index=True)
    df_name = "/home/mzvyagin/ephemeral_streams_code/theta_gpu/"
    split_name = args.out.split('.csv')[0]
    df_name += split_name
    df_name += "_space"
    df_name += "_allresults"
    df_name += ".csv"
    results.to_csv(df_name)
    print("Saved results to "+df_name)


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
