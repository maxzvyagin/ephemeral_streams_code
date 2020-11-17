"""Ray Tune and HyperSpace implementation to optimize UNet model"""
from pytorch_unet import segmentation_pt_objective
from argparse import ArgumentParser
import argparse
from hyperspace import create_hyperspace
import ray
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
from tqdm import tqdm
import time

def tune_unet(config):
    pt_test_acc, pt_model = segmentation_pt_objective(config)
    search_results = {'test_acc': pt_test_acc}
    tune.report(**search_results)
    return search_results


if __name__ == "__main__":
    ray.init()
    parser = ArgumentParser()
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-t", "--trials", required=True)
    args = parser.parse_args()
    # set up hyperparameter spaces
    hyperparameters = [(0.00001, 0.1),  # learning_rate
                       (10, 100),  # epochs
                       (100, 1000)]  # batch size
    space = create_hyperspace(hyperparameters)

    # Run and aggregate the results
    results = []
    i = 0
    error_name = args.out.split(".csv")[0]
    error_name += "_error.txt"
    error_file = open(error_name, "w")
    for section in tqdm(space):
        # create a skopt gp minimize object
        optimizer = Optimizer(section)
        search_algo = SkOptSearch(optimizer, ['learning_rate', 'epochs', 'batch_size'],
                                  metric='average_res', mode='max')
        try:
            analysis = tune.run(tune_unet, search_alg=search_algo, num_samples=args.trials,
                                resources_per_trial={'cpu': 25, 'gpu': 1},
                                local_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results")
            # analysis = tune.run(multi_train, search_alg=search_algo, num_samples=TRIALS,
            #                     resources_per_trial={'cpu': 25, 'gpu': 1})
            results.append(analysis)
        except Exception as e:
            error_file.write("Unable to complete trials in space " + str(i) + "... Exception below.")
            error_file.write(str(e))
            error_file.write("\n\n")
            print("Unable to complete trials in space " + str(i) + "... Continuing with other trials.")
        i += 1

    error_file.close()

    # save results to specified csv file
    all_results = results[0].results_df
    for i in range(1, len(results)):
        all_results = all_results.append(results[i].results_df)

    all_results.to_csv(args.out)
    print("Ray Tune results have been saved at " + args.out + " .")
    print("Error file has been saved at " + error_name + " .")