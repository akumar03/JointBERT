import argparse
import shutil
import os
import numpy as np

folder_data = "data"
def main(args):
    np.random.seed(20200312)
    task = args.task
    pct_noise = args.noise
    folder_random = os.path.join(folder_data, task+"_rand_"+str(pct_noise))
    if not os.path.isdir(folder_random):
        shutil.copytree(os.path.join(folder_data,task), folder_random)
    with open(os.path.join(folder_random,"intent_label.txt")) as fp_intent_label:
        list_intents = fp_intent_label.read().splitlines()
    cnt_replace = 1
    with open(os.path.join(folder_random,"train","label")) as fp_label:
        with open(os.path.join(folder_random, "train","label_rnd"),"w") as fp_label_rnd:
            for line in fp_label:
                rnd_value = np.random.rand()
                if rnd_value < (float(pct_noise)/100):
                    print("{}:{}. Replacing {}".format(cnt_replace,rnd_value,line))
                    line = np.random.choice(list_intents)
                    line += "\n"
                    print("With {}".format(line))
                    cnt_replace += 1
                fp_label_rnd.write(line)
                #fp_label_rnd.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--noise", default=None, required=True, type=int, help="Percentage of noise")
    args = parser.parse_args()
    main(args)