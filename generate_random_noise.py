import argparse
import shutil
import os
import numpy as np

folder_data = "data"
def main(args):
    task = args.task
    pct_noise = args.noise
    update_dataset(task,"train",pct_noise,args.seed)
    update_dataset(task, "dev", pct_noise, args.seed)

def update_dataset(task,partition, pct_noise,seed):
    np.random.seed(seed)
    folder_random = os.path.join(folder_data, task + "_rand_" + str(pct_noise))
    if not os.path.isdir(folder_random):
        shutil.copytree(os.path.join(folder_data, task), folder_random)
    with open(os.path.join(folder_random, "intent_label.txt")) as fp_intent_label:
        list_intents = fp_intent_label.read().splitlines()
    del list_intents[0]
    cnt_replace = 1
    file_label = os.path.join(folder_random, partition, "label")
    file_label_rnd = os.path.join(folder_random, partition, "label_rnd")
    with open(file_label) as fp_label:
        with open(file_label_rnd, "w") as fp_label_rnd:
            for line in fp_label:
                rnd_value = np.random.rand()
                if rnd_value < (float(pct_noise) / 100):
                    print("{}:{}. Replacing {}".format(cnt_replace, rnd_value, line))
                    intent = line.rstrip()
                    new_intent = np.random.choice(list_intents)
                    while new_intent == intent:
                        new_intent = np.random.choice(list_intents)
                    line = new_intent
                    line += "\n"
                    print("With {}".format(line))
                    cnt_replace += 1
                fp_label_rnd.write(line)
                # fp_label_rnd.write("\n")
    os.replace(file_label_rnd, file_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--noise", default=None, required=True, type=int, help="Percentage of noise")
    parser.add_argument("--seed", default=1, required=True, type=int, help="Percentage of noise")
    args = parser.parse_args()
    main(args)