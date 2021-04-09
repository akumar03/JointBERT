import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_loader import load_and_cache_examples
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from tqdm import tqdm, trange
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class IntentSimilarityTSNE(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.bert = BertModel(config=self.config)
        self.bert.eval()
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.bert.to(self.device)

    def compute_similarity(self):

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        epoch = 1
        n_classes = 22
        np_embedding = np.empty((0, 768))
        np_count = np.array([])
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                input_ids = batch[0]
                attention_mask = batch[1]
                token_type_ids = batch[2]
                intent_label_ids = batch[3]

                outputs = self.bert(input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
                intent_id = intent_label_ids[0]
                np_pooled_output = outputs[1].cpu().detach().numpy()
                np_embedding = np.concatenate((np_embedding, np_pooled_output))
                np_count = np.concatenate((np_count, np.array([intent_id])))

        print("OUTPUT")
        print(np_embedding)
        print(np_count)
        tsne_embedding = TSNE(n_components=2).fit_transform(np_embedding)
        plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=np_count)
        plt.savefig("tsne_" + self.args.task + ".png")
        np.savetxt("tsne_embedding_" + self.args.task + ".csv", tsne_embedding, delimiter=',')
        np.savetxt("label_" + self.args.task + ".csv", np_count, delimiter=',')
        c_count = np.zeros(n_classes)
        tsne_2 = np.zeros((n_classes,2))
        for x,y in zip(tsne_embedding,np_count):
            c_count[y] += 1
            tsne_2[y] = tsne_2[y] + x

        for i in range(n_classes):
            if c_count[i] != 0:
                tsne_2[i] = tsne_2[i] / c_count[i]
        with open("sim_"+self.args.task + ".csv","a") as fp_sim:
            for i in range(n_classes):
                for j in range(n_classes):
                    cos_sim = np.dot(tsne_2[i], tsne_2[j]) / (np.sqrt(np.dot(tsne_2[i], tsne_2[i])) * np.sqrt(np.dot(tsne_2[j], tsne_2[j])))
                    n_cos_sim = (cos_sim+1)/2
                    fp_sim.write(("{},{},{}".format(i, j, n_cos_sim)))
                    fp_sim.write("\n")

        # tensor_embedding = tensor_embedding/tensor_count


def main(args):
    set_seed(args)
    tokenizer = load_tokenizer(args)
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    intent_similarity = IntentSimilarityTSNE(args, train_dataset=train_dataset)
    intent_similarity.compute_similarity()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignore when calculate loss)")
    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
