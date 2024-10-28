import collections
import random
import torch
import sys
import os

from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from io_new import tokenization
from common.tools import logger
from callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset


class InputExample(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, attention_mask, token_type_ids, labels, next_sentence_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.next_sentence_labels = next_sentence_labels


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, vocab_path, args):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path)
        self.args = args

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_masked_lm_data(self, tokens, masked_lm_prob, max_predictions_per_seq, vocab_words):
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[NAN]":
                continue
            cand_indexes.append(i)
        random.shuffle(cand_indexes)
        output_tokens = list(tokens)
        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))
        # 记录目前已添加掩码的数量
        masked_lms = []
        # 记录目前已被掩码的token在tokens数组中的索引
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                # 掩码的数量够了就break
                break
            if random.random() < 0.8:
                # [mask]掩盖
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    # 原值
                    masked_token = tokens[index_set]
                    # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]
            output_tokens[index_set] = masked_token
            masked_lms.append(MaskedLmInstance(index=int(index_set), label=tokens[index_set]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)
        return (output_tokens, masked_lm_positions, masked_lm_labels)

    def create_examples_from_document(self, all_documents, document_index, max_seq_length, short_seq_prob,
                                      masked_lm_prob, max_predictions_per_seq, vocab_words):
        document = all_documents[document_index]
        max_num_tokens = max_seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    tokens_b = []
                    is_random_next = False
                    # len(current_chunk)一直都没变，这里指的是document只有一句无法分割为两个句子，或者概率<0.5，在 50% 的情况下构建随机句子对
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)
                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break
                        random_document = all_documents[random_document_index]
                        # 生成符合索引条件的随机整数
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    # 抛弃a b中较长的句子超长的部分
                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    tokens = []
                    segment_ids = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)

                    tokens.append("[SEP]")
                    segment_ids.append(0)

                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    segment_ids.append(1)

                    # masked工作
                    # 现在tokens是一个token数组，包含了a b句和[CLS]，[SEP]，[SEP]，以[SEP]为句子结尾
                    (output_tokens, masked_lm_positions, masked_lm_labels) = self.create_masked_lm_data(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words)

                    ### process to examples
                    input_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
                    input_mask = [1] * len(input_ids)
                    assert len(input_ids) <= max_seq_length
                    # 对于过于短的，给其填充到max_seq_length，但是mask为0（为什么要填充到max_seq_length）
                    while len(input_ids) < max_seq_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                    assert len(input_ids) == max_seq_length
                    assert len(input_mask) == max_seq_length
                    assert len(segment_ids) == max_seq_length
                    next_sentence_label = 1 if is_random_next else 0
                    labels = [-100] * len(input_ids)
                    assert len(masked_lm_positions) == len(masked_lm_labels)
                    for i in range(len(masked_lm_positions)):
                        labels[masked_lm_positions[i]] = self.tokenizer.convert_token_to_id(masked_lm_labels[i])

                    instance = InputExample(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=labels,
                        next_sentence_labels=next_sentence_label
                    )
                    # 一个instance是一个包含了a,b句子对和掩码的训练数据，从一个document中可以得到所有instance，其中document中的每一个句子都在a句中作为组成部分。
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

    def get_documents(self, input_file):
        all_documents = [[]]
        with open(input_file, "r") as reader:
            while True:
                # 处理utf-8编码
                line = reader.readline()
                if not line:
                    break
                # 去除行空格
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    # 开启下一个document
                    all_documents.append([])
                # 行分词
                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    # 往最后一个document对应的数组里添加行对应tokens
                    all_documents[-1].append(tokens)
        all_documents = [x for x in all_documents if x]
        random.shuffle(all_documents)
        return all_documents

    def create_examples(self, all_documents, cached_examples_file):
        '''
        Creates examples for data
        '''
        # load examples from cache.
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            vocab_words = list(self.tokenizer.vocab.keys())
            # masked and get the two sentences
            for i in range(self.args.dupe_factor):
                pbar = ProgressBar(n_total=len(all_documents))
                for document_index in range(len(all_documents)):
                    examples.extend(
                        self.create_examples_from_document(all_documents, document_index, self.args.train_max_seq_len,
                                                           self.args.short_seq_prob, self.args.masked_lm_prob,
                                                           self.args.max_predictions_per_seq, vocab_words))
                    pbar.batch_step(step=i, info={}, bar_type='create examples')
            random.shuffle(examples)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_dataset(self, examples):
        print("run create_dataset function")
        all_input_ids = torch.tensor([f.input_ids for f in examples], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in examples], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in examples], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in examples], dtype=torch.long)
        all_next_sentence_labels = torch.tensor([f.next_sentence_labels for f in examples])
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_next_sentence_labels)
        return dataset

    def create_train_dataset(self, examples):
        all_input_ids = [f.input_ids for f in examples]
        all_attention_mask = [f.attention_mask for f in examples]
        all_token_type_ids = [f.token_type_ids for f in examples]
        all_labels = [f.labels for f in examples]
        all_next_sentence_labels = [f.next_sentence_labels for f in examples]

        # 创建字典
        data_dict = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'token_type_ids': all_token_type_ids,
            'labels': all_labels,
            'next_sentence_labels': all_next_sentence_labels,
        }

        # 使用 datasets.Dataset.from_dict() 创建数据集
        dataset = Dataset.from_dict(data_dict)
        return dataset
