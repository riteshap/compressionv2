
import os.path
import nltk
import torch
from nltk.tokenize import word_tokenize
from transformers import logging
from collections import Counter
from transformers import pipeline
from sentence_transformers import SentenceTransformer


class CompressV2:
    def __init__(self, mood_root):
        self.model_root = mood_root
        self.model_name = "Llama-3.2-3B-Instruct"
        self.model_path = os.path.join(self.model_root, self.model_name)
        self.compress_turn = 0

        self.qa_model_name = os.path.join(self.model_root, r"multi-qa-mpnet-base-cos-v1")
        self.similarity_model_name = os.path.join(self.model_root, r"all-mpnet-base-v2")
        self.device_map = "cuda"

        self.rate = None
        self.pipe = None
        self.device = None
        self.qa_model = None
        self.similarity_model = None
        self.check_flag = False
        logging.set_verbosity_error()

    def prepare_load(self):
        self.load_llama_model(self.model_path, self.device_map)
        self.qa_model = SentenceTransformer(self.qa_model_name)
        self.similarity_model = SentenceTransformer(self.similarity_model_name)

    def load_llama_model(self, model_name: str, device_map: str = "cuda"):
        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )

        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.pipe = pipe

    def cut_sentence(self, content: list[str]):
        content = [cont.replace("\n", ". ") for cont in content]

        def sync_sentence(sentences, text):
            seen_text = 0
            sentence_num = len(sentences)
            new_sentences = []
            for i, s in enumerate(sentences):
                assert s == text[seen_text: seen_text + len(s)]
                if i == sentence_num - 1:
                    new_sentences.append(text[seen_text:])
                    break
                next_sentence_start = text.find(
                    sentences[i + 1][:5], seen_text + len(s)
                )
                new_sentences.append(text[seen_text:next_sentence_start])
                seen_text = next_sentence_start
            if "".join(new_sentences) != text:
                print("*******")
                print("".join(new_sentences))
                print("*******")
                print(text)
                raise Exception("different")
            return new_sentences

        sentences = [nltk.sent_tokenize(c) for c in content]
        sentences = [sync_sentence(s, c) for s, c in zip(sentences, content)]

        return sentences

    def get_sentence_distance(self, model_name, corpus: list[str], query: str):
        model = model_name
        query_embedding = model.encode(query)
        passage_embeddings = model.encode(corpus)
        similarities = model.similarity(query_embedding, passage_embeddings)
        return similarities

    def combine_sentence(self, sentences: list[str], question: str, max_len=200):
        def combine(cb_model, i):
            add_num = 0
            max_score = 0
            while True:
                aim_sent = [" ".join(sentences[i: i + add_num + 1])]
                if len(word_tokenize(aim_sent[0])) > max_len:
                    break
                score = self.get_sentence_distance(cb_model, aim_sent, question)[0]
                if score > max_score:
                    max_score = score
                    add_num += 1
                else:
                    break
            return add_num

        if self.compress_turn == 0:
            cb_model = self.qa_model
        else:
            cb_model = self.qa_model
        unit_list = []
        skip_num = 0
        for i, sent in enumerate(sentences):
            if skip_num > 0:
                skip_num -= 1
                continue
            if len(word_tokenize(sent)) >= max_len:
                unit_list.append([sent])
                continue
            skip_num = combine(cb_model, i) - 1
            unit_list.append(sentences[i: i + skip_num + 1])

        return unit_list

    def prompts_0(self, unit_list: list[list[str]], question: str, batch_size=50):
        batch_list = []
        for unit in unit_list:
            long_sent = " ".join(unit)
            prompts = ("document:{}\nquestion:{}\n"
                       "Does the document is useful to answering the question?"
                       " yes or no or not sure\nanswer:").format(long_sent, question)
            batch_list.append(prompts)
        # print("batch:{}".format(len(batch_list)))
        new_unit_list_y = []
        new_unit_list_ns = []
        for i in range(len(batch_list) // batch_size + int(len(batch_list) % batch_size != 0)):
            batch_ans = self.pipe(batch_list[i * batch_size: (i + 1) * batch_size],
                                  max_new_tokens=2,
                                  continue_final_message=True,
                                  return_full_text=False,
                                  pad_token_id=None,
                                  eos_token_id=None)
            for j, ans in enumerate(batch_ans):
                if "yes" in ans[0]["generated_text"].lower():
                    new_unit_list_y.append(unit_list[i * batch_size + j])
                elif "not sure" in ans[0]["generated_text"].lower():
                    new_unit_list_ns.append(unit_list[i * batch_size + j])
                elif "no" in ans[0]["generated_text"].lower():
                    pass
                else:
                    new_unit_list_ns.append(unit_list[i * batch_size + j])
        return new_unit_list_y, new_unit_list_ns

    def extract_0(self, unit_list: list[list[str]], question, max_len=200, threshold=0.8):
        def fun_0(ans_list: list[str]):
            emp_ans = []
            for i, aa in enumerate(ans_list):
                aac = word_tokenize(aa.lower())
                save_flag = False
                for bb in emp_ans:
                    bbc = word_tokenize(bb.lower())
                    same_words_num = sum((Counter(aac) & Counter(bbc)).values())
                    ratio_a = same_words_num / len(aac)
                    if threshold <= ratio_a <= 1.0:
                        save_flag = True
                        break
                if not save_flag:
                    emp_ans.append(aa)
            return emp_ans

        def fun_1(ans_list: list[str], sent_str: str):
            emp_ans = []
            for aa in ans_list:
                aac = word_tokenize(aa.lower())
                same_words_num = sum((Counter(aac) & Counter(word_tokenize(sent_str.lower()))).values())
                ratio_a = same_words_num / len(aac)
                if threshold <= ratio_a <= 1.0:
                    emp_ans.append(aa)
            return emp_ans

        new_unit_list = []
        for unit in unit_list:
            long_sent = " ".join(unit)
            max_new_tokens = len(word_tokenize(long_sent))
            if max_new_tokens < max_len:
                new_unit_list.append(unit)
                continue
            prompts = ("document:{}\nquestion:{}\n"
                       "Copy sentences from the document that are useful in answering this question\n"
                       .format(long_sent, question))
            ans = self.pipe(prompts,
                            max_new_tokens=max_new_tokens,
                            continue_final_message=True,
                            return_full_text=False)

            ans = [a.strip() for a in ans[0]["generated_text"].split("\n") if len(a.strip()) > 0]
            new_ans = []
            for an in ans:
                sent = self.cut_sentence([an])[0]
                new_ans.extend([a for a in sent])
            ans = new_ans

            new_ans = fun_0(ans)
            ans = new_ans

            new_ans = fun_1(ans, long_sent)

            new_unit_list.append(new_ans)

        return new_unit_list

    def get_target_token(self, context: list[str], llm_limit=None):
        if llm_limit is None:
            llm_limit = 10000000

        alpha = 1.0 if self.compress_turn == 0 else 1.0
        ori_len = sum([len(a) for a in context])
        target_token = min(llm_limit, int(self.rate * alpha * ori_len))
        return target_token

    def sort_and_remove_min_values(self, similarities, sim_lens, target_token):

        indexed_lst = list(enumerate(similarities))

        sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)

        remaining_lst = []
        remain_lens = 0
        for ele in sorted_lst:
            remain_lens += sim_lens[ele[0]][0]
            remaining_lst.append(ele)
            if remain_lens > target_token:
                break

        remaining_lst.sort(key=lambda x: x[0])
        return remaining_lst

    def get_round_0_answer(self, prompts):
        ans = self.pipe(prompts,
                        max_new_tokens=30,
                        continue_final_message=True,
                        return_full_text=False,
                        pad_token_id=None,
                        eos_token_id=None,
                        num_return_sequences=1
                        )

        return ans[0]["generated_text"]

    def check_answer_simlarity(self, answer: list[list[str]], sentences: list[str], similarities):
        new_answer = {}
        for words in answer:
            for word in words:
                if word not in new_answer:
                    new_answer[word + "bbb"] = "0:0"
                    new_answer[word] = "0:0"
                    new_answer[word + "fff"] = "0:0"
        for i, (sen, sim) in enumerate(zip(sentences, similarities)):
            for word in new_answer.keys():
                if word.endswith("bbb") or word.endswith("fff"):
                    continue
                try:
                    if word.lower() in sen.lower() and sim > float(new_answer[word].split(":")[0]):
                        new_answer[word + "bbb"] = str(similarities[i - 1]) + ": " + sentences[i - 1]
                        new_answer[word] = str(sim) + ": " + sen
                        new_answer[word + "fff"] = str(similarities[i + 1]) + ": " + sentences[i + 1]
                except:
                    print("error" + word)

        for a in new_answer.keys():
            print("{}: {}".format(a, new_answer[a]))

    def compress_v2(self,
                    context: list[str],
                    question: str,
                    rate: float,
                    max_len=200,
                    threshold=0.8,
                    llm_limit=None,
                    extract_flag=True
                    ):
        if rate is not None and type(rate) is float:
            self.rate = rate
        if self.pipe is None:
            self.prepare_load()

        sentences_0 = self.cut_sentence(context)
        sentences = [sen for li in sentences_0 for sen in li]
        question = question + "?" if not question.strip().endswith("?") else question
        round_1_ans = ""
        ret_content0 = ""
        prompts_0 = ""
        for i in range(2):
            self.compress_turn = i

            target_token = self.get_target_token(context, llm_limit)

            unit_list = self.combine_sentence(sentences, question + round_1_ans, max_len)

            input_question = question if self.compress_turn == 0 \
                else question + "\nTips:{}".format(round_1_ans.replace("\n", ""))

            new_unit_list_y, new_unit_list_ns = self.prompts_0(unit_list, input_question)

            if extract_flag:
                new_unit_list_y = self.extract_0(new_unit_list_y, input_question, max_len=max_len, threshold=threshold)

            new_unit_list_y.extend(new_unit_list_ns)
            unit_sentences = [" ".join(aa) for aa in new_unit_list_y]

            model_name = self.qa_model if self.compress_turn == 0 else self.qa_model
            similarities = self.get_sentence_distance(model_name, unit_sentences, question + round_1_ans)
            similarities = similarities.numpy()[0]

            sen_lens = [len(a) for a in unit_sentences]

            sim_lens = [(a, b) for a, b in zip(sen_lens, similarities)]

            remaining_lst = self.sort_and_remove_min_values(similarities, sim_lens, target_token)

            ret_content = " ".join([unit_sentences[i] for i, _ in remaining_lst])

            prompts = ("Document:{}\nQuestion:{}\nanswer for the given Question "
                       "using only the Document:\n").format(ret_content, question)

            if self.compress_turn == 0:
                print("get round0 answer")
                ret_content0 = ret_content
                prompts_0 = prompts
                round_1_ans = self.get_round_0_answer(prompts)
                print(round_1_ans)
            else:
                return ret_content0, ret_content, prompts, prompts_0


if __name__ == "__main__":
    ss = CompressV2()
