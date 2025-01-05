# 处理NQ数据集，清除html标签
import os.path
import json
import re


class NQDataset:
    def __init__(self):
        self.dev_data_path = "./NQdataset/v1.0-simplified_nq-dev-all.jsonl"
        self.save_root_path = "./data_test"

    def original_preprocess(self, ori_file_name, limit_num=5):
        from NQdataset.natural_questions_master.text_utils import simplify_nq_example
        file = open(self.dev_data_path, "r", encoding="utf-8")
        line_num = 0
        save_path = os.path.join(self.save_root_path, ori_file_name)
        n_file = open(save_path, "w", encoding="utf-8")
        for line in file:
            a_sample = json.loads(line)
            simplified_nq_example = simplify_nq_example(a_sample)
            line_num += 1
            simplified_nq_example = json.dumps(simplified_nq_example)
            n_file.write(simplified_nq_example + "\n")
            if line_num > limit_num:
                break
        n_file.close()

    def fun0(self, ori_name, out_name, limit_num=5):
        ori_path = os.path.join(self.save_root_path, ori_name)
        file = open(ori_path, "r", encoding="utf-8")
        new_sample_list = []
        line_num = 0
        for line in file:
            line_num += 1
            a_sample = json.loads(line)
            document_text = a_sample["document_text"]
            question_text = a_sample["question_text"]
            annotations = a_sample["annotations"]
            doculent_url = a_sample["document_url"]
            example_id = a_sample["example_id"]
            tokens = document_text.split(" ")

            short_answers = []
            for anno in annotations:
                if len(anno["short_answers"]) > 0:
                    for s_a in anno["short_answers"]:
                        short_answers.append(tokens[s_a["start_token"]: s_a["end_token"]])

            remove_tage_list = ["<H1>", "</H1>", "<Table>", "</Table>", "<Tr>", "</Tr>",
                                "<Td>", "</Td>", "<Li>", "</Li>", "<Ul>", "</Ul>",
                                "<Th>", "</Th>", "<P>", "</P>", "<H2>", "</H2>",
                                "<H3>", "</H3>"]
            n_tokens = []
            j = 0
            index_dic = {}
            for i, t in enumerate(tokens):

                if t not in remove_tage_list and not t.startswith("<") and not t.endswith(">"):

                    n_t = re.sub(r"[\u200B-\u200F]", "", t)
                    if len(n_t) == 1 and ord(n_t) >= 128:
                        n_t = ""
                    if len(n_t) > 0:
                        n_tokens.append(n_t)
                        j += 1
                        add_index = -1
                    else:
                        add_index = 0
                else:
                    add_index = 0
                index_dic[str(i)] = j + add_index

            for anno in annotations:
                if anno["long_answer"]["end_token"] != -1:
                    anno["long_answer"]["end_token"] = index_dic[str(anno["long_answer"]["end_token"])]
                    anno["long_answer"]["start_token"] = index_dic[str(anno["long_answer"]["start_token"])]
                    for a in anno["short_answers"]:
                        a["end_token"] = index_dic[str(a["end_token"])]
                        a["start_token"] = index_dic[str(a["start_token"])]

            short_answers_2 = []
            for anno in annotations:
                if len(anno["short_answers"]) > 0:
                    for s_a in anno["short_answers"]:
                        short_answers_2.append(n_tokens[s_a["start_token"]: s_a["end_token"]])

            n_sample = {}
            a_sample["document_text"] = " ".join(n_tokens)
            n_sample["document_text"] = a_sample["document_text"]
            n_sample["annotations"] = a_sample["annotations"]
            n_sample["question_text"] = a_sample["question_text"]

            new_sample_list.append(n_sample)
            if line_num > limit_num:
                break
        out_path = os.path.join(self.save_root_path, out_name)
        new_file = open(out_path, "w", encoding="utf-8")
        for line in new_sample_list:
            line = json.dumps(line)
            new_file.write(line + "\n")
        new_file.close()

    def read_data(self, file_name):
        file_path = os.path.join(self.save_root_path, file_name)
        file = open(file_path, "r", encoding="utf-8")
        for line in file:
            a_sample = json.loads(line)
            document_text = a_sample["document_text"]
            question_text = a_sample["question_text"]
            annotations = a_sample["annotations"]
            try:
                start = annotations[0]["short_answers"][0]["start_token"]
                end = annotations[0]["short_answers"][0]["end_token"]
                short_answer = document_text.split(" ")[start:end]
                print(question_text)
                print(" ".join(short_answer))
            except Exception:
                pass

    def run_preprocess(self):
        ori_file_name = "ori_nq.json"
        self.original_preprocess(ori_file_name, 5)
        dev_file_name = "nq_test_data.json"
        self.fun0(ori_file_name, dev_file_name, 5)


class TriviaDataset:
    def __init__(self):
        self.save_root_path = "./data_test"
        self.dataset_path = "./trivia_dataset/triviaqa-rc"
        self.web_dev_path = os.path.join(self.dataset_path, "qa/web-dev.json")
        self.wiki_dev_path = os.path.join(self.dataset_path, "qa/wikipedia-dev.json")
        self.web_evidence_path = os.path.join(self.dataset_path, "evidence/web")
        self.wiki_evidence_path = os.path.join(self.dataset_path, "evidence/wikipedia")

    def preprocess(self, save_file_name, data_type="web", limit_number=5):
        if data_type not in ["web", "wiki"]:
           raise Exception("type error")
        save_file_path = os.path.join(self.save_root_path, save_file_name)
        new_file = open(save_file_path, "w", encoding="utf-8")
        original_file = self.web_dev_path if data_type == "web" else self.wiki_dev_path
        file = json.load(open(original_file, "r", encoding="utf-8"))

        for i, sample in enumerate(file["Data"]):
            if i >= limit_number:
                break
            question = sample["Question"]
            answers = [a.split(" ") for a in sample["Answer"]["NormalizedAliases"]]
            if data_type == "web":
                document_paths = [{"Title": result["Title"],
                                   "name": os.path.join(self.web_evidence_path, result["Filename"])}
                                  for result in sample["SearchResults"]]
            else:
                document_paths = []

            document_paths_wiki = [{"Title": result["Title"],
                                    "name": os.path.join(self.wiki_evidence_path, result["Filename"].replace(":", "_"))}
                                   for result in sample["EntityPages"]]
            document_paths.extend(document_paths_wiki)

            new_line = json.dumps({"question": question,
                                   "answers": answers,
                                   "document_paths": document_paths})
            new_file.write(new_line + "\n")
        new_file.close()

    def run_preprocess(self, data_type="web"):
        if data_type == "web":
            self.preprocess("trivia_web.json", data_type="web")
        else:
            self.preprocess("trivia_wiki.json", data_type="wiki")


if __name__ == "__main__":
    class_nq = NQDataset()
    class_nq.run_preprocess()
    class_trivia = TriviaDataset()
    class_trivia.run_preprocess(data_type="web")
    class_trivia.run_preprocess(data_type="wiki")

    pass
